# main.py
"""
Railway-ready Shoonya paper-trading bot entrypoint.

Usage:
- Put sensitive keys in Railway environment variables (or .env for local testing)
- Deploy on Railway with Procfile: `worker: python main.py`
- Optional: include a sim_ticks.csv for offline PAPER testing (columns: timestamp,symbol,ltp)
"""

import os
import json
import time
import logging
import bisect
from datetime import datetime, timedelta
from email.mime.text import MIMEText
import smtplib

# third-party imports (in requirements.txt)
try:
    import numpy as np
    import pandas as pd
    import pyotp
    from NorenRestApiPy.NorenApi import NorenApi
except Exception:
    # If imports fail, we keep going — PAPER mode with CSV simulator will work.
    np = None
    pd = None
    pyotp = None
    NorenApi = None

# -------------------------
# Logging
# -------------------------
LOGFILE = os.getenv("LOGFILE", "strategy.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOGFILE), logging.StreamHandler()],
)

# -------------------------
# Load config from env
# -------------------------
def env(key, default=None, cast=str):
    v = os.getenv(key)
    if v is None:
        return default
    try:
        if cast == bool:
            return v.lower() in ("1", "true", "yes", "y")
        return cast(v)
    except Exception:
        return default

CONFIG = {
    "USER_ID": env("USER_ID", None, str),
    "PASSWORD": env("PASSWORD", None, str),
    "TOTP_SECRET": env("TOTP_SECRET", "", str),
    "API_KEY": env("API_KEY", None, str),
    "VC": env("VC", None, str),
    "LOT_SIZE": env("LOT_SIZE", 75, int),
    "MARKET_OPEN_HOUR": env("MARKET_OPEN_HOUR", 9, int),
    "MARKET_OPEN_MIN": env("MARKET_OPEN_MIN", 15, int),
    "MARKET_CLOSE_HOUR": env("MARKET_CLOSE_HOUR", 15, int),
    "MARKET_CLOSE_MIN": env("MARKET_CLOSE_MIN", 30, int),
    "EMAIL_SENDER": env("EMAIL_SENDER", None, str),
    "EMAIL_PASSWORD": env("EMAIL_PASSWORD", None, str),
    "EMAIL_RECEIVER": env("EMAIL_RECEIVER", None, str),
    "EXPIRY_DAY": env("EXPIRY_DAY", 1, int),  # default Thursday equivalent per earlier code
    "PAPER_TRADE": env("PAPER_TRADE", "true", str).lower() in ("1", "true", "yes", "y"),
    "MARGIN_PER_LOT": env("MARGIN_PER_LOT", 300000, int),
    "RISK_PERCENT": env("RISK_PERCENT", 1.8, float),
    "SELECTED_LOTS": env("SELECTED_LOTS", 1, int),
    "SIM_TICKS_CSV": env("SIM_TICKS_CSV", "sim_ticks.csv", str),
}

logging.info("Config loaded (sensitive values hidden in logs).")

# -------------------------
# State files (persist in container)
# -------------------------
STATE_FILE = os.path.join(os.getcwd(), "strategy_state.json")
MONTHLY_PNL_FILE = os.path.join(os.getcwd(), "monthly_pnl.json")


def convert_to_serializable(obj):
    try:
        import numpy as _np
    except Exception:
        _np = None
    if _np and isinstance(obj, _np.integer):
        return int(obj)
    if _np and isinstance(obj, _np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    return obj


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(convert_to_serializable(state), f, indent=4)


def load_state():
    if not os.path.exists(STATE_FILE) or os.path.getsize(STATE_FILE) == 0:
        return {
            "position_open": False,
            "entry_time": None,
            "buy_ce_strike": None,
            "buy_pe_strike": None,
            "short_ce_strike": None,
            "short_pe_strike": None,
            "buy_ce_prem": None,
            "buy_pe_prem": None,
            "current_expiry": None,
            "entry_ce_ltp": None,
            "entry_short_ce_ltp": None,
            "entry_pe_ltp": None,
            "entry_short_pe_ltp": None,
            "cooldown_end": None,
            "trades": [],
            "realized_mtm": 0.0,
        }
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        logging.error("Failed to load state: %s", e)
        return load_state_default()


def save_monthly_pnl(monthly_pnl):
    with open(MONTHLY_PNL_FILE, "w") as f:
        json.dump(monthly_pnl, f, indent=4)


def load_monthly_pnl():
    if not os.path.exists(MONTHLY_PNL_FILE) or os.path.getsize(MONTHLY_PNL_FILE) == 0:
        return {}
    try:
        with open(MONTHLY_PNL_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        logging.error("Failed to load monthly pnl: %s", e)
        return {}


# -------------------------
# Email helper
# -------------------------
def send_email(subject, body):
    sender = CONFIG.get("EMAIL_SENDER")
    pwd = CONFIG.get("EMAIL_PASSWORD")
    recv = CONFIG.get("EMAIL_RECEIVER")
    if not (sender and pwd and recv):
        logging.debug("Email credentials missing — skipping email.")
        return
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = recv
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender, pwd)
            server.send_message(msg)
        logging.info("Email sent: %s", subject)
    except Exception as e:
        logging.error("Failed to send email: %s", e)


# -------------------------
# Shoonya API initialization (optional)
# -------------------------
api = None
if NorenApi and CONFIG.get("API_KEY") and CONFIG.get("USER_ID"):
    try:
        api = NorenApi(
            host="https://api.shoonya.com/NorenWClientTP/",
            websocket="wss://api.shoonya.com/NorenWSTP/",
        )
        logging.info("NorenApi client initialized.")
    except Exception as e:
        logging.error("Failed to init NorenApi: %s", e)
        api = None
else:
    logging.info("NorenApi not initialized (missing library or credentials).")


# -------------------------
# Optional CSV LTP simulator for PAPER mode
# sim_ticks.csv columns: timestamp (ISO), symbol, ltp
# -------------------------
_sim_index = {}
_sim_loaded = False


def load_sim_ticks(path):
    global _sim_loaded, _sim_index
    _sim_index = {}
    _sim_loaded = False
    if not pd:
        logging.debug("Pandas not available — skipping sim ticks load.")
        return
    if not os.path.exists(path):
        logging.info("Sim ticks file not found: %s", path)
        return
    try:
        df = pd.read_csv(path, parse_dates=["timestamp"])
        df.sort_values("timestamp", inplace=True)
        for sym, g in df.groupby("symbol"):
            times = list((g["timestamp"].astype("int64") // 10 ** 9).tolist())
            ltps = list(g["ltp"].astype(float).tolist())
            _sim_index[sym] = {"times": times, "ltps": ltps}
        _sim_loaded = True
        logging.info("Sim ticks loaded for symbols: %s", list(_sim_index.keys()))
    except Exception as e:
        logging.error("Failed to load sim ticks: %s", e)


def get_sim_ltp(symbol, now_ts=None):
    if not _sim_loaded or symbol not in _sim_index:
        return None
    if now_ts is None:
        now_ts = int(datetime.now().timestamp())
    else:
        now_ts = int(now_ts.timestamp())
    arr = _sim_index[symbol]["times"]
    idx = bisect.bisect_right(arr, now_ts) - 1
    if idx < 0:
        return None
    return float(_sim_index[symbol]["ltps"][idx])


# Load sim ticks if PAPER and CSV exists
if CONFIG.get("PAPER_TRADE"):
    load_sim_ticks(CONFIG.get("SIM_TICKS_CSV", "sim_ticks.csv"))


# -------------------------
# LTP fetch wrapper
# -------------------------
def get_shoonya_ltp(tsym, retries=2, delay=1):
    """
    Attempts:
      1) If PAPER and sim loaded -> return sim LTP
      2) If NorenApi available -> query API
      3) else return None
    """
    # 1) sim mode
    if CONFIG.get("PAPER_TRADE") and _sim_loaded:
        val = get_sim_ltp(tsym)
        if val is not None:
            return val

    # 2) try live API
    if api is None:
        logging.debug("No API client available for LTP: %s", tsym)
        return None

    for attempt in range(retries):
        try:
            time.sleep(delay)
            res = api.searchscrip(exchange="NFO", searchtext=tsym)
            if res and res.get("values"):
                token = res["values"][0]["token"]
                quote = api.get_quotes(exchange="NFO", token=token)
                if quote and "lp" in quote:
                    return float(quote["lp"])
        except Exception as e:
            logging.error("LTP attempt %d failed for %s: %s", attempt + 1, tsym, e)
            if attempt == retries - 1 and ("auth" in str(e).lower() or "session" in str(e).lower()):
                logging.info("Session expired — attempting re-login.")
                login_shoonya()
        time.sleep(delay * (attempt + 1))
    return None


# -------------------------
# Option chain builder
# -------------------------
def get_option_chain(current_expiry=None):
    future_symbols = ["NIFTY28OCT25F", "NIFTY25NOV25F", "NIFTY30DEC25F"]
    future_ltp = None
    for sym in future_symbols:
        future_ltp = get_shoonya_ltp(sym)
        if future_ltp:
            break
    if not future_ltp:
        return None, None, None, None
    atm = round(future_ltp / 50) * 50
    expiry = current_expiry if current_expiry else datetime.now().strftime("%d-%b-%Y")
    strike_range = range(int(atm - 500), int(atm + 501), 50)
    ce_data = []
    pe_data = []
    for strike in strike_range:
        ce_sym = f"NIFTY {expiry} {strike} CE"
        pe_sym = f"NIFTY {expiry} {strike} PE"
        ce_ltp = get_shoonya_ltp(ce_sym)
        pe_ltp = get_shoonya_ltp(pe_sym)
        if ce_ltp is not None:
            ce_data.append({"strike": strike, "ltp": float(ce_ltp)})
        if pe_ltp is not None:
            pe_data.append({"strike": strike, "ltp": float(pe_ltp)})
    # Use pandas if available, else simple lists
    if pd:
        df_ce = pd.DataFrame(ce_data)
        df_pe = pd.DataFrame(pe_data)
    else:
        df_ce = ce_data
        df_pe = pe_data
    return future_ltp, df_ce, df_pe, expiry


# -------------------------
# Utils
# -------------------------
def get_atm_strike(spot):
    return round(spot / 50) * 50


def find_nearest_premium(df, target, direction):
    """df expected as pandas DataFrame with columns strike, ltp"""
    if not pd or df is None or len(df) == 0:
        return None, None
    if direction == "higher":
        cand = df[df["ltp"] >= target]
        if cand.empty:
            return None, None
        idx = cand.index.min()
    else:
        cand = df[df["ltp"] <= target]
        if cand.empty:
            return None, None
        idx = cand.index.max()
    return int(df.loc[idx, "strike"]), float(df.loc[idx, "ltp"])


def calculate_monthly_expiry():
    today = datetime.now()
    last_day = (today.replace(day=1) + timedelta(days=40)).replace(day=1) - timedelta(days=1)
    while last_day.weekday() != 1:
        last_day -= timedelta(days=1)
    return last_day.strftime("%d-%b-%Y")


# -------------------------
# Order placement simulator / live placeholder
# -------------------------
PAPER_TRADE = CONFIG.get("PAPER_TRADE", True)
LOT_SIZE = int(CONFIG.get("LOT_SIZE", 75))
RISK_PERCENT = float(CONFIG.get("RISK_PERCENT", 1.8))
MARGIN_PER_LOT = int(CONFIG.get("MARGIN_PER_LOT", 300000))
SELECTED_LOTS = int(CONFIG.get("SELECTED_LOTS", 1))

BUY_QTY = LOT_SIZE * SELECTED_LOTS
SELL_QTY = 2 * LOT_SIZE * SELECTED_LOTS
required_capital = MARGIN_PER_LOT * SELECTED_LOTS
max_loss_value = required_capital * (RISK_PERCENT / 100.0)

logging.info(
    "BOT PARAMS BUY_QTY=%s SELL_QTY=%s required_capital=%s max_loss_value=%s",
    BUY_QTY,
    SELL_QTY,
    required_capital,
    max_loss_value,
)


def place_order(symbol, action, qty, strike, option_type, expiry, price, state=None):
    order_desc = f"{'Paper' if PAPER_TRADE else 'Live'} {action} {qty} {symbol} {strike} {option_type} @ {price}"
    logging.info(order_desc)
    if PAPER_TRADE:
        sym_name = f"NIFTY {expiry} {strike} {option_type}"
        trade = {
            "symbol": sym_name,
            "strike": int(strike),
            "option_type": option_type,
            "action": "B" if action == "B" else "S",
            "qty": int(qty),
            "price": float(price),
            "filled_at": datetime.now().isoformat(),
        }
        if state is not None:
            trades = state.get("trades", [])
            trades.append(trade)
            state["trades"] = trades
            save_state(state)
        return trade
    else:
        # TODO: implement real order placement via api
        return None


# -------------------------
# MTM / Exit helpers
# -------------------------
def get_mtm(state, df_ce=None, df_pe=None):
    total_mtm = 0.0
    seen = set()
    trades = state.get("trades", []) if state else []
    for trade in trades:
        key = (trade["symbol"], trade["strike"], trade["option_type"])
        if key in seen:
            continue
        seen.add(key)
        qty = trade["qty"] if trade["action"] == "B" else -trade["qty"]
        entry = float(trade["price"])
        # Prefer df lookups (fast), else attempt live fetch
        ltp = None
        if pd and isinstance(df_ce, pd.DataFrame) and trade["option_type"] == "CE":
            row = df_ce[df_ce["strike"] == trade["strike"]]
            if not row.empty:
                ltp = float(row["ltp"].values[0])
        if pd and isinstance(df_pe, pd.DataFrame) and trade["option_type"] == "PE":
            row = df_pe[df_pe["strike"] == trade["strike"]]
            if not row.empty:
                ltp = float(row["ltp"].values[0])
        if ltp is None:
            ltp = get_shoonya_ltp(trade["symbol"])
        if ltp is None:
            logging.debug("LTP missing for %s — assuming 0", trade["symbol"])
            continue
        total_mtm += (ltp - entry) * qty
    return total_mtm


def exit_position(state, monthly_pnl):
    mtm = get_mtm(state)
    logging.info("Exit triggered. PNL: %s", mtm)
    send_email("Position Closed", f"PNL: {mtm}")
    current_month = datetime.now().strftime("%Y-%m")
    monthly_pnl[current_month] = monthly_pnl.get(current_month, 0) + mtm
    save_monthly_pnl(monthly_pnl)
    state["position_open"] = False
    state["cooldown_end"] = None
    save_state(state)


# -------------------------
# Login helper
# -------------------------
def login_shoonya():
    if api is None:
        logging.debug("No API client — skipping login.")
        return None
    try:
        totp = ""
        if CONFIG.get("TOTP_SECRET") and pyotp:
            try:
                totp = pyotp.TOTP(CONFIG.get("TOTP_SECRET")).now()
            except Exception as e:
                logging.debug("TOTP gen failed: %s", e)
        logging.info("Attempting Shoonya login...")
        ret = api.login(
            userid=CONFIG.get("USER_ID"),
            password=CONFIG.get("PASSWORD"),
            twoFA=totp,
            vendor_code=CONFIG.get("VC"),
            api_secret=CONFIG.get("API_KEY"),
            imei="cloudbot01",
        )
        logging.info("Login response: %s", ret)
        if ret and ret.get("stat") == "Ok":
            send_email("Shoonya Login", f"Logged in at {datetime.now()}")
            return ret
        else:
            logging.warning("Login failed or returned unexpected response.")
            send_email("Login Failed", f"Response: {ret}")
            return ret
    except Exception as e:
        logging.error("Login error: %s", e)
        send_email("Login Error", str(e))
        return None


# -------------------------
# Market open helper
# -------------------------
def is_market_open():
    now = datetime.now().time()
    open_t = datetime.strptime(
        f"{CONFIG.get('MARKET_OPEN_HOUR')}:{CONFIG.get('MARKET_OPEN_MIN')}", "%H:%M"
    ).time()
    close_t = datetime.strptime(
        f"{CONFIG.get('MARKET_CLOSE_HOUR')}:{CONFIG.get('MARKET_CLOSE_MIN')}", "%H:%M"
    ).time()
    return open_t <= now <= close_t


# -------------------------
# Main loop
# -------------------------
def main():
    logging.info("Starting Shoonya Bot (cloud-ready). PAPER_TRADE=%s", PAPER_TRADE)
    if api is not None:
        login_shoonya()

    state = load_state()
    monthly_pnl = load_monthly_pnl()

    while True:
        try:
            now = datetime.now()
            if not is_market_open():
                logging.debug("Market closed. Sleeping 30s.")
                time.sleep(30)
                save_state(state)
                continue

            # Daily MTM notifications
            if now.hour == CONFIG.get("MARKET_OPEN_HOUR") and now.minute < 5:
                mtm = get_mtm(state)
                send_email("Daily MTM (Market Open)", f"Current MTM: {mtm:.2f}")
                logging.info("MTM Open: %s", mtm)
            if now.hour == CONFIG.get("MARKET_CLOSE_HOUR") - 1 and now.minute >= 55:
                mtm = get_mtm(state)
                send_email("Daily MTM (Before Close)", f"Current MTM: {mtm:.2f}")
                logging.info("MTM CloseWarn: %s", mtm)

            spot, df_ce, df_pe, expiry = get_option_chain(state.get("current_expiry"))
            if not all([spot is not None, df_ce is not None, df_pe is not None, expiry]):
                logging.debug("Option chain incomplete — sleeping 30s")
                time.sleep(30)
                continue

            remainder = int(spot) % 100
            if not (35 <= remainder <= 65):
                logging.debug("Spot %s not in zone — sleeping 30s", spot)
                time.sleep(30)
                continue

            exp_date = datetime.strptime(expiry, "%d-%b-%Y")
            days_to_exp = (exp_date - now).days

            # Expiry roll
            if state.get("position_open") and days_to_exp < 7:
                old = state.get("current_expiry")
                state["current_expiry"] = calculate_monthly_expiry()
                save_state(state)
                logging.info("Expiry shifted from %s to %s", old, state["current_expiry"])
                send_email("Expiry Shifted", f"Expiry shifted from {old} to {state['current_expiry']}")

            # Entry logic
            if not state.get("position_open"):
                atm = get_atm_strike(spot)
                state["buy_ce_strike"] = int(atm - 50)
                state["buy_pe_strike"] = int(atm + 50)

                # Data rows
                if pd:
                    ce_row = df_ce[df_ce["strike"] == state["buy_ce_strike"]]
                    pe_row = df_pe[df_pe["strike"] == state["buy_pe_strike"]]
                else:
                    ce_row = [r for r in df_ce if r["strike"] == state["buy_ce_strike"]]
                    pe_row = [r for r in df_pe if r["strike"] == state["buy_pe_strike"]]

                if (pd and (ce_row.empty or pe_row.empty)) or (not pd and (len(ce_row) == 0 or len(pe_row) == 0)):
                    logging.debug("Required strikes not found — sleeping")
                    time.sleep(30)
                    continue

                buy_ce_prem = float(ce_row["ltp"].values[0]) if pd else float(ce_row[0]["ltp"])
                buy_pe_prem = float(pe_row["ltp"].values[0]) if pd else float(pe_row[0]["ltp"])
                state["buy_ce_prem"] = buy_ce_prem
                state["buy_pe_prem"] = buy_pe_prem

                half_ce = buy_ce_prem / 2.0
                half_pe = buy_pe_prem / 2.0

                short_ce_strike, _ = find_nearest_premium(df_ce, half_ce, "higher")
                short_pe_strike, _ = find_nearest_premium(df_pe, half_pe, "lower")

                if not (short_ce_strike and short_pe_strike):
                    logging.debug("Short strikes not found — sleeping")
                    time.sleep(30)
                    continue

                state["short_ce_strike"] = int(short_ce_strike)
                state["short_pe_strike"] = int(short_pe_strike)
                state["entry_ce_ltp"] = buy_ce_prem
                state["entry_pe_ltp"] = buy_pe_prem

                # record short entry ltps from df
                state["entry_short_ce_ltp"] = float(df_ce[df_ce["strike"] == state["short_ce_strike"]]["ltp"].values[0]) if pd else None
                state["entry_short_pe_ltp"] = float(df_pe[df_pe["strike"] == state["short_pe_strike"]]["ltp"].values[0]) if pd else None

                # Place simulated orders (paper)
                place_order("NIFTY", "B", BUY_QTY, state["buy_ce_strike"], "CE", expiry, state["entry_ce_ltp"], state)
                place_order("NIFTY", "S", SELL_QTY, state["short_ce_strike"], "CE", expiry, state["entry_short_ce_ltp"], state)
                place_order("NIFTY", "B", BUY_QTY, state["buy_pe_strike"], "PE", expiry, state["entry_pe_ltp"], state)
                place_order("NIFTY", "S", SELL_QTY, state["short_pe_strike"], "PE", expiry, state["entry_short_pe_ltp"], state)

                state["position_open"] = True
                state["entry_time"] = now.isoformat()
                state["current_expiry"] = expiry
                save_state(state)
                send_email("Position Opened", f"Entry at {now.strftime('%H:%M')}")
                logging.info("Position opened at %s", now)

            else:
                # Exit checks
                buy_ce_ltp = (
                    float(df_ce[df_ce["strike"] == state["buy_ce_strike"]]["ltp"].values[0])
                    if pd and not df_ce[df_ce["strike"] == state["buy_ce_strike"]].empty
                    else None
                )
                buy_pe_ltp = (
                    float(df_pe[df_pe["strike"] == state["buy_pe_strike"]]["ltp"].values[0])
                    if pd and not df_pe[df_pe["strike"] == state["buy_pe_strike"]].empty
                    else None
                )

                if buy_ce_ltp and buy_pe_ltp and buy_ce_ltp + buy_pe_ltp < 150:
                    send_email("Exit: Premium < 150", f"CE:{buy_ce_ltp}, PE:{buy_pe_ltp}")
                    exit_position(state, monthly_pnl)
                    continue

                mtm = get_mtm(state, df_ce, df_pe)
                if mtm <= -max_loss_value:
                    send_email(f"Exit: Loss >= {RISK_PERCENT}% of margin", f"MTM:{mtm:.2f}")
                    exit_position(state, monthly_pnl)
                    continue

                future_symbols = ["NIFTY28OCT25F", "NIFTY25NOV25F", "NIFTY30DEC25F"]
                future_price = None
                for sym in future_symbols:
                    future_price = get_shoonya_ltp(sym)
                    if future_price:
                        break

                if not future_price:
                    logging.warning("Unable to fetch future price for exit check.")
                    time.sleep(30)
                    continue

                if future_price >= state.get("short_ce_strike", 999999):
                    send_email("Exit: CE Breach", f"Future {future_price}>={state.get('short_ce_strike')}")
                    exit_position(state, monthly_pnl)
                    continue
                if future_price <= state.get("short_pe_strike", -1):
                    send_email("Exit: PE Breach", f"Future {future_price}<={state.get('short_pe_strike')}")
                    exit_position(state, monthly_pnl)
                    continue

                if (
                    datetime.now().weekday() == CONFIG.get("EXPIRY_DAY", 1)
                    and datetime.now().time() >= datetime.strptime("15:25", "%H:%M").time()
                ):
                    send_email("Auto Exit", "Expiry day auto-exit at 3:25 PM")
                    exit_position(state, monthly_pnl)
                    continue

            time.sleep(30)
            save_state(state)

        except Exception as e:
            logging.exception("Main loop error: %s", e)
            time.sleep(10)


if __name__ == "__main__":
    main()

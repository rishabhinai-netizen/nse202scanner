"""
Auto Scanner — Standalone script for GitHub Actions
=====================================================
Runs all daily strategies against NSE 500, saves signals to CSV.
No Streamlit dependency. Uses yfinance for data.

Usage:
    python auto_scanner.py                    # Scan + record signals
    python auto_scanner.py --track            # Update tracker (check SL/T1 hits)
    python auto_scanner.py --mode eod         # End-of-day full scan
    python auto_scanner.py --mode intraday    # Quick scan (fewer stocks)
"""

import pandas as pd
import numpy as np
import os, sys, json, csv, logging
from datetime import datetime, date, timedelta
from pathlib import Path
import argparse

# Add parent dir for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_engine import fetch_batch_daily, fetch_nifty_data, Indicators
from stock_universe import get_stock_universe, get_sector, NIFTY_50
from scanners import (
    STRATEGY_PROFILES, DAILY_SCANNERS,
    run_all_scanners, detect_market_regime
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

SIGNALS_DIR = Path(__file__).parent / "signals"
TRACKER_FILE = SIGNALS_DIR / "tracker.csv"

# ============================================================================
# SIGNAL RECORDING
# ============================================================================

def scan_and_record(mode="eod", universe="nifty500"):
    """Run all scanners, save unique signals to today's CSV."""
    SIGNALS_DIR.mkdir(exist_ok=True)
    today = date.today().isoformat()
    today_file = SIGNALS_DIR / f"{today}_signals.csv"
    
    log.info(f"=== Auto Scan: {mode} mode, universe={universe} ===")
    
    # Fetch data
    syms = get_stock_universe(universe)
    log.info(f"Fetching data for {len(syms)} stocks...")
    
    data = fetch_batch_daily(syms, "1y")
    log.info(f"Loaded {len(data)} stocks")
    
    nifty = fetch_nifty_data()
    if nifty is None:
        log.error("Failed to fetch Nifty data. Aborting.")
        return
    
    # Detect regime
    regime = detect_market_regime(nifty)
    log.info(f"Regime: {regime['regime_display']} ({regime['score']}/{regime['max_score']})")
    
    # Run all daily scanners (no regime blocking for recording — we want all signals)
    results = run_all_scanners(
        data, nifty, daily_only=True,
        regime=None,  # Don't block — record everything
        has_intraday=False,
        sector_rankings={},
        min_rs=0,  # Don't filter — record everything
    )
    
    total = sum(len(v) for v in results.values())
    log.info(f"Found {total} total signals")
    
    # Load existing signals for today (dedup)
    existing = set()
    if today_file.exists():
        df_existing = pd.read_csv(today_file)
        for _, row in df_existing.iterrows():
            existing.add(f"{row['Strategy']}_{row['Symbol']}")
    
    # Build new rows
    now_str = datetime.now().strftime("%H:%M")
    new_rows = []
    
    for strategy, signals in results.items():
        for r in signals:
            key = f"{strategy}_{r.symbol}"
            if key in existing:
                continue  # Already recorded today
            
            new_rows.append({
                "Date": today,
                "Time": now_str,
                "Strategy": strategy,
                "Strategy_Name": STRATEGY_PROFILES.get(strategy, {}).get("name", strategy),
                "Symbol": r.symbol,
                "Signal": r.signal,
                "CMP": round(r.cmp, 2),
                "Entry": round(r.entry, 2),
                "SL": round(r.stop_loss, 2),
                "T1": round(r.target_1, 2),
                "T2": round(r.target_2, 2),
                "RR": round(r.risk_reward, 1),
                "Confidence": r.confidence,
                "RS": round(r.rs_rating, 0),
                "Sector": r.sector or get_sector(r.symbol),
                "Regime": regime["regime"],
                "Regime_Score": regime["score"],
                "Regime_Fit": r.regime_fit,
                "Status": "OPEN",
                "Exit_Date": "",
                "Exit_Price": "",
                "Exit_Reason": "",
                "PnL_Pct": "",
            })
    
    if new_rows:
        df_new = pd.DataFrame(new_rows)
        if today_file.exists():
            df_old = pd.read_csv(today_file)
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_combined = df_new
        df_combined.to_csv(today_file, index=False)
        log.info(f"Recorded {len(new_rows)} NEW signals (total today: {len(df_combined)})")
    else:
        log.info("No new unique signals to record.")
    
    # Write summary
    summary_file = SIGNALS_DIR / f"{today}_summary.json"
    summary = {
        "date": today,
        "scan_time": now_str,
        "mode": mode,
        "universe": universe,
        "regime": regime["regime"],
        "regime_score": regime["score"],
        "nifty": regime["nifty_close"],
        "stocks_scanned": len(data),
        "total_signals": total,
        "new_signals": len(new_rows),
        "strategies": {s: len(sigs) for s, sigs in results.items() if sigs},
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    log.info(f"Summary saved: {summary_file}")
    return today_file


# ============================================================================
# SIGNAL TRACKER — Check SL/T1 hits
# ============================================================================

def update_tracker():
    """Check all OPEN signals and update status based on current prices."""
    SIGNALS_DIR.mkdir(exist_ok=True)
    
    # Find all signal files
    signal_files = sorted(SIGNALS_DIR.glob("*_signals.csv"))
    if not signal_files:
        log.info("No signal files found.")
        return
    
    # Collect all OPEN signals
    all_signals = []
    _str_cols = ["Status", "Exit_Date", "Exit_Price", "Exit_Reason", "PnL_Pct"]
    for f in signal_files:
        try:
            df = pd.read_csv(f)
            # Force string dtype on columns that may be empty (read as float64)
            for c in _str_cols:
                if c in df.columns:
                    df[c] = df[c].fillna("").astype(object)
                    df[c] = df[c].replace("nan", "")
            all_signals.append((f, df))
        except:
            continue
    
    if not all_signals:
        return
    
    # Get unique symbols that are OPEN
    open_symbols = set()
    for f, df in all_signals:
        open_mask = df["Status"] == "OPEN"
        open_symbols.update(df[open_mask]["Symbol"].unique())
    
    if not open_symbols:
        log.info("No OPEN signals to track.")
        return
    
    log.info(f"Tracking {len(open_symbols)} symbols with OPEN signals...")
    
    # Fetch current prices
    prices = {}
    for sym in open_symbols:
        try:
            import yfinance as yf
            tk = yf.Ticker(f"{sym}.NS")
            hist = tk.history(period="5d")
            if not hist.empty:
                latest = hist.iloc[-1]
                prices[sym] = {
                    "close": latest["Close"],
                    "high": hist["High"].max(),  # Max high in last 5 days
                    "low": hist["Low"].min(),     # Min low in last 5 days
                    "today_high": latest["High"],
                    "today_low": latest["Low"],
                }
        except:
            continue
    
    log.info(f"Fetched prices for {len(prices)} symbols")
    today = date.today().isoformat()
    updated_count = 0
    
    # Update each signal file
    for filepath, df in all_signals:
        changed = False
        for idx, row in df.iterrows():
            if row["Status"] != "OPEN":
                continue
            
            sym = row["Symbol"]
            if sym not in prices:
                continue
            
            p = prices[sym]
            signal_date = row["Date"]
            entry = float(row["Entry"])
            sl = float(row["SL"])
            t1 = float(row["T1"])
            
            # Check days since signal
            try:
                sig_date = datetime.strptime(signal_date, "%Y-%m-%d").date()
                days_held = (date.today() - sig_date).days
            except:
                days_held = 0
            
            if row["Signal"] == "BUY":
                # Check SL hit (did price go below SL?)
                if p["low"] <= sl or p["today_low"] <= sl:
                    df.at[idx, "Status"] = "STOPPED"
                    df.at[idx, "Exit_Date"] = today
                    df.at[idx, "Exit_Price"] = round(sl, 2)
                    df.at[idx, "Exit_Reason"] = "Stop Loss Hit"
                    df.at[idx, "PnL_Pct"] = round((sl / entry - 1) * 100, 2)
                    changed = True
                    updated_count += 1
                    log.info(f"  🔴 {sym} BUY STOPPED at {sl:.0f} (entry {entry:.0f})")
                
                # Check T1 hit
                elif p["high"] >= t1 or p["today_high"] >= t1:
                    df.at[idx, "Status"] = "TARGET"
                    df.at[idx, "Exit_Date"] = today
                    df.at[idx, "Exit_Price"] = round(t1, 2)
                    df.at[idx, "Exit_Reason"] = "Target 1 Hit"
                    df.at[idx, "PnL_Pct"] = round((t1 / entry - 1) * 100, 2)
                    changed = True
                    updated_count += 1
                    log.info(f"  🟢 {sym} BUY TARGET at {t1:.0f} (entry {entry:.0f})")
                
                # Expire after 30 days
                elif days_held > 30:
                    df.at[idx, "Status"] = "EXPIRED"
                    df.at[idx, "Exit_Date"] = today
                    df.at[idx, "Exit_Price"] = round(p["close"], 2)
                    df.at[idx, "Exit_Reason"] = f"Expired ({days_held}d)"
                    df.at[idx, "PnL_Pct"] = round((p["close"] / entry - 1) * 100, 2)
                    changed = True
                    updated_count += 1
                
            elif row["Signal"] == "SHORT":
                if p["high"] >= sl or p["today_high"] >= sl:
                    df.at[idx, "Status"] = "STOPPED"
                    df.at[idx, "Exit_Date"] = today
                    df.at[idx, "Exit_Price"] = round(sl, 2)
                    df.at[idx, "Exit_Reason"] = "Stop Loss Hit"
                    df.at[idx, "PnL_Pct"] = round((entry / sl - 1) * 100, 2)
                    changed = True
                    updated_count += 1
                    log.info(f"  🔴 {sym} SHORT STOPPED at {sl:.0f}")
                
                elif p["low"] <= t1 or p["today_low"] <= t1:
                    df.at[idx, "Status"] = "TARGET"
                    df.at[idx, "Exit_Date"] = today
                    df.at[idx, "Exit_Price"] = round(t1, 2)
                    df.at[idx, "Exit_Reason"] = "Target 1 Hit"
                    df.at[idx, "PnL_Pct"] = round((entry / t1 - 1) * 100, 2)
                    changed = True
                    updated_count += 1
                    log.info(f"  🟢 {sym} SHORT TARGET at {t1:.0f}")
                
                elif days_held > 30:
                    df.at[idx, "Status"] = "EXPIRED"
                    df.at[idx, "Exit_Date"] = today
                    df.at[idx, "Exit_Price"] = round(p["close"], 2)
                    df.at[idx, "Exit_Reason"] = f"Expired ({days_held}d)"
                    df.at[idx, "PnL_Pct"] = round((entry / p["close"] - 1) * 100, 2)
                    changed = True
                    updated_count += 1
        
        if changed:
            df.to_csv(filepath, index=False)
    
    log.info(f"Updated {updated_count} signals")
    
    # Build consolidated tracker
    _rebuild_tracker(all_signals)


def _rebuild_tracker(all_signals=None):
    """Build consolidated tracker.csv from all signal files."""
    if all_signals is None:
        signal_files = sorted(SIGNALS_DIR.glob("*_signals.csv"))
        all_signals = []
        _str_cols = ["Status", "Exit_Date", "Exit_Price", "Exit_Reason", "PnL_Pct"]
        for f in signal_files:
            try:
                df = pd.read_csv(f)
                for c in _str_cols:
                    if c in df.columns:
                        df[c] = df[c].fillna("").astype(object)
                        df[c] = df[c].replace("nan", "")
                all_signals.append((f, df))
            except:
                continue
    
    if not all_signals:
        return
    
    frames = [df for _, df in all_signals]
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(TRACKER_FILE, index=False)
    
    # Stats
    total = len(combined)
    targets = len(combined[combined["Status"] == "TARGET"])
    stopped = len(combined[combined["Status"] == "STOPPED"])
    open_count = len(combined[combined["Status"] == "OPEN"])
    expired = len(combined[combined["Status"] == "EXPIRED"])
    
    log.info(f"Tracker: {total} total | 🟢 {targets} TARGET | 🔴 {stopped} STOPPED | "
             f"⏳ {open_count} OPEN | ⏰ {expired} EXPIRED")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NSE Scanner Auto-Runner")
    parser.add_argument("--mode", choices=["eod", "intraday"], default="eod",
                        help="eod=full NSE500 scan, intraday=quick Nifty200")
    parser.add_argument("--track", action="store_true",
                        help="Update tracker (check SL/T1 hits)")
    parser.add_argument("--universe", default=None,
                        help="Override universe: nifty50, nifty200, nifty500")
    args = parser.parse_args()
    
    if args.track:
        update_tracker()
    else:
        universe = args.universe or ("nifty500" if args.mode == "eod" else "nifty200")
        scan_and_record(mode=args.mode, universe=universe)
        # Always update tracker after scanning
        update_tracker()

"""
Signal Tracker — Read signal logs, track outcomes, generate reports.
Used by both the app (display) and auto_scanner (update).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Optional, Dict, List, Tuple
import os


SIGNALS_DIR = Path(__file__).parent / "signals"


def get_signal_dates() -> List[str]:
    """Get all dates that have signal files."""
    SIGNALS_DIR.mkdir(exist_ok=True)
    files = sorted(SIGNALS_DIR.glob("*_signals.csv"), reverse=True)
    dates = []
    for f in files:
        name = f.stem.replace("_signals", "")
        dates.append(name)
    return dates


def load_signals(date_str: str = None) -> Optional[pd.DataFrame]:
    """Load signals for a specific date, or all if date_str is None."""
    SIGNALS_DIR.mkdir(exist_ok=True)
    
    if date_str:
        filepath = SIGNALS_DIR / f"{date_str}_signals.csv"
        if filepath.exists():
            return pd.read_csv(filepath)
        return None
    
    # Load all
    files = sorted(SIGNALS_DIR.glob("*_signals.csv"))
    if not files:
        return None
    
    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f))
        except:
            continue
    
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def load_tracker() -> Optional[pd.DataFrame]:
    """Load the consolidated tracker."""
    tracker = SIGNALS_DIR / "tracker.csv"
    if tracker.exists():
        return pd.read_csv(tracker)
    # Fallback: build from individual files
    return load_signals()


def save_signals_today(signals_list: list, regime: dict = None):
    """Save signals from an in-app scan to today's CSV."""
    SIGNALS_DIR.mkdir(exist_ok=True)
    today = date.today().isoformat()
    today_file = SIGNALS_DIR / f"{today}_signals.csv"
    now_str = datetime.now().strftime("%H:%M")
    
    # Load existing for dedup
    existing = set()
    if today_file.exists():
        try:
            df_old = pd.read_csv(today_file)
            for _, row in df_old.iterrows():
                existing.add(f"{row['Strategy']}_{row['Symbol']}")
        except:
            df_old = pd.DataFrame()
    else:
        df_old = pd.DataFrame()
    
    new_rows = []
    for r in signals_list:
        key = f"{r.strategy}_{r.symbol}"
        if key in existing:
            continue
        
        new_rows.append({
            "Date": today,
            "Time": now_str,
            "Strategy": getattr(r, 'strategy', ''),
            "Strategy_Name": getattr(r, 'strategy_name', getattr(r, 'strategy', '')),
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
            "Sector": getattr(r, 'sector', ''),
            "Regime": regime.get("regime", "") if regime else "",
            "Regime_Score": regime.get("score", 0) if regime else 0,
            "Regime_Fit": getattr(r, 'regime_fit', ''),
            "Status": "OPEN",
            "Exit_Date": "",
            "Exit_Price": "",
            "Exit_Reason": "",
            "PnL_Pct": "",
        })
    
    if new_rows:
        df_new = pd.DataFrame(new_rows)
        combined = pd.concat([df_old, df_new], ignore_index=True) if not df_old.empty else df_new
        combined.to_csv(today_file, index=False)
        return len(new_rows)
    return 0


def compute_tracker_stats(df: pd.DataFrame = None) -> Dict:
    """Compute summary statistics from tracker data."""
    if df is None:
        df = load_tracker()
    if df is None or df.empty:
        return {}
    
    total = len(df)
    targets = len(df[df["Status"] == "TARGET"])
    stopped = len(df[df["Status"] == "STOPPED"])
    open_count = len(df[df["Status"] == "OPEN"])
    expired = len(df[df["Status"] == "EXPIRED"])
    
    # Win rate (excluding OPEN and EXPIRED)
    closed = targets + stopped
    win_rate = round(targets / closed * 100, 1) if closed > 0 else 0
    
    # P&L stats
    pnl_col = pd.to_numeric(df["PnL_Pct"], errors="coerce")
    closed_pnl = pnl_col[df["Status"].isin(["TARGET", "STOPPED"])].dropna()
    
    total_pnl = round(closed_pnl.sum(), 2) if len(closed_pnl) > 0 else 0
    avg_win = round(closed_pnl[closed_pnl > 0].mean(), 2) if len(closed_pnl[closed_pnl > 0]) > 0 else 0
    avg_loss = round(closed_pnl[closed_pnl < 0].mean(), 2) if len(closed_pnl[closed_pnl < 0]) > 0 else 0
    
    # Strategy breakdown
    strategy_stats = {}
    for strat in df["Strategy"].unique():
        sdf = df[df["Strategy"] == strat]
        s_closed = sdf[sdf["Status"].isin(["TARGET", "STOPPED"])]
        s_targets = len(sdf[sdf["Status"] == "TARGET"])
        s_stopped = len(sdf[sdf["Status"] == "STOPPED"])
        s_total = s_targets + s_stopped
        strategy_stats[strat] = {
            "total": len(sdf),
            "open": len(sdf[sdf["Status"] == "OPEN"]),
            "targets": s_targets,
            "stopped": s_stopped,
            "win_rate": round(s_targets / s_total * 100, 1) if s_total > 0 else 0,
        }
    
    # Date range
    dates = pd.to_datetime(df["Date"], errors="coerce").dropna()
    
    return {
        "total": total,
        "targets": targets,
        "stopped": stopped,
        "open": open_count,
        "expired": expired,
        "closed": closed,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "strategy_stats": strategy_stats,
        "first_date": dates.min().strftime("%Y-%m-%d") if len(dates) > 0 else "",
        "last_date": dates.max().strftime("%Y-%m-%d") if len(dates) > 0 else "",
        "trading_days": len(df["Date"].unique()),
    }


def update_open_signals_live(data_dict: dict) -> int:
    """Update OPEN signals with current prices from loaded data. Returns count of updates."""
    SIGNALS_DIR.mkdir(exist_ok=True)
    signal_files = sorted(SIGNALS_DIR.glob("*_signals.csv"))
    if not signal_files:
        return 0
    
    updated = 0
    today = date.today().isoformat()
    
    for filepath in signal_files:
        try:
            df = pd.read_csv(filepath)
        except:
            continue
        
        changed = False
        for idx, row in df.iterrows():
            if row["Status"] != "OPEN":
                continue
            
            sym = row["Symbol"]
            if sym not in data_dict:
                continue
            
            stock_df = data_dict[sym]
            if stock_df is None or stock_df.empty:
                continue
            
            latest = stock_df.iloc[-1]
            entry = float(row["Entry"])
            sl = float(row["SL"])
            t1 = float(row["T1"])
            
            # Get recent high/low for SL/T1 check
            recent = stock_df.iloc[-5:]  # last 5 days
            recent_high = recent["high"].max() if "high" in recent.columns else recent["High"].max()
            recent_low = recent["low"].min() if "low" in recent.columns else recent["Low"].min()
            
            close = latest.get("close", latest.get("Close", 0))
            
            if row["Signal"] == "BUY":
                if recent_low <= sl:
                    df.at[idx, "Status"] = "STOPPED"
                    df.at[idx, "Exit_Date"] = today
                    df.at[idx, "Exit_Price"] = round(sl, 2)
                    df.at[idx, "Exit_Reason"] = "Stop Loss Hit"
                    df.at[idx, "PnL_Pct"] = round((sl / entry - 1) * 100, 2)
                    changed = True; updated += 1
                elif recent_high >= t1:
                    df.at[idx, "Status"] = "TARGET"
                    df.at[idx, "Exit_Date"] = today
                    df.at[idx, "Exit_Price"] = round(t1, 2)
                    df.at[idx, "Exit_Reason"] = "Target 1 Hit"
                    df.at[idx, "PnL_Pct"] = round((t1 / entry - 1) * 100, 2)
                    changed = True; updated += 1
            
            elif row["Signal"] == "SHORT":
                if recent_high >= sl:
                    df.at[idx, "Status"] = "STOPPED"
                    df.at[idx, "Exit_Date"] = today
                    df.at[idx, "Exit_Price"] = round(sl, 2)
                    df.at[idx, "Exit_Reason"] = "Stop Loss Hit"
                    df.at[idx, "PnL_Pct"] = round((entry / sl - 1) * 100, 2)
                    changed = True; updated += 1
                elif recent_low <= t1:
                    df.at[idx, "Status"] = "TARGET"
                    df.at[idx, "Exit_Date"] = today
                    df.at[idx, "Exit_Price"] = round(t1, 2)
                    df.at[idx, "Exit_Reason"] = "Target 1 Hit"
                    df.at[idx, "PnL_Pct"] = round((entry / t1 - 1) * 100, 2)
                    changed = True; updated += 1
        
        if changed:
            df.to_csv(filepath, index=False)
    
    return updated


def generate_csv_download() -> Optional[bytes]:
    """Generate full tracker CSV as bytes for download."""
    df = load_tracker()
    if df is None or df.empty:
        return None
    return df.to_csv(index=False).encode("utf-8")

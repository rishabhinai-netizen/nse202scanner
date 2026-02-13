"""
Backtesting Engine — v4
========================
Run any scanner against historical data. Walk-forward simulation.
No lookahead bias — each day only sees data available up to that point.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import json

from data_engine import Indicators


@dataclass
class BacktestTrade:
    symbol: str
    strategy: str
    signal: str
    entry_date: str
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    exit_date: str = ""
    exit_price: float = 0.0
    exit_reason: str = ""
    pnl_pct: float = 0.0
    pnl_abs: float = 0.0
    holding_days: int = 0
    max_favorable: float = 0.0  # Best % during trade
    max_adverse: float = 0.0    # Worst % during trade


@dataclass  
class BacktestResult:
    strategy: str
    symbol: str
    period: str
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    total_pnl_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    max_drawdown_pct: float
    expectancy_pct: float
    avg_holding_days: float
    best_trade_pct: float
    worst_trade_pct: float
    trades: List[BacktestTrade]
    equity_curve: List[dict]


def backtest_strategy(df: pd.DataFrame, symbol: str, strategy: str,
                      lookback_days: int = 500, max_hold: int = 30) -> Optional[BacktestResult]:
    """
    Walk-forward backtest of a strategy on a single stock.
    
    Process:
    1. For each historical day, check if scanner would have triggered
    2. If triggered, simulate entry/SL/target management
    3. Track P&L, holding period, max adverse excursion
    
    No lookahead bias — enrichment uses only data up to each point.
    """
    if df is None or len(df) < 220:
        return None
    
    # Use available data (yfinance 1y ≈ 248 days)
    test_data = df.iloc[-min(lookback_days, len(df)):]
    trades: List[BacktestTrade] = []
    in_trade = False
    current_trade = None
    
    # We need at least 200 days for indicators, so start scanning from day 200
    start_idx = 200
    
    for i in range(start_idx, len(test_data)):
        # Slice data up to this point (no lookahead)
        hist = test_data.iloc[:i+1].copy()
        
        if len(hist) < 200:
            continue
        
        try:
            enriched = Indicators.enrich_dataframe(hist)
        except:
            continue
        
        latest = enriched.iloc[-1]
        date_str = str(enriched.index[-1].date())
        
        if in_trade and current_trade:
            # === MANAGE EXISTING TRADE ===
            current_trade.holding_days += 1
            
            if current_trade.signal == "BUY":
                # Check stop loss hit (use low of day)
                if latest["low"] <= current_trade.stop_loss:
                    current_trade.exit_date = date_str
                    current_trade.exit_price = current_trade.stop_loss
                    current_trade.exit_reason = "STOP LOSS"
                    current_trade.pnl_pct = (current_trade.exit_price / current_trade.entry_price - 1) * 100
                    trades.append(current_trade)
                    in_trade = False
                    current_trade = None
                    continue
                
                # Check T1 hit (use high of day)
                if latest["high"] >= current_trade.target_1:
                    # Exit at T1 (conservative)
                    current_trade.exit_date = date_str
                    current_trade.exit_price = current_trade.target_1
                    current_trade.exit_reason = "TARGET 1"
                    current_trade.pnl_pct = (current_trade.exit_price / current_trade.entry_price - 1) * 100
                    trades.append(current_trade)
                    in_trade = False
                    current_trade = None
                    continue
                
                # Track max favorable/adverse
                day_pnl = (latest["close"] / current_trade.entry_price - 1) * 100
                current_trade.max_favorable = max(current_trade.max_favorable, day_pnl)
                current_trade.max_adverse = min(current_trade.max_adverse, day_pnl)
                
            else:  # SHORT
                if latest["high"] >= current_trade.stop_loss:
                    current_trade.exit_date = date_str
                    current_trade.exit_price = current_trade.stop_loss
                    current_trade.exit_reason = "STOP LOSS"
                    current_trade.pnl_pct = (current_trade.entry_price / current_trade.exit_price - 1) * 100
                    trades.append(current_trade)
                    in_trade = False
                    current_trade = None
                    continue
                
                if latest["low"] <= current_trade.target_1:
                    current_trade.exit_date = date_str
                    current_trade.exit_price = current_trade.target_1
                    current_trade.exit_reason = "TARGET 1"
                    current_trade.pnl_pct = (current_trade.entry_price / current_trade.exit_price - 1) * 100
                    trades.append(current_trade)
                    in_trade = False
                    current_trade = None
                    continue
                
                day_pnl = (current_trade.entry_price / latest["close"] - 1) * 100
                current_trade.max_favorable = max(current_trade.max_favorable, day_pnl)
                current_trade.max_adverse = min(current_trade.max_adverse, day_pnl)
            
            # Max holding period exit
            if current_trade.holding_days >= max_hold:
                current_trade.exit_date = date_str
                current_trade.exit_price = latest["close"]
                current_trade.exit_reason = f"MAX HOLD ({max_hold}d)"
                if current_trade.signal == "BUY":
                    current_trade.pnl_pct = (current_trade.exit_price / current_trade.entry_price - 1) * 100
                else:
                    current_trade.pnl_pct = (current_trade.entry_price / current_trade.exit_price - 1) * 100
                trades.append(current_trade)
                in_trade = False
                current_trade = None
            
            continue
        
        # === CHECK FOR NEW SIGNAL ===
        signal = _check_signal(enriched, latest, strategy, symbol)
        if signal:
            in_trade = True
            current_trade = signal
    
    # Close any remaining open trade
    if in_trade and current_trade:
        last = test_data.iloc[-1]
        current_trade.exit_date = str(test_data.index[-1].date())
        current_trade.exit_price = last["close"]
        current_trade.exit_reason = "END OF DATA"
        if current_trade.signal == "BUY":
            current_trade.pnl_pct = (current_trade.exit_price / current_trade.entry_price - 1) * 100
        else:
            current_trade.pnl_pct = (current_trade.entry_price / current_trade.exit_price - 1) * 100
        trades.append(current_trade)
    
    if not trades:
        return None
    
    # === COMPUTE STATISTICS ===
    return _compute_stats(trades, strategy, symbol, lookback_days)


def _check_signal(df: pd.DataFrame, latest, strategy: str, 
                  symbol: str) -> Optional[BacktestTrade]:
    """Check if a strategy signal fires on the latest bar. Simplified for speed."""
    
    if strategy == "VCP":
        return _check_vcp(df, latest, symbol)
    elif strategy == "EMA21_Bounce":
        return _check_ema21(df, latest, symbol)
    elif strategy == "52WH_Breakout":
        return _check_52wh(df, latest, symbol)
    elif strategy == "Failed_Breakout_Short":
        return _check_short(df, latest, symbol)
    elif strategy == "Last30Min_ATH":
        return _check_ath(df, latest, symbol)
    return None


def _check_vcp(df, latest, symbol):
    """VCP signal check for backtesting."""
    if len(df) < 200: return None
    if latest["close"] < latest["sma_50"] or latest["sma_50"] < latest["sma_200"]: return None
    if latest["pct_from_52w_high"] < -25: return None
    pct_above_low = (latest["close"] - latest["low_52w"]) / latest["low_52w"] * 100
    if pct_above_low < 30: return None
    
    recent_range = df["high"].iloc[-10:].max() - df["low"].iloc[-10:].min()
    wide_range = df["high"].iloc[-40:].max() - df["low"].iloc[-40:].min()
    if wide_range == 0: return None
    contraction = recent_range / wide_range
    if contraction > 0.65: return None
    
    recent_vol = df["volume"].iloc[-10:].mean()
    avg_vol = latest["vol_sma_20"]
    if avg_vol > 0 and recent_vol / avg_vol > 1.2: return None
    
    pivot = df["high"].iloc[-10:].max()
    entry = latest["close"] if latest["close"] >= pivot else pivot
    atr = latest["atr_14"]
    sl = df["low"].iloc[-10:].min() - 0.5 * atr
    risk = entry - sl
    if risk <= 0: return None
    
    return BacktestTrade(
        symbol=symbol, strategy="VCP", signal="BUY",
        entry_date=str(df.index[-1].date()), entry_price=round(entry, 2),
        stop_loss=round(sl, 2), target_1=round(entry + 2 * risk, 2),
        target_2=round(entry + 3.5 * risk, 2),
    )


def _check_ema21(df, latest, symbol):
    """EMA21 Bounce signal check."""
    if len(df) < 60: return None
    if latest["close"] < latest["sma_50"]: return None
    if latest["sma_50"] < latest["sma_200"]: return None
    
    ema_dist = (latest["close"] - latest["ema_21"]) / latest["ema_21"] * 100
    if ema_dist > 3 or ema_dist < -2: return None
    if latest["low"] > latest["ema_21"] * 1.01: return None
    if latest["close"] < latest["ema_21"]: return None
    if latest["close"] < latest["open"]: return None
    
    entry = latest["close"]
    atr = latest["atr_14"]
    sl = min(latest["low"], latest["ema_21"]) - 0.5 * atr
    risk = entry - sl
    if risk <= 0: return None
    
    return BacktestTrade(
        symbol=symbol, strategy="EMA21_Bounce", signal="BUY",
        entry_date=str(df.index[-1].date()), entry_price=round(entry, 2),
        stop_loss=round(sl, 2), target_1=round(entry + 1.5 * risk, 2),
        target_2=round(entry + 2.5 * risk, 2),
    )


def _check_52wh(df, latest, symbol):
    """52WH Breakout signal check."""
    if len(df) < 200: return None
    if latest["pct_from_52w_high"] < -3: return None
    prev_high = df["high"].iloc[-20:-1].max()
    if latest["close"] <= prev_high: return None
    if latest["close"] < latest["sma_50"]: return None
    if latest["sma_50"] < latest["sma_200"]: return None
    vol_ratio = latest["volume"] / (latest["vol_sma_20"] + 1)
    if vol_ratio < 1.5: return None
    
    entry = latest["close"]
    atr = latest["atr_14"]
    sl = prev_high - 0.5 * atr
    risk = entry - sl
    if risk <= 0: sl = entry - 1.5 * atr; risk = entry - sl
    if risk <= 0: return None
    
    return BacktestTrade(
        symbol=symbol, strategy="52WH_Breakout", signal="BUY",
        entry_date=str(df.index[-1].date()), entry_price=round(entry, 2),
        stop_loss=round(sl, 2), target_1=round(entry + 2 * risk, 2),
        target_2=round(entry + 3.5 * risk, 2),
    )


def _check_short(df, latest, symbol):
    """Failed Breakout Short signal check."""
    if len(df) < 60: return None
    prev = df.iloc[-2]
    resistance = df["high"].iloc[-20:-2].max()
    if prev["high"] < resistance: return None
    if latest["close"] >= resistance: return None
    if latest["close"] >= latest["open"]: return None
    if latest["close"] >= prev["close"]: return None
    
    entry = latest["close"]
    sl = max(latest["high"], prev["high"]) + latest["atr_14"] * 0.3
    risk = sl - entry
    if risk <= 0: return None
    
    return BacktestTrade(
        symbol=symbol, strategy="Failed_Breakout_Short", signal="SHORT",
        entry_date=str(df.index[-1].date()), entry_price=round(entry, 2),
        stop_loss=round(sl, 2), target_1=round(entry - 1.5 * risk, 2),
        target_2=round(entry - 2.5 * risk, 2),
    )


def _check_ath(df, latest, symbol):
    """Last 30 Min ATH signal check."""
    if len(df) < 50: return None
    if latest["high"] == 0: return None
    close_vs_high = (latest["high"] - latest["close"]) / latest["high"] * 100
    if close_vs_high > 1.0: return None
    if latest["close"] < latest["ema_21"]: return None
    if latest["pct_from_52w_high"] < -5: return None
    if latest["close"] < latest["open"]: return None
    vol_ratio = latest["volume"] / (latest["vol_sma_20"] + 1)
    if vol_ratio < 1.0: return None
    
    entry = latest["close"]
    sl = latest["low"]
    risk = entry - sl
    if risk <= 0: risk = latest["atr_14"]; sl = entry - risk
    if risk <= 0: return None
    
    return BacktestTrade(
        symbol=symbol, strategy="Last30Min_ATH", signal="BUY",
        entry_date=str(df.index[-1].date()), entry_price=round(entry, 2),
        stop_loss=round(sl, 2), target_1=round(entry + 1.5 * risk, 2),
        target_2=round(entry + 2.5 * risk, 2),
    )


def _compute_stats(trades: List[BacktestTrade], strategy: str, 
                   symbol: str, lookback: int) -> BacktestResult:
    """Compute backtest statistics from trades list."""
    wins = [t for t in trades if t.pnl_pct > 0]
    losses = [t for t in trades if t.pnl_pct <= 0]
    
    total_pnl = sum(t.pnl_pct for t in trades)
    avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0
    avg_loss = np.mean([abs(t.pnl_pct) for t in losses]) if losses else 0
    pf = (sum(t.pnl_pct for t in wins) / abs(sum(t.pnl_pct for t in losses))) if losses and sum(t.pnl_pct for t in losses) != 0 else 0
    
    # Equity curve + max drawdown
    equity = []
    running = 0
    peak = 0
    max_dd = 0
    for t in trades:
        running += t.pnl_pct
        peak = max(peak, running)
        dd = peak - running
        max_dd = max(max_dd, dd)
        equity.append({"date": t.exit_date, "equity": round(running, 2), "drawdown": round(dd, 2)})
    
    return BacktestResult(
        strategy=strategy, symbol=symbol,
        period=f"Last {lookback} trading days",
        total_trades=len(trades),
        wins=len(wins), losses=len(losses),
        win_rate=round(len(wins) / len(trades) * 100, 1) if trades else 0,
        total_pnl_pct=round(total_pnl, 2),
        avg_win_pct=round(avg_win, 2),
        avg_loss_pct=round(avg_loss, 2),
        profit_factor=round(pf, 2),
        max_drawdown_pct=round(max_dd, 2),
        expectancy_pct=round(total_pnl / len(trades), 2) if trades else 0,
        avg_holding_days=round(np.mean([t.holding_days for t in trades]), 1),
        best_trade_pct=round(max(t.pnl_pct for t in trades), 2),
        worst_trade_pct=round(min(t.pnl_pct for t in trades), 2),
        trades=trades,
        equity_curve=equity,
    )


def backtest_multi_stock(symbols: List[str], data_dict: Dict[str, pd.DataFrame],
                         strategy: str, lookback: int = 500,
                         max_hold: int = 30) -> Optional[BacktestResult]:
    """Run backtest across multiple stocks, aggregate results."""
    all_trades = []
    
    for symbol in symbols:
        if symbol not in data_dict:
            continue
        result = backtest_strategy(data_dict[symbol], symbol, strategy, lookback, max_hold)
        if result:
            all_trades.extend(result.trades)
    
    if not all_trades:
        return None
    
    # Sort by entry date
    all_trades.sort(key=lambda t: t.entry_date)
    
    return _compute_stats(all_trades, strategy, f"{len(symbols)} stocks", lookback)

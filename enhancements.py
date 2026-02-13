"""
World-Class Enhancements — Charts, Heatmaps, Journal, FII/DII, Multi-Timeframe
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import os
import requests
from typing import Dict, List, Optional, Tuple
from data_engine import Indicators, IST, now_ist
from scanners import ScanResult


# ============================================================================
# 1. CANDLESTICK CHARTS with Entry/SL/Target overlay
# ============================================================================

def plot_candlestick(df: pd.DataFrame, symbol: str, 
                     entry: float = None, stop_loss: float = None,
                     target1: float = None, target2: float = None,
                     signal: str = "BUY", days: int = 90) -> go.Figure:
    """
    Professional candlestick chart with EMA overlays, volume, and trade levels.
    """
    data = df.tail(days).copy()
    
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.60, 0.20, 0.20],
        subplot_titles=None
    )
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=data.index, open=data["open"], high=data["high"],
        low=data["low"], close=data["close"],
        increasing_line_color="#00d26a", decreasing_line_color="#ff4757",
        increasing_fillcolor="#00d26a", decreasing_fillcolor="#ff4757",
        name="Price", whiskerwidth=0.5
    ), row=1, col=1)
    
    # EMAs
    if "ema_9" in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data["ema_9"], name="EMA 9",
            line=dict(color="#ffd700", width=1), opacity=0.8
        ), row=1, col=1)
    if "ema_21" in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data["ema_21"], name="EMA 21",
            line=dict(color="#ff6b35", width=1.5), opacity=0.8
        ), row=1, col=1)
    if "sma_50" in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data["sma_50"], name="SMA 50",
            line=dict(color="#5dade2", width=1.5, dash="dot"), opacity=0.7
        ), row=1, col=1)
    if "sma_200" in data.columns:
        valid_sma200 = data["sma_200"].dropna()
        if len(valid_sma200) > 0:
            fig.add_trace(go.Scatter(
                x=valid_sma200.index, y=valid_sma200, name="SMA 200",
                line=dict(color="#888", width=1, dash="dash"), opacity=0.5
            ), row=1, col=1)
    
    # Trade levels
    if entry:
        fig.add_hline(y=entry, line_color="#ffffff", line_width=2,
                      line_dash="dash", annotation_text=f"Entry ₹{entry:,.0f}",
                      annotation_position="right", row=1, col=1)
    if stop_loss:
        fig.add_hline(y=stop_loss, line_color="#ff4757", line_width=2,
                      line_dash="dash", annotation_text=f"SL ₹{stop_loss:,.0f}",
                      annotation_position="right", row=1, col=1)
    if target1:
        fig.add_hline(y=target1, line_color="#00d26a", line_width=1.5,
                      line_dash="dot", annotation_text=f"T1 ₹{target1:,.0f}",
                      annotation_position="right", row=1, col=1)
    if target2:
        fig.add_hline(y=target2, line_color="#00d26a", line_width=1,
                      line_dash="dot", annotation_text=f"T2 ₹{target2:,.0f}",
                      annotation_position="right", row=1, col=1)
    
    # Volume bars (colored by direction)
    colors = ["#00d26a" if c >= o else "#ff4757" for c, o in zip(data["close"], data["open"])]
    fig.add_trace(go.Bar(
        x=data.index, y=data["volume"], name="Volume",
        marker_color=colors, opacity=0.6, showlegend=False
    ), row=2, col=1)
    
    # Volume SMA
    if "vol_sma_20" in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data["vol_sma_20"], name="Vol SMA20",
            line=dict(color="#ffd700", width=1), showlegend=False
        ), row=2, col=1)
    
    # RSI
    if "rsi_14" in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data["rsi_14"], name="RSI 14",
            line=dict(color="#ce93d8", width=1.5)
        ), row=3, col=1)
        fig.add_hline(y=70, line_color="#ff4757", line_width=0.5,
                      line_dash="dot", row=3, col=1)
        fig.add_hline(y=30, line_color="#00d26a", line_width=0.5,
                      line_dash="dot", row=3, col=1)
        fig.add_hrect(y0=30, y1=70, fillcolor="#333", opacity=0.1,
                      line_width=0, row=3, col=1)
    
    fig.update_layout(
        title=dict(text=f"{symbol} — {signal if signal else 'Chart'}", 
                   font=dict(size=16, color="#fafafa")),
        template="plotly_dark",
        paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
        height=550, margin=dict(l=50, r=20, t=40, b=30),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02, x=0, font=dict(size=10)),
        xaxis3=dict(title=""),
        yaxis=dict(title="Price (₹)", gridcolor="#222"),
        yaxis2=dict(title="Vol", gridcolor="#222"),
        yaxis3=dict(title="RSI", gridcolor="#222", range=[0, 100]),
    )
    
    # Remove weekends from x-axis
    fig.update_xaxes(
        rangebreaks=[dict(bounds=["sat", "mon"])],
        gridcolor="#222"
    )
    
    return fig


# ============================================================================
# 2. SECTOR ROTATION HEATMAP
# ============================================================================

def compute_sector_performance(stock_data: Dict[str, pd.DataFrame], 
                                get_sector_fn) -> pd.DataFrame:
    """Compute sector-wise performance for 1W, 1M, 3M periods."""
    records = []
    for symbol, df in stock_data.items():
        if len(df) < 65:
            continue
        sector = get_sector_fn(symbol)
        close = df["close"].iloc[-1]
        
        pct_1w = (close / df["close"].iloc[-5] - 1) * 100 if len(df) >= 5 else 0
        pct_1m = (close / df["close"].iloc[-22] - 1) * 100 if len(df) >= 22 else 0
        pct_3m = (close / df["close"].iloc[-65] - 1) * 100 if len(df) >= 65 else 0
        
        records.append({
            "symbol": symbol, "sector": sector,
            "1W %": pct_1w, "1M %": pct_1m, "3M %": pct_3m,
            "close": close
        })
    
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame()
    
    # Aggregate by sector
    sector_df = df.groupby("sector").agg(
        stocks=("symbol", "count"),
        avg_1w=("1W %", "mean"),
        avg_1m=("1M %", "mean"),
        avg_3m=("3M %", "mean"),
    ).round(2).sort_values("avg_1m", ascending=False)
    
    return sector_df


def plot_sector_heatmap(sector_df: pd.DataFrame) -> go.Figure:
    """Sector rotation heatmap."""
    if sector_df.empty:
        return None
    
    sectors = sector_df.index.tolist()
    periods = ["1 Week", "1 Month", "3 Month"]
    z = [
        sector_df["avg_1w"].tolist(),
        sector_df["avg_1m"].tolist(),
        sector_df["avg_3m"].tolist(),
    ]
    
    # Text annotations
    text = []
    for row in z:
        text.append([f"{v:+.1f}%" for v in row])
    
    fig = go.Figure(data=go.Heatmap(
        z=z, x=sectors, y=periods,
        colorscale=[[0, "#ff4757"], [0.5, "#1a1d23"], [1, "#00d26a"]],
        zmid=0, text=text, texttemplate="%{text}",
        textfont=dict(size=11, color="white"),
        hovertemplate="Sector: %{x}<br>Period: %{y}<br>Return: %{z:.1f}%<extra></extra>",
        colorbar=dict(title="Return %", tickformat="+.1f"),
    ))
    
    fig.update_layout(
        title="Sector Rotation Heatmap",
        template="plotly_dark",
        paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
        height=280, margin=dict(l=80, r=20, t=40, b=80),
        xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
    )
    return fig


# ============================================================================
# 3. RELATIVE STRENGTH HEATMAP
# ============================================================================

def compute_rs_rankings(stock_data: Dict[str, pd.DataFrame],
                        nifty_data: pd.DataFrame,
                        get_sector_fn) -> pd.DataFrame:
    """Compute RS rating for all stocks and rank them."""
    if nifty_data is None or len(nifty_data) < 65:
        return pd.DataFrame()
    
    records = []
    nifty_3m = (nifty_data["close"].iloc[-1] / nifty_data["close"].iloc[-65] - 1) * 100
    nifty_1m = (nifty_data["close"].iloc[-1] / nifty_data["close"].iloc[-22] - 1) * 100
    
    for symbol, df in stock_data.items():
        if len(df) < 65:
            continue
        close = df["close"].iloc[-1]
        pct_1m = (close / df["close"].iloc[-22] - 1) * 100
        pct_3m = (close / df["close"].iloc[-65] - 1) * 100
        
        # RS = stock return - nifty return (relative outperformance)
        rs_1m = pct_1m - nifty_1m
        rs_3m = pct_3m - nifty_3m
        # Composite RS: 40% 1M + 60% 3M
        rs_composite = 0.4 * rs_1m + 0.6 * rs_3m
        
        records.append({
            "Symbol": symbol,
            "Sector": get_sector_fn(symbol),
            "CMP": round(close, 2),
            "1M %": round(pct_1m, 1),
            "3M %": round(pct_3m, 1),
            "RS vs Nifty (1M)": round(rs_1m, 1),
            "RS vs Nifty (3M)": round(rs_3m, 1),
            "RS Score": round(rs_composite, 1),
        })
    
    df = pd.DataFrame(records)
    if df.empty:
        return df
    
    # Percentile rank
    df["RS Rank"] = df["RS Score"].rank(pct=True).mul(100).round(0).astype(int)
    df = df.sort_values("RS Rank", ascending=False)
    return df


def plot_rs_scatter(rs_df: pd.DataFrame) -> go.Figure:
    """RS scatter plot: 1M vs 3M performance colored by RS rank."""
    if rs_df.empty:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=rs_df["3M %"], y=rs_df["1M %"],
        mode="markers+text",
        text=rs_df["Symbol"],
        textposition="top center",
        textfont=dict(size=8, color="#aaa"),
        marker=dict(
            size=8,
            color=rs_df["RS Rank"],
            colorscale=[[0, "#ff4757"], [0.5, "#ffd700"], [1, "#00d26a"]],
            colorbar=dict(title="RS Rank"),
            line=dict(width=0.5, color="#333"),
        ),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "3M: %{x:.1f}%<br>"
            "1M: %{y:.1f}%<br>"
            "RS Rank: %{marker.color:.0f}<extra></extra>"
        ),
    ))
    
    # Quadrant lines
    fig.add_hline(y=0, line_color="#555", line_width=1)
    fig.add_vline(x=0, line_color="#555", line_width=1)
    
    # Quadrant labels
    fig.add_annotation(x=0.95, y=0.95, xref="paper", yref="paper",
                       text="⭐ LEADERS", font=dict(color="#00d26a", size=11),
                       showarrow=False)
    fig.add_annotation(x=0.05, y=0.95, xref="paper", yref="paper",
                       text="🔄 IMPROVING", font=dict(color="#ffd700", size=11),
                       showarrow=False)
    fig.add_annotation(x=0.95, y=0.05, xref="paper", yref="paper",
                       text="⚠️ WEAKENING", font=dict(color="#ff8a65", size=11),
                       showarrow=False)
    fig.add_annotation(x=0.05, y=0.05, xref="paper", yref="paper",
                       text="🔴 LAGGARDS", font=dict(color="#ff4757", size=11),
                       showarrow=False)
    
    fig.update_layout(
        title="Relative Strength Map (vs Nifty 50)",
        xaxis_title="3-Month Return %", yaxis_title="1-Month Return %",
        template="plotly_dark",
        paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
        height=500, margin=dict(l=50, r=20, t=40, b=50),
    )
    return fig


# ============================================================================
# 4. TRADE JOURNAL
# ============================================================================

JOURNAL_FILE = "trade_journal.json"

def load_journal() -> List[dict]:
    if os.path.exists(JOURNAL_FILE):
        try:
            with open(JOURNAL_FILE, "r") as f:
                return json.load(f)
        except:
            return []
    return []

def save_journal(entries: List[dict]):
    with open(JOURNAL_FILE, "w") as f:
        json.dump(entries, f, indent=2, default=str)

def add_journal_entry(entry: dict):
    journal = load_journal()
    entry["id"] = len(journal) + 1
    entry["timestamp"] = str(now_ist())
    journal.append(entry)
    save_journal(journal)
    return entry

def compute_journal_analytics(journal: List[dict]) -> dict:
    """Compute trading statistics from journal."""
    if not journal:
        return {}
    
    closed = [e for e in journal if e.get("status") == "closed"]
    if not closed:
        return {"total_trades": len(journal), "open_trades": len(journal)}
    
    wins = [e for e in closed if e.get("pnl", 0) > 0]
    losses = [e for e in closed if e.get("pnl", 0) <= 0]
    
    total_pnl = sum(e.get("pnl", 0) for e in closed)
    avg_win = np.mean([e["pnl"] for e in wins]) if wins else 0
    avg_loss = np.mean([abs(e["pnl"]) for e in losses]) if losses else 0
    
    # By strategy
    strategy_stats = {}
    for e in closed:
        strat = e.get("strategy", "Unknown")
        if strat not in strategy_stats:
            strategy_stats[strat] = {"wins": 0, "losses": 0, "pnl": 0, "trades": 0}
        strategy_stats[strat]["trades"] += 1
        strategy_stats[strat]["pnl"] += e.get("pnl", 0)
        if e.get("pnl", 0) > 0:
            strategy_stats[strat]["wins"] += 1
        else:
            strategy_stats[strat]["losses"] += 1
    
    # Equity curve
    cumulative = []
    running = 0
    for e in sorted(closed, key=lambda x: x.get("exit_date", x.get("timestamp", ""))):
        running += e.get("pnl", 0)
        cumulative.append({"date": e.get("exit_date", ""), "equity": running})
    
    return {
        "total_trades": len(journal),
        "closed_trades": len(closed),
        "open_trades": len(journal) - len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(closed) * 100, 1) if closed else 0,
        "total_pnl": round(total_pnl, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(avg_win / avg_loss, 2) if avg_loss > 0 else 0,
        "expectancy": round(total_pnl / len(closed), 2) if closed else 0,
        "max_win": round(max([e["pnl"] for e in wins], default=0), 2),
        "max_loss": round(min([e["pnl"] for e in losses], default=0), 2),
        "strategy_stats": strategy_stats,
        "equity_curve": cumulative,
    }


def plot_equity_curve(analytics: dict) -> Optional[go.Figure]:
    """Plot equity curve from journal analytics."""
    curve = analytics.get("equity_curve", [])
    if not curve:
        return None
    
    dates = [c["date"] for c in curve]
    equity = [c["equity"] for c in curve]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=equity, mode="lines+markers",
        line=dict(color="#00d26a" if equity[-1] >= 0 else "#ff4757", width=2),
        marker=dict(size=5),
        fill="tozeroy",
        fillcolor="rgba(0,210,106,0.1)" if equity[-1] >= 0 else "rgba(255,71,87,0.1)",
        name="Cumulative P&L"
    ))
    fig.add_hline(y=0, line_color="#555", line_width=1)
    
    fig.update_layout(
        title="Equity Curve",
        yaxis_title="Cumulative P&L (₹)",
        template="plotly_dark",
        paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
        height=300, margin=dict(l=50, r=20, t=40, b=30),
    )
    return fig


# ============================================================================
# 5. FII/DII DATA
# ============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fii_dii_data() -> Optional[pd.DataFrame]:
    """Fetch FII/DII activity data from public API."""
    try:
        # Try NSDL/public endpoint
        headers = {"User-Agent": "Mozilla/5.0"}
        url = "https://www.moneycontrol.com/stocks/marketstats/fii_dii_activity/data.json"
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if data:
                df = pd.DataFrame(data)
                return df
    except:
        pass
    
    # Fallback: scrape from NSE (simplified)
    try:
        url = "https://archives.nseindia.com/content/fo/fii_stats.csv"
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/csv",
            "Referer": "https://www.nseindia.com/"
        }
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            import io
            df = pd.read_csv(io.StringIO(r.text))
            return df
    except:
        pass
    return None


def format_fii_dii_summary() -> str:
    """Get FII/DII summary text."""
    data = fetch_fii_dii_data()
    if data is not None and not data.empty:
        return f"FII/DII data loaded: {len(data)} records"
    return "FII/DII data unavailable (check moneycontrol.com manually)"


# ============================================================================
# 6. MULTI-TIMEFRAME CONFIRMATION
# ============================================================================

def check_weekly_alignment(df: pd.DataFrame) -> dict:
    """
    Check if weekly timeframe confirms daily signal.
    Resample daily data to weekly, compute indicators, check alignment.
    """
    if df is None or len(df) < 50:
        return {"aligned": False, "reason": "Insufficient data"}
    
    # Resample to weekly
    weekly = df.resample("W").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna()
    
    if len(weekly) < 20:
        return {"aligned": False, "reason": "Not enough weekly data"}
    
    # Weekly indicators
    weekly["ema_10"] = weekly["close"].ewm(span=10, adjust=False).mean()
    weekly["ema_30"] = weekly["close"].ewm(span=30, adjust=False).mean()
    weekly["rsi"] = Indicators.rsi(weekly["close"], 14)
    
    latest = weekly.iloc[-1]
    reasons = []
    score = 0
    
    # Weekly close > 10 EMA
    if latest["close"] > latest["ema_10"]:
        score += 1
        reasons.append("✅ Weekly close > 10 EMA")
    else:
        reasons.append("❌ Weekly close < 10 EMA")
    
    # 10 EMA > 30 EMA (trend)
    if latest["ema_10"] > latest["ema_30"]:
        score += 1
        reasons.append("✅ Weekly 10 EMA > 30 EMA (uptrend)")
    else:
        reasons.append("❌ Weekly 10 EMA < 30 EMA (downtrend)")
    
    # Weekly RSI > 50
    if latest["rsi"] > 50:
        score += 1
        reasons.append(f"✅ Weekly RSI {latest['rsi']:.0f} > 50")
    else:
        reasons.append(f"❌ Weekly RSI {latest['rsi']:.0f} < 50")
    
    # Weekly candle is bullish
    if latest["close"] > latest["open"]:
        score += 1
        reasons.append("✅ Weekly candle is bullish")
    else:
        reasons.append("❌ Weekly candle is bearish")
    
    return {
        "aligned": score >= 3,
        "score": score,
        "max_score": 4,
        "reasons": reasons,
        "weekly_rsi": round(latest["rsi"], 1),
        "weekly_trend": "UP" if latest["ema_10"] > latest["ema_30"] else "DOWN",
    }


# ============================================================================
# 7. MARKET BREADTH (advance/decline from loaded data)
# ============================================================================

def compute_market_breadth(stock_data: Dict[str, pd.DataFrame]) -> dict:
    """Compute advance/decline and other breadth indicators from loaded stocks."""
    if not stock_data:
        return {}
    
    advancing = 0
    declining = 0
    unchanged = 0
    above_200sma = 0
    above_50sma = 0
    new_52w_high = 0
    new_52w_low = 0
    total = 0
    
    for symbol, df in stock_data.items():
        if len(df) < 2:
            continue
        total += 1
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        if latest["close"] > prev["close"]:
            advancing += 1
        elif latest["close"] < prev["close"]:
            declining += 1
        else:
            unchanged += 1
        
        # Check SMAs if enriched
        if "sma_200" in df.columns:
            enriched = df
        else:
            enriched = Indicators.enrich_dataframe(df)
        
        lat = enriched.iloc[-1]
        if pd.notna(lat.get("sma_200")) and lat["close"] > lat["sma_200"]:
            above_200sma += 1
        if pd.notna(lat.get("sma_50")) and lat["close"] > lat["sma_50"]:
            above_50sma += 1
        if pd.notna(lat.get("high_52w")) and lat["close"] >= lat["high_52w"] * 0.98:
            new_52w_high += 1
        if pd.notna(lat.get("low_52w")) and lat["close"] <= lat["low_52w"] * 1.02:
            new_52w_low += 1
    
    ad_ratio = advancing / declining if declining > 0 else advancing
    
    return {
        "total": total,
        "advancing": advancing,
        "declining": declining,
        "unchanged": unchanged,
        "ad_ratio": round(ad_ratio, 2),
        "above_200sma": above_200sma,
        "above_200sma_pct": round(above_200sma / total * 100, 1) if total else 0,
        "above_50sma": above_50sma,
        "above_50sma_pct": round(above_50sma / total * 100, 1) if total else 0,
        "new_52w_high": new_52w_high,
        "new_52w_low": new_52w_low,
    }


def plot_breadth_gauge(breadth: dict) -> go.Figure:
    """Market breadth gauge chart."""
    adv = breadth.get("advancing", 0)
    dec = breadth.get("declining", 0)
    total = adv + dec
    if total == 0:
        return None
    
    pct = adv / total * 100
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pct,
        title=dict(text="Market Breadth (% Advancing)", font=dict(size=14)),
        number=dict(suffix="%", font=dict(size=24)),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor="#333"),
            bar=dict(color="#00d26a" if pct > 50 else "#ff4757"),
            bgcolor="#1a1d23",
            borderwidth=2,
            bordercolor="#333",
            steps=[
                dict(range=[0, 30], color="#3d1a1a"),
                dict(range=[30, 50], color="#2d2a1a"),
                dict(range=[50, 70], color="#1a2d1a"),
                dict(range=[70, 100], color="#0d3320"),
            ],
            threshold=dict(
                line=dict(color="#ffd700", width=3),
                thickness=0.75, value=50
            ),
        ),
    ))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        height=250, margin=dict(l=30, r=30, t=50, b=10),
    )
    return fig

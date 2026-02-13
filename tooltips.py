"""
Tooltips — Hover explanations for every metric, indicator, and strategy.
Used throughout the app for ? icons and help text.
"""

TIPS = {
    # === REGIME ===
    "regime": "Market Regime classifies the overall market into 4 states based on Nifty trend, momentum, volatility, and breadth. EXPANSION = buy breakouts aggressively. ACCUMULATION = be selective, base building. DISTRIBUTION = defensive, favor shorts. PANIC = cash is king.",
    "regime_score": "Composite score from -5 to +12 based on: Golden/Death Cross (+/-2), Price vs 50 DMA (+/-1), Price vs 21 EMA (+/-1), RSI momentum (+/-2), MACD (+/-1), 52W position (+/-2), Volatility (+/-3), ADX (+1), Breadth (+/-2).",
    "position_size": "Regime-adjusted position sizing. EXPANSION=100%, ACCUMULATION=70%, DISTRIBUTION=40%, PANIC=15%. Multiplies your per-trade risk allocation to match market conditions.",
    
    # === INDICATORS ===
    "rs_rating": "Relative Strength (RS) Rating measures how a stock performed vs Nifty over 1-3 months. RS 70 = outperformed 70% of all stocks. RS 90 = top 10%. RS < 30 = laggard. Long signals require RS > 70 (configurable) because 95% of big winners have high RS before their move.",
    "confidence": "Signal quality score (0-100%) based on how many conditions are met and how strongly. Factors: price pattern quality, volume confirmation, trend alignment, sector strength, regime fit. Higher = more criteria satisfied = higher probability.",
    "risk_reward": "Risk-to-Reward ratio. 1:2.5 means for every ₹1 risked (Entry to SL), you expect ₹2.5 profit (Entry to Target). Professional minimum is 1:2. Never take trades below 1:1.5.",
    "regime_fit": "How well this strategy matches the current market regime. IDEAL = best conditions for this strategy. OK = acceptable. CAUTION = reduced probability, use smaller size. BLOCKED = this strategy historically fails in this regime.",
    "weekly_aligned": "Multi-timeframe confirmation. Checks 4 conditions on the WEEKLY chart: (1) Close > 10 EMA, (2) 10 EMA > 30 EMA, (3) RSI > 50, (4) Bullish candle. Score 3-4/4 = ✅ aligned = higher probability trade.",
    
    # === STRATEGIES ===
    "VCP": "Volatility Contraction Pattern (Mark Minervini). Stock in Stage 2 uptrend with price range contracting (base getting tighter). Volume dries up before breakout. Best in EXPANSION regime. Swing hold: 15-40 days.",
    "EMA21_Bounce": "21 EMA Bounce. Stock pulls back to the 21-day EMA in a confirmed uptrend (above 50 & 200 SMA), then bounces with a bullish candle. Classic buy-the-dip in strong stocks. Swing hold: 5-15 days.",
    "52WH_Breakout": "52-Week High Breakout. Stock breaks to new 52-week highs with 1.5x+ volume surge. Institutional accumulation signal. Best in EXPANSION. Positional hold: 20-60 days.",
    "Last30Min_ATH": "Last 30 Min ATH. Stock closing at the day's high (within 1%) near 52W highs. Strong demand into close = overnight gap-up probability. BTST trade. Best entry: 3:25 PM.",
    "Failed_Breakout_Short": "Failed Breakout Short. Stock attempted to break above resistance but reversed sharply on volume. Trapped longs create selling pressure. Works in ALL regimes, especially DISTRIBUTION/PANIC.",
    "ORB": "Opening Range Breakout. Price breaks above the first 15-min high with volume + VWAP confirmation. REQUIRES live intraday data (Breeze API). Cannot be approximated from daily candles. Intraday: 2-6 hours.",
    "VWAP_Reclaim": "VWAP Reclaim. Price dips below VWAP then reclaims it with volume surge. Mean-reversion play. REQUIRES Breeze API for live intraday data. Hold: 2-4 hours.",
    "Lunch_Low": "Lunch Low Buy. Hammer reversal at lunch-hour support with low volume on dip. REQUIRES live intraday data. Best: 12:30-1:30 PM. Hold: 2-3 hours.",
    
    # === COLUMNS ===
    "CMP": "Current Market Price — last traded price.",
    "Entry": "Recommended entry price. 'AT CMP' = buy now. 'ABOVE ₹X' = wait for price to cross pivot.",
    "SL": "Stop Loss — exit price if trade goes wrong. Always honor your stop loss.",
    "T1": "Target 1 — first profit booking level (usually 1.5x risk).",
    "T2": "Target 2 — extended target (usually 2.5x risk). Book partial profits at T1, trail rest.",
    "sector": "Stock's industry sector. Sector alignment matters: buying a strong stock in a weak sector reduces probability. Top 30% sectors get confidence boost, bottom 30% get penalty.",
    "ad_ratio": "Advance/Decline Ratio. Advancing stocks ÷ Declining stocks. > 1.5 = broad participation (healthy). < 0.7 = narrow market (risky). Measures market breadth.",
    "above_200sma": "% of stocks trading above their 200-day SMA. > 60% = broad bull market. < 35% = bear market. Shows how many stocks participate in the trend.",
    
    # === BACKTEST ===
    "win_rate": "Percentage of trades that were profitable. Above 50% is good. But win rate alone doesn't matter — a 40% win rate with 3:1 R:R is better than 60% win rate with 1:1 R:R.",
    "profit_factor": "Total gross profit ÷ Total gross loss. PF > 1.5 = good strategy. PF > 2.0 = excellent. PF < 1.0 = losing strategy.",
    "expectancy": "Average P&L per trade. Positive = strategy has edge. Formula: (Win% × Avg Win) - (Loss% × Avg Loss). This is the most important backtest metric.",
    "max_drawdown": "Largest peak-to-trough decline in equity curve. Shows worst-case scenario. Max DD > 20% = strategy needs better risk management.",
}


def tip(key: str) -> str:
    """Return tooltip text for a key, or empty string if not found."""
    return TIPS.get(key, "")


def tip_icon(key: str) -> str:
    """Return a small help icon with tooltip text for Streamlit."""
    t = TIPS.get(key, "")
    return f" ℹ️" if t else ""

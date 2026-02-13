# 🎯 NSE Scanner Pro v4.0 — Regime-Aware + Backtesting

## Setup
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Secrets (Streamlit Cloud → Settings → Secrets)
Paste ONLY these lines (**no backticks**):
```
BREEZE_API_KEY = "your_key"
BREEZE_API_SECRET = "your_secret"
BREEZE_SESSION_TOKEN = "daily_token"
TELEGRAM_BOT_TOKEN = "optional"
TELEGRAM_CHAT_ID = "optional"
```

## What's New in v4.0

| Feature | Details |
|---------|---------|
| 🧠 **Regime Engine (Recalibrated)** | 4 regimes scored against 12-point scale. Nifty at ₹25,694 with golden cross = ACCUMULATION (4/12), not DISTRIBUTION. |
| 🧪 **Backtesting Engine** | Walk-forward backtest any strategy on single stock or full portfolio. Equity curve, win rate, profit factor, trade log. |
| ℹ️ **Tooltips Everywhere** | Hover any ? icon to see what RS Rating, Confidence, Regime Fit, R:R ratio mean. Glossary in Scanner Hub. |
| 🔧 **Regime Recalibration** | Fixed harsh scoring: RSI 45-60 = neutral (no penalty), vol threshold raised to 1.5x, golden cross properly weighted at +2. |
| 🚫 **Intraday Proxies Killed** | ORB, VWAP, Lunch Low return nothing without Breeze. No fake signals. |
| 💪 **RS > 70 Filter** | Long signals require top-30% relative strength vs Nifty. |
| 🎯 **Daily Focus Panel** | Time-aware panel: what to do NOW based on market hours. |

## Regime Behavior (Recalibrated)

| Regime | Score Range | Position | Example Market State |
|--------|---:|---:|---|
| 🟢 EXPANSION | ≥ 6/12 | 100% | All DMAs aligned up, RSI > 60 |
| 🟡 ACCUMULATION | 2 to 5 | 70% | Golden cross, below 50 DMA, near highs |
| 🟠 DISTRIBUTION | -2 to 1 | 40% | Death cross forming, breadth weakening |
| 🔴 PANIC | < -2 | 15% | Vol spike, deep correction |

## Backtest Results (Nifty 50, ~1 year)

| Strategy | Trades | Win % | P&L | PF | Avg Hold |
|----------|---:|---:|---:|---:|---:|
| 52WH Breakout | 11 | 63.6% | +18.5% | 3.01 | 5d |
| ATH Close | 30 | 50.0% | +13.1% | 1.63 | 5d |
| Failed Short | 29 | 55.2% | +12.6% | 1.49 | 5d |
| EMA21 Bounce | 34 | 38.2% | +10.5% | 1.28 | 6d |
| VCP | 23 | 26.1% | -34.9% | 0.49 | 13d |

*VCP underperforms in ACCUMULATION regime (current). Best in EXPANSION.*

## Pages (10)

📊 Dashboard · 🔍 Scanner Hub · 📈 Charts & RS · 🧪 Backtest · 📐 Trade Planner · ⭐ Watchlist · 📓 Journal · 📋 Workflow · 🔔 Alerts · ⚙️ Settings

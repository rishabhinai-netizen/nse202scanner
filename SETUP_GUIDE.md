# 🎯 NSE Scanner Pro v5.0 — Complete Setup Guide

## What's New in v5.0

| Feature | What It Does |
|---------|-------------|
| **Sidebar fix** | Page no longer jumps when you interact with sliders/buttons |
| **📋 Signal Log** | Every scan auto-records signals with timestamp. Browse by date. |
| **📊 Tracker** | Forward-tests all recorded signals — shows which hit Target vs Stop Loss |
| **🤖 GitHub Actions** | Auto-scans NSE 500 daily at 4:30 PM and 7:00 PM IST (free) |
| **📥 CSV Download** | Download full signal history for any date or the entire tracker |
| **🔐 Password Gate** | Optional — password-protect your app for sharing |
| **Breeze token detection** | Clear error message when session token expires |

---

## Part 1: Update Your Streamlit App (5 minutes)

### Step 1: Download and extract the zip

Extract `nse_scanner_pro_v5.zip`. You'll see these NEW files:

```
NEW FILES:
  auto_scanner.py          ← Standalone scanner for GitHub Actions
  signal_tracker.py        ← Signal recording + tracking module
  SETUP_GUIDE.md           ← This file
  signals/.gitkeep         ← Empty folder for signal CSVs
  .github/workflows/daily_scan.yml  ← GitHub Actions workflow

UPDATED FILES:
  app.py                   ← v5.0 with Signal Log, Tracker, sidebar fix
```

### Step 2: Push to GitHub

```bash
cd /path/to/nse101scanner

# Verify the workflow file exists:
ls .github/workflows/daily_scan.yml
# If folder missing: mkdir -p .github/workflows
# Then copy daily_scan.yml into it

git add .
git commit -m "v5.0: Signal tracking + GitHub Actions auto-scanner"
git push origin main
```

### Step 3: Verify on Streamlit Cloud

App auto-redeploys. Check:
- Sidebar now has 10 pages (Signal Log and Tracker are new)
- Version shows "v5.0 — Signal Tracker"
- Run a scan → signals auto-save to Signal Log tab

---

## Part 2: GitHub Actions Auto-Scanner (10 minutes)

**Cost: FREE.** Public repos = unlimited minutes. Private repos = 2,000 free min/month (you'll use ~220).

**No subscriptions needed.**

### Step 1: Enable GitHub Actions

1. Go to `https://github.com/YOUR_USERNAME/nse101scanner`
2. Click **Actions** tab
3. If yellow banner appears → click **"I understand my workflows, go ahead and enable them"**
4. You should see **"Daily NSE Scanner"** in the left sidebar

### Step 2: Give Actions write permission

1. Repo → **Settings** → left sidebar → **Actions** → **General**
2. Scroll to **"Workflow permissions"**
3. Select **"Read and write permissions"**
4. Click **Save**

⚠️ Without this, the workflow runs but can't save results!

### Step 3: Test with Manual Trigger

1. **Actions** tab → click **"Daily NSE Scanner"** on left
2. Click **"Run workflow"** dropdown (blue button, right side)
3. Branch: **main**, Mode: **eod**
4. Click **"Run workflow"**
5. Click the new run to watch. Takes 3-5 minutes.
6. Green checkmark = success!

### Step 4: Verify Results

1. Repo → **Code** tab → open `signals/` folder
2. You should see `YYYY-MM-DD_signals.csv` and `tracker.csv`
3. Open Streamlit app → **📋 Signal Log** → select today → signals appear!

### Step 5: Done! Auto-scans now run every weekday:

| IST Time | What |
|---:|---|
| **4:30 PM** | Quick scan (Nifty 200) |
| **7:00 PM** | Full scan (NSE 500) + tracker update |

---

## Part 3: Daily Workflow

### After Market Close
- GitHub Actions runs automatically
- Open **📋 Signal Log** → today's signals
- Click **🔄 Update Tracker** → checks SL/T1 hits
- **📊 Tracker** → cumulative win rate by strategy

### Weekly Review
- **📊 Tracker** → Strategy Performance table
- **📥 Download CSV** → analyze in Excel
- Compare forward-test vs backtest (🧪 tab)

### CSV Columns
```
Date, Time, Strategy, Symbol, Signal, Entry, SL, T1, T2,
Confidence, RS, Sector, Regime, Status, Exit_Date, Exit_Reason, PnL_Pct
```

Status: OPEN → TARGET (win) / STOPPED (loss) / EXPIRED (30d timeout)

---

## Part 4: Optional Features

### Password Protection
Streamlit Cloud → Settings → Secrets:
```toml
APP_PASSWORD = "your_chosen_password"
```

### Telegram Alerts
```toml
TELEGRAM_BOT_TOKEN = "your_bot_token"
TELEGRAM_CHAT_ID = "your_chat_id"
```

### Change Schedule
Edit `.github/workflows/daily_scan.yml`:
```yaml
schedule:
  - cron: '0 11 * * 1-5'    # 4:30 PM IST (= 11:00 UTC)
  - cron: '30 13 * * 1-5'   # 7:00 PM IST (= 13:30 UTC)
```
Use https://crontab.guru/ — cron uses UTC (IST minus 5h30m).

---

## FAQ

**Will GitHub Actions cost anything?** No. Free for public repos. Private repos: 2,000 free min/month.

**Can yfinance ban me?** Very unlikely. ~500 stocks, twice daily. Well within limits.

**Why not Breeze for auto-scans?** Breeze tokens expire daily — can't automate headlessly. When you use the app with Breeze connected, those signals also get recorded.

**How to see which strategy is profitable?** Run scans 2-3 weeks → **📊 Tracker** → Strategy Performance.

**Where is data stored?** `signals/` folder in your GitHub repo. No external database.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Actions not running | Enable in Actions tab. Set write permissions in Settings. |
| No signals in Signal Log | Run a scan first, or trigger Actions manually. |
| Sidebar jumps | Clear browser cache. v5 query_params fix prevents this. |
| Breeze token expired | Generate new one from ICICI Direct portal. |

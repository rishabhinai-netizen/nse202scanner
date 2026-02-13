"""
NSE SCANNER PRO v5.1 — Smart Targets + Confluence + F&O
============================================================
v5.1 changes:
- Smart R:R targets (resistance-based T1, Fibonacci T2 — stock-specific)
- Multi-strategy confluence detection (🔥 when stock appears in 2+ strategies)
- F&O stock tagging on all signals
- Telegram alerts include RS, sector performance, F&O status
- Sector RS shown alongside signals
- PWA manifest for mobile install (Add to Home Screen)
- GitHub Actions fix (streamlit conditional import)
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time as dtime, date
import pytz, json, os

from stock_universe import get_stock_universe, get_sector, NIFTY_50
from data_engine import (
    fetch_batch_daily, fetch_nifty_data,
    Indicators, BreezeEngine, now_ist, IST
)
from scanners import (
    STRATEGY_PROFILES, DAILY_SCANNERS, INTRADAY_SCANNERS,
    run_scanner, run_all_scanners, detect_market_regime, ScanResult
)
from risk_manager import RiskManager
from enhancements import (
    plot_candlestick, compute_sector_performance, plot_sector_heatmap,
    compute_rs_rankings, plot_rs_scatter,
    load_journal, save_journal, add_journal_entry, compute_journal_analytics, plot_equity_curve,
    check_weekly_alignment, compute_market_breadth, plot_breadth_gauge,
)
from backtester import backtest_strategy, backtest_multi_stock, BacktestResult
from tooltips import TIPS, tip
from signal_tracker import (
    load_signals, load_tracker, save_signals_today,
    compute_tracker_stats, update_open_signals_live,
    generate_csv_download, get_signal_dates,
)
from fno_list import is_fno, get_fno_tag

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="NSE Scanner Pro", page_icon="🎯", layout="wide",
    initial_sidebar_state="auto",  # Auto-collapse on mobile
    menu_items={"Get Help": None, "Report a bug": None, "About": None}
)

# ============================================================================
# PASSWORD GATE (optional — set APP_PASSWORD in Streamlit secrets to enable)
# ============================================================================
try:
    _app_pw = st.secrets.get("APP_PASSWORD", "")
except Exception:
    _app_pw = ""
if _app_pw and _app_pw != "your_password":
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if not st.session_state.authenticated:
        st.markdown("## 🎯 NSE Scanner Pro")
        st.markdown("---")
        pw = st.text_input("Enter access password", type="password", key="pw_input")
        if st.button("Login", key="pw_btn"):
            if pw == _app_pw:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Wrong password.")
        st.stop()
else:
    st.session_state.authenticated = True

# ============================================================================
# CSS — Mobile-friendly, overflow-safe, branding hidden
# ============================================================================
st.markdown("""
<style>
    /* === NUCLEAR BRANDING HIDE — covers all Streamlit versions === */
    #MainMenu, footer, header, 
    .stDeployButton,
    [data-testid="stToolbar"],
    [data-testid="stStatusWidget"],
    [data-testid="stHeader"],
    [data-testid="manage-app-button"],
    #stDecoration,
    /* Class-based (varies by Streamlit version) */
    .viewerBadge_container__r5tak, .styles_viewerBadge__CvC9N,
    ._profileContainer_51w34_53, ._profilePreview_51w34_63,
    .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_link__qRIco,
    .styles_viewerBadge__1yB5_, .viewerBadge_text__1JaDK,
    /* Catch-all for ANY link to github or streamlit */
    a[href*="github.com"],
    a[href*="streamlit.io"],
    a[href*="github"],
    /* Footer and deploy elements */
    .stApp > footer,
    div[class*="stDeployButton"],
    div[class*="viewerBadge"],
    div[class*="StatusWidget"],
    button[kind="header"],
    /* Mobile-specific toolbar */
    section[data-testid="stSidebar"] a[href*="github"],
    .stApp header,
    .stApp [data-testid="stHeader"] {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        min-height: 0 !important;
        max-height: 0 !important;
        overflow: hidden !important;
        position: absolute !important;
        top: -9999px !important;
    }
    
    /* Also hide the top padding that header leaves behind */
    .stApp > header + div { padding-top: 0 !important; }
    .block-container { padding-top: 1rem !important; }
    
    /* Metric boxes: compact, no overflow */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1d23, #252830);
        border: 1px solid #333; border-radius: 8px; padding: 8px 10px; overflow: hidden;
    }
    div[data-testid="stMetric"] label { font-size: 0.7rem !important; color: #888 !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1rem !important; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    
    /* Custom cards */
    .pc { background: linear-gradient(135deg,#1a1d23,#252830); border:1px solid #333;
          border-radius:8px; padding:8px 12px; margin:3px 0; }
    .pc .lb { font-size:0.68rem; color:#888; margin-bottom:1px; }
    .pc .vl { font-size:0.95rem; font-weight:600; color:#fafafa; }
    .pc .vl.g { color:#00d26a; } .pc .vl.r { color:#ff4757; } .pc .vl.o { color:#FF6B35; }
    .pc .vl.y { color:#ffd700; }
    
    /* Focus panel */
    .focus { background:linear-gradient(135deg,#0d1117,#161b22); border:1px solid #FF6B35;
             border-radius:12px; padding:16px 20px; margin:10px 0; }
    .focus h3 { margin:0 0 8px 0; color:#FF6B35; font-size:1.1rem; }
    .focus .regime { font-size:1.3rem; font-weight:700; margin:4px 0; }
    .focus .tip { color:#aaa; font-size:0.82rem; margin-top:6px; }
    
    /* Strategy cards */
    .sc { background:#1a1d23; border:1px solid #333; border-radius:10px; padding:12px; margin:5px 0; }
    .sc:hover { border-color:#FF6B35; }
    .sc.blocked { opacity:0.4; border-color:#ff4757; }
    .sc.ideal { border-left:3px solid #00d26a; }
    .sc.caution { border-left:3px solid #ffd700; }
    
    /* Badges */
    .bg { display:inline-block; padding:2px 7px; border-radius:10px; font-size:0.65em; }
    .bg-s { background:#1e3a5f; color:#5dade2; } .bg-i { background:#3e2723; color:#ff8a65; }
    .bg-p { background:#1b5e20; color:#81c784; } .bg-o { background:#4a148c; color:#ce93d8; }
    .bg-blocked { background:#3d1a1a; color:#ff4757; }
    .bg-ideal { background:#0d3320; color:#00d26a; }
    .bg-caution { background:#3d3a1a; color:#ffd700; }
    
    /* Stale data warning */
    .stale { background:#3d2a1a; border:1px solid #ff8a65; border-radius:8px;
             padding:8px 14px; color:#ff8a65; font-size:0.85rem; }
    
    /* Workflow */
    .ws { border-left:3px solid #FF6B35; padding:7px 12px; margin:5px 0;
          background:#1a1d23; border-radius:0 8px 8px 0; }
    
    /* Breeze banner */
    .bb { padding:6px 12px; border-radius:8px; margin:6px 0; font-size:0.8rem; }
    .bb-on { background:#0d3320; border:1px solid #1b5e20; color:#81c784; }
    .bb-off { background:#1a1d23; border:1px solid #333; color:#888; }
    
    .dataframe { font-size:0.78rem !important; }
    
    /* Mobile: ensure sidebar works */
    @media (max-width: 768px) {
        .pc .vl { font-size:0.85rem; }
        .pc .lb { font-size:0.62rem; }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-size:0.85rem !important; }
        .focus { padding:12px 14px; }
        .sc { padding:10px; }
        /* Extra mobile branding kill */
        header[data-testid="stHeader"] { display:none !important; }
        .stApp > header { display:none !important; }
    }
</style>
""", unsafe_allow_html=True)

# PWA — Add to Home Screen support for mobile
st.markdown("""
<link rel="manifest" href="data:application/json;base64,eyJuYW1lIjogIk5TRSBTY2FubmVyIFBybyIsICJzaG9ydF9uYW1lIjogIlNjYW5uZXIiLCAic3RhcnRfdXJsIjogIi4iLCAiZGlzcGxheSI6ICJzdGFuZGFsb25lIiwgImJhY2tncm91bmRfY29sb3IiOiAiIzBFMTExNyIsICJ0aGVtZV9jb2xvciI6ICIjRkY2QjAwIiwgIm9yaWVudGF0aW9uIjogInBvcnRyYWl0IiwgImljb25zIjogW3sic3JjIjogImh0dHBzOi8vZW0tY29udGVudC56b2JqLm5ldC9zb3VyY2UvYXBwbGUvMzU0L2RpcmVjdC1oaXRfMWYzYWYucG5nIiwgInNpemVzIjogIjEyMHgxMjAiLCAidHlwZSI6ICJpbWFnZS9wbmcifV19" />
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="NSE Scanner">
<meta name="theme-color" content="#FF6B00">
""", unsafe_allow_html=True)
# ============================================================================
for k, v in {
    "watchlist":[], "scan_results":{}, "regime":None,
    "data_loaded":False, "stock_data":{}, "enriched_data":{},
    "nifty_data":None, "capital":500000,
    "breeze_connected":False, "breeze_engine":None, "breeze_msg":"",
    "workflow_checks":{}, "universe_size":"nifty200",
    "telegram_token":"", "telegram_chat_id":"",
    "journal":None, "last_scan_time":None, "sector_rankings":{},
    "rs_filter": 70, "regime_filter": True,
}.items():
    if k not in st.session_state: st.session_state[k] = v

if st.session_state.journal is None:
    st.session_state.journal = load_journal()

# Breeze auto-connect
def try_breeze():
    if st.session_state.breeze_connected: return
    try:
        ak = st.secrets.get("BREEZE_API_KEY","")
        asc = st.secrets.get("BREEZE_API_SECRET","")
        st_ = st.secrets.get("BREEZE_SESSION_TOKEN","")
        if ak and asc and st_ and "your_" not in ak:
            import signal as _sig
            def _timeout_handler(signum, frame):
                raise TimeoutError("Breeze connection timed out")
            # Set 10-second timeout for Breeze connection
            old_handler = None
            try:
                old_handler = _sig.signal(_sig.SIGALRM, _timeout_handler)
                _sig.alarm(10)
            except (AttributeError, OSError):
                pass  # Windows or restricted environment
            try:
                e = BreezeEngine()
                ok, msg = e.connect(ak, asc, st_)
                st.session_state.breeze_connected = ok
                st.session_state.breeze_msg = msg
                if ok: st.session_state.breeze_engine = e
            finally:
                try:
                    _sig.alarm(0)
                    if old_handler: _sig.signal(_sig.SIGALRM, old_handler)
                except (AttributeError, OSError):
                    pass
    except TimeoutError:
        st.session_state.breeze_msg = "Breeze connection timed out (10s)"
    except Exception as ex:
        st.session_state.breeze_msg = f"Breeze: {str(ex)[:80]}"
try_breeze()

# ============================================================================
# HELPERS
# ============================================================================
def pc(label, value, css=""):
    if isinstance(value,(int,float)):
        v = f"₹{value:,.0f}" if value>=10000 else (f"₹{value:,.1f}" if value>=100 else f"₹{value:,.2f}")
    else: v = str(value)
    st.markdown(f'<div class="pc"><div class="lb">{label}</div><div class="vl {css}">{v}</div></div>', unsafe_allow_html=True)

def fp(v):
    if v>=10000: return f"₹{v:,.0f}"
    elif v>=100: return f"₹{v:,.1f}"
    else: return f"₹{v:,.2f}"

def send_tg(msg):
    try:
        tk = st.session_state.telegram_token or st.secrets.get("TELEGRAM_BOT_TOKEN","")
        ci = st.session_state.telegram_chat_id or st.secrets.get("TELEGRAM_CHAT_ID","")
    except Exception:
        tk, ci = st.session_state.get("telegram_token",""), st.session_state.get("telegram_chat_id","")
    if not tk or not ci: return False
    try:
        import requests
        return requests.post(f"https://api.telegram.org/bot{tk}/sendMessage",
                             json={"chat_id":ci,"text":msg,"parse_mode":"HTML"},timeout=10).status_code == 200
    except: return False

def fmt_alert(r, is_confluence=False):
    fno = "F&O ✓" if is_fno(r.symbol) else "Cash"
    sec_perf = _get_sector_perf_str(r.sector)
    conf_tag = "🔥 CONFLUENCE — " if is_confluence else ""
    return (f"{conf_tag}🎯 <b>{r.strategy}</b> — {r.signal}\n"
            f"📈 <b>{r.symbol}</b> ({r.sector}) [{fno}]\n"
            f"💰 CMP: {fp(r.cmp)} | Entry: {fp(r.entry)}\n"
            f"🛑 SL: {fp(r.stop_loss)} | T1: {fp(r.target_1)} | R:R 1:{r.risk_reward:.1f}\n"
            f"📊 Conf: {r.confidence}% | RS: {int(r.rs_rating)} | {sec_perf}\n"
            f"🏷️ Regime: {r.regime_fit}")

def _get_sector_perf_str(sector):
    """Get sector performance string for display."""
    ranks = st.session_state.get("sector_rankings", {})
    if sector and sector in ranks:
        rank_pct = ranks[sector]
        return f"Sector: #{int(100-rank_pct)+1}"
    return "Sector: -"

def results_df(results):
    rows = []
    for r in results:
        fno = "F&O" if is_fno(r.symbol) else "Cash"
        rows.append({
            "Symbol": r.symbol, "Signal": r.signal, "CMP": fp(r.cmp), "Entry": fp(r.entry),
            "SL": fp(r.stop_loss), "T1": fp(r.target_1), "R:R": f"1:{r.risk_reward:.1f}",
            "Conf": f"{r.confidence}%", "RS": int(r.rs_rating), "Mkt": fno,
            "Regime": r.regime_fit, "Sector": r.sector, "Hold": r.hold_type,
        })
    return pd.DataFrame(rows)


def detect_confluence(scan_results: dict) -> dict:
    """Find stocks appearing in 2+ strategies. Returns {symbol: [strategies]}."""
    symbol_strategies = {}
    for strategy, signals in scan_results.items():
        for r in signals:
            if r.symbol not in symbol_strategies:
                symbol_strategies[r.symbol] = []
            symbol_strategies[r.symbol].append(strategy)
    return {sym: strats for sym, strats in symbol_strategies.items() if len(strats) >= 2}


def send_scan_alerts(scan_results: dict):
    """Send telegram alerts with confluence awareness."""
    confluence = detect_confluence(scan_results)
    confluence_symbols = set(confluence.keys())
    
    # Send confluence alerts first (higher priority)
    if confluence:
        msg = "🔥🔥 <b>CONFLUENCE ALERT</b> 🔥🔥\n"
        for sym, strats in confluence.items():
            strat_names = [STRATEGY_PROFILES.get(s, {}).get("name", s) for s in strats]
            fno = "F&O ✓" if is_fno(sym) else "Cash"
            msg += f"\n📈 <b>{sym}</b> [{fno}] — {len(strats)} strategies:\n"
            msg += ", ".join(strat_names) + "\n"
        send_tg(msg)
    
    # Then individual signals
    for strat, signals in scan_results.items():
        for r in signals:
            is_conf = r.symbol in confluence_symbols
            send_tg(fmt_alert(r, is_confluence=is_conf))

def is_data_stale():
    """Check if scan data is stale (>15 min old)."""
    if not st.session_state.last_scan_time: return True
    age = (now_ist() - st.session_state.last_scan_time).total_seconds() / 60
    return age > 15

def compute_sector_ranks():
    """Build sector ranking dict for filtering."""
    data = st.session_state.enriched_data or st.session_state.stock_data
    if not data: return {}
    perf = compute_sector_performance(data, get_sector)
    if perf.empty: return {}
    perf["rank"] = perf["avg_1m"].rank(pct=True) * 100
    return perf["rank"].to_dict()

def load_data():
    syms = get_stock_universe(st.session_state.universe_size)
    pb = st.progress(0, "Starting...")
    def cb(p, t): pb.progress(min(p, 0.95), t)
    data = fetch_batch_daily(syms, "1y", cb)
    pb.progress(0.96, "Fetching Nifty...")
    nifty = fetch_nifty_data()
    pb.progress(0.97, "Computing indicators...")
    enriched = {}
    for s, df in data.items():
        try: enriched[s] = Indicators.enrich_dataframe(df)
        except: enriched[s] = df
    pb.progress(1.0, f"✅ {len(data)} stocks loaded!")
    return data, nifty, enriched

# ============================================================================
# SIDEBAR — with query_params persistence (fixes page jump on interaction)
# ============================================================================
PAGES = [
    "📊 Dashboard", "🔍 Scanner Hub", "📈 Charts & RS",
    "🧪 Backtest", "📋 Signal Log", "📊 Tracker",
    "📐 Trade Planner", "⭐ Watchlist", "📓 Journal", "⚙️ Settings"
]

# Persist page selection in URL query params
qp = st.query_params
default_page = qp.get("page", PAGES[0])
if default_page not in PAGES:
    default_page = PAGES[0]

with st.sidebar:
    st.markdown("## 🎯 NSE Scanner Pro")
    st.caption("v5.1 — Smart Targets + Confluence")
    st.markdown("---")
    page = st.radio("Navigation", PAGES,
                    index=PAGES.index(default_page),
                    label_visibility="collapsed", key="nav_radio")
    
    # Persist to URL so page survives reruns
    if page != qp.get("page"):
        st.query_params["page"] = page
    
    st.markdown("---")
    ist = now_ist()
    is_mkt = dtime(9,15) <= ist.time() <= dtime(15,30) and ist.weekday() < 5
    st.caption(f"{'🟢' if is_mkt else '🔴'} {ist.strftime('%d %b, %I:%M %p IST')}")
    
    if st.session_state.regime:
        rg = st.session_state.regime
        st.markdown(f"**{rg['regime_display']}**")
        nv = rg.get("nifty_close", 0)
        if isinstance(nv,(int,float)): st.caption(f"Nifty ₹{nv:,.0f} | Pos {rg['position_multiplier']*100:.0f}%")
    
    sigs = sum(len(v) for v in st.session_state.scan_results.values())
    st.caption(f"Signals: {sigs} | Watch: {len(st.session_state.watchlist)}")
    
    # Signal tracker summary in sidebar
    tracker_df = load_tracker()
    if tracker_df is not None and not tracker_df.empty:
        open_count = len(tracker_df[tracker_df["Status"] == "OPEN"])
        target_count = len(tracker_df[tracker_df["Status"] == "TARGET"])
        stop_count = len(tracker_df[tracker_df["Status"] == "STOPPED"])
        st.caption(f"📊 Tracked: ⏳{open_count} 🟢{target_count} 🔴{stop_count}")
    
    st.markdown("---")
    if st.session_state.breeze_connected:
        st.markdown('<div class="bb bb-on">✅ Breeze Connected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="bb bb-off">⚪ Breeze Off — intraday disabled</div>', unsafe_allow_html=True)
    
    if st.session_state.last_scan_time:
        age = (now_ist() - st.session_state.last_scan_time).total_seconds() / 60
        color = "color:#00d26a" if age < 15 else "color:#ff8a65"
        st.markdown(f'<small style="{color}">Last scan: {int(age)}m ago</small>', unsafe_allow_html=True)


# ============================================================================
# DAILY FOCUS PANEL — "What matters NOW"
# ============================================================================
def render_focus_panel():
    """The #1 thing the user should see."""
    ist = now_ist()
    hour = ist.hour
    minute = ist.minute
    t = ist.time()
    is_weekend = ist.weekday() >= 5
    regime = st.session_state.regime
    
    # Determine what matters NOW based on time
    if is_weekend:
        focus_title = "📅 Weekend Review"
        focus_action = "Run full Nifty 500 VCP scan. Review journal. Analyze sector rotation."
        focus_tip = "Best time for deep analysis without market noise."
    elif t < dtime(9,15):
        focus_title = "🌅 Pre-Market Prep"
        focus_action = "Load data → Check market health → Review global cues → Plan watchlist"
        focus_tip = "Don't trade the first candle. Observe, then act."
    elif t < dtime(9,45):
        focus_title = "🔔 Market Open — OBSERVE"
        focus_action = "Watch first 15-30 min candle. DO NOT trade. Let the noise settle."
        focus_tip = "Most false breakouts happen in the first 15 minutes."
    elif t < dtime(10,30):
        if st.session_state.breeze_connected:
            focus_title = "🔓 ORB Window (LIVE)"
            focus_action = "Run ORB scanner → Enter confirmed breakouts with volume"
        else:
            focus_title = "⏳ Morning Session"
            focus_action = "ORB requires Breeze API. Focus on swing setups instead."
        focus_tip = "Only take trades with volume confirmation > 1.5x average."
    elif t < dtime(12,30):
        focus_title = "📈 Mid-Morning"
        focus_action = "VWAP Reclaim window (needs Breeze). Trail morning stops."
        focus_tip = "If morning trades are green, trail stop to breakeven."
    elif t < dtime(13,30):
        focus_title = "🍽️ Lunch Session"
        focus_action = "Lunch Low reversal window (needs Breeze). Low-volume = traps."
        focus_tip = "Lunch hour is the least reliable time. Be extra selective."
    elif t < dtime(15,0):
        focus_title = "⏳ Afternoon"
        focus_action = "Review open positions. Prepare for close."
        focus_tip = "Start identifying BTST candidates for 3:20 PM entry."
    elif t < dtime(15,30):
        focus_title = "⭐ Power Hour — Last 30 Min ATH"
        focus_action = "Run ATH scanner NOW → Buy strongest stocks at 3:25 PM"
        focus_tip = "This is the BTST window. Overnight gap-up probability is highest here."
    else:
        focus_title = "📋 Post-Market — Swing Scans"
        focus_action = "Run VCP, EMA21, 52WH, Short scanners → Build tomorrow's watchlist"
        focus_tip = "Post-market is the best time for swing analysis. No noise, clean data."
    
    # Regime context
    regime_str = ""
    if regime:
        r = regime["regime"]
        if r == "EXPANSION":
            regime_str = "🟢 EXPANSION — Go aggressive on breakouts. Full position sizing."
        elif r == "ACCUMULATION":
            regime_str = "🟡 ACCUMULATION — Be selective. Focus on VCP and EMA21 setups."
        elif r == "DISTRIBUTION":
            regime_str = "🟠 DISTRIBUTION — Defensive mode. Prefer shorts or skip breakouts."
        elif r == "PANIC":
            regime_str = "🔴 PANIC — Cash is king. Only short setups or sit out completely."
        else:
            regime_str = "⚪ Unknown regime — Load data to detect."
    
    st.markdown(f"""<div class="focus">
        <h3>🎯 {focus_title}</h3>
        <div style="color:#fafafa; font-size:0.9rem;">{focus_action}</div>
        {"<div class='regime'>" + regime_str + "</div>" if regime_str else ""}
        <div class="tip">💡 {focus_tip}</div>
    </div>""", unsafe_allow_html=True)


# ============================================================================
# DASHBOARD
# ============================================================================
def page_dashboard():
    st.markdown("# 📊 Dashboard")
    
    # FOCUS PANEL — #1 thing user sees
    render_focus_panel()
    
    # Data staleness check
    if st.session_state.data_loaded and is_data_stale():
        st.markdown('<div class="stale">⚠️ Data may be stale. Click Refresh to update.</div>', unsafe_allow_html=True)
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.session_state.universe_size = st.selectbox("Universe", ["nifty50","nifty200","nifty500"], index=1,
            format_func=lambda x: {"nifty50":"Nifty 50","nifty200":"Nifty 200","nifty500":"Nifty 500"}[x])
    with c2:
        if st.button("🔄 Load / Refresh Data", type="primary", use_container_width=True):
            st.cache_data.clear()
            data, nifty, enriched = load_data()
            st.session_state.stock_data = data
            st.session_state.nifty_data = nifty
            st.session_state.enriched_data = enriched
            st.session_state.data_loaded = True
            st.session_state.last_scan_time = now_ist()
            # Detect regime with breadth
            breadth = compute_market_breadth(enriched)
            st.session_state.regime = detect_market_regime(nifty, breadth)
            st.session_state.sector_rankings = compute_sector_ranks()
            st.rerun()
    
    if not st.session_state.data_loaded:
        st.info("👆 Click **Load / Refresh Data** to start.")
        # Show strategies with regime compatibility
        cols = st.columns(4)
        for i, (k, p) in enumerate(STRATEGY_PROFILES.items()):
            with cols[i%4]:
                nb = " 🔌" if p.get("requires_intraday") else ""
                st.markdown(f'<div class="sc"><strong>{p["icon"]} {p["name"]}</strong>{nb}<br>'
                    f'<span class="bg bg-{"i" if p["type"]=="Intraday" else "s"}">{p["type"]}</span> '
                    f'<span style="color:#888;font-size:0.7em">{p["hold"]}</span><br>'
                    f'<span style="color:#00d26a">Win {p["win_rate"]}%</span></div>', unsafe_allow_html=True)
        return
    
    # === REGIME ===
    rg = st.session_state.regime
    if rg:
        st.markdown("### 🧠 Market Regime")
        st.caption(tip("regime"))
        c1,c2,c3,c4 = st.columns(4)
        with c1: pc("Regime", rg["regime_display"])
        with c2: pc("Regime Score", f"{rg['score']}/{rg['max_score']}")
        nv = rg.get("nifty_close", 0)
        with c3: pc("Nifty", f"₹{nv:,.0f}" if isinstance(nv,(int,float)) else str(nv))
        with c4: pc("Position Size", f"{rg['position_multiplier']*100:.0f}%",
                     "g" if rg["position_multiplier"]>=0.6 else "r")
        
        # Show allowed/blocked strategies
        c1,c2,c3 = st.columns(3)
        with c1:
            st.markdown("**✅ Ideal now:**")
            for s in rg.get("allowed_strategies",[]):
                st.markdown(f"  {STRATEGY_PROFILES.get(s,{}).get('icon','')} {STRATEGY_PROFILES.get(s,{}).get('name',s)}")
        with c2:
            st.markdown("**⚠️ Use caution:**")
            for s in rg.get("caution_strategies",[]):
                st.markdown(f"  {STRATEGY_PROFILES.get(s,{}).get('icon','')} {STRATEGY_PROFILES.get(s,{}).get('name',s)}")
        with c3:
            st.markdown("**🚫 Blocked:**")
            for s in rg.get("blocked_strategies",[]):
                st.markdown(f"  {STRATEGY_PROFILES.get(s,{}).get('icon','')} {STRATEGY_PROFILES.get(s,{}).get('name',s)}")
        
        with st.expander("📊 Regime Scoring Breakdown"):
            for d in rg.get("details",[]):
                st.markdown(f"  {d}")
            st.caption(f"Score: {rg['score']}/{rg['max_score']} | Vol expansion: {rg.get('vol_expansion','-')}x | "
                       f"RSI: {rg.get('nifty_rsi','-')} | Dist from 52WH: {rg.get('nifty_pct_52wh','-')}%")
    
    # === BREADTH ===
    breadth = compute_market_breadth(st.session_state.enriched_data or st.session_state.stock_data)
    if breadth:
        st.markdown("### 📊 Market Breadth")
        st.caption(tip("ad_ratio"))
        c1,c2,c3,c4 = st.columns(4)
        with c1: pc("Advancing", str(breadth["advancing"]), "g")
        with c2: pc("Declining", str(breadth["declining"]), "r")
        with c3: pc("A/D Ratio", str(breadth["ad_ratio"]))
        with c4: pc("> 200 SMA", f"{breadth['above_200sma_pct']}%",
                     "g" if breadth["above_200sma_pct"]>50 else "r")
        fig = plot_breadth_gauge(breadth)
        if fig: st.plotly_chart(fig, use_container_width=True)
    
    # === SECTOR ===
    st.markdown("### 🗺️ Sector Rotation")
    sector_df = compute_sector_performance(st.session_state.enriched_data or st.session_state.stock_data, get_sector)
    if not sector_df.empty:
        fig = plot_sector_heatmap(sector_df)
        if fig: st.plotly_chart(fig, use_container_width=True)
    
    # === QUICK SCAN ===
    st.markdown("### ⚡ Quick Scan")
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.rs_filter = st.slider("Min RS Rating (Long signals only)", 0, 95,
            st.session_state.rs_filter, 5, key="dash_rs",
            help=tip("rs_rating") + " **Note:** SHORT signals are never filtered by RS — weak stocks are ideal short candidates.")
    with c2:
        st.session_state.regime_filter = st.checkbox("Block strategies not suited for current regime",
            value=st.session_state.regime_filter, key="dash_regime",
            help=tip("regime_fit"))
    
    if st.button("🚀 Run All Swing Scanners", type="primary"):
        with st.spinner("Scanning with regime + RS filters..."):
            results = run_all_scanners(
                st.session_state.stock_data, st.session_state.nifty_data, True,
                regime=st.session_state.regime if st.session_state.regime_filter else None,
                has_intraday=st.session_state.breeze_connected,
                sector_rankings=st.session_state.sector_rankings,
                min_rs=st.session_state.rs_filter,
            )
            st.session_state.scan_results = results
            st.session_state.last_scan_time = now_ist()
            # Auto-save signals to log
            all_sigs = [r for sigs in results.values() for r in sigs]
            saved = save_signals_today(all_sigs, st.session_state.regime)
            send_scan_alerts(results)
            st.rerun()
    
    if st.session_state.scan_results:
        # Confluence section
        confluence = detect_confluence(st.session_state.scan_results)
        if confluence:
            st.markdown("### 🔥 Confluence — Multiple Strategy Confirmation")
            for sym, strats in confluence.items():
                strat_names = [STRATEGY_PROFILES.get(s, {}).get("icon", "") + " " +
                              STRATEGY_PROFILES.get(s, {}).get("name", s) for s in strats]
                fno = get_fno_tag(sym)
                st.success(f"**{sym}** [{fno}] — {len(strats)} strategies: {', '.join(strat_names)}")
        
        st.markdown("### 📋 Results")
        for strat, results in st.session_state.scan_results.items():
            if not results: continue
            p = STRATEGY_PROFILES.get(strat, {})
            with st.expander(f"{p.get('icon','')} {p.get('name',strat)} — {len(results)}", expanded=True):
                st.dataframe(results_df(results), use_container_width=True, hide_index=True)


# ============================================================================
# SCANNER HUB
# ============================================================================
def page_scanner_hub():
    st.markdown("# 🔍 Scanner Hub")
    if not st.session_state.data_loaded:
        st.warning("Load data from Dashboard first.")
        return
    
    render_focus_panel()
    
    with st.expander("ℹ️ What do these terms mean?"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**RS Rating:** {tip('rs_rating')}")
            st.markdown(f"**Confidence:** {tip('confidence')}")
            st.markdown(f"**Regime Fit:** {tip('regime_fit')}")
        with c2:
            st.markdown(f"**R:R Ratio:** {tip('risk_reward')}")
            st.markdown(f"**Weekly Aligned:** {tip('weekly_aligned')}")
            st.markdown(f"**Sector Filter:** {tip('sector')}")
    
    # === FILTER CONTROLS (visible on Scanner Hub too) ===
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.rs_filter = st.slider("Min RS Rating (Long signals only)", 0, 95,
            st.session_state.rs_filter, 5, key="hub_rs",
            help=tip("rs_rating") + " **Note:** This filter only applies to BUY signals. SHORT signals are NOT filtered by RS — weak stocks (low RS) are ideal short candidates.")
    with c2:
        st.session_state.regime_filter = st.checkbox("Block strategies not suited for current regime",
            value=st.session_state.regime_filter, key="hub_regime",
            help=tip("regime_fit"))
    
    rg = st.session_state.regime
    cols = st.columns(4)
    selected = None
    
    for i, (k, p) in enumerate(STRATEGY_PROFILES.items()):
        with cols[i%4]:
            needs_breeze = p.get("requires_intraday", False)
            # Regime fit
            fit_class = ""
            fit_badge = ""
            if rg:
                if k in rg.get("allowed_strategies",[]): fit_class = "ideal"; fit_badge = '<span class="bg bg-ideal">IDEAL</span>'
                elif k in rg.get("blocked_strategies",[]): fit_class = "blocked"; fit_badge = '<span class="bg bg-blocked">BLOCKED</span>'
                elif k in rg.get("caution_strategies",[]): fit_class = "caution"; fit_badge = '<span class="bg bg-caution">CAUTION</span>'
            
            bc = {"Swing":"bg-s","Intraday":"bg-i","Positional":"bg-p","Overnight":"bg-o"}.get(p["type"],"bg-s")
            
            if needs_breeze and not st.session_state.breeze_connected:
                data_tag = '<small style="color:#ff4757">🔌 Needs Breeze</small>'
            elif needs_breeze:
                data_tag = '<small style="color:#00d26a">🔴 LIVE</small>'
            else:
                data_tag = '<small style="color:#5dade2">📊 Daily</small>'
            
            st.markdown(f'<div class="sc {fit_class}"><strong>{p["icon"]} {p["name"]}</strong><br>'
                f'<span class="bg {bc}">{p["type"]}</span> {fit_badge}<br>'
                f'<span style="color:#00d26a;font-size:0.8em">Win {p["win_rate"]}%</span> · '
                f'<span style="color:#FF6B35;font-size:0.8em">+{p["expectancy"]}%</span><br>{data_tag}</div>',
                unsafe_allow_html=True)
            
            # Disable button if blocked or needs breeze
            disabled = (fit_class == "blocked" and st.session_state.regime_filter) or \
                       (needs_breeze and not st.session_state.breeze_connected)
            if st.button("Scan" if not disabled else "🚫", key=f"s_{k}",
                         use_container_width=True, disabled=disabled,
                         help=tip(k)):
                selected = k
    
    st.markdown("---")
    if st.button("🚀 All Allowed Scanners", type="primary", use_container_width=True):
        selected = "ALL"
    
    if selected:
        n = st.session_state.nifty_data
        if selected == "ALL":
            with st.spinner("Scanning (regime + RS filtered)..."):
                st.session_state.scan_results = run_all_scanners(
                    st.session_state.stock_data, n, True,
                    regime=st.session_state.regime if st.session_state.regime_filter else None,
                    has_intraday=st.session_state.breeze_connected,
                    sector_rankings=st.session_state.sector_rankings,
                    min_rs=st.session_state.rs_filter)
        else:
            with st.spinner(f"Running {STRATEGY_PROFILES[selected]['name']}..."):
                st.session_state.scan_results[selected] = run_scanner(
                    selected, st.session_state.stock_data, n,
                    regime=st.session_state.regime if st.session_state.regime_filter else None,
                    has_intraday=st.session_state.breeze_connected,
                    sector_rankings=st.session_state.sector_rankings,
                    min_rs=st.session_state.rs_filter)
        st.session_state.last_scan_time = now_ist()
        # Auto-save signals to log
        all_sigs = [r for sigs in st.session_state.scan_results.values() for r in sigs]
        saved = save_signals_today(all_sigs, st.session_state.regime)
        send_scan_alerts(st.session_state.scan_results)
        st.rerun()
    
    if not st.session_state.scan_results:
        st.info("Select a strategy.")
        return
    
    # Confluence section
    confluence = detect_confluence(st.session_state.scan_results)
    if confluence:
        st.markdown("#### 🔥 Confluence — Multiple Strategy Confirmation")
        for sym, strats in confluence.items():
            strat_names = [STRATEGY_PROFILES.get(s, {}).get("icon", "") + " " +
                          STRATEGY_PROFILES.get(s, {}).get("name", s) for s in strats]
            fno = get_fno_tag(sym)
            st.success(f"**{sym}** [{fno}] — {len(strats)} strategies: {', '.join(strat_names)}")
    
    for strategy, results in st.session_state.scan_results.items():
        if not results: continue
        p = STRATEGY_PROFILES.get(strategy, {})
        st.markdown(f"#### {p.get('icon','')} {p.get('name',strategy)} — {len(results)}")
        st.dataframe(results_df(results), use_container_width=True, hide_index=True)
        
        for r in results:
            with st.expander(f"📋 {r.symbol} — {r.signal} | {fp(r.cmp)} | Conf {r.confidence}% | RS {r.rs_rating:.0f}"):
                c1,c2,c3,c4,c5,c6 = st.columns(6)
                with c1: pc("CMP", fp(r.cmp))
                with c2: pc("Entry", fp(r.entry))
                with c3: pc("SL", fp(r.stop_loss), "r")
                with c4: pc("T1", fp(r.target_1), "g")
                with c5: pc("T2", fp(r.target_2), "g")
                with c6: pc("Regime", r.regime_fit, "g" if r.regime_fit=="IDEAL" else ("y" if r.regime_fit=="CAUTION" else ""))
                
                # Multi-timeframe
                if r.symbol in (st.session_state.enriched_data or {}):
                    mtf = check_weekly_alignment(st.session_state.enriched_data[r.symbol])
                    if mtf["aligned"]:
                        st.success(f"✅ Weekly confirms ({mtf['score']}/4)")
                    else:
                        st.warning(f"⚠️ Weekly not aligned ({mtf['score']}/4)")
                
                # Chart
                if r.symbol in (st.session_state.enriched_data or {}):
                    fig = plot_candlestick(st.session_state.enriched_data[r.symbol], r.symbol,
                        entry=r.entry, stop_loss=r.stop_loss, target1=r.target_1, target2=r.target_2, signal=r.signal)
                    st.plotly_chart(fig, use_container_width=True)
                
                # "Why This Stock" — explainability
                st.markdown("**Why this stock qualified:**")
                for reason in r.reasons: st.markdown(f"  • {reason}")
                
                c1,c2,c3 = st.columns(3)
                with c1:
                    if st.button("⭐ Watch", key=f"a_{strategy}_{r.symbol}"):
                        ent = {"symbol":r.symbol,"strategy":strategy,"cmp":r.cmp,"entry":r.entry,
                               "stop":r.stop_loss,"target1":r.target_1,"target2":r.target_2,
                               "confidence":r.confidence,"date":r.timestamp,"entry_type":r.entry_type,"regime":r.regime_fit}
                        if not any(w["symbol"]==r.symbol and w["strategy"]==strategy for w in st.session_state.watchlist):
                            st.session_state.watchlist.append(ent)
                            st.success("Added!")
                with c2:
                    if st.button("📱 TG", key=f"t_{strategy}_{r.symbol}"):
                        if send_tg(fmt_alert(r)): st.success("Sent!")
                        else: st.warning("Setup Telegram first")
                with c3:
                    if st.button("📓 Journal", key=f"j_{strategy}_{r.symbol}"):
                        add_journal_entry({"symbol":r.symbol,"strategy":strategy,"signal":r.signal,
                            "entry":r.entry,"stop":r.stop_loss,"target1":r.target_1,"cmp":r.cmp,
                            "confidence":r.confidence,"status":"open","entry_date":r.timestamp,"reasons":r.reasons[:3]})
                        st.session_state.journal = load_journal()
                        st.success("Journaled!")


# ============================================================================
# CHARTS & RS
# ============================================================================
def page_charts_rs():
    st.markdown("# 📈 Charts & Relative Strength")
    if not st.session_state.data_loaded:
        st.warning("Load data first.")
        return
    tab1, tab2, tab3 = st.tabs(["📊 Chart", "💪 RS Rankings", "🗺️ Sectors"])
    with tab1:
        enriched = st.session_state.enriched_data or st.session_state.stock_data
        sel = st.selectbox("Stock", sorted(enriched.keys()))
        days = st.slider("Days", 30, 250, 90)
        if sel in enriched:
            fig = plot_candlestick(enriched[sel], sel, days=days)
            st.plotly_chart(fig, use_container_width=True)
            lat = enriched[sel].iloc[-1]
            c1,c2,c3,c4 = st.columns(4)
            with c1: pc("CMP", fp(lat["close"]))
            with c2: pc("RSI", f"{lat.get('rsi_14',0):.0f}")
            with c3: pc("52W High", fp(lat.get("high_52w",0)))
            with c4:
                mtf = check_weekly_alignment(enriched[sel])
                pc("Weekly", f"{'✅' if mtf['aligned'] else '❌'} {mtf['score']}/4")
    with tab2:
        rs_df = compute_rs_rankings(st.session_state.enriched_data or st.session_state.stock_data,
                                     st.session_state.nifty_data, get_sector)
        if not rs_df.empty:
            fig = plot_rs_scatter(rs_df)
            if fig: st.plotly_chart(fig, use_container_width=True)
            st.markdown("#### ⭐ Top 20 RS Leaders")
            st.dataframe(rs_df.head(20)[["Symbol","Sector","CMP","1M %","3M %","RS Score","RS Rank"]], use_container_width=True, hide_index=True)
            st.markdown("#### 🔴 Bottom 20")
            st.dataframe(rs_df.tail(20)[["Symbol","Sector","CMP","1M %","3M %","RS Score","RS Rank"]], use_container_width=True, hide_index=True)
    with tab3:
        sector_df = compute_sector_performance(st.session_state.enriched_data or st.session_state.stock_data, get_sector)
        if not sector_df.empty:
            fig = plot_sector_heatmap(sector_df)
            if fig: st.plotly_chart(fig, use_container_width=True)
            st.dataframe(sector_df.reset_index().rename(columns={"index":"Sector","stocks":"#","avg_1w":"1W%","avg_1m":"1M%","avg_3m":"3M%"}),
                         use_container_width=True, hide_index=True)


# ============================================================================
# TRADE PLANNER
# ============================================================================
def page_trade_planner():
    st.markdown("# 📐 Trade Planner")
    c1,c2 = st.columns(2)
    with c1:
        capital = st.number_input("Capital (₹)", value=st.session_state.capital, step=50000, min_value=10000)
        st.session_state.capital = capital
        risk_pct = st.slider("Risk %", 0.5, 3.0, 2.0, 0.25)
        sigs = [(f"{r.symbol} ({s}) {fp(r.cmp)}", s, r) for s, res in st.session_state.scan_results.items() for r in res]
        mode = st.radio("Input", ["Scanner","Manual"], horizontal=True)
        if mode == "Scanner" and sigs:
            sel = st.selectbox("Signal", [s[0] for s in sigs])
            r = sigs[[s[0] for s in sigs].index(sel)][2]
            entry, sl, short = r.entry, r.stop_loss, r.signal == "SHORT"
            st.info(f"**{r.symbol}** {r.signal} | CMP {fp(r.cmp)} | Conf {r.confidence}% | Regime: {r.regime_fit}")
        else:
            entry = st.number_input("Entry ₹", value=100.0, step=1.0)
            sl = st.number_input("SL ₹", value=95.0, step=1.0)
            short = st.checkbox("Short")
    with c2:
        mult = st.session_state.regime.get("position_multiplier", 1.0) if st.session_state.regime else 1.0
        if mult < 0.6: st.warning(f"⚠️ Regime: positions at {mult*100:.0f}%")
        if entry > 0 and sl > 0 and entry != sl:
            pos = RiskManager.calculate_position(capital, risk_pct, entry, sl, mult)
            tgt = RiskManager.calculate_targets(entry, sl, short)
            c1,c2 = st.columns(2)
            with c1: pc("Shares", f"{pos.shares:,}"); pc("Position", f"₹{pos.position_value:,.0f}")
            with c2: pc("Risk", f"₹{pos.risk_amount:,.0f}"); pc("% Portfolio", f"{pos.pct_of_portfolio:.1f}%")
            for w in pos.warnings: st.warning(w)
            st.markdown("### 🎯 Targets")
            c1,c2,c3 = st.columns(3)
            with c1: pc("T1 (1.5R)", fp(tgt.t1), "g")
            with c2: pc("T2 (2.5R)", fp(tgt.t2), "g")
            with c3: pc("T3 (4R)", fp(tgt.t3), "g")
            st.markdown(f"**Trail at** {fp(tgt.trailing_trigger)} → SL to breakeven")
            for lb, m in [("SL",-1),("T1",1.5),("T2",2.5),("T3",4)]:
                pnl = pos.shares * m * tgt.risk_per_share
                st.markdown(f"{'🟢' if pnl>0 else '🔴'} **{lb}:** ₹{pnl:+,.0f}")


# ============================================================================
# WATCHLIST
# ============================================================================
def page_watchlist():
    st.markdown("# ⭐ Watchlist")
    if not st.session_state.watchlist:
        st.info("Empty. Add from Scanner Hub.")
        return
    rows = [{"#":i+1,"Symbol":w["symbol"],"Strategy":w["strategy"],"CMP":fp(w.get("cmp",w["entry"])),
             "Entry":fp(w["entry"]),"SL":fp(w["stop"]),"T1":fp(w["target1"]),
             "Conf":f"{w['confidence']}%","Regime":w.get("regime","")
    } for i, w in enumerate(st.session_state.watchlist)]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    c1,c2 = st.columns(2)
    with c1:
        syms = [f"{w['symbol']} ({w['strategy']})" for w in st.session_state.watchlist]
        to_rm = st.selectbox("Remove", syms)
        if st.button("🗑️ Remove"):
            st.session_state.watchlist.pop(syms.index(to_rm)); st.rerun()
    with c2:
        if st.button("🗑️ Clear All"):
            st.session_state.watchlist = []; st.rerun()


# ============================================================================
# JOURNAL
# ============================================================================
def page_journal():
    st.markdown("# 📓 Trade Journal")
    journal = st.session_state.journal
    analytics = compute_journal_analytics(journal)
    if analytics and analytics.get("closed_trades", 0) > 0:
        c1,c2,c3,c4 = st.columns(4)
        with c1: pc("Win Rate", f"{analytics['win_rate']}%", "g" if analytics["win_rate"]>55 else "r")
        with c2: pc("Total P&L", f"₹{analytics['total_pnl']:+,.0f}", "g" if analytics["total_pnl"]>0 else "r")
        with c3: pc("Profit Factor", str(analytics["profit_factor"]))
        with c4: pc("Expectancy", f"₹{analytics['expectancy']:,.0f}/trade")
        fig = plot_equity_curve(analytics)
        if fig: st.plotly_chart(fig, use_container_width=True)
        if analytics.get("strategy_stats"):
            rows = []
            for s, d in analytics["strategy_stats"].items():
                wr = d["wins"]/d["trades"]*100 if d["trades"] else 0
                rows.append({"Strategy":s,"Trades":d["trades"],"Win%":f"{wr:.0f}%","P&L":f"₹{d['pnl']:+,.0f}"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    
    with st.form("add"):
        c1,c2,c3 = st.columns(3)
        with c1:
            sym = st.text_input("Symbol"); strat = st.selectbox("Strategy", list(STRATEGY_PROFILES.keys()))
            sig = st.selectbox("Signal", ["BUY","SHORT"])
        with c2:
            ep = st.number_input("Entry ₹",min_value=0.0,step=1.0)
            sl_ = st.number_input("SL ₹",min_value=0.0,step=1.0)
            tg = st.number_input("Target ₹",min_value=0.0,step=1.0)
        with c3:
            qty = st.number_input("Qty",min_value=1,value=1)
            status = st.selectbox("Status",["open","closed"])
            ex = st.number_input("Exit ₹",min_value=0.0,step=1.0)
        notes = st.text_area("Notes")
        if st.form_submit_button("Add"):
            pnl = (ex-ep)*qty if status=="closed" and ex>0 else 0
            if sig=="SHORT" and status=="closed" and ex>0: pnl=(ep-ex)*qty
            add_journal_entry({"symbol":sym.upper(),"strategy":strat,"signal":sig,"entry":ep,"stop":sl_,
                "target1":tg,"qty":qty,"status":status,"exit":ex if ex>0 else None,"pnl":pnl,
                "notes":notes,"entry_date":str(now_ist().date()),
                "exit_date":str(now_ist().date()) if status=="closed" else None})
            st.session_state.journal = load_journal()
            st.success(f"Added! P&L: ₹{pnl:+,.0f}" if pnl else "Added!"); st.rerun()
    
    if journal:
        rows = [{"#":e.get("id",""),"Symbol":e.get("symbol",""),"Strategy":e.get("strategy",""),
                 "Status":e.get("status",""),"P&L":f"₹{e.get('pnl',0):+,.0f}" if e.get("status")=="closed" else "—",
                 "Notes":e.get("notes","")[:30]} for e in reversed(journal)]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        
        open_t = [e for e in journal if e.get("status")=="open"]
        if open_t:
            st.markdown("### Close Trade")
            labels = [f"#{e['id']} {e['symbol']}" for e in open_t]
            sel = st.selectbox("Trade", labels)
            trade = open_t[labels.index(sel)]
            ex = st.number_input("Exit ₹", min_value=0.0, step=1.0, key="cx")
            if st.button("Close"):
                if ex > 0:
                    pnl = (ex-trade["entry"])*trade.get("qty",1)
                    if trade.get("signal")=="SHORT": pnl=(trade["entry"]-ex)*trade.get("qty",1)
                    for e in journal:
                        if e.get("id")==trade["id"]:
                            e["status"]="closed"; e["exit"]=ex; e["pnl"]=pnl; e["exit_date"]=str(now_ist().date())
                    save_journal(journal); st.session_state.journal=journal
                    st.success(f"P&L: ₹{pnl:+,.0f}"); st.rerun()



# ============================================================================
# BACKTEST ENGINE
# ============================================================================
def page_backtest():
    st.markdown("# 🧪 Backtest Engine")
    
    if not st.session_state.data_loaded:
        st.warning("Load data from Dashboard first.")
        return
    
    # === CLEAR EXPLANATION ===
    with st.expander("ℹ️ How does backtesting work?", expanded=False):
        st.markdown(f"""
**What it does:** Simulates running a strategy on historical data to see how it would have performed.

**Period:** Uses ~1 year of daily data from Yahoo Finance (approximately 248 trading days). This is the maximum available without a premium data feed.

**How it works (walk-forward, no cheating):**
1. For each historical day, the engine checks if the scanner would have triggered a signal using ONLY data available up to that day (no future peeking)
2. If a signal fires, it simulates the trade with the exact Entry, Stop Loss, and Target levels
3. Trade exits when: Stop Loss hit → loss, Target 1 hit → win, or Max Hold days reached → exit at close
4. All trades are logged and statistics computed

**Key metrics explained:**
- **{tip('win_rate')}**
- **{tip('profit_factor')}**
- **{tip('expectancy')}**
- **{tip('max_drawdown')}**

**Limitations:** Only ~1 year of data. Strategies like VCP need bull markets to shine — if the test period is sideways, VCP may underperform. Always consider the market regime during the test period.
""")
    
    tab1, tab2 = st.tabs(["📊 Single Stock", "📈 Multi-Stock Portfolio"])
    
    with tab1:
        c1, c2, c3 = st.columns(3)
        with c1:
            stock_data = st.session_state.stock_data or {}
            sym = st.selectbox("Stock", sorted(stock_data.keys()), key="bt_sym") if stock_data else None
        with c2:
            strat = st.selectbox("Strategy", list(DAILY_SCANNERS.keys()),
                format_func=lambda x: f"{STRATEGY_PROFILES.get(x,{}).get('icon','')} {STRATEGY_PROFILES.get(x,{}).get('name',x)}",
                key="bt_strat", help="Only daily strategies can be backtested (intraday needs live Breeze data).")
        with c3:
            max_hold = st.number_input("Max Hold (days)", 5, 60, 20, 5, key="bt_hold",
                help="Force exit after N days if neither SL nor T1 is hit. Shorter = more trades, less risk per trade.")
        
        if sym and st.button("🧪 Run Backtest", type="primary", key="bt_run1"):
            if sym not in stock_data:
                st.error(f"{sym} data not available."); return
            
            with st.spinner(f"Backtesting {STRATEGY_PROFILES[strat]['name']} on {sym} (~1 year data)..."):
                result = backtest_strategy(stock_data[sym], sym, strat,
                                          lookback_days=500, max_hold=max_hold)
            
            if result and result.total_trades > 0:
                _render_backtest_result(result)
            else:
                st.info(f"No {STRATEGY_PROFILES[strat]['name']} signals found for {sym} in ~1 year of data. "
                        "This strategy may need different market conditions to trigger.")
    
    with tab2:
        c1, c2, c3 = st.columns(3)
        with c1:
            strat2 = st.selectbox("Strategy", list(DAILY_SCANNERS.keys()),
                format_func=lambda x: f"{STRATEGY_PROFILES.get(x,{}).get('icon','')} {STRATEGY_PROFILES.get(x,{}).get('name',x)}",
                key="bt_strat2")
        with c2:
            max_hold2 = st.number_input("Max Hold (days)", 5, 60, 20, 5, key="bt_hold2")
        with c3:
            stock_count = len(st.session_state.stock_data) if st.session_state.stock_data else 0
            st.metric("Stocks in Universe", stock_count)
        
        if st.button("🧪 Run Portfolio Backtest", type="primary", key="bt_run2"):
            stock_data = st.session_state.stock_data or {}
            with st.spinner(f"Backtesting {STRATEGY_PROFILES[strat2]['name']} across {len(stock_data)} stocks (~1 year each)..."):
                result2 = backtest_multi_stock(
                    list(stock_data.keys()),
                    stock_data,
                    strat2, lookback=500, max_hold=max_hold2)
            
            if result2 and result2.total_trades > 0:
                _render_backtest_result(result2)
            else:
                st.info(f"No {STRATEGY_PROFILES[strat2]['name']} signals found across the portfolio.")
    
    # Strategy comparison
    st.markdown("---")
    st.markdown("### 📊 Compare All Strategies")
    st.caption("Runs each daily strategy across your entire loaded universe (~1 year) and compares performance side by side.")
    if st.button("🔬 Run All Strategy Backtests", key="bt_compare"):
        stock_data = st.session_state.stock_data or {}
        if not stock_data:
            st.warning("No data loaded."); return
        comparison = []
        progress = st.progress(0)
        strats = list(DAILY_SCANNERS.keys())
        for idx, s in enumerate(strats):
            progress.progress((idx + 1) / len(strats), f"Testing {STRATEGY_PROFILES[s]['name']}...")
            r = backtest_multi_stock(list(stock_data.keys()),
                                     stock_data, s, lookback=500, max_hold=20)
            if r and r.total_trades > 0:
                comparison.append({
                    "Strategy": f"{STRATEGY_PROFILES[s]['icon']} {STRATEGY_PROFILES[s]['name']}",
                    "Trades": r.total_trades,
                    "Win Rate": f"{r.win_rate}%",
                    "Total P&L": f"{r.total_pnl_pct:+.1f}%",
                    "Avg Win": f"+{r.avg_win_pct:.1f}%",
                    "Avg Loss": f"-{r.avg_loss_pct:.1f}%",
                    "Profit Factor": r.profit_factor,
                    "Max DD": f"{r.max_drawdown_pct:.1f}%",
                    "Expectancy": f"{r.expectancy_pct:+.2f}%",
                    "Avg Hold": f"{r.avg_holding_days:.0f}d",
                })
            else:
                comparison.append({
                    "Strategy": f"{STRATEGY_PROFILES[s]['icon']} {STRATEGY_PROFILES[s]['name']}",
                    "Trades": 0, "Win Rate": "-", "Total P&L": "-", "Avg Win": "-",
                    "Avg Loss": "-", "Profit Factor": "-", "Max DD": "-", 
                    "Expectancy": "-", "Avg Hold": "-",
                })
        progress.empty()
        if comparison:
            st.dataframe(pd.DataFrame(comparison), use_container_width=True, hide_index=True)
            st.caption("📌 Data period: ~1 year per stock from Yahoo Finance. "
                       "Strategies with 0 trades had no matching signals in this period.")


def _render_backtest_result(result):
    """Render backtest results with metrics, equity curve, and trade log."""
    st.markdown(f"### Results: {result.strategy} on {result.symbol}")
    st.caption(f"Period: {result.period} | Max hold: inferred from trades")
    
    # Key metrics row
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1: pc("Trades", str(result.total_trades))
    with c2: pc("Win Rate", f"{result.win_rate}%", "g" if result.win_rate > 50 else "r")
    with c3: pc("Total P&L", f"{result.total_pnl_pct:+.1f}%", "g" if result.total_pnl_pct > 0 else "r")
    with c4: pc("Profit Factor", str(result.profit_factor), "g" if result.profit_factor > 1.5 else ("y" if result.profit_factor > 1 else "r"))
    with c5: pc("Max Drawdown", f"{result.max_drawdown_pct:.1f}%", "r")
    with c6: pc("Expectancy", f"{result.expectancy_pct:+.2f}%/trade", "g" if result.expectancy_pct > 0 else "r")
    
    # Interpretation
    if result.profit_factor > 1.5 and result.win_rate > 45:
        st.success(f"✅ **Strong edge detected.** PF {result.profit_factor} with {result.win_rate}% win rate.")
    elif result.profit_factor > 1:
        st.info(f"➖ **Marginal edge.** PF {result.profit_factor} — consider with regime filter for better results.")
    else:
        st.warning(f"⚠️ **No edge in this period.** PF {result.profit_factor} — strategy may need different market conditions.")
    
    c1,c2 = st.columns(2)
    with c1: pc("Avg Win", f"+{result.avg_win_pct:.1f}%", "g"); pc("Best Trade", f"{result.best_trade_pct:+.1f}%", "g")
    with c2: pc("Avg Loss", f"-{result.avg_loss_pct:.1f}%", "r"); pc("Worst Trade", f"{result.worst_trade_pct:+.1f}%", "r")
    
    # Equity curve
    if result.equity_curve:
        import plotly.graph_objects as go
        eq_df = pd.DataFrame(result.equity_curve)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=eq_df["date"], y=eq_df["equity"], mode="lines+markers",
                                  name="Cumulative P&L %", line=dict(color="#FF6B35", width=2),
                                  marker=dict(size=5)))
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.update_layout(template="plotly_dark", title="Equity Curve (Cumulative P&L %)",
                          xaxis_title="Date", yaxis_title="Cumulative P&L %",
                          height=350, margin=dict(t=40, b=30, l=40, r=20))
        st.plotly_chart(fig, use_container_width=True)
    
    # Trade log
    with st.expander(f"📋 Trade Log ({result.total_trades} trades)", expanded=False):
        rows = []
        for t in result.trades:
            pnl_color = "🟢" if t.pnl_pct > 0 else "🔴"
            rows.append({
                "": pnl_color,
                "Symbol": t.symbol,
                "Entry Date": t.entry_date,
                "Entry ₹": fp(t.entry_price),
                "Exit Date": t.exit_date,
                "Exit ₹": fp(t.exit_price),
                "P&L %": f"{t.pnl_pct:+.1f}%",
                "Hold": f"{t.holding_days}d",
                "Exit Reason": t.exit_reason,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ============================================================================
# SIGNAL LOG — Daily signal recording
# ============================================================================
def page_signal_log():
    st.markdown("# 📋 Signal Log")
    st.caption("Every scan auto-records signals here. Deduped by Strategy + Symbol per day.")
    
    # Date selector
    dates = get_signal_dates()
    if not dates:
        st.info("No signals recorded yet. Run a scan from Dashboard or Scanner Hub — signals auto-save here.")
        st.markdown("""
        **How it works:**
        1. Every time you run a scan (Dashboard or Scanner Hub), signals are automatically saved with timestamp
        2. GitHub Actions runs EOD scans daily at 4:30 PM and 7:00 PM IST (if configured)
        3. Duplicate signals (same strategy + stock on same day) are ignored
        """)
        return
    
    c1, c2 = st.columns([2, 1])
    with c1:
        selected_date = st.selectbox("Date", dates, key="sl_date",
            format_func=lambda d: f"{d} {'(today)' if d == date.today().isoformat() else ''}")
    with c2:
        if st.button("🔄 Update Tracker", key="sl_update", help="Check if any OPEN signals hit SL or T1"):
            if st.session_state.stock_data:
                updated = update_open_signals_live(st.session_state.stock_data)
                st.success(f"Updated {updated} signals" if updated else "No changes — all signals unchanged")
                st.rerun()
            else:
                st.warning("Load data from Dashboard first to update live prices.")
    
    df = load_signals(selected_date)
    if df is None or df.empty:
        st.info(f"No signals for {selected_date}.")
        return
    
    # Summary
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: pc("Total Signals", str(len(df)))
    with c2: pc("Strategies", str(df["Strategy"].nunique()))
    with c3: pc("Stocks", str(df["Symbol"].nunique()))
    open_c = len(df[df["Status"]=="OPEN"])
    with c4: pc("Open", str(open_c), "o" if open_c > 0 else "")
    regime_val = df["Regime"].iloc[0] if "Regime" in df.columns and len(df) > 0 else "-"
    with c5: pc("Regime", str(regime_val))
    
    # Grouped by strategy
    for strat in df["Strategy"].unique():
        sdf = df[df["Strategy"] == strat]
        p = STRATEGY_PROFILES.get(strat, {})
        name = p.get("name", strat)
        icon = p.get("icon", "")
        
        with st.expander(f"{icon} {name} — {len(sdf)} signals", expanded=True):
            display_cols = ["Time","Symbol","Signal","CMP","Entry","SL","T1","RR",
                          "Confidence","RS","Sector","Regime_Fit","Status"]
            cols_present = [c for c in display_cols if c in sdf.columns]
            st.dataframe(sdf[cols_present], use_container_width=True, hide_index=True)
    
    # Download
    st.markdown("---")
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(f"📥 Download {selected_date} signals (CSV)", csv_data,
                       f"signals_{selected_date}.csv", "text/csv", key="sl_dl")


# ============================================================================
# SIGNAL TRACKER — Forward testing performance
# ============================================================================
def page_tracker():
    st.markdown("# 📊 Signal Tracker")
    st.caption("Forward-test: did scanner signals actually hit their targets?")
    
    tracker_df = load_tracker()
    if tracker_df is None or tracker_df.empty:
        st.info("No tracked signals yet. Run scans to start recording, then check back to see outcomes.")
        st.markdown("""
        **How forward testing works:**
        1. Every scan auto-records signals with Entry, SL, and T1 prices
        2. Each day, the system checks if price hit SL (loss) or T1 (win)
        3. After 30 days, unresolved signals expire at current price
        4. Over time, this builds a real track record for each strategy
        """)
        return
    
    stats = compute_tracker_stats(tracker_df)
    if not stats:
        st.info("Insufficient data for statistics.")
        return
    
    # Header metrics
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1: pc("Total Signals", str(stats["total"]))
    with c2: pc("🟢 Targets Hit", str(stats["targets"]), "g")
    with c3: pc("🔴 Stopped Out", str(stats["stopped"]), "r")
    with c4: pc("⏳ Open", str(stats["open"]), "o")
    with c5: pc("Win Rate", f"{stats['win_rate']}%", "g" if stats["win_rate"] > 50 else "r")
    with c6: pc("Total P&L", f"{stats['total_pnl']:+.1f}%", "g" if stats["total_pnl"] > 0 else "r")
    
    # Update button
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🔄 Refresh Tracker (check SL/T1 hits)", key="tr_refresh", type="primary"):
            if st.session_state.stock_data:
                updated = update_open_signals_live(st.session_state.stock_data)
                st.success(f"Updated {updated} signals") if updated else st.info("No changes")
                st.rerun()
            else:
                st.warning("Load data from Dashboard first.")
    with c2:
        st.caption(f"Period: {stats.get('first_date','')} to {stats.get('last_date','')}")
    with c3:
        st.caption(f"Trading days: {stats.get('trading_days',0)}")
    
    # Strategy performance table
    st.markdown("### 📊 Strategy Performance (Forward Test)")
    strat_rows = []
    for strat, s in stats.get("strategy_stats", {}).items():
        p = STRATEGY_PROFILES.get(strat, {})
        strat_rows.append({
            "Strategy": f"{p.get('icon','')} {p.get('name',strat)}",
            "Total": s["total"],
            "Open": s["open"],
            "🟢 Target": s["targets"],
            "🔴 Stopped": s["stopped"],
            "Win Rate": f"{s['win_rate']}%" if s["targets"] + s["stopped"] > 0 else "-",
        })
    if strat_rows:
        st.dataframe(pd.DataFrame(strat_rows), use_container_width=True, hide_index=True)
    
    # Detailed signal table with filters
    st.markdown("### 📋 All Tracked Signals")
    c1, c2, c3 = st.columns(3)
    with c1:
        status_filter = st.multiselect("Status", ["OPEN","TARGET","STOPPED","EXPIRED"],
                                        default=["OPEN","TARGET","STOPPED"], key="tr_status")
    with c2:
        strat_options = list(tracker_df["Strategy"].unique())
        strat_filter = st.multiselect("Strategy", strat_options, default=strat_options, key="tr_strat")
    with c3:
        signal_filter = st.multiselect("Signal Type", ["BUY","SHORT"],
                                        default=["BUY","SHORT"], key="tr_signal")
    
    filtered = tracker_df[
        (tracker_df["Status"].isin(status_filter)) &
        (tracker_df["Strategy"].isin(strat_filter)) &
        (tracker_df["Signal"].isin(signal_filter))
    ].copy()
    
    # Color code status
    display_cols = ["Date","Time","Strategy_Name","Symbol","Signal","Entry","SL","T1",
                   "Confidence","RS","Status","Exit_Date","Exit_Reason","PnL_Pct"]
    cols_present = [c for c in display_cols if c in filtered.columns]
    
    st.dataframe(filtered[cols_present].sort_values("Date", ascending=False),
                 use_container_width=True, hide_index=True)
    
    # Download full tracker
    st.markdown("---")
    csv_bytes = generate_csv_download()
    if csv_bytes:
        st.download_button("📥 Download Full Signal History (CSV)", csv_bytes,
                           f"signal_tracker_{date.today().isoformat()}.csv", "text/csv",
                           key="tr_dl")


# ============================================================================
# SETTINGS
# ============================================================================
def page_settings():
    st.markdown("# ⚙️ Settings")
    st.session_state.capital = st.number_input("Capital ₹", value=st.session_state.capital, step=50000)
    
    st.markdown("### 🔌 Breeze API")
    if st.session_state.breeze_connected:
        st.success("✅ Breeze Connected! Intraday scanners (ORB, VWAP, Lunch Low) are LIVE.")
    else:
        if st.session_state.breeze_msg:
            msg_lower = st.session_state.breeze_msg.lower()
            if "session" in msg_lower or "token" in msg_lower or "expired" in msg_lower:
                st.error("🔑 Breeze session token expired! Generate a new one from ICICI Direct and update Streamlit secrets.")
            else:
                st.error(st.session_state.breeze_msg)
        st.markdown("**Without Breeze:** ORB, VWAP Reclaim, Lunch Low are **disabled** (not proxied)."
                     " VCP, EMA21, 52WH, Short, ATH work fine with daily data.")
        st.warning("⚠️ Paste ONLY these lines in Streamlit Settings → Secrets (no backticks!):")
        st.code('BREEZE_API_KEY = "your_key"\nBREEZE_API_SECRET = "your_secret"\nBREEZE_SESSION_TOKEN = "daily_token"', language="toml")
        st.info("⚠️ Session Token expires daily. Regenerate each morning from ICICI Direct portal.")
        with st.expander("Manual Connect"):
            with st.form("bf"):
                ak = st.text_input("Key", type="password"); asc = st.text_input("Secret", type="password")
                st_ = st.text_input("Token", type="password")
                if st.form_submit_button("Connect"):
                    if ak and asc and st_:
                        e = BreezeEngine(); ok, msg = e.connect(ak, asc, st_)
                        if ok: st.success(msg); st.session_state.breeze_connected=True; st.session_state.breeze_engine=e
                        else: st.error(msg)
    
    st.session_state.universe_size = st.selectbox("Universe", ["nifty50","nifty200","nifty500"],
        index=["nifty50","nifty200","nifty500"].index(st.session_state.universe_size))
    
    st.markdown("### 🔐 Access Control")
    st.markdown("Add this to Streamlit secrets to require a password:")
    st.code('APP_PASSWORD = "your_chosen_password"', language="toml")
    st.caption("Leave blank or remove to disable. Share the password with trusted users.")
    
    st.markdown("### 🤖 GitHub Actions (Auto-Scanner)")
    st.markdown("""
    Auto-scans run daily via GitHub Actions (**free** on public repos):
    - **4:30 PM IST** — Quick Nifty 200 scan
    - **7:00 PM IST** — Full NSE 500 EOD scan + signal tracking
    - Results auto-committed to `signals/` folder in your repo
    
    See **SETUP_GUIDE.md** in your repo for step-by-step setup instructions.
    """)
    
    st.markdown("### 🧠 Regime Behavior (Recalibrated)")
    st.markdown("""
    | Regime | Score Range | Position Size | Ideal For |
    |--------|---:|---:|---|
    | 🟢 EXPANSION | ≥ 6/12 | 100% | VCP, 52WH, ORB, ATH — buy breakouts |
    | 🟡 ACCUMULATION | 2 to 5 | 70% | VCP, EMA21, VWAP — be selective |
    | 🟠 DISTRIBUTION | -2 to 1 | 40% | Shorts, mean-reversion — defensive |
    | 🔴 PANIC | < -2 | 15% | Shorts only — protect capital |
    """)


# ============================================================================
# ROUTER
# ============================================================================
page_map = {
    "📊 Dashboard": page_dashboard,
    "🔍 Scanner Hub": page_scanner_hub,
    "📈 Charts & RS": page_charts_rs,
    "🧪 Backtest": page_backtest,
    "📋 Signal Log": page_signal_log,
    "📊 Tracker": page_tracker,
    "📐 Trade Planner": page_trade_planner,
    "⭐ Watchlist": page_watchlist,
    "📓 Journal": page_journal,
    "⚙️ Settings": page_settings,
}
page_func = page_map.get(page, page_dashboard)
page_func()

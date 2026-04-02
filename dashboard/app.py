"""
Streamlit Dashboard — Real-time trading control center.
Fully dynamic: pulls live MT5 data, saves signal state, shows Claude debate.

Run with: streamlit run dashboard/app.py
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import json
import os
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import (
    ACTIVE_MARKET, STARTING_CAPITAL, JOURNAL_DIR, MEMORY_DIR,
    DATA_DIR, LOGS_DIR, TRADING_MODE
)
from core.self_learning import TradingJournal


# ─── Page Config ─────────────────────────────────────────
st.set_page_config(
    page_title="ML Trading System v2",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Dashboard Authentication ────────────────────────────
DASHBOARD_PASSWORD_HASH = os.environ.get("DASHBOARD_PASSWORD_HASH", "")

if DASHBOARD_PASSWORD_HASH:
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.title("🔒 ML Trading Dashboard")
        password = st.text_input("Enter dashboard password:", type="password")
        if password:
            import hashlib
            entered_hash = hashlib.sha256(password.encode()).hexdigest()
            if entered_hash == DASHBOARD_PASSWORD_HASH:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password.")
        st.stop()

# ─── Persistent State ─────────────────────────────────
STATE_FILE = LOGS_DIR / "dashboard_state.json"

def load_dashboard_state():
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"market": ACTIVE_MARKET, "trading_mode": "Manual"}

def save_dashboard_state(state):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

dash_state = load_dashboard_state()

if "market" not in st.session_state:
    st.session_state.market = dash_state.get("market", ACTIVE_MARKET)
if "trading_mode" not in st.session_state:
    st.session_state.trading_mode = dash_state.get("trading_mode", "Manual")

# ─── Load latest analysis result ────────────────────────
LATEST_SIGNAL_FILE = LOGS_DIR / "latest_signal.json"

def load_latest_signal():
    """Load the most recent analysis from main.py."""
    if LATEST_SIGNAL_FILE.exists():
        try:
            with open(LATEST_SIGNAL_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return None

def get_live_balance():
    """Get live balance from MT5 or latest signal."""
    signal = load_latest_signal()
    if signal and "account" in signal:
        return signal["account"].get("balance", STARTING_CAPITAL)
    # Try MT5 direct
    try:
        from data.mt5_connector import connect_mt5, disconnect_mt5, get_account_info
        if connect_mt5():
            info = get_account_info()
            disconnect_mt5()
            return info.get("balance", STARTING_CAPITAL)
    except Exception:
        pass
    return STARTING_CAPITAL


# ─── Custom CSS ──────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    .trade-card { background: #161b22; padding: 20px; border-radius: 12px;
                  border: 1px solid #30363d; margin: 10px 0; }
    .bull-card { border-left: 4px solid #00ff88; }
    .bear-card { border-left: 4px solid #ff4444; }
    .judge-card { border-left: 4px solid #ffcc00; }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 ML Trading v2")
    st.markdown("---")

    # Market selector — saves to session state
    market_options = ["EURUSD", "XAUUSD", "BTCUSD"]
    current_idx = market_options.index(st.session_state.market) if st.session_state.market in market_options else 0
    market = st.selectbox("Active Market", market_options, index=current_idx, key="market_select")
    if market != st.session_state.market:
        st.session_state.market = market
        dash_state["market"] = market
        save_dashboard_state(dash_state)

    # Mode toggle — saves to session state
    mode_idx = 0 if st.session_state.trading_mode == "Manual" else 1
    trading_mode = st.radio("Trading Mode", ["Manual", "Auto"], index=mode_idx, key="mode_select")
    if trading_mode != st.session_state.trading_mode:
        st.session_state.trading_mode = trading_mode
        dash_state["trading_mode"] = trading_mode
        save_dashboard_state(dash_state)

    if trading_mode == "Auto":
        st.warning("⚠️ Auto-mode: Green signals execute automatically!")

    st.markdown("---")

    # LIVE risk status from journal
    journal = TradingJournal()
    trades_today = [t for t in journal.journal if t.get("timestamp", "").startswith(datetime.now().strftime("%Y-%m-%d"))]
    daily_pnl = sum(t.get("pnl", 0) for t in trades_today)
    consec_losses = 0
    for t in reversed(journal.journal):
        if t.get("pnl", 0) < 0:
            consec_losses += 1
        else:
            break

    st.subheader("⚡ Risk Status")
    st.metric("Trades Today", f"{len(trades_today)} / 4")
    pnl_delta = f"+${daily_pnl:.2f}" if daily_pnl >= 0 else f"-${abs(daily_pnl):.2f}"
    st.metric("Daily PnL", f"${daily_pnl:.2f}", delta=pnl_delta)
    st.metric("Consecutive Losses", str(consec_losses))

    st.markdown("---")
    st.subheader("🕐 Trading Sessions")
    sessions = {
        "EURUSD": "London-NY Overlap: 5 PM - 9 PM PKT",
        "XAUUSD": "New York: 6 PM - 11 PM PKT",
        "BTCUSD": "24/7 (Best: 6 PM - 1 AM PKT)",
    }
    st.info(sessions.get(market, ""))

    st.markdown("---")
    st.subheader("⏰ System Status")
    st.caption(f"Mode: {'🟡 DEMO' if TRADING_MODE == 'demo' else '🔴 LIVE'}")
    st.caption(f"Market: {market}")
    st.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")
    if st.button("🔄 Refresh Data"):
        st.rerun()


# ─── LIVE Balance from MT5 ───────────────────────────────
balance = get_live_balance()
total_pnl = sum(t.get("pnl", 0) for t in journal.journal)
win_trades = [t for t in journal.journal if t.get("pnl", 0) > 0]
loss_trades = [t for t in journal.journal if t.get("pnl", 0) < 0]
win_rate = len(win_trades) / len(journal.journal) * 100 if journal.journal else 0
gross_profit = sum(t.get("pnl", 0) for t in win_trades) if win_trades else 0
gross_loss = abs(sum(t.get("pnl", 0) for t in loss_trades)) if loss_trades else 0
profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

# Load latest signal for Aegis Score display
latest = load_latest_signal()
try:
    latest_aegis = float(latest.get("aegis", {}).get("score", 0)) if latest and latest.get("aegis") else 0
except (TypeError, ValueError):
    latest_aegis = 0

col1, col2, col3, col4 = st.columns(4)
with col1:
    pnl_str = f"+${total_pnl:.2f}" if total_pnl >= 0 else f"-${abs(total_pnl):.2f}"
    st.metric("Balance", f"${balance:.2f}", pnl_str)
with col2:
    wr_str = f"{win_rate:.1f}%" if journal.journal else "—"
    st.metric("Win Rate", wr_str, help=f"{len(win_trades)}W / {len(loss_trades)}L" if journal.journal else "Need trades")
with col3:
    pf_str = f"{profit_factor:.2f}" if profit_factor > 0 else "—"
    st.metric("Profit Factor", pf_str)
with col4:
    st.metric("Aegis Score", f"{latest_aegis:.0f}" if latest_aegis else "—",
              help="Composite confidence 0-100")


st.markdown("---")

# ─── Latest Signal (DYNAMIC) ────────────────────────────
st.header("📡 Latest Signal")

signal_col1, signal_col2 = st.columns([2, 1])

with signal_col1:
    if latest:
        verdict = latest.get("verdict", {})
        decision = verdict.get("decision", "HOLD")
        reasoning = verdict.get("reasoning", "No reasoning available")
        confidence = verdict.get("confidence", 0)
        timestamp = latest.get("timestamp", "")

        emoji = "🟢 BUY" if decision == "BUY" else ("🔴 SELL" if decision == "SELL" else "⏸️ HOLD")
        color = "#00ff88" if decision == "BUY" else ("#ff4444" if decision == "SELL" else "#ffcc00")

        price_info = latest.get("current_price", {})
        bid = price_info.get("bid", "N/A") if price_info else "N/A"

        st.markdown(f"""
        <div class="trade-card" style="border-left: 4px solid {color};">
            <h3>{emoji} — {latest.get('market', market)}</h3>
            <p><b>Price:</b> {bid} | <b>Claude Confidence:</b> {confidence}/10</p>
            <p><b>Reasoning:</b> {reasoning[:300]}{'...' if len(reasoning) > 300 else ''}</p>
            <p style="color: #666; font-size: 12px;">Analysis at: {timestamp[:19] if timestamp else 'N/A'}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="trade-card">
            <h3>⏸️ No Signal Yet</h3>
            <p>System is monitoring. Signals appear when ML models detect a setup
            that passes the Aegis Score threshold (≥65).</p>
            <p>Make sure main.py is running.</p>
        </div>
        """, unsafe_allow_html=True)

with signal_col2:
    st.subheader("⭐ Aegis Score")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=latest_aegis,
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={
            "axis": {"range": [None, 100]},
            "bar": {"color": "#333"},
            "steps": [
                {"range": [0, 65], "color": "#ff4444"},
                {"range": [65, 80], "color": "#ffcc00"},
                {"range": [80, 100], "color": "#00ff88"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 2},
                "thickness": 0.8,
                "value": 65,
            },
        },
        title={"text": "Confidence"},
    ))
    fig_gauge.update_layout(
        height=250, margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)", font={"color": "white"},
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Aegis breakdown
    if latest and "aegis" in latest:
        aegis = latest["aegis"]
        breakdown = aegis.get("breakdown", {})
        if breakdown:
            st.caption(f"ML: {breakdown.get('ml', 0):.0f}/30 | Sent: {breakdown.get('sentiment', 0):.0f}/15 | "
                       f"Regime: {breakdown.get('regime', 0):.0f}/20 | Claude: {breakdown.get('claude', 0):.0f}/20")


st.markdown("---")

# ─── Trade Controls ──────────────────────────────────────
st.header("🎮 Trade Controls")
can_trade = latest.get("can_trade", False) if latest else False

ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns(4)
with ctrl_col1:
    st.button("✅ Approve Trade", type="primary", disabled=not can_trade)
with ctrl_col2:
    st.button("❌ Reject Trade", disabled=not can_trade)
with ctrl_col3:
    if st.button("🔄 Force Scan Now"):
        st.info("Scan requested — main.py will process on next cycle.")
with ctrl_col4:
    if st.button("🛑 Emergency Stop", type="secondary"):
        st.error("Emergency stop — close all positions manually in MT5!")


st.markdown("---")

# ─── Claude Debate Panel (DYNAMIC) ──────────────────────
st.header("🧠 Claude Multi-Agent Debate")

debate_col1, debate_col2, debate_col3 = st.columns(3)

if latest and "debate" in latest:
    debate = latest["debate"]
    with debate_col1:
        st.markdown("### 🟢 Agent Bull")
        bull_text = debate.get("bull_case", debate.get("bull", "No analysis available"))
        st.success(bull_text[:500] if len(bull_text) > 500 else bull_text)
    with debate_col2:
        st.markdown("### 🔴 Agent Bear")
        bear_text = debate.get("bear_case", debate.get("bear", "No analysis available"))
        st.error(bear_text[:500] if len(bear_text) > 500 else bear_text)
    with debate_col3:
        st.markdown("### ⚖️ Agent Judge")
        verdict = latest.get("verdict", {})
        judge_text = verdict.get("reasoning", "No verdict yet")
        st.warning(judge_text[:500] if len(judge_text) > 500 else judge_text)
else:
    with debate_col1:
        st.markdown("### 🟢 Agent Bull")
        st.info("Waiting for analysis... Run main.py")
    with debate_col2:
        st.markdown("### 🔴 Agent Bear")
        st.info("Waiting for analysis...")
    with debate_col3:
        st.markdown("### ⚖️ Agent Judge")
        st.info("Waiting for analysis...")


st.markdown("---")

# ─── Trade Journal ───────────────────────────────────────
st.header("📓 Trade Journal")

trades = journal.journal
if trades:
    df_trades = pd.DataFrame(trades)
    display_cols = [c for c in ["timestamp", "market", "direction", "entry_price",
                                 "exit_price", "pnl", "result", "aegis_score"] if c in df_trades.columns]
    st.dataframe(df_trades[display_cols].sort_values("timestamp", ascending=False), use_container_width=True)
else:
    st.info("No trades recorded yet. Trades appear here after the system executes on your demo account.")


st.markdown("---")

# ─── Pattern Memory ──────────────────────────────────────
st.header("🧬 Pattern Memory")

patterns = journal.patterns
if patterns:
    pattern_data = []
    for key, data in patterns.items():
        pattern_data.append({
            "Pattern": key,
            "Trades": data["total"],
            "Win Rate": f"{data.get('win_rate', 0):.0%}",
            "Total PnL": f"${data.get('total_pnl', 0):.2f}",
            "Avg Aegis": data.get("avg_aegis", 0),
        })
    st.dataframe(pd.DataFrame(pattern_data), use_container_width=True)
else:
    st.info("Pattern memory builds as the system takes trades — tracks what works and what doesn't.")


st.markdown("---")

# ─── Session Heatmap ─────────────────────────────────────
st.header("🗓️ Session Performance Heatmap")
try:
    from dashboard.analytics import (
        generate_session_heatmap, generate_win_rate_heatmap,
        generate_confidence_trend, generate_equity_curve,
    )
    heatmap_col1, heatmap_col2 = st.columns(2)
    with heatmap_col1:
        st.plotly_chart(generate_session_heatmap(journal), use_container_width=True)
    with heatmap_col2:
        st.plotly_chart(generate_win_rate_heatmap(journal), use_container_width=True)
except Exception:
    st.info("Heatmap populates after 10+ trades — shows which hours/days perform best.")


# ─── Confidence Calibration ──────────────────────────────
st.header("🎯 Confidence Calibration")
try:
    st.plotly_chart(generate_confidence_trend(journal), use_container_width=True)
except Exception:
    pass

calibration = journal.get_confidence_calibration()
if "status" in calibration:
    st.info(calibration["status"])
else:
    cal_data = []
    for bucket, data in calibration.items():
        cal_data.append({
            "Bucket": bucket.upper(),
            "Trades": data["trades"],
            "Expected WR": f"{data['expected_win_rate']:.0%}",
            "Actual WR": f"{data['actual_win_rate']:.0%}",
            "Calibrated": "✅" if data["calibrated"] else "⚠️",
        })
    st.dataframe(pd.DataFrame(cal_data), use_container_width=True)


# ─── Equity Curve + Drawdown ────────────────────────────
st.header("📈 Equity Curve & Drawdown")
try:
    st.plotly_chart(generate_equity_curve(journal, initial_capital=balance), use_container_width=True)
except Exception:
    st.info("Equity curve will appear after your first trade.")


st.markdown("---")

# ─── 📚 Book Knowledge Upload ───────────────────────────
st.header("📚 Trading Book Upload")
st.caption("Upload trading strategy books (PDF/TXT). The system extracts key patterns and teaches Claude.")

BOOKS_DIR = Path("data/books")
BOOKS_DIR.mkdir(parents=True, exist_ok=True)

book_col1, book_col2 = st.columns([1, 1])

with book_col1:
    uploaded_file = st.file_uploader(
        "Upload a trading book",
        type=["pdf", "txt", "md"],
        help="PDF or text files with trading strategies"
    )

    if uploaded_file is not None:
        save_path = BOOKS_DIR / uploaded_file.name
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"✅ Saved: {uploaded_file.name}")

        # Extract text from the book
        if uploaded_file.name.endswith(".txt") or uploaded_file.name.endswith(".md"):
            content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
            # Save extracted strategies
            strategies_file = BOOKS_DIR / f"{Path(uploaded_file.name).stem}_strategies.json"
            strategies = {
                "source": uploaded_file.name,
                "uploaded": datetime.now().isoformat(),
                "content_preview": content[:2000],
                "full_text": content,
            }
            with open(strategies_file, "w") as f:
                json.dump(strategies, f, indent=2)
            st.info(f"📖 Extracted {len(content)} characters from {uploaded_file.name}")

        elif uploaded_file.name.endswith(".pdf"):
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(uploaded_file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                strategies_file = BOOKS_DIR / f"{Path(uploaded_file.name).stem}_strategies.json"
                strategies = {
                    "source": uploaded_file.name,
                    "uploaded": datetime.now().isoformat(),
                    "pages": len(reader.pages),
                    "content_preview": text[:2000],
                    "full_text": text,
                }
                with open(strategies_file, "w") as f:
                    json.dump(strategies, f, indent=2)
                st.info(f"📖 Extracted {len(text)} chars from {len(reader.pages)} pages")
            except ImportError:
                st.warning("Install PyPDF2: pip install PyPDF2")

with book_col2:
    st.subheader("📖 Loaded Books")
    book_files = list(BOOKS_DIR.glob("*_strategies.json"))
    if book_files:
        for bf in book_files:
            with open(bf) as f:
                book_data = json.load(f)
            source = book_data.get("source", bf.stem)
            preview = book_data.get("content_preview", "")[:200]
            pages = book_data.get("pages", "N/A")
            st.markdown(f"**{source}** ({pages} pages)")
            st.caption(preview + "..." if len(preview) >= 200 else preview)
            st.markdown("---")
    else:
        st.info("No books uploaded yet. Upload PDF/TXT trading books to teach the system.")

    # Show built-in book knowledge
    with st.expander("📗 Built-in Knowledge (ICT + Nison)"):
        from models.book_knowledge import BookKnowledge
        bk = BookKnowledge()
        st.markdown(bk.get_claude_knowledge())


st.markdown("---")

# ─── Signal Version Tracking ────────────────────────────
st.header("🔢 Model Version Performance")
try:
    from core.signal_versioning import SignalVersioner
    versioner = SignalVersioner()
    perf = versioner.get_version_performance()
    if isinstance(perf, dict) and "message" not in perf:
        version_data = []
        for ver, data in perf.items():
            version_data.append({
                "Version": ver,
                "Trades": data["trades"],
                "Win Rate": f"{data['win_rate']:.0%}",
                "Total PnL": f"${data['total_pnl']:.2f}",
            })
        st.dataframe(pd.DataFrame(version_data), use_container_width=True)
    else:
        st.info("Version tracking starts after model-generated trades.")
except Exception:
    st.info("Signal versioning activates once models are trained.")


# ─── Footer ──────────────────────────────────────────────
st.markdown("---")
st.caption("ML Trading System v2 — Built with PyTorch, XGBoost, Claude AI, and ❤️")

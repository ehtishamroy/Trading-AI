"""
Analytics Module — Session Heatmap + Confidence Trend.
Provides visual analytics for the Streamlit dashboard.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from loguru import logger
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.self_learning import TradingJournal


def generate_session_heatmap(journal: TradingJournal) -> go.Figure:
    """
    Generate a heatmap showing PnL by day-of-week × hour-of-day.
    Shows which trading sessions are most profitable.

    Returns: Plotly Figure object
    """
    trades = journal.journal
    if len(trades) < 10:
        return _empty_heatmap("Need 10+ trades for heatmap")

    df = pd.DataFrame(trades)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day_name()

    # Create pivot table: day × hour → average PnL
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    pivot = df.pivot_table(
        values="pnl",
        index="day",
        columns="hour",
        aggfunc="mean",
        fill_value=0,
    )

    # Reindex to ensure consistent order
    pivot = pivot.reindex(days_order)

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[f"{h}:00" for h in pivot.columns],
        y=pivot.index,
        colorscale=[
            [0, "#ff4444"],      # Red = losing
            [0.5, "#1a1a2e"],    # Dark = breakeven
            [1, "#00ff88"],      # Green = winning
        ],
        colorbar_title="Avg PnL ($)",
        hoverongaps=False,
        text=np.round(pivot.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10, "color": "white"},
    ))

    fig.update_layout(
        title="Session Performance Heatmap (Avg PnL by Day × Hour)",
        xaxis_title="Hour (UTC)",
        yaxis_title="Day of Week",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        height=350,
    )

    return fig


def generate_win_rate_heatmap(journal: TradingJournal) -> go.Figure:
    """
    Win rate by day × hour — shows when our signals are most accurate.
    """
    trades = journal.journal
    if len(trades) < 10:
        return _empty_heatmap("Need 10+ trades for win rate heatmap")

    df = pd.DataFrame(trades)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day_name()
    df["win"] = (df["pnl"] > 0).astype(int)

    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    pivot = df.pivot_table(
        values="win",
        index="day",
        columns="hour",
        aggfunc="mean",
        fill_value=0.5,
    )
    pivot = pivot.reindex(days_order)

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values * 100,
        x=[f"{h}:00" for h in pivot.columns],
        y=pivot.index,
        colorscale=[[0, "#ff4444"], [0.5, "#ffcc00"], [1, "#00ff88"]],
        colorbar_title="Win Rate %",
        text=np.round(pivot.values * 100, 0).astype(int),
        texttemplate="%{text}%",
        textfont={"size": 10, "color": "white"},
    ))

    fig.update_layout(
        title="Win Rate Heatmap (by Day × Hour)",
        xaxis_title="Hour (UTC)",
        yaxis_title="Day of Week",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        height=350,
    )

    return fig


def generate_confidence_trend(journal: TradingJournal) -> go.Figure:
    """
    Track Aegis Score confidence over time vs actual outcome.
    Shows if confidence calibration is improving or drifting.
    """
    trades = journal.journal
    if len(trades) < 5:
        return _empty_chart("Need 5+ trades for confidence trend")

    df = pd.DataFrame(trades)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["win"] = (df["pnl"] > 0).astype(int)

    # Rolling win rate (last 10 trades)
    df["rolling_wr"] = df["win"].rolling(10, min_periods=3).mean() * 100

    # Rolling average Aegis Score
    df["rolling_aegis"] = df["aegis_score"].rolling(10, min_periods=3).mean()

    fig = go.Figure()

    # Aegis Score trend
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["rolling_aegis"],
        name="Avg Aegis Score",
        line=dict(color="#00ccff", width=2),
        fill="tozeroy",
        fillcolor="rgba(0,204,255,0.1)",
    ))

    # Win rate trend
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["rolling_wr"],
        name="Win Rate %",
        line=dict(color="#00ff88", width=2, dash="dot"),
        yaxis="y2",
    ))

    # Individual trade Aegis scores
    colors = ["#00ff88" if w else "#ff4444" for w in df["win"]]
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["aegis_score"],
        name="Trade Aegis Score",
        mode="markers",
        marker=dict(color=colors, size=8, opacity=0.7),
    ))

    fig.update_layout(
        title="Confidence Calibration Trend",
        xaxis_title="Time",
        yaxis_title="Aegis Score",
        yaxis2=dict(title="Win Rate %", overlaying="y", side="right", range=[0, 100]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        height=400,
        legend=dict(x=0, y=1.1, orientation="h"),
    )

    return fig


def generate_equity_curve(journal: TradingJournal, initial_capital: float = 200) -> go.Figure:
    """Generate detailed equity curve with drawdown overlay."""
    trades = journal.journal
    if not trades:
        return _empty_chart("No trades yet")

    df = pd.DataFrame(trades)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["equity"] = df["pnl"].cumsum() + initial_capital

    # Calculate drawdown
    df["peak"] = df["equity"].cummax()
    df["drawdown"] = (df["equity"] - df["peak"]) / df["peak"] * 100

    fig = go.Figure()

    # Equity curve
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["equity"],
        name="Balance",
        line=dict(color="#00ff88", width=2),
        fill="tozeroy",
        fillcolor="rgba(0,255,136,0.1)",
    ))

    # Peak
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["peak"],
        name="Peak",
        line=dict(color="#666", width=1, dash="dot"),
    ))

    # Drawdown on second axis
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["drawdown"],
        name="Drawdown %",
        line=dict(color="#ff4444", width=1),
        fill="tozeroy",
        fillcolor="rgba(255,68,68,0.15)",
        yaxis="y2",
    ))

    # Starting capital line
    fig.add_hline(y=initial_capital, line_dash="dash", line_color="gray",
                  annotation_text=f"Starting: ${initial_capital}")

    fig.update_layout(
        title="Equity Curve & Drawdown",
        xaxis_title="Time",
        yaxis_title="Balance ($)",
        yaxis2=dict(title="Drawdown %", overlaying="y", side="right", range=[-20, 5]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        height=400,
    )

    return fig


def _empty_heatmap(message: str) -> go.Figure:
    """Return placeholder heatmap."""
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font=dict(size=16, color="gray"))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=300, font={"color": "white"})
    return fig


def _empty_chart(message: str) -> go.Figure:
    return _empty_heatmap(message)

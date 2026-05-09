"""Plotting helpers for replication outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_returns(
    returns: pd.Series,
    output_path: str | Path | None = None,
) -> go.Figure:
    """Plot a return series with Plotly and optionally save the figure."""
    series = returns.dropna()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series.to_numpy(),
            mode="lines",
            name=series.name or "returns",
            line={"width": 1.2},
        )
    )
    fig.update_layout(
        title=series.name or "Returns",
        template="plotly_white",
        xaxis_title="",
        yaxis_title="Return",
        hovermode="x unified",
        margin={"l": 56, "r": 24, "t": 56, "b": 40},
    )
    if output_path is not None:
        save_plotly_figure(fig, output_path)
    return fig


def save_plotly_figure(fig: go.Figure, output_path: str | Path) -> Path:
    """Save a Plotly figure as HTML, JSON, or a static image."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

    if suffix == ".html":
        fig.write_html(path)
    elif suffix == ".json":
        fig.write_json(path)
    else:
        fig.write_image(path)

    return path


def plot_price_evolution(
    prices: pd.DataFrame,
    output_path: str | Path | None = None,
) -> go.Figure:
    """Reproduce Figure 1: time evolution of NASDAQ and S&P 500."""
    frame = prices.dropna()

    fig = go.Figure()

    for column in frame.columns:
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame[column],
                mode="lines",
                name=column,
                line={"width": 2},
            )
        )

    fig.update_layout(
        title="Time evolution of NASDAQ Composite index and S&P 500",
        template="plotly_white",
        xaxis_title="",
        yaxis_title="Price (USD)",
        hovermode="x unified",
        legend={
            "x": 0.02,
            "y": 0.98,
            "xanchor": "left",
            "yanchor": "top",
            "bgcolor": "rgba(255,255,255,0.8)",
            "bordercolor": "black",
            "borderwidth": 1,
        },
        margin={"l": 60, "r": 30, "t": 60, "b": 50},
    )

    if output_path is not None:
        save_plotly_figure(fig, output_path)

    return fig


def plot_returns_and_squared_returns(
    returns: pd.DataFrame,
    output_path: str | Path | None = None,
) -> go.Figure:
    """Reproduce Figure 2: daily returns and squared returns."""
    frame = returns.dropna()
    squared = frame**2

    if frame.shape[1] != 2:
        raise ValueError("This figure expects exactly two return series.")

    name_1, name_2 = frame.columns

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            f"Return-{name_1}",
            f"Return-{name_2}",
            f"Squared Return-{name_1}",
            f"Squared Return-{name_2}",
        ],
        vertical_spacing=0.16,
        horizontal_spacing=0.12,
    )

    fig.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame[name_1],
            mode="lines",
            name=f"Return-{name_1}",
            line={"width": 1},
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame[name_2],
            mode="lines",
            name=f"Return-{name_2}",
            line={"width": 1},
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=squared.index,
            y=squared[name_1],
            mode="lines",
            name=f"Squared Return-{name_1}",
            line={"width": 1},
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=squared.index,
            y=squared[name_2],
            mode="lines",
            name=f"Squared Return-{name_2}",
            line={"width": 1},
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    fig.update_yaxes(title_text="Returns (%)", row=1, col=1)
    fig.update_yaxes(title_text="Returns (%)", row=1, col=2)
    fig.update_yaxes(title_text="Squared Returns", row=2, col=1)
    fig.update_yaxes(title_text="Squared Returns", row=2, col=2)

    fig.update_layout(
        title="Daily returns and squared returns",
        template="plotly_white",
        hovermode="x unified",
        height=720,
        width=1000,
        margin={"l": 70, "r": 30, "t": 80, "b": 50},
    )

    if output_path is not None:
        save_plotly_figure(fig, output_path)

    return fig
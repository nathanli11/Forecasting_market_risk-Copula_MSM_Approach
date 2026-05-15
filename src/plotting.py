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


def plot_var_forecasts(
    portfolio_returns: pd.Series,
    var_forecasts: pd.DataFrame | pd.Series,
    output_path: str | Path | None = None,
    title: str = "VaR forecasts vs portfolio returns",
    positive_loss_var: bool = False,
) -> go.Figure:
    """Plot portfolio returns and VaR forecasts.

    This is intended to reproduce Figure 3-style VaR plots.

    Parameters
    ----------
    portfolio_returns:
        Portfolio return series, usually in percentage points.
    var_forecasts:
        VaR forecasts. If VaR is stored as a positive loss number, set
        positive_loss_var=True so that the plotted threshold is -VaR.
    output_path:
        Optional path used to save the figure.
    title:
        Figure title.
    positive_loss_var:
        If True, VaR forecasts are interpreted as positive losses and plotted
        with a negative sign. This matches the return-axis convention of the
        paper's VaR figures.

    Returns
    -------
    go.Figure
        Plotly figure.
    """
    returns = pd.to_numeric(portfolio_returns, errors="coerce").dropna()
    returns.name = returns.name or "Portfolio returns"

    if isinstance(var_forecasts, pd.Series):
        var_frame = var_forecasts.to_frame()
    else:
        var_frame = var_forecasts.copy()

    var_frame = var_frame.apply(pd.to_numeric, errors="coerce")

    common_index = returns.index.intersection(var_frame.dropna(how="all").index)
    returns = returns.loc[common_index]
    var_frame = var_frame.loc[common_index]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=returns.index,
            y=returns.to_numpy(),
            mode="lines",
            name=returns.name,
            line={"width": 1.2},
        )
    )

    for column in var_frame.columns:
        series = var_frame[column].dropna()

        y_values = -series.to_numpy() if positive_loss_var else series.to_numpy()

        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=y_values,
                mode="lines",
                name=str(column),
                line={"width": 1.4},
            )
        )

    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis_title="",
        yaxis_title="Return (%)",
        hovermode="x unified",
        legend={
            "x": 0.02,
            "y": 0.02,
            "xanchor": "left",
            "yanchor": "bottom",
            "bgcolor": "rgba(255,255,255,0.8)",
            "bordercolor": "black",
            "borderwidth": 1,
        },
        margin={"l": 70, "r": 30, "t": 70, "b": 50},
    )

    if output_path is not None:
        save_plotly_figure(fig, output_path)

    return fig
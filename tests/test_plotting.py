import pandas as pd
import plotly.graph_objects as go

from src.plotting import plot_returns


def test_plot_returns_returns_plotly_figure() -> None:
    returns = pd.Series(
        [0.01, -0.02, 0.015],
        index=pd.date_range("2020-01-01", periods=3),
        name="NASDAQCOM",
    )

    fig = plot_returns(returns)

    assert isinstance(fig, go.Figure)
    assert fig.data[0].name == "NASDAQCOM"
    assert fig.layout.template.layout.paper_bgcolor == "white"

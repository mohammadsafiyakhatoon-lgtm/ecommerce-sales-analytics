"""Plotly chart builders. Every function takes real dataframes and returns a styled figure."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import utils

ACCENT = "#3B5BDB"
ACCENT_SOFT = "#98ABF0"
INK = "#1F2328"
MUTED = "#6B7280"
GRID = "#ECECE8"
NEUTRAL = "#9AA3B2"
DONUT = ["#3B5BDB", "#6C86E8", "#98ABF0", "#C2CDF6", "#DDE3FA"]
FONT = "Inter, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"

LABELS = {
    "Sales_Amount": "Sales amount (₹)", "Profit": "Profit (₹)", "Quantity": "Quantity",
    "Discount": "Discount", "Profit_Margin": "Profit margin",
}


def style(fig: go.Figure, title: str, xlabel: str | None = None, ylabel: str | None = None,
          height: int = 340, legend: bool = False) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15, color=INK)),
        height=height, margin=dict(l=8, r=8, t=56, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color=INK),
        showlegend=legend, hoverlabel=dict(bgcolor="white", font_size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, title_text=""),
    )
    fig.update_xaxes(title_text=xlabel, gridcolor=GRID, zeroline=False, automargin=True, linecolor=GRID)
    fig.update_yaxes(title_text=ylabel, gridcolor=GRID, zeroline=False, automargin=True)
    return fig


# --------------------------------------------------------------------------- #
# Distributions
# --------------------------------------------------------------------------- #
def histogram(df: pd.DataFrame, col: str, title: str, nbins: int = 40, money: bool = False) -> go.Figure:
    fig = px.histogram(df, x=col, nbins=nbins)
    fig.update_traces(marker_color=ACCENT, marker_line_color="white", marker_line_width=1)
    style(fig, title, LABELS.get(col, col), "Number of orders")
    if money:
        fig.update_xaxes(tickprefix="₹", tickformat="~s")
    return fig


def count_bar(df: pd.DataFrame, col: str, title: str) -> go.Figure:
    g = df[col].value_counts().sort_index().reset_index()
    g.columns = [col, "Orders"]
    g[col] = g[col].astype(str)
    fig = px.bar(g, x=col, y="Orders")
    fig.update_traces(marker_color=ACCENT, texttemplate="%{y:,}", textposition="outside", cliponaxis=False)
    return style(fig, title, LABELS.get(col, col), "Number of orders")


# --------------------------------------------------------------------------- #
# Aggregated bars
# --------------------------------------------------------------------------- #
def bar_by(df: pd.DataFrame, group: str, value: str, title: str, agg: str = "sum",
           horizontal: bool = False, top: int | None = None, xlabel: str | None = None,
           ylabel: str | None = None, money: bool = True, height: int = 340) -> go.Figure:
    g = (df.groupby(group)[value].agg(agg).reset_index()
         .sort_values(value, ascending=False))
    if top:
        g = g.head(top)
    label = ylabel or LABELS.get(value, value)
    pref = "₹" if money else ""
    if horizontal:
        g = g.sort_values(value, ascending=True)
        fig = px.bar(g, x=value, y=group, orientation="h")
        fig.update_traces(texttemplate=pref + "%{x:.3s}", textposition="outside", cliponaxis=False)
        style(fig, title, xlabel or label, group.replace("_", " "), height)
        fig.update_xaxes(tickprefix=pref, tickformat="~s")
        fig.update_yaxes(title_text=None)
    else:
        fig = px.bar(g, x=group, y=value)
        fig.update_traces(texttemplate=pref + "%{y:.3s}", textposition="outside", cliponaxis=False)
        style(fig, title, xlabel or group.replace("_", " "), label, height)
        fig.update_yaxes(tickprefix=pref, tickformat="~s")
    fig.update_traces(marker_color=ACCENT)
    return fig


def payment_donut(df: pd.DataFrame, title: str = "Payment mode distribution") -> go.Figure:
    g = df["Payment_Mode"].value_counts().reset_index()
    g.columns = ["Payment_Mode", "Orders"]
    fig = go.Figure(go.Pie(
        labels=g["Payment_Mode"], values=g["Orders"], hole=0.62, sort=False,
        marker=dict(colors=DONUT[: len(g)], line=dict(color="white", width=2)),
        textinfo="percent", hovertemplate="%{label}<br>%{value:,} orders (%{percent})<extra></extra>"))
    style(fig, title, legend=True)
    fig.update_layout(legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02))
    return fig


# --------------------------------------------------------------------------- #
# Trends
# --------------------------------------------------------------------------- #
def monthly_trend(df: pd.DataFrame, value: str, title: str, height: int = 340) -> go.Figure:
    g = df.groupby("Year_Month")[value].sum().reset_index().sort_values("Year_Month")
    g["Month"] = pd.to_datetime(g["Year_Month"] + "-01")
    fig = px.line(g, x="Month", y=value, markers=True)
    fig.update_traces(line=dict(color=ACCENT, width=2.5), marker=dict(size=6, color=ACCENT),
                      hovertemplate="%{x|%b %Y}<br>₹%{y:,.0f}<extra></extra>")
    style(fig, title, "Month", LABELS.get(value, value), height)
    fig.update_xaxes(tickformat="%b %Y")
    fig.update_yaxes(tickprefix="₹", tickformat="~s", rangemode="tozero")
    return fig


# --------------------------------------------------------------------------- #
# Relationships
# --------------------------------------------------------------------------- #
def scatter(df: pd.DataFrame, x: str, y: str, title: str, sample: int = 3000, height: int = 340) -> go.Figure:
    plot_df = df.sample(min(len(df), sample), random_state=42)
    fig = px.scatter(plot_df, x=x, y=y, opacity=0.35)
    fig.update_traces(marker=dict(size=5, color=ACCENT))
    if len(df) > 2 and df[x].std() > 0:
        slope, intercept = np.polyfit(df[x].astype(float), df[y].astype(float), 1)
        xs = np.array([df[x].min(), df[x].max()], dtype=float)
        fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines", name="Linear trend",
                                 line=dict(color=INK, width=2, dash="dash")))
    style(fig, title, LABELS.get(x, x), LABELS.get(y, y), height, legend=True)
    if x == "Sales_Amount":
        fig.update_xaxes(tickprefix="₹", tickformat="~s")
    if y in ("Profit", "Sales_Amount"):
        fig.update_yaxes(tickprefix="₹", tickformat="~s")
    return fig


def quantity_sales_box(df: pd.DataFrame, title: str = "Sales amount by quantity") -> go.Figure:
    plot_df = df.assign(Qty=df["Quantity"].astype(int).astype(str))
    order = [str(q) for q in sorted(df["Quantity"].astype(int).unique())]
    fig = px.box(plot_df, x="Qty", y="Sales_Amount", category_orders={"Qty": order})
    fig.update_traces(marker_color=ACCENT, line_color=ACCENT, fillcolor="rgba(59,91,219,0.15)")
    style(fig, title, "Quantity per order", LABELS["Sales_Amount"])
    fig.update_yaxes(tickprefix="₹", tickformat="~s")
    return fig


def discount_band_margin(df: pd.DataFrame, title: str = "Average profit margin by discount band") -> go.Figure:
    work = df[["Discount", "Profit_Margin"]].copy()
    try:
        work["Band"] = pd.qcut(work["Discount"], q=5, duplicates="drop")
    except ValueError:
        work["Band"] = pd.cut(work["Discount"], bins=1)
    g = work.groupby("Band", observed=True)["Profit_Margin"].mean().reset_index()
    g["Band"] = g["Band"].astype(str)
    fig = px.bar(g, x="Band", y="Profit_Margin")
    fig.update_traces(marker_color=ACCENT, texttemplate="%{y:.1%}", textposition="outside", cliponaxis=False)
    style(fig, title, "Discount band", "Average profit margin")
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    return fig


def correlation_heatmap(df: pd.DataFrame, title: str = "Correlation heatmap") -> go.Figure:
    cols = [c for c in ["Quantity", "Sales_Amount", "Discount", "Profit", "Profit_Margin"] if c in df.columns]
    corr = df[cols].corr().round(2)
    fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, aspect="auto",
                    color_continuous_scale=[[0, NEUTRAL], [0.5, "#FFFFFF"], [1, ACCENT]])
    style(fig, title, height=380)
    fig.update_coloraxes(colorbar_title="r")
    return fig


# --------------------------------------------------------------------------- #
# Machine-learning charts
# --------------------------------------------------------------------------- #
def model_comparison(metrics: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, column_widths=[0.6, 0.4],
                        subplot_titles=("Error in ₹ (lower is better)", "R² (higher is better)"))
    colors = {"Linear Regression": NEUTRAL, "Random Forest": ACCENT}
    for _, r in metrics.iterrows():
        color = colors.get(r["Model"], ACCENT)
        fig.add_trace(go.Bar(name=r["Model"], x=["MAE", "RMSE"], y=[r["MAE"], r["RMSE"]], marker_color=color,
                             text=[f"{r['MAE']:.1f}", f"{r['RMSE']:.1f}"], textposition="outside",
                             cliponaxis=False), row=1, col=1)
        fig.add_trace(go.Bar(name=r["Model"], x=["R²"], y=[r["R2"]], marker_color=color, showlegend=False,
                             text=[f"{r['R2']:.3f}"], textposition="outside", cliponaxis=False), row=1, col=2)
    style(fig, "Model comparison on the test set", height=380, legend=True)
    fig.update_layout(barmode="group")
    fig.update_yaxes(rangemode="tozero")
    fig.update_yaxes(range=[0, 1], row=1, col=2)
    return fig


def actual_vs_predicted(pred_df: pd.DataFrame, model: str, sample: int = 3000) -> go.Figure:
    col = utils.PRED_COLUMNS[model]
    plot_df = pred_df.sample(min(len(pred_df), sample), random_state=42)
    fig = px.scatter(plot_df, x="Actual_Profit", y=col, opacity=0.4)
    fig.update_traces(marker=dict(size=5, color=ACCENT))
    lo = float(min(pred_df["Actual_Profit"].min(), pred_df[col].min()))
    hi = float(max(pred_df["Actual_Profit"].max(), pred_df[col].max()))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect prediction",
                             line=dict(color=INK, dash="dash", width=2)))
    style(fig, f"Actual vs predicted profit - {model}", "Actual profit (₹)", "Predicted profit (₹)",
          height=380, legend=True)
    return fig


def residual_scatter(pred_df: pd.DataFrame, model: str, sample: int = 3000) -> go.Figure:
    col = utils.PRED_COLUMNS[model]
    work = pd.DataFrame({"Predicted": pred_df[col], "Residual": pred_df["Actual_Profit"] - pred_df[col]})
    work = work.sample(min(len(work), sample), random_state=42)
    fig = px.scatter(work, x="Predicted", y="Residual", opacity=0.4)
    fig.update_traces(marker=dict(size=5, color=ACCENT))
    fig.add_hline(y=0, line_dash="dash", line_color=INK)
    return style(fig, f"Residuals vs predicted - {model}", "Predicted profit (₹)",
                 "Residual (actual - predicted, ₹)", height=340)


def residual_hist(pred_df: pd.DataFrame, model: str) -> go.Figure:
    col = utils.PRED_COLUMNS[model]
    work = pd.DataFrame({"Residual": pred_df["Actual_Profit"] - pred_df[col]})
    fig = px.histogram(work, x="Residual", nbins=50)
    fig.update_traces(marker_color=ACCENT, marker_line_color="white", marker_line_width=1)
    return style(fig, f"Residual distribution - {model}", "Residual (₹)", "Number of orders", height=340)


def feature_importance_chart(importance: pd.DataFrame, by_original: bool = True, top: int = 15) -> go.Figure:
    if by_original:
        g = importance.groupby("Original_Feature")["Importance"].sum().sort_values(ascending=False).head(top)
        g = g.reset_index().rename(columns={"Original_Feature": "Feature"})
        title = "Random Forest feature importance (grouped by input)"
    else:
        g = importance.sort_values("Importance", ascending=False).head(top)[["Feature", "Importance"]]
        title = f"Random Forest feature importance (top {top} encoded features)"
    g = g.sort_values("Importance", ascending=True)
    g["Feature"] = g["Feature"].str.replace("_", " ")
    fig = px.bar(g, x="Importance", y="Feature", orientation="h")
    fig.update_traces(marker_color=ACCENT, texttemplate="%{x:.1%}", textposition="outside", cliponaxis=False)
    style(fig, title, "Importance (share of total)", None, height=max(320, 26 * len(g) + 90))
    fig.update_xaxes(tickformat=".0%")
    return fig


def profit_gauge(value: float, average: float, max_profit: float) -> go.Figure:
    upper = max(max_profit, value, average) * 1.05
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=value,
        number=dict(prefix="₹", valueformat=",.2f", font=dict(size=34, color=INK)),
        delta=dict(reference=average, valueformat=",.2f", prefix="₹",
                   increasing=dict(color=ACCENT), decreasing=dict(color=MUTED)),
        gauge=dict(
            axis=dict(range=[min(0, value), upper], tickprefix="₹", tickformat="~s", tickcolor=MUTED),
            bar=dict(color=ACCENT, thickness=0.28), bgcolor="#F4F5F7", borderwidth=0,
            threshold=dict(line=dict(color=INK, width=3), thickness=0.8, value=average)),
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(family=FONT, color=INK))
    return fig

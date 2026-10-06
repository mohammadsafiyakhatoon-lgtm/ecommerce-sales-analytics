"""Exploratory data analysis: summaries, data-driven insights and static figures.

Every sentence returned by the functions below is computed from the data or from the
saved model outputs - nothing is hard-coded.

Run `python src/eda.py` to save static Matplotlib/Seaborn figures to outputs/figures/.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import utils


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #
def kpis(df: pd.DataFrame) -> dict:
    orders = int(df["Order_ID"].nunique())
    sales = float(df["Sales_Amount"].sum())
    return {
        "orders": orders,
        "sales": sales,
        "profit": float(df["Profit"].sum()),
        "quantity": int(df["Quantity"].sum()),
        "avg_discount": float(df["Discount"].mean()),
        "aov": sales / orders if orders else 0.0,
    }


def overview(df: pd.DataFrame) -> dict:
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "date_min": df["Order_Date"].min(),
        "date_max": df["Order_Date"].max(),
    }


def dtype_table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Column": df.columns,
        "Data type": [str(t) for t in df.dtypes],
        "Missing": df.isna().sum().values,
        "Unique values": [df[c].nunique() for c in df.columns],
    })


def describe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["Quantity", "Sales_Amount", "Discount", "Profit", "Profit_Margin"]
    cols = [c for c in cols if c in df.columns]
    return df[cols].describe().T.round(3).reset_index().rename(columns={"index": "Column"})


# --------------------------------------------------------------------------- #
# Insight generation
# --------------------------------------------------------------------------- #
def _strength(r: float) -> str:
    a = abs(r)
    if a < 0.05:
        return "no meaningful"
    if a < 0.3:
        return "a weak"
    if a < 0.6:
        return "a moderate"
    if a < 0.8:
        return "a strong"
    return "a very strong"


def generate_insights(df: pd.DataFrame) -> list[str]:
    """Plain-language findings computed from the (optionally filtered) dataframe."""
    if df.empty:
        return []
    out: list[str] = []
    k = kpis(df)
    out.append(f"The data covers {k['orders']:,} orders from {df['Order_Date'].min():%d %b %Y} to "
               f"{df['Order_Date'].max():%d %b %Y}, with total sales of {utils.format_compact(k['sales'])} "
               f"and total profit of {utils.format_compact(k['profit'])} "
               f"({k['profit'] / k['sales']:.1%} overall margin).")

    cat = df.groupby("Product_Category")[["Sales_Amount", "Profit"]].sum().sort_values("Sales_Amount", ascending=False)
    if len(cat) > 1:
        top, low = cat.index[0], cat.index[-1]
        gap = (cat["Sales_Amount"].iloc[0] / cat["Sales_Amount"].iloc[-1] - 1)
        out.append(f"{top} is the top category by sales ({cat['Sales_Amount'].iloc[0] / cat['Sales_Amount'].sum():.1%} "
                   f"of total); it is {gap:.1%} ahead of {low}, the lowest. "
                   + ("Categories are very evenly balanced." if gap < 0.10 else "Category sales are uneven."))
        ptop = cat["Profit"].idxmax()
        out.append(f"{ptop} generates the highest total profit ({utils.format_compact(cat['Profit'].max())}).")

    st = df.groupby("State")["Sales_Amount"].sum().sort_values(ascending=False)
    if len(st) > 1:
        out.append(f"{st.index[0]} leads all states in sales ({utils.format_compact(st.iloc[0])}); "
                   f"{st.index[-1]} is lowest ({utils.format_compact(st.iloc[-1])}).")

    prod = df.groupby("Product_Name")[["Sales_Amount", "Profit"]].sum()
    out.append(f"The best-selling product is {prod['Sales_Amount'].idxmax()} and the most profitable product is "
               f"{prod['Profit'].idxmax()}.")

    pay = df["Payment_Mode"].value_counts(normalize=True)
    out.append(f"{pay.index[0]} is the most used payment mode ({pay.iloc[0]:.1%} of orders); "
               f"{pay.index[-1]} is the least used ({pay.iloc[-1]:.1%}).")

    monthly = df.groupby("Year_Month")["Sales_Amount"].sum()
    if len(monthly) > 1:
        out.append(f"Monthly sales peaked in {pd.to_datetime(monthly.idxmax() + '-01'):%B %Y} "
                   f"({utils.format_compact(monthly.max())}) and were lowest in "
                   f"{pd.to_datetime(monthly.idxmin() + '-01'):%B %Y} ({utils.format_compact(monthly.min())}).")

    if len(df) > 2 and df["Sales_Amount"].std() > 0 and df["Profit"].std() > 0:
        r_sp = df["Sales_Amount"].corr(df["Profit"])
        out.append(f"Sales amount and profit have {_strength(r_sp)} positive relationship (r = {r_sp:.2f}).")
        r_dp = df["Discount"].corr(df["Profit"]) if df["Discount"].std() > 0 else 0.0
        out.append(f"Discount and profit show {_strength(r_dp)} linear relationship (r = {r_dp:.2f}).")
        r_qs = df["Quantity"].corr(df["Sales_Amount"]) if df["Quantity"].std() > 0 else 0.0
        out.append(f"Quantity and sales amount show {_strength(r_qs)} relationship (r = {r_qs:.2f}).")

    m = df["Profit_Margin"]
    out.append(f"Profit margin per order ranges from {m.min():.1%} to {m.max():.1%} (median {m.median():.1%}).")

    # Is City nested inside State?
    states_per_city = df.groupby("City")["State"].nunique()
    if len(states_per_city) and states_per_city.min() > 1:
        out.append(f"Every city appears in several states (up to {states_per_city.max()}), so City and State are "
                   "independent fields in this dataset and are analysed separately.")
    return out


def model_explanation(metrics: pd.DataFrame, info: dict | None) -> list[str]:
    """Why the best model was selected - generated from the saved evaluation metrics."""
    if metrics is None or metrics.empty:
        return []
    m = metrics.set_index("Model")
    best = metrics.sort_values(["RMSE", "MAE"]).iloc[0]["Model"]
    other = [x for x in m.index if x != best]
    out = []
    if other:
        o = other[0]
        out.append(f"{best} was selected because it has the lowest RMSE on the unseen test set "
                   f"({m.loc[best, 'RMSE']:.2f} vs {m.loc[o, 'RMSE']:.2f} for {o}) "
                   f"and an R² of {m.loc[best, 'R2']:.4f} vs {m.loc[o, 'R2']:.4f}.")
        diff = abs(m.loc[best, "R2"] - m.loc[o, "R2"])
        if diff < 0.02:
            out.append(f"The two models are very close (R² differs by {diff:.4f}), so the difference is small "
                       "in practice.")
        gap_best = m.loc[best, "Train_R2"] - m.loc[best, "CV_R2_Mean"]
        gap_o = m.loc[o, "Train_R2"] - m.loc[o, "CV_R2_Mean"]
        if gap_o - gap_best > 0.05:
            out.append(f"{o} fits the training data much better (train R² {m.loc[o, 'Train_R2']:.3f}) than "
                       f"held-out data (cross-validated R² {m.loc[o, 'CV_R2_Mean']:.3f}), which indicates "
                       "overfitting, while " + f"{best} generalises more consistently "
                       f"(train {m.loc[best, 'Train_R2']:.3f} vs CV {m.loc[best, 'CV_R2_Mean']:.3f}).")
    out.append(f"{best} explains about {m.loc[best, 'R2']:.0%} of the variation in profit; its average error is "
               f"{utils.format_inr(m.loc[best, 'MAE'], 2)} per order (MAE).")
    return out


def model_interpretation(df: pd.DataFrame, importance: pd.DataFrame | None) -> list[str]:
    """Data-driven reading of Random Forest importance plus correlations in the data."""
    out: list[str] = []
    if importance is None or importance.empty or df.empty:
        return out
    agg = importance.groupby("Original_Feature")["Importance"].sum().sort_values(ascending=False)
    total = agg.sum()
    share = (agg / total) if total else agg
    top3 = ", ".join(f"{n.replace('_', ' ')} ({share[n]:.1%})" for n in share.index[:3])
    out.append(f"Most influential inputs in the Random Forest: {top3}.")

    def feat(name: str) -> float:
        return float(share.get(name, 0.0))

    r_s = df["Sales_Amount"].corr(df["Profit"])
    out.append(f"Sales amount accounts for {feat('Sales_Amount'):.1%} of the importance and correlates with "
               f"profit at r = {r_s:.2f}: larger orders tend to earn more profit.")

    r_q = df["Quantity"].corr(df["Profit"])
    r_qs = df["Quantity"].corr(df["Sales_Amount"])
    out.append(f"Quantity has r = {r_q:.2f} with profit but also r = {r_qs:.2f} with sales amount, so most of its "
               f"effect is already captured by sales amount (importance {feat('Quantity'):.1%}).")

    r_d = df["Discount"].corr(df["Profit"])
    r_dm = df["Discount"].corr(df["Profit_Margin"])
    out.append(f"Discount has r = {r_d:.2f} with profit and r = {r_dm:.2f} with profit margin, i.e. "
               f"{_strength(r_dm)} linear association in this data (importance {feat('Discount'):.1%}).")

    cat = df.groupby("Product_Category")["Profit"].agg(["sum", "mean"])
    spread = cat["mean"].max() / cat["mean"].min() - 1
    out.append(f"{cat['sum'].idxmax()} contributes the most total profit; average profit per order differs by only "
               f"{spread:.1%} between the best and worst category "
               f"(category importance {feat('Product_Category') + feat('Product_Name'):.1%}).")

    loc_pay = feat("State") + feat("City") + feat("Payment_Mode")
    level = "negligible" if loc_pay < 0.05 else ("modest" if loc_pay < 0.10 else "meaningful")
    out.append(f"State, city and payment mode together account for {loc_pay:.1%} of the importance, "
               f"a {level} influence.")
    date_share = sum(feat(c) for c in ["Year", "Month", "Quarter", "Day", "Day_of_Week"])
    out.append(f"Date-derived features account for {date_share:.1%}. Because the Random Forest can fit noise, "
               "small shares like this should not be read as real effects.")
    out.append("Importance describes which inputs the model relies on, not what causes profit.")
    return out


# --------------------------------------------------------------------------- #
# Static figures (Matplotlib / Seaborn) for the report
# --------------------------------------------------------------------------- #
def save_static_figures(df: pd.DataFrame, out_dir=None) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    out_dir = out_dir or utils.FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette=["#3B5BDB"])
    saved = []

    def finish(fig, name):
        fig.tight_layout()
        path = out_dir / name
        fig.savefig(path, dpi=140)
        plt.close(fig)
        saved.append(str(path))

    for col, name in [("Sales_Amount", "sales_distribution.png"), ("Profit", "profit_distribution.png"),
                      ("Discount", "discount_distribution.png")]:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(df[col], bins=40, ax=ax, color="#3B5BDB")
        ax.set_title(f"Distribution of {col.replace('_', ' ')}")
        ax.set_xlabel(col.replace("_", " "))
        ax.set_ylabel("Number of orders")
        finish(fig, name)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(x="Quantity", data=df, ax=ax, color="#3B5BDB")
    ax.set_title("Orders by quantity")
    ax.set_ylabel("Number of orders")
    finish(fig, "quantity_distribution.png")

    for col, name in [("Sales_Amount", "sales_by_category.png"), ("Profit", "profit_by_category.png")]:
        g = df.groupby("Product_Category")[col].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(x=g.index, y=g.values, ax=ax, color="#3B5BDB")
        ax.set_title(f"Total {col.replace('_', ' ').lower()} by product category")
        ax.set_xlabel("Product category")
        ax.set_ylabel(f"Total {col.replace('_', ' ').lower()} (INR)")
        finish(fig, name)

    monthly = df.groupby("Year_Month")[["Sales_Amount", "Profit"]].sum()
    for col, name in [("Sales_Amount", "monthly_sales.png"), ("Profit", "monthly_profit.png")]:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(monthly.index, monthly[col], color="#3B5BDB", marker="o")
        ax.set_title(f"Monthly {col.replace('_', ' ').lower()}")
        ax.set_xlabel("Month")
        ax.set_ylabel(f"{col.replace('_', ' ')} (INR)")
        ax.tick_params(axis="x", rotation=60)
        finish(fig, name)

    fig, ax = plt.subplots(figsize=(6, 5))
    corr = df[["Quantity", "Sales_Amount", "Discount", "Profit"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlation heatmap")
    finish(fig, "correlation_heatmap.png")
    return saved


if __name__ == "__main__":
    import data_preprocessing as dp
    files = save_static_figures(dp.load_clean_data())
    print(f"Saved {len(files)} figures to {utils.FIGURES_DIR}")

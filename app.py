"""E-Commerce Sales Analytics & Profit Prediction - Streamlit application.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import streamlit as st

st.set_page_config(page_title="E-Commerce Analytics", layout="wide", initial_sidebar_state="expanded")

import charts  # noqa: E402
import data_preprocessing as dp  # noqa: E402
import eda  # noqa: E402
import prediction  # noqa: E402
import ui  # noqa: E402
import utils  # noqa: E402

PAGES = ["Home", "Dashboard", "Data Exploration", "Machine Learning", "Profit Prediction", "About Project"]
FILTER_KEYS = ["f_date", "f_cat", "f_prod", "f_state", "f_city", "f_pay"]


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading data ...")
def load_data() -> pd.DataFrame:
    return dp.load_clean_data()


@st.cache_data(show_spinner=False)
def load_artifacts() -> dict:
    return {
        "metrics": utils.read_csv_if_exists(utils.METRICS_PATH),
        "importance": utils.read_csv_if_exists(utils.IMPORTANCE_PATH),
        "predictions": utils.read_csv_if_exists(utils.PREDICTIONS_PATH),
        "info": utils.read_json(utils.TRAIN_INFO_PATH),
        "quality": utils.read_json(utils.QUALITY_REPORT_PATH),
    }


@st.cache_resource(show_spinner="Loading trained models ...")
def get_models() -> dict:
    return prediction.load_models()


def go(page: str) -> None:
    st.session_state["page"] = page


def retrain_button(key: str) -> None:
    """Fallback when models / metrics are missing: retrain from the app."""
    if st.button("Retrain models now", key=key, type="primary"):
        ok = True
        with st.spinner("Training both models - this takes about a minute ..."):
            try:
                import train_models
                train_models.main()
            except Exception as exc:  # noqa: BLE001
                ok = False
                st.error(f"Training failed: {exc}")
        if ok:
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()


def ml_ready(art: dict) -> bool:
    return all(art.get(k) is not None for k in ["metrics", "importance", "predictions", "info"])


def best_model_name(art: dict) -> str:
    if art.get("info") and art["info"].get("best_model") in utils.MODEL_NAMES:
        return art["info"]["best_model"]
    return utils.MODEL_NAMES[0]


# --------------------------------------------------------------------------- #
# Login / sidebar
# --------------------------------------------------------------------------- #
def login_page() -> None:
    try:
        n_orders = f"{load_data()['Order_ID'].nunique():,}"
    except Exception:  # noqa: BLE001
        n_orders = None

    left, right = st.columns([1.15, 1], gap="large")
    with left:
        stats = ""
        if n_orders:
            stats = (f'<div class="login-stats"><div class="login-stat"><b>{n_orders}</b><span>Orders analysed</span></div>'
                     '<div class="login-stat"><b>2</b><span>ML models compared</span></div>'
                     '<div class="login-stat"><b>Live</b><span>Profit prediction</span></div></div>')
        st.markdown(
            '<div class="login-brand"><div class="brand" style="margin:0"><div class="brand-mark">EA</div>'
            '<div><div class="brand-name">E-Commerce Analytics</div>'
            '<div class="brand-sub">Sales &amp; profit intelligence</div></div></div>'
            '<h1>Turn your sales data into smarter decisions.</h1>'
            '<p>Explore sales performance, compare machine learning models and estimate the expected '
            'profit of a new order - all in one place.</p>' + stats + '</div>',
            unsafe_allow_html=True)
    with right:
        st.markdown("<div style='height:3.2rem'></div>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("<h3 style='margin:.2rem 0 .1rem 0'>Sign in</h3>"
                        "<p style='color:#6B7280;font-size:.88rem;margin:0 0 .8rem 0'>"
                        "Enter your credentials to open the dashboard.</p>", unsafe_allow_html=True)
            with st.form("login_form", clear_on_submit=False):
                username = st.text_input("Username", placeholder="admin")
                password = st.text_input("Password", type="password", placeholder="Password")
                submitted = st.form_submit_button("Login", type="primary")
            if submitted:
                if utils.check_credentials(username, password):
                    st.session_state["authenticated"] = True
                    st.session_state["username"] = username.strip()
                    st.session_state["page"] = "Home"
                    st.rerun()
                else:
                    st.error("Incorrect username or password.")
            st.markdown(f'<div class="login-hint">Demo access: {utils.DEMO_USERNAME} / {utils.DEMO_PASSWORD}</div>',
                        unsafe_allow_html=True)


def sidebar() -> None:
    with st.sidebar:
        ui.brand()
        ui.nav_label("Navigation")
        st.radio("Navigation", PAGES, key="page", label_visibility="collapsed")
        st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
        ui.user_chip(st.session_state.get("username", "admin"))
        if st.button("Sign out", key="logout"):
            st.session_state.clear()
            st.rerun()


# --------------------------------------------------------------------------- #
# HOME
# --------------------------------------------------------------------------- #
def kpi_items(df: pd.DataFrame) -> list[tuple]:
    k = eda.kpis(df)
    return [
        ("Total orders", utils.format_int(k["orders"])),
        ("Total sales", utils.format_compact(k["sales"]), utils.format_inr(k["sales"])),
        ("Total profit", utils.format_compact(k["profit"]), utils.format_inr(k["profit"])),
        ("Average discount", f"{k['avg_discount']:.1%}", f"Mean of {len(df):,} orders"),
        ("Total quantity", utils.format_int(k["quantity"]), "Units sold"),
        ("Average order value", utils.format_inr(k["aov"]), "Sales per order"),
    ]


def page_home(df: pd.DataFrame, art: dict) -> None:
    n = len(df)
    ui.hero("E-Commerce Sales Analytics & Profit Prediction System",
            f"Analyse {n:,} orders from {df['Order_Date'].min():%b %Y} to {df['Order_Date'].max():%b %Y}, "
            "compare two regression models and estimate the expected profit of a new order.",
            eyebrow="Data Analytics & Data Visualization")
    ui.kpi_grid(kpi_items(df))

    ui.section("How it works", "From raw records to a live profit estimate")
    ui.steps([
        ("Data", f"{n:,} order records loaded from Excel"),
        ("Preprocessing", "Cleaning, type fixes, date features"),
        ("EDA", "Distributions, relationships, insights"),
        ("Visualization", "Interactive, filterable dashboard"),
        ("ML model", "Linear Regression vs Random Forest"),
        ("Profit prediction", "Estimate profit for a new order"),
    ])

    ui.section("Explore")
    cards = [
        ("Dashboard", "Dashboard", "Filter by date, category, product, state, city and payment mode."),
        ("Data Exploration", "Exploratory analysis", "Data quality, wrangling summary, distributions and insights."),
        ("Machine Learning", "Model comparison", "MAE, RMSE and R² for both models, with diagnostic charts."),
        ("Profit Prediction", "Predict profit", "Enter an order and get an estimate from the trained model."),
    ]
    cols = st.columns(4, gap="medium")
    for col, (page, title, body) in zip(cols, cards):
        with col:
            ui.info_card(title, body)
            st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
            st.button("Open", key=f"home_open_{page}", on_click=go, args=(page,))


# --------------------------------------------------------------------------- #
# DASHBOARD
# --------------------------------------------------------------------------- #
def reset_filters() -> None:
    for key in FILTER_KEYS:
        st.session_state.pop(key, None)


def filter_panel(df: pd.DataFrame) -> pd.DataFrame:
    dmin, dmax = df["Order_Date"].min().date(), df["Order_Date"].max().date()
    with st.container(border=True):
        c1, c2, c3 = st.columns([1.2, 1, 1], gap="medium")
        date_val = c1.date_input("Date range", value=(dmin, dmax), min_value=dmin, max_value=dmax, key="f_date")
        sel_cat = c2.multiselect("Product category", sorted(df["Product_Category"].unique()), key="f_cat",
                                 placeholder="All categories")
        pool = df[df["Product_Category"].isin(sel_cat)] if sel_cat else df
        prod_opts = sorted(pool["Product_Name"].unique())
        st.session_state["f_prod"] = [p for p in st.session_state.get("f_prod", []) if p in prod_opts]
        sel_prod = c3.multiselect("Product", prod_opts, key="f_prod", placeholder="All products")

        c4, c5, c6, c7 = st.columns([1, 1, 1, 0.5], gap="medium")
        sel_state = c4.multiselect("State", sorted(df["State"].unique()), key="f_state", placeholder="All states")
        sel_city = c5.multiselect("City", sorted(df["City"].unique()), key="f_city", placeholder="All cities")
        sel_pay = c6.multiselect("Payment mode", sorted(df["Payment_Mode"].unique()), key="f_pay",
                                 placeholder="All payment modes")
        c7.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
        c7.button("Reset", key="reset_filters", on_click=reset_filters)

    if isinstance(date_val, (tuple, list)):
        if len(date_val) == 2:
            start, end = date_val
        elif len(date_val) == 1:
            start = end = date_val[0]
        else:
            start, end = dmin, dmax
    else:
        start = end = date_val

    mask = (df["Order_Date"].dt.date >= start) & (df["Order_Date"].dt.date <= end)
    for col, sel in [("Product_Category", sel_cat), ("Product_Name", sel_prod), ("State", sel_state),
                     ("City", sel_city), ("Payment_Mode", sel_pay)]:
        if sel:
            mask &= df[col].isin(sel)
    return df[mask]


def page_dashboard(df: pd.DataFrame, art: dict) -> None:
    ui.page_header("Sales Dashboard", "Interactive overview of sales, profit and orders. "
                   "Every chart and KPI responds to the filters below.", eyebrow="Dashboard")
    fdf = filter_panel(df)
    st.caption(f"Showing {len(fdf):,} of {len(df):,} orders")
    if fdf.empty:
        ui.empty_state("No orders match these filters", "Widen the date range or clear some filters, "
                       "or press Reset.")
        return

    st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
    ui.kpi_grid(kpi_items(fdf))

    ui.section("Trends")
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        ui.show(charts.monthly_trend(fdf, "Sales_Amount", "Monthly sales trend"), "d_ms")
    with c2:
        ui.show(charts.monthly_trend(fdf, "Profit", "Monthly profit trend"), "d_mp")

    ui.section("Categories")
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        ui.show(charts.bar_by(fdf, "Product_Category", "Sales_Amount", "Sales by product category"), "d_sc")
    with c2:
        ui.show(charts.bar_by(fdf, "Product_Category", "Profit", "Profit by product category"), "d_pc")

    ui.section("Geography and payments")
    c1, c2 = st.columns([1.35, 1], gap="medium")
    with c1:
        ui.show(charts.bar_by(fdf, "State", "Sales_Amount", "Sales by state", horizontal=True), "d_ss")
    with c2:
        ui.show(charts.payment_donut(fdf), "d_pay")

    ui.section("Products")
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        ui.show(charts.bar_by(fdf, "Product_Name", "Sales_Amount", "Top 10 products by sales",
                              horizontal=True, top=10, height=420), "d_tps")
    with c2:
        ui.show(charts.bar_by(fdf, "Product_Name", "Profit", "Top 10 products by profit",
                              horizontal=True, top=10, height=420), "d_tpp")

    ui.section("Volume")
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        ui.show(charts.bar_by(fdf, "Product_Category", "Quantity", "Quantity sold by category",
                              ylabel="Units sold", money=False), "d_qc")
    with c2:
        ui.show(charts.bar_by(fdf, "City", "Sales_Amount", "Sales by city", horizontal=True), "d_city")


# --------------------------------------------------------------------------- #
# DATA EXPLORATION
# --------------------------------------------------------------------------- #
def page_eda(df: pd.DataFrame, art: dict) -> None:
    ui.page_header("Data Exploration", "Data quality, wrangling summary, distributions and relationships - "
                   "all computed from the cleaned dataset.", eyebrow="Exploratory data analysis")
    ov = eda.overview(df)
    ui.kpi_grid([
        ("Rows", f"{ov['rows']:,}"), ("Columns", f"{ov['columns']}"),
        ("Missing values", f"{ov['missing']:,}"), ("Duplicate rows", f"{ov['duplicates']:,}"),
        ("Date range", f"{ov['date_min']:%b %Y} - {ov['date_max']:%b %Y}",
         f"{ov['date_min']:%d %b %Y} to {ov['date_max']:%d %b %Y}"),
    ])

    ui.section("Key data insights", "Generated automatically from the data")
    ui.insight_list(eda.generate_insights(df))

    tabs = st.tabs(["Overview", "Data wrangling", "Distributions", "Categories and locations",
                    "Trends", "Relationships"])

    with tabs[0]:
        ui.section("Data types", "Column types, missing values and unique counts")
        ui.show_df(eda.dtype_table(df))
        ui.section("Descriptive statistics")
        ui.show_df(eda.describe_numeric(df))
        ui.section("Cleaned dataset preview", "First 100 rows")
        ui.show_df(df.head(100), height=360)

    with tabs[1]:
        q = art.get("quality")
        if not q:
            ui.empty_state("Data-quality report not found", "Run `python src/data_preprocessing.py` to create it.")
        else:
            b, a = q["before"], q["after"]
            ui.section("Before and after cleaning")
            table = pd.DataFrame([
                ("Rows", b["rows"], a["rows"]), ("Columns (raw schema)", b["columns"], a["columns"]),
                ("Missing values", b["missing_total"], a["missing_total"]),
                ("Duplicate rows", b["duplicate_rows"], a["duplicate_rows"]),
                ("Duplicate Order_ID", b["duplicate_order_ids"], a["duplicate_order_ids"]),
                ("Invalid dates", b["invalid_dates"], a["invalid_dates"]),
            ], columns=["Check", "Before", "After"])
            ui.html_table(table, formats={"Before": lambda v: f"{int(v):,}", "After": lambda v: f"{int(v):,}"})
            st.caption(f"Records removed in total: {q['records_removed']:,}. "
                       f"Final modelling table: {q['final_shape'][0]:,} rows x {q['final_shape'][1]} columns "
                       "(raw columns plus derived date features and Profit_Margin).")
            ui.section("Wrangling steps", "Every step with the number of records it affected")
            ui.html_table(pd.DataFrame(q["steps"]))
            ui.section("Outliers (IQR rule)", "Flagged for information; retained because they are valid orders")
            out = pd.DataFrame({"Column": list(b["outliers_iqr"].keys()),
                                "Outliers flagged": list(b["outliers_iqr"].values())})
            ui.html_table(out)

    with tabs[2]:
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            ui.show(charts.histogram(df, "Sales_Amount", "Sales amount distribution", money=True), "e_hs")
        with c2:
            ui.show(charts.histogram(df, "Profit", "Profit distribution", money=True), "e_hp")
        c3, c4 = st.columns(2, gap="medium")
        with c3:
            ui.show(charts.count_bar(df, "Quantity", "Quantity distribution"), "e_hq")
        with c4:
            ui.show(charts.histogram(df, "Discount", "Discount distribution", nbins=26), "e_hd")

    with tabs[3]:
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            ui.show(charts.bar_by(df, "Product_Category", "Sales_Amount", "Sales by product category"), "e_sc")
        with c2:
            ui.show(charts.bar_by(df, "Product_Category", "Profit", "Profit by product category"), "e_pc")
        c3, c4 = st.columns(2, gap="medium")
        with c3:
            ui.show(charts.bar_by(df, "State", "Sales_Amount", "Sales by state", horizontal=True), "e_ss")
        with c4:
            ui.show(charts.bar_by(df, "State", "Profit", "Profit by state", horizontal=True), "e_ps")
        c5, c6 = st.columns(2, gap="medium")
        with c5:
            ui.show(charts.bar_by(df, "Product_Name", "Sales_Amount", "Top 10 products by sales",
                                  horizontal=True, top=10, height=420), "e_tps")
        with c6:
            ui.show(charts.bar_by(df, "Product_Name", "Profit", "Top 10 products by profit",
                                  horizontal=True, top=10, height=420), "e_tpp")
        c7, c8 = st.columns(2, gap="medium")
        with c7:
            ui.show(charts.payment_donut(df), "e_pay")
        with c8:
            ui.show(charts.bar_by(df, "Product_Category", "Quantity", "Quantity sold by category",
                                  ylabel="Units sold", money=False), "e_qc")
        ui.show(charts.bar_by(df, "City", "Sales_Amount", "Sales by city", horizontal=True), "e_city")

    with tabs[4]:
        ui.show(charts.monthly_trend(df, "Sales_Amount", "Monthly sales trend"), "e_ms")
        ui.show(charts.monthly_trend(df, "Profit", "Monthly profit trend"), "e_mp")

    with tabs[5]:
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            ui.show(charts.scatter(df, "Sales_Amount", "Profit", "Sales amount vs profit"), "e_sp")
        with c2:
            ui.show(charts.scatter(df, "Discount", "Profit", "Discount vs profit"), "e_dp")
        c3, c4 = st.columns(2, gap="medium")
        with c3:
            ui.show(charts.quantity_sales_box(df, "Quantity vs sales amount"), "e_qs")
        with c4:
            ui.show(charts.discount_band_margin(df), "e_dm")
        ui.show(charts.correlation_heatmap(df), "e_corr")


# --------------------------------------------------------------------------- #
# MACHINE LEARNING
# --------------------------------------------------------------------------- #
def page_ml(df: pd.DataFrame, art: dict) -> None:
    ui.page_header("Machine Learning", "Predicting Profit for an E-Commerce Order - Linear Regression compared "
                   "with Random Forest Regressor.", eyebrow="Model comparison")
    if not ml_ready(art):
        ui.empty_state("Model results not found", "Run `python src/train_models.py`, or retrain below.")
        retrain_button("retrain_ml")
        return

    metrics, info, preds, imp = art["metrics"], art["info"], art["predictions"], art["importance"]
    best = best_model_name(art)

    ui.kpi_grid([
        ("Training records", f"{info['n_train']:,}", f"{1 - info['test_size']:.0%} of data"),
        ("Testing records", f"{info['n_test']:,}", f"{info['test_size']:.0%} of data"),
        ("Input features", f"{info['n_input_features']}", f"{info['n_encoded_features']} after encoding"),
        ("Target", info["target"], "Continuous (regression)"),
        ("Best model", best, "Lowest test RMSE"),
    ])

    ui.section("Models")
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        ui.info_card("Linear Regression", "A simple, interpretable baseline that fits a straight-line relationship. "
                     "Numeric inputs are scaled and categories are one-hot encoded.")
    with c2:
        ui.info_card("Random Forest Regressor", "An ensemble of decision trees that can capture non-linear patterns "
                     "and interactions. No scaling is needed.")

    ui.section("Model comparison", "Evaluated on the same unseen test set")
    show = metrics.copy()
    show["Status"] = show["Model"].map(lambda m: "Best model" if m == best else "")
    show = show[["Model", "MAE", "RMSE", "R2", "CV_R2_Mean", "Train_R2", "Status"]].rename(
        columns={"R2": "R²", "CV_R2_Mean": "CV R² (5-fold)", "Train_R2": "Train R²"})
    ui.html_table(show, highlight_col="Model", highlight_value=best, formats={
        "MAE": lambda v: f"{v:,.2f}", "RMSE": lambda v: f"{v:,.2f}", "R²": lambda v: f"{v:.4f}",
        "CV R² (5-fold)": lambda v: f"{v:.4f}", "Train R²": lambda v: f"{v:.4f}"})
    st.caption("MAE and RMSE are in rupees per order. Higher R² and lower MAE/RMSE are better.")

    ui.section("Why was this model selected?")
    ui.insight_list(eda.model_explanation(metrics, info))

    ui.section("Performance charts")
    ui.show(charts.model_comparison(metrics), "m_cmp")

    sel = st.radio("Model for diagnostic charts", utils.MODEL_NAMES, horizontal=True,
                   index=utils.MODEL_NAMES.index(best), key="ml_diag_model")
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        ui.show(charts.actual_vs_predicted(preds, sel), "m_avp")
    with c2:
        ui.show(charts.residual_scatter(preds, sel), "m_res")
    ui.show(charts.residual_hist(preds, sel), "m_resh")

    ui.section("Random Forest feature importance")
    t1, t2 = st.tabs(["By input feature", "By encoded feature"])
    with t1:
        ui.show(charts.feature_importance_chart(imp, by_original=True), "m_fi1")
    with t2:
        ui.show(charts.feature_importance_chart(imp, by_original=False), "m_fi2")

    ui.section("What the model tells us", "Computed from feature importance and the data")
    ui.insight_list(eda.model_interpretation(df, imp))


# --------------------------------------------------------------------------- #
# PROFIT PREDICTION
# --------------------------------------------------------------------------- #
def page_predict(df: pd.DataFrame, art: dict) -> None:
    ui.page_header("Predict Expected Profit", "Describe an order and the trained model will estimate its profit.",
                   eyebrow="Profit prediction")
    try:
        models = get_models()
    except prediction.ModelLoadError as exc:
        st.error(str(exc))
        retrain_button("retrain_pred")
        return
    except Exception as exc:  # noqa: BLE001
        st.error(f"Unexpected error while loading models: {exc}")
        retrain_button("retrain_pred2")
        return
    warn = prediction.version_warning()
    if warn:
        st.warning(warn)

    product_map = {c: sorted(g["Product_Name"].unique()) for c, g in df.groupby("Product_Category")}
    best = best_model_name(art)

    left, right = st.columns([1, 1.05], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("<h4 style='margin:.1rem 0 .6rem 0'>Order details</h4>", unsafe_allow_html=True)
            model_name = st.radio("Model", utils.MODEL_NAMES, horizontal=True,
                                  index=utils.MODEL_NAMES.index(best), key="p_model")
            c1, c2 = st.columns(2)
            order_date = c1.date_input("Order date", value=df["Order_Date"].max().date(), key="p_date")
            category = c2.selectbox("Product category", sorted(product_map), key="p_cat")
            opts = product_map.get(category, [])
            if st.session_state.get("p_prod") not in opts and opts:
                st.session_state["p_prod"] = opts[0]
            c3, c4 = st.columns(2)
            product = c3.selectbox("Product name", opts, key="p_prod")
            payment = c4.selectbox("Payment mode", sorted(df["Payment_Mode"].unique()), key="p_pay")
            c5, c6 = st.columns(2)
            state = c5.selectbox("State", sorted(df["State"].unique()), key="p_state")
            city = c6.selectbox("City", sorted(df["City"].unique()), key="p_city")
            c7, c8, c9 = st.columns(3)
            quantity = c7.number_input("Quantity", min_value=1, max_value=1000, step=1,
                                       value=int(df["Quantity"].median()), key="p_qty")
            sales = c8.number_input("Sales amount (₹)", min_value=1.0, max_value=10_000_000.0, step=100.0,
                                    value=float(df["Sales_Amount"].median()), key="p_sales")
            discount = c9.number_input("Discount (0-1)", min_value=0.0, max_value=1.0, step=0.01, format="%.2f",
                                       value=round(float(df["Discount"].median()), 2), key="p_disc")
            clicked = st.button("Predict Profit", type="primary", key="p_go")

        if clicked:
            errors, _ = prediction.validate_inputs(order_date, category, product, state, city, payment,
                                                   quantity, sales, discount, product_map)
            if errors:
                for e in errors:
                    st.error(e)
            else:
                try:
                    frame = prediction.build_input_frame(order_date, category, product, state, city, payment,
                                                         quantity, sales, discount)
                    values = {m: prediction.predict_profit(models[m], frame) for m in utils.MODEL_NAMES}
                    st.session_state["last_prediction"] = {
                        "model": model_name, "values": values,
                        "inputs": {"Order date": f"{pd.Timestamp(order_date):%d %b %Y}", "Category": category,
                                   "Product": product, "State": state, "City": city, "Payment mode": payment,
                                   "Quantity": int(quantity), "Sales amount": utils.format_inr(sales, 2),
                                   "Discount": f"{discount:.0%}"},
                        "sales": float(sales), "category": category,
                        "notes": prediction.range_notes(df, order_date, quantity, sales, discount),
                    }
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Prediction failed: {exc}")

    with right:
        res = st.session_state.get("last_prediction")
        if not res:
            ui.empty_state("No prediction yet", "Fill in the order details and press Predict Profit.")
            return
        render_prediction(df, art, res)


def render_prediction(df: pd.DataFrame, art: dict, res: dict) -> None:
    pred = res["values"][res["model"]]
    avg, pmin, pmax = float(df["Profit"].mean()), float(df["Profit"].min()), float(df["Profit"].max())
    if pred < 0:
        status, tone = "Below zero - outside the observed pattern", "warn"
    elif pred < pmin:
        status, tone = "Below the observed profit range", "warn"
    elif pred > pmax:
        status, tone = "Above the observed profit range", "warn"
    else:
        status, tone = "Within the observed profit range", "good"

    st.markdown(
        '<div class="result"><div class="result-label">Predicted profit</div>'
        f'<div class="result-value">{utils.format_inr(pred, 2)}</div>'
        f'{ui.pill(res["model"], "accent")} &nbsp; {ui.pill(status, tone)}</div>', unsafe_allow_html=True)
    ui.show(charts.profit_gauge(pred, avg, pmax), "p_gauge")

    cat_avg = float(df.loc[df["Product_Category"] == res["category"], "Profit"].mean())
    diff = (pred / avg - 1) if avg else 0.0
    margin = pred / res["sales"] if res["sales"] else 0.0
    ui.kpi_grid([
        ("Dataset average profit", utils.format_inr(avg, 2), f"{diff:+.1%} vs this estimate"),
        (f"{res['category']} average", utils.format_inr(cat_avg, 2), "Average order in category"),
        ("Implied margin", f"{margin:.1%}", f"Dataset median {df['Profit_Margin'].median():.1%}"),
    ])

    other = [m for m in utils.MODEL_NAMES if m != res["model"]][0]
    mae_txt = ""
    if art.get("metrics") is not None:
        row = art["metrics"].loc[art["metrics"]["Model"] == res["model"]]
        if not row.empty:
            mae_txt = (f" On unseen test orders this model's average error was "
                       f"{utils.format_inr(float(row['MAE'].iloc[0]), 2)} (MAE), so treat the figure as an estimate.")
    ui.section("Interpretation")
    ui.info_card(
        "What this means",
        f"Based on the entered order characteristics, the {res['model']} model estimates an expected profit of "
        f"{utils.format_inr(pred, 2)}, which is {abs(diff):.1%} {'above' if diff >= 0 else 'below'} the dataset "
        f"average of {utils.format_inr(avg, 2)}. For comparison, {other} estimates "
        f"{utils.format_inr(res['values'][other], 2)}.{mae_txt} "
        "This reflects patterns in historical data, not a causal relationship.")
    for note in res.get("notes", []):
        st.warning(note)

    ui.section("Input summary")
    summary = pd.DataFrame(list(res["inputs"].items()), columns=["Field", "Value"])
    ui.html_table(summary)


# --------------------------------------------------------------------------- #
# ABOUT
# --------------------------------------------------------------------------- #
def page_about(df: pd.DataFrame, art: dict) -> None:
    ui.page_header("About Project", "Smart E-Commerce Sales Analytics and Profit Prediction System",
                   eyebrow="Project overview")
    ui.kpi_grid([
        ("Domain", "E-Commerce"), ("Dataset size", f"{len(df):,} records"),
        ("Columns (raw)", f"{len(utils.RAW_COLUMNS)}"), ("Target variable", utils.TARGET),
        ("Models", "2 regression models"),
    ])

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        ui.section("Objectives")
        ui.insight_list([
            "Clean and wrangle the raw e-commerce order data and document every change.",
            "Explore sales, profit, quantity and discount patterns with interactive visualizations.",
            "Rebuild the key Power BI dashboard views as a filterable web dashboard.",
            "Train and compare Linear Regression and Random Forest to predict profit.",
            "Provide a live prediction form that uses the saved trained models.",
        ])
    with c2:
        ui.section("Workflow")
        ui.insight_list([
            "Excel dataset - Pandas DataFrame",
            "Cleaning and wrangling - cleaned CSV and data-quality report",
            "Exploratory data analysis and visualization",
            "80/20 train-test split with preprocessing fitted on training data only",
            "Model training, evaluation and comparison - saved with Joblib",
            "Streamlit web application with live prediction",
        ])

    ui.section("Technologies")
    ui.html_table(pd.DataFrame([
        ("Python", "Programming language"), ("Pandas / NumPy", "Data handling and numerical computing"),
        ("Matplotlib / Seaborn", "Static report figures"), ("Plotly", "Interactive charts"),
        ("Scikit-learn", "Pipelines, preprocessing, models, metrics"), ("Joblib", "Saving and loading models"),
        ("Streamlit", "Web application"),
    ], columns=["Technology", "Purpose"]))

    ui.section("Limitations")
    lim = [
        "The dataset covers a fixed period; the models do not account for seasonality beyond date features.",
        "Only the order attributes available in the dataset are used - no customer history or product cost data.",
        "City and State are independent in this dataset, so location effects cannot be interpreted geographically.",
        "Predictions are statistical estimates and do not imply causation.",
    ]
    m = df["Profit_Margin"]
    msg = (f"Profit margin per order varies between {m.min():.1%} and {m.max():.1%}")
    if art.get("metrics") is not None and not art["metrics"].empty:
        msg += f" and the best model reaches R² = {art['metrics']['R2'].max():.2f}, so part of the variation in " \
               "profit is not explained by the available inputs."
    else:
        msg += "."
    lim.insert(0, msg)
    ui.insight_list(lim)

    ui.section("Future enhancements")
    ui.insight_list([
        "Hyperparameter tuning and additional models such as Gradient Boosting.",
        "Time-based train/test split and forecasting of future sales.",
        "Customer segmentation using Customer_ID history.",
        "Cloud deployment with real authentication.",
    ])


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.session_state.setdefault("authenticated", False)
    ui.inject_css(show_sidebar=bool(st.session_state["authenticated"]))
    if not st.session_state["authenticated"]:
        login_page()
        return

    try:
        df = load_data()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load the dataset: {exc}")
        st.stop()
        return
    if df.empty:
        st.error("The dataset is empty after cleaning.")
        st.stop()
        return

    st.session_state.setdefault("page", "Home")
    sidebar()
    art = load_artifacts()
    page = st.session_state.get("page", "Home")
    {
        "Home": page_home, "Dashboard": page_dashboard, "Data Exploration": page_eda,
        "Machine Learning": page_ml, "Profit Prediction": page_predict, "About Project": page_about,
    }.get(page, page_home)(df, art)


main()

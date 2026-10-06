"""Data inspection, cleaning and wrangling for the e-commerce dataset.

Run directly to (re)create the cleaned dataset and the data-quality report:
    python src/data_preprocessing.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import utils


# --------------------------------------------------------------------------- #
# Loading and inspection
# --------------------------------------------------------------------------- #
def load_raw_data(path=None) -> pd.DataFrame:
    """Read the Excel dataset and verify that all expected columns exist."""
    path = path or utils.RAW_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place 'ecommerce_sales_dataset.xlsx' in the data/ folder."
        )
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in utils.RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")
    return df[utils.RAW_COLUMNS].copy()


def iqr_outlier_count(series: pd.Series) -> int:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 0
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())


def quality_snapshot(df: pd.DataFrame) -> dict:
    """Shape, types, missing values, duplicates, unique counts, date range, outliers."""
    dates = pd.to_datetime(df["Order_Date"], errors="coerce")
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isna().sum().sum()),
        "missing_by_column": {c: int(v) for c, v in df.isna().sum().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_order_ids": int(df["Order_ID"].duplicated().sum()),
        "invalid_dates": int(dates.isna().sum()),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "unique_values": {c: int(df[c].nunique()) for c in df.columns},
        "date_min": None if dates.dropna().empty else str(dates.min().date()),
        "date_max": None if dates.dropna().empty else str(dates.max().date()),
        "outliers_iqr": {c: iqr_outlier_count(df[c]) for c in utils.NUMERIC_COLUMNS
                         if c in df.columns},
    }


# --------------------------------------------------------------------------- #
# Feature engineering (shared by training AND the prediction page)
# --------------------------------------------------------------------------- #
def add_date_features(df: pd.DataFrame, date_col: str = "Order_Date") -> pd.DataFrame:
    """Derive Year, Month, Month_Name, Quarter, Day, Day_of_Week (+ Day_Name, Year_Month)."""
    out = df.copy()
    d = pd.to_datetime(out[date_col], errors="coerce")
    out["Year"] = d.dt.year
    out["Month"] = d.dt.month
    out["Month_Name"] = d.dt.month.map(lambda m: utils.MONTH_NAMES[int(m) - 1] if pd.notna(m) else np.nan)
    out["Quarter"] = d.dt.quarter
    out["Day"] = d.dt.day
    out["Day_of_Week"] = d.dt.dayofweek          # 0 = Monday
    out["Day_Name"] = d.dt.dayofweek.map(lambda x: utils.DAY_NAMES[int(x)] if pd.notna(x) else np.nan)
    out["Year_Month"] = d.dt.strftime("%Y-%m")
    return out


def prepare_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return ONLY the model input columns, with missing categoricals as NaN (object dtype).

    Profit, Profit_Margin, Order_ID and Customer_ID are never part of the output.
    """
    X = df[utils.FEATURES].copy()
    for c in utils.CATEGORICAL_FEATURES:
        X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
    for c in utils.NUMERIC_FEATURES:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def clean_data(raw: pd.DataFrame):
    """Clean the raw dataset. Returns (clean_df, step_log).

    Every step records how many records were affected, so nothing is silently changed.
    Outliers are only flagged (not removed): high-value orders are legitimate sales.
    """
    df = raw.copy()
    steps: list[dict] = []

    def log(step: str, affected: int, action: str):
        steps.append({"Step": step, "Records affected": int(affected), "Action": action})

    # 1. Text columns: trim whitespace, collapse repeated spaces, blank -> missing
    modified = 0
    for c in utils.TEXT_COLUMNS:
        s = df[c].astype("object")
        cleaned = s.map(lambda v: " ".join(str(v).split()) if pd.notna(v) else v)
        cleaned = cleaned.map(lambda v: np.nan if isinstance(v, str) and v == "" else v)
        modified += int(((s.notna()) & (s != cleaned) & ~(cleaned.isna())).sum())
        df[c] = cleaned
    log("Trim and normalise text", modified, "Removed stray/duplicate spaces in text columns")

    # 2. Data types: dates and numerics
    parsed = pd.to_datetime(df["Order_Date"], errors="coerce")
    invalid_dates = int(parsed.isna().sum() - df["Order_Date"].isna().sum())
    df["Order_Date"] = parsed
    coerced = 0
    for c in utils.NUMERIC_COLUMNS:
        before_na = df[c].isna().sum()
        df[c] = pd.to_numeric(df[c], errors="coerce")
        coerced += int(df[c].isna().sum() - before_na)
    log("Convert data types", invalid_dates + coerced,
        "Order_Date parsed to datetime; numeric columns coerced (unparseable -> missing)")

    # 3. Rows that cannot be used at all: no valid date (no date features) or no target
    unusable = df["Order_Date"].isna() | df["Profit"].isna()
    log("Drop rows without valid Order_Date or Profit", unusable.sum(),
        "Rows removed (date features / target cannot be derived)")
    df = df.loc[~unusable].copy()

    # 4. Missing categoricals -> "Unknown"; missing numeric features -> median
    cat_missing = int(df[utils.CATEGORICAL_COLUMNS + ["Order_ID", "Customer_ID"]].isna().sum().sum())
    df[utils.CATEGORICAL_COLUMNS] = df[utils.CATEGORICAL_COLUMNS].fillna("Unknown")
    df[["Order_ID", "Customer_ID"]] = df[["Order_ID", "Customer_ID"]].fillna("Unknown")
    log("Fill missing categorical values", cat_missing, "Filled with 'Unknown'")

    num_missing = int(df[["Quantity", "Sales_Amount", "Discount"]].isna().sum().sum())
    for c in ["Quantity", "Sales_Amount", "Discount"]:
        df[c] = df[c].fillna(df[c].median())
    log("Fill missing numeric values", num_missing, "Filled with column median")

    # 5. Invalid values (impossible business values)
    invalid = (
        (df["Quantity"] <= 0) | (df["Sales_Amount"] <= 0)
        | (df["Discount"] < 0) | (df["Discount"] > 1)
    )
    log("Remove invalid values", invalid.sum(),
        "Quantity <= 0, Sales_Amount <= 0 or Discount outside 0-1 removed")
    df = df.loc[~invalid].copy()

    # 6. Duplicates
    dup_rows = int(df.duplicated().sum())
    df = df.drop_duplicates()
    dup_ids = int(df["Order_ID"].duplicated().sum())
    df = df.drop_duplicates(subset="Order_ID", keep="first")
    log("Remove duplicate records", dup_rows + dup_ids,
        f"{dup_rows} fully duplicated rows + {dup_ids} repeated Order_ID removed")

    # 7. Outliers: flagged only
    out_total = sum(iqr_outlier_count(df[c]) for c in ["Sales_Amount", "Profit"])
    log("Flag numeric outliers (IQR rule)", out_total,
        "Sales_Amount/Profit outliers retained - they are genuine high-value orders")

    # 8. Whole-number columns
    df["Quantity"] = df["Quantity"].round().astype(int)

    # 9. Wrangling: derived columns
    df = add_date_features(df)
    df["Profit_Margin"] = df["Profit"] / df["Sales_Amount"]   # analysis only, NEVER a model feature
    df = df.sort_values("Order_Date").reset_index(drop=True)
    log("Create derived features", len(df),
        "Year, Month, Month_Name, Quarter, Day, Day_of_Week, Day_Name, Year_Month, Profit_Margin")
    return df, steps


def build_quality_report(raw: pd.DataFrame, clean: pd.DataFrame, steps: list[dict]) -> dict:
    before = quality_snapshot(raw)
    after = quality_snapshot(clean[utils.RAW_COLUMNS])
    return {
        "before": before,
        "after": after,
        "steps": steps,
        "records_removed": int(before["rows"] - after["rows"]),
        "final_shape": [int(clean.shape[0]), int(clean.shape[1])],
    }


def run_pipeline(save: bool = True) -> pd.DataFrame:
    """Load -> clean -> save cleaned CSV and quality report."""
    raw = load_raw_data()
    clean, steps = clean_data(raw)
    report = build_quality_report(raw, clean, steps)
    if save:
        utils.DATA_DIR.mkdir(parents=True, exist_ok=True)
        clean.to_csv(utils.CLEAN_DATA_PATH, index=False)
        utils.write_json(utils.QUALITY_REPORT_PATH, report)
    return clean


def load_clean_data() -> pd.DataFrame:
    """Load the cleaned CSV; create it from the Excel file if it does not exist yet."""
    if not utils.CLEAN_DATA_PATH.exists():
        return run_pipeline(save=True)
    df = pd.read_csv(utils.CLEAN_DATA_PATH, parse_dates=["Order_Date"])
    if df.empty or any(c not in df.columns for c in utils.FEATURES + [utils.TARGET]):
        return run_pipeline(save=True)
    return df


if __name__ == "__main__":
    data = run_pipeline(save=True)
    rep = utils.read_json(utils.QUALITY_REPORT_PATH)
    print(f"Raw rows: {rep['before']['rows']}  ->  Clean rows: {rep['after']['rows']}  "
          f"(removed {rep['records_removed']})")
    for s in rep["steps"]:
        print(f"- {s['Step']}: {s['Records affected']} | {s['Action']}")
    print(f"Saved: {utils.CLEAN_DATA_PATH}")

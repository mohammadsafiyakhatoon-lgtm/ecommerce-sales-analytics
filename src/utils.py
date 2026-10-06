"""Shared constants, paths and helper functions for the E-Commerce DAV project."""
from __future__ import annotations

import hmac
import json
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

RAW_DATA_PATH = DATA_DIR / "ecommerce_sales_dataset.xlsx"
CLEAN_DATA_PATH = DATA_DIR / "cleaned_ecommerce_data.csv"

LR_MODEL_PATH = MODELS_DIR / "linear_regression_model.joblib"
RF_MODEL_PATH = MODELS_DIR / "random_forest_model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessing_pipeline.joblib"

METRICS_PATH = OUTPUTS_DIR / "model_metrics.csv"
IMPORTANCE_PATH = OUTPUTS_DIR / "feature_importance.csv"
PREDICTIONS_PATH = OUTPUTS_DIR / "test_predictions.csv"
TRAIN_INFO_PATH = OUTPUTS_DIR / "training_info.json"
QUALITY_REPORT_PATH = OUTPUTS_DIR / "data_quality_report.json"

# --------------------------------------------------------------------------- #
# Dataset schema (exactly the columns in the uploaded Excel file)
# --------------------------------------------------------------------------- #
RAW_COLUMNS = [
    "Order_ID", "Order_Date", "Customer_ID", "Product_Category", "Product_Name",
    "State", "City", "Payment_Mode", "Quantity", "Sales_Amount", "Discount", "Profit",
]
TEXT_COLUMNS = ["Order_ID", "Customer_ID", "Product_Category", "Product_Name",
                "State", "City", "Payment_Mode"]
CATEGORICAL_COLUMNS = ["Product_Category", "Product_Name", "State", "City", "Payment_Mode"]
NUMERIC_COLUMNS = ["Quantity", "Sales_Amount", "Discount", "Profit"]

# --------------------------------------------------------------------------- #
# Machine-learning schema
#   Target  : Profit
#   Excluded: Order_ID, Customer_ID (identifiers), Profit, Profit_Margin (target-derived)
# --------------------------------------------------------------------------- #
TARGET = "Profit"
NUMERIC_FEATURES = ["Quantity", "Sales_Amount", "Discount",
                    "Year", "Month", "Quarter", "Day", "Day_of_Week"]
CATEGORICAL_FEATURES = ["Product_Category", "Product_Name", "State", "City", "Payment_Mode"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
RANDOM_STATE = 42
TEST_SIZE = 0.2

MODEL_NAMES = ["Linear Regression", "Random Forest"]
PRED_COLUMNS = {
    "Linear Regression": "Pred_Linear_Regression",
    "Random Forest": "Pred_Random_Forest",
}

MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# --------------------------------------------------------------------------- #
# Demo authentication (academic project only - not real security)
# --------------------------------------------------------------------------- #
DEMO_USERNAME = "admin"
DEMO_PASSWORD = "admin123"


def check_credentials(username: str, password: str) -> bool:
    """Constant-time comparison against the demo credentials."""
    user_ok = hmac.compare_digest((username or "").strip().encode(), DEMO_USERNAME.encode())
    pass_ok = hmac.compare_digest((password or "").encode(), DEMO_PASSWORD.encode())
    return bool(user_ok and pass_ok)


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def format_inr(value: float, decimals: int = 0) -> str:
    """Format a number with the Indian digit grouping, e.g. 4,93,84,840."""
    if value is None or pd.isna(value):
        return "-"
    negative = value < 0
    text = f"{abs(float(value)):.{decimals}f}"
    whole, _, frac = text.partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        whole = ",".join(groups + [tail])
    out = "₹" + whole + (("." + frac) if frac else "")
    return "-" + out if negative else out


def format_compact(value: float, prefix: str = "₹") -> str:
    """Compact number: 49384840 -> ₹49.38M, 12500 -> ₹12.5K."""
    if value is None or pd.isna(value):
        return "-"
    sign = "-" if value < 0 else ""
    v = abs(float(value))
    if v >= 1e9:
        s = f"{v / 1e9:.2f}B"
    elif v >= 1e6:
        s = f"{v / 1e6:.2f}M"
    elif v >= 1e3:
        s = f"{v / 1e3:.1f}K"
    else:
        s = f"{v:.0f}"
    return f"{sign}{prefix}{s}"


def format_int(value: float) -> str:
    return "-" if value is None or pd.isna(value) else f"{int(round(value)):,}"


# --------------------------------------------------------------------------- #
# File helpers
# --------------------------------------------------------------------------- #
def read_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def write_json(path: Path, payload) -> None:
    def _default(obj):
        if hasattr(obj, "item"):
            return obj.item()
        return str(obj)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=_default)


def read_csv_if_exists(path: Path):
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return None

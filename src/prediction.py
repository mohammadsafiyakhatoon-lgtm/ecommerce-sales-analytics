"""Load the saved models and predict Profit for a new order."""
from __future__ import annotations

import warnings
from datetime import date, datetime

import joblib
import numpy as np
import pandas as pd
import sklearn

import data_preprocessing as dp
import utils

MODEL_FILES = {
    "Linear Regression": utils.LR_MODEL_PATH,
    "Random Forest": utils.RF_MODEL_PATH,
}


class ModelLoadError(RuntimeError):
    """Raised when a saved model is missing or cannot be loaded."""


def missing_model_files() -> list[str]:
    return [p.name for p in MODEL_FILES.values() if not p.exists()]


def load_models() -> dict:
    """Load both saved pipelines. Raises ModelLoadError with a readable message."""
    missing = missing_model_files()
    if missing:
        raise ModelLoadError(
            "Saved model file(s) not found: " + ", ".join(missing)
            + ". Run `python src/train_models.py` to create them.")
    models = {}
    for name, path in MODEL_FILES.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                models[name] = joblib.load(path)
        except Exception as exc:  # corrupt file, version mismatch, ...
            raise ModelLoadError(
                f"Could not load {path.name} ({type(exc).__name__}). It may have been created with a "
                "different scikit-learn version. Run `python src/train_models.py` to retrain.") from exc
    return models


def version_warning() -> str | None:
    info = utils.read_json(utils.TRAIN_INFO_PATH)
    if info and info.get("sklearn_version") != sklearn.__version__:
        return (f"Models were trained with scikit-learn {info.get('sklearn_version')} but "
                f"{sklearn.__version__} is installed. Retrain if predictions fail.")
    return None


def build_input_frame(order_date, category, product, state, city, payment_mode,
                      quantity, sales_amount, discount) -> pd.DataFrame:
    """One-row frame with exactly the model's input columns (date features derived here)."""
    row = pd.DataFrame([{
        "Order_Date": pd.Timestamp(order_date),
        "Product_Category": category, "Product_Name": product, "State": state,
        "City": city, "Payment_Mode": payment_mode,
        "Quantity": quantity, "Sales_Amount": sales_amount, "Discount": discount,
    }])
    row = dp.add_date_features(row)
    return dp.prepare_model_frame(row)


def validate_inputs(order_date, category, product, state, city, payment_mode,
                    quantity, sales_amount, discount, product_map: dict | None = None):
    """Return (errors, warnings). Errors block prediction; warnings flag extrapolation."""
    errors, notes = [], []
    if not isinstance(order_date, (date, datetime, pd.Timestamp)):
        errors.append("Please select a valid order date.")
    for label, val in [("Product category", category), ("Product name", product),
                       ("State", state), ("City", city), ("Payment mode", payment_mode)]:
        if val is None or (isinstance(val, str) and not val.strip()):
            errors.append(f"{label} is required.")
    for label, val in [("Quantity", quantity), ("Sales amount", sales_amount), ("Discount", discount)]:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            errors.append(f"{label} is required.")
    if errors:
        return errors, notes
    if quantity < 1:
        errors.append("Quantity must be at least 1.")
    if sales_amount <= 0:
        errors.append("Sales amount must be greater than 0.")
    if not 0 <= discount <= 1:
        errors.append("Discount must be between 0 and 1 (e.g. 0.15 for 15%).")
    if product_map and category in product_map and product not in product_map[category]:
        errors.append(f"'{product}' does not belong to the '{category}' category in the dataset.")
    return errors, notes


def range_notes(df: pd.DataFrame, order_date, quantity, sales_amount, discount) -> list[str]:
    """Warn when inputs fall outside the range the models were trained on."""
    notes = []
    for label, col, val in [("Quantity", "Quantity", quantity), ("Sales amount", "Sales_Amount", sales_amount),
                            ("Discount", "Discount", discount)]:
        lo, hi = df[col].min(), df[col].max()
        if val < lo or val > hi:
            notes.append(f"{label} is outside the range seen in the data ({lo:g} to {hi:g}); "
                         "the estimate is less reliable.")
    y = pd.Timestamp(order_date).year
    if y < df["Year"].min() or y > df["Year"].max():
        notes.append(f"Order year is outside the data period ({df['Year'].min()}-{df['Year'].max()}).")
    return notes


def predict_profit(model, frame: pd.DataFrame) -> float:
    """Run the saved pipeline (imputation, encoding, scaling and model are all inside it)."""
    value = float(model.predict(frame)[0])
    if not np.isfinite(value):
        raise ValueError("Model returned a non-finite value.")
    return value

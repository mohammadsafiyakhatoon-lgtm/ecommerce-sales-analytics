"""Train, evaluate and save the two regression models that predict Profit.

Usage:
    python src/train_models.py

Models : Linear Regression (with scaling) and Random Forest Regressor
Target : Profit
Split  : 80% train / 20% test, random_state=42
All preprocessing lives inside sklearn Pipelines, so imputers, scalers and the
one-hot encoder are fitted on the TRAINING data only (no train/test leakage).
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import data_preprocessing as dp
import utils


def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    """Numeric: median imputation (+ scaling for Linear Regression).
    Categorical: most-frequent imputation + OneHotEncoder(handle_unknown='ignore')."""
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    categorical_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
    return ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), utils.NUMERIC_FEATURES),
            ("cat", Pipeline(categorical_steps), utils.CATEGORICAL_FEATURES),
        ]
    )


def build_models() -> dict[str, Pipeline]:
    return {
        "Linear Regression": Pipeline([
            ("preprocessor", build_preprocessor(scale_numeric=True)),
            ("model", LinearRegression()),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", build_preprocessor(scale_numeric=False)),
            ("model", RandomForestRegressor(
                n_estimators=200, max_depth=16, min_samples_leaf=5,
                random_state=utils.RANDOM_STATE, n_jobs=-1)),
        ]),
    }


def evaluate(model: Pipeline, X_train, y_train, X_test, y_test) -> tuple[dict, np.ndarray]:
    pred = model.predict(X_test)
    cv = cross_val_score(
        model, X_train, y_train, scoring="r2",
        cv=KFold(n_splits=5, shuffle=True, random_state=utils.RANDOM_STATE))
    metrics = {
        "MAE": float(mean_absolute_error(y_test, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
        "R2": float(r2_score(y_test, pred)),
        "Train_R2": float(r2_score(y_train, model.predict(X_train))),
        "CV_R2_Mean": float(cv.mean()),
        "CV_R2_Std": float(cv.std()),
    }
    return metrics, pred


def aggregate_importance(names: list[str], values: np.ndarray) -> pd.DataFrame:
    """Map encoded feature names back to the original input column."""
    originals = sorted(utils.FEATURES, key=len, reverse=True)
    rows = []
    for name, val in zip(names, values):
        clean = name.split("__", 1)[1] if "__" in name else name
        origin = next((o for o in originals if clean == o or clean.startswith(o + "_")), clean)
        rows.append({"Feature": clean, "Original_Feature": origin, "Importance": float(val)})
    return pd.DataFrame(rows).sort_values("Importance", ascending=False).reset_index(drop=True)


def main() -> dict:
    print("Loading and cleaning data ...")
    df = dp.run_pipeline(save=True)

    X = dp.prepare_model_frame(df)
    y = df[utils.TARGET].astype(float)

    # Leakage guards
    forbidden = {"Profit", "Profit_Margin", "Order_ID", "Customer_ID"}
    assert not forbidden.intersection(X.columns), "Target-derived or ID column found in features"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=utils.TEST_SIZE, random_state=utils.RANDOM_STATE)
    print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows | "
          f"Input columns: {X.shape[1]}")

    utils.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    utils.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    models = build_models()
    metric_rows, predictions = [], {}
    for name, pipe in models.items():
        print(f"Training {name} ...")
        pipe.fit(X_train, y_train)               # preprocessing fitted on training data only
        metrics, pred = evaluate(pipe, X_train, y_train, X_test, y_test)
        predictions[name] = pred
        metric_rows.append({"Model": name, **metrics})
        print(f"  MAE={metrics['MAE']:.2f}  RMSE={metrics['RMSE']:.2f}  R2={metrics['R2']:.4f}  "
              f"(train R2={metrics['Train_R2']:.4f}, 5-fold CV R2={metrics['CV_R2_Mean']:.4f})")

    metrics_df = pd.DataFrame(metric_rows)
    best = metrics_df.sort_values(["RMSE", "MAE"]).iloc[0]["Model"]
    metrics_df["Best"] = metrics_df["Model"] == best
    metrics_df.to_csv(utils.METRICS_PATH, index=False)

    # Test-set predictions (used by the app's actual-vs-predicted and residual charts)
    pred_df = X_test.copy()
    pred_df.insert(0, "Actual_Profit", y_test.values)
    for name, col in utils.PRED_COLUMNS.items():
        pred_df[col] = predictions[name]
    pred_df.to_csv(utils.PREDICTIONS_PATH, index=False)

    # Random Forest feature importance
    rf = models["Random Forest"]
    names = list(rf.named_steps["preprocessor"].get_feature_names_out())
    importance = aggregate_importance(names, rf.named_steps["model"].feature_importances_)
    importance.to_csv(utils.IMPORTANCE_PATH, index=False)
    n_encoded = len(names)

    # Save models. preprocessing_pipeline.joblib = the fitted Random Forest preprocessor
    joblib.dump(models["Linear Regression"], utils.LR_MODEL_PATH, compress=3)
    joblib.dump(rf, utils.RF_MODEL_PATH, compress=3)
    joblib.dump(rf.named_steps["preprocessor"], utils.PREPROCESSOR_PATH, compress=3)

    info = {
        "target": utils.TARGET,
        "n_total": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_input_features": int(X.shape[1]),
        "n_encoded_features": int(n_encoded),
        "numeric_features": utils.NUMERIC_FEATURES,
        "categorical_features": utils.CATEGORICAL_FEATURES,
        "best_model": best,
        "random_state": utils.RANDOM_STATE,
        "test_size": utils.TEST_SIZE,
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    utils.write_json(utils.TRAIN_INFO_PATH, info)

    print(f"\nBest model: {best}")
    print(metrics_df.round(4).to_string(index=False))
    print(f"\nSaved models to {utils.MODELS_DIR}")
    return info


if __name__ == "__main__":
    main()

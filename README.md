# E-Commerce Sales Analytics & Profit Prediction

*Smart E-Commerce Sales Analytics and Profit Prediction System* - a Data Analytics and Data Visualization (DAV) mini project.

A Streamlit web application that cleans an e-commerce order dataset, explores it with interactive charts, compares two regression models and predicts the **Profit** of a new order.

## Objectives
- Clean and wrangle the raw order data and report every change.
- Explore sales, profit, quantity and discount patterns with meaningful visualizations.
- Rebuild the key views of the Power BI dashboard as an interactive, filterable web dashboard.
- Train and compare **Linear Regression** and **Random Forest Regressor** to predict profit.
- Offer a live prediction page that uses the saved, trained models.

## Dataset
`data/ecommerce_sales_dataset.xlsx` - 15,000 orders, 12 columns:

| Column | Meaning |
|---|---|
| Order_ID, Customer_ID | Identifiers (kept for analysis, **not** used as ML features) |
| Order_Date | Date of the order (2023-01-01 to 2024-12-31) |
| Product_Category, Product_Name | 5 categories, 25 products |
| State, City | 7 states, 7 cities (every city appears in every state, so they are independent fields) |
| Payment_Mode | 5 modes |
| Quantity, Sales_Amount, Discount | Order values (Discount is a fraction, 0.15 = 15%) |
| Profit | **Target variable** |

## Data preprocessing and wrangling (`src/data_preprocessing.py`)
Inspection of shape, types, missing values, duplicates, unique values, date range, outliers and invalid values, followed by:
text trimming, date/number type conversion, removal of rows without a valid date or profit, `Unknown` for missing categories, median for missing numbers, removal of impossible values (quantity/sales <= 0, discount outside 0-1), removal of duplicate rows and repeated Order_IDs, and IQR outlier **flagging** (outliers are kept - they are real high-value orders).

Derived columns: `Year, Month, Month_Name, Quarter, Day, Day_of_Week, Day_Name, Year_Month, Profit_Margin`.
`Profit_Margin` is used for analysis only and is excluded from the model because it is derived from the target.

Every step logs how many records it affected; the before/after table is shown in the app under *Data Exploration - Data wrangling* and saved to `outputs/data_quality_report.json`.
On this dataset the raw file had no missing values, duplicates or invalid values, so 0 records were removed (688 Sales_Amount/Profit outliers were flagged and kept).

## Exploratory data analysis and visualizations
Distributions of sales, profit, quantity and discount; sales and profit by category, state and city; top 10 products by sales and profit; payment mode share; quantity by category; monthly sales and profit trends; sales vs profit, discount vs profit, quantity vs sales; correlation heatmap. The *Key data insights* text is generated from the data at run time. Static Matplotlib/Seaborn figures are saved to `outputs/figures/`.

## Machine learning
- **Target:** `Profit` (regression). **Split:** 80% train / 20% test, `random_state=42`.
- **Features (13):** Quantity, Sales_Amount, Discount, Year, Month, Quarter, Day, Day_of_Week, Product_Category, Product_Name, State, City, Payment_Mode.
- **Preprocessing:** `ColumnTransformer` inside each `Pipeline` - median imputation (+ `StandardScaler` for Linear Regression) for numbers; most-frequent imputation + `OneHotEncoder(handle_unknown="ignore")` for categories. Fitted on training data only, so there is no leakage.
- **Models:** `LinearRegression`, and `RandomForestRegressor(n_estimators=200, max_depth=16, min_samples_leaf=5, random_state=42)`.
- **Metrics:** MAE, RMSE, R² on the test set (plus 5-fold CV R² on the training set). The best model is the one with the lowest test RMSE.

Results from the last training run (re-created every time you run `train_models.py`; the app always reads the saved files, never these numbers):

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 164.70 | 233.43 | 0.6912 |
| Random Forest | 166.02 | 235.51 | 0.6857 |

The two models perform almost identically; Random Forest fits the training data more closely (train R² 0.84) but generalises no better (CV R² 0.67), so Linear Regression is selected. Sales amount is by far the most important input. Profit margin varies between 5% and 25% per order and is essentially unrelated to the other columns, which limits how much of the profit variation any model can explain.

## Web application
Pages: Login, Home, Dashboard, Data Exploration, Machine Learning, Profit Prediction, About Project. Minimal light theme with a single accent colour, KPI cards, filter panel (date, category, product, state, city, payment mode), Plotly charts, and a prediction form that loads the saved model files.

## Project structure
```
ecommerce_dav_project/
├── app.py                      Streamlit application
├── requirements.txt
├── README.md
├── data/
│   ├── ecommerce_sales_dataset.xlsx
│   └── cleaned_ecommerce_data.csv
├── models/
│   ├── linear_regression_model.joblib
│   ├── random_forest_model.joblib
│   └── preprocessing_pipeline.joblib   (fitted preprocessor, for inspection)
├── notebooks/ecommerce_analysis.ipynb
├── src/
│   ├── data_preprocessing.py   cleaning, wrangling, date features
│   ├── eda.py                  statistics, dynamic insights, static figures
│   ├── charts.py               Plotly chart builders
│   ├── train_models.py         training, evaluation, saving
│   ├── prediction.py           model loading, validation, inference
│   ├── ui.py                   CSS theme and UI components
│   └── utils.py                paths, schema, formatting
└── outputs/
    ├── figures/                static PNG charts
    ├── model_metrics.csv
    ├── feature_importance.csv
    ├── test_predictions.csv
    ├── training_info.json
    └── data_quality_report.json
```

## Installation
Python 3.10+ recommended.
```bash
python -m venv venv
venv\Scripts\activate          # Windows   (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

## Train the models
```bash
python src/train_models.py
```
This cleans the data, trains both models, evaluates them and writes the models to `models/` and the metrics to `outputs/`. Run it once on your machine (it takes about a minute) so the saved models match your scikit-learn version.

## Run the application
```bash
streamlit run app.py
```
Open http://localhost:8501.

## Demo login
Username: `admin` - Password: `admin123` (local demo authentication only, not secure).

## Limitations
- One fixed period of data; no customer history or product cost information.
- Profit margin appears largely random relative to the available inputs, capping model accuracy.
- Saved models depend on the scikit-learn version; retrain if the app reports a load error.
- Demo login is not real authentication.

## Future enhancements
Hyperparameter tuning, Gradient Boosting, time-based validation and sales forecasting, customer segmentation, cloud deployment with real authentication.

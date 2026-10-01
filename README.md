# Indian Pre-Owned Car Price Prediction

End-to-end ML pipeline (Component 1A) + Streamlit deployment (Component 1B)
for the Unlox Capstone: **Indian Pre-Owned Car Price Prediction**, built
on the **official capstone dataset**.

## Objective

Predict the `Price` of a pre-owned car in India from vehicle
characteristics, and deploy the trained model as an interactive
Streamlit web application.

## Official Datasets Used

| File | Role | Shape | Notes |
|---|---|---|---|
| `Cap_Training_Data_2025.csv` | Training | 53,515 rows × 17 cols | Includes `Price` target |
| `Cap_Test_Data_2025.csv` | Test | 5,615 rows × 16 cols | No `Price` — predictions required |
| `Cap_Sample_Submission_2025.csv` | Submission template | 5,615 rows × 2 cols | `ID, Price`; `ID` order verified to exactly match the test set |

**Features:** `Maker`, `model`, `Location`, `Distance`, `Owner Type`,
`manufacture_year`, `Age of car`, `engine_displacement`, `engine_power`,
`body_type`, `Vroom Audit Rating`, `transmission`, `door_count`,
`seat_count`, `fuel_type`. **Target:** `Price` (training set only).

## Data Quality Issues Found & How They Were Handled

All findings below come from actually inspecting the official files
(documented in `project_notebook.ipynb`, Sections 3–4):

- **Column name whitespace:** the raw header has `"Distance "` (trailing
  space) — stripped from all column names on load.
- **114 duplicate rows** (identical on every feature + `Price`, only
  `ID` differs) — dropped, keeping the first occurrence.
- **17 rows with `Price < ₹1,000`** (e.g. `Price = 3.00`) — physically
  impossible; dropped from training since the target can't be imputed.
- **`engine_displacement == 32000`** (29 train / 3 test rows) — an
  exact, repeated sentinel value (e.g. implies a 32-litre engine on a
  Skoda Octavia); set to missing, then median-imputed.
- **`engine_power > 500 bhp`** (6 train / 1 test rows) — e.g. 896 bhp
  from a 74cc engine is impossible; set to missing, then median-imputed.
- **`Distance > 500,000 km`** (141 train / 21 test rows, up to 9.9
  million km) — implausible for these passenger-car makers; set to
  missing, then median-imputed.
- **`body_type` is 92.3% missing**, and only holds two real categories
  (`compact`, `van`) where present. Rather than most-frequent
  imputation (which would force an assumed category onto ~92% of
  rows), missing values are imputed with an explicit `"Unknown"`
  category.
- **`manufacture_year`** is dropped in feature engineering — it is
  perfectly redundant with `Age of car` (0 mismatches found), and
  `Age of car` has zero missing values vs. 2 for `manufacture_year`.
- Legitimate extreme-but-real values (a ₹2.2 crore BMW X3, a 5.2-litre
  Audi engine, a 449 bhp performance variant) were **kept** — only
  physically impossible values were corrected.

## Actual Results (from the executed notebook — not fabricated)

3-fold CV + `RandomizedSearchCV` hyperparameter tuning (Random Forest
tuned on a seeded 8,000-row subsample for runtime practicality, then
refit on the full training set; XGBoost/LightGBM tuned directly on the
full training set), evaluated on a held-out 20% validation split:

| Model | RMSE | MAE | R² |
|---|---|---|---|
| **LightGBM (tuned)** ✅ selected | 309,464.41 | 113,507.52 | 0.8772 |
| XGBoost (tuned) | 309,929.20 | 112,664.29 | 0.8768 |
| Random Forest (tuned) | 323,118.89 | 128,479.22 | 0.8661 |
| Linear Regression (baseline) | 459,980.84 | 243,978.55 | 0.7286 |

**Selected model:** LightGBM (tuned) — lowest validation RMSE, refit on
all 53,384 cleaned official training rows before deployment.

Residual analysis (`outputs/residual_analysis.png`): mean residual
≈ ₹4,970 (near-zero, no systematic bias), residual std ≈ ₹309,425.

## Project Structure

```text
project/
│
├── project_notebook.ipynb          # Component 1A — full executed notebook
├── app.py                          # Component 1B — Streamlit application
├── requirements.txt
├── README.md
│
├── data/
│   ├── Cap_Training_Data_2025.csv          # official training set (unmodified)
│   ├── Cap_Test_Data_2025.csv              # official test set (unmodified)
│   ├── Cap_Sample_Submission_2025.csv      # official submission template (unmodified)
│   └── Sample_Submission_Predictions.csv   # ID, Price — actual model predictions
│
├── model/
│   ├── trained_model.joblib        # fitted sklearn Pipeline (preprocessing + LightGBM)
│   └── model_metadata.joblib       # feature-column metadata + validation metrics
│
└── outputs/                        # EDA / evaluation charts from the executed notebook
    ├── missing_values.png
    ├── distributions.png
    ├── outliers.png
    ├── categorical_counts.png
    ├── correlation_matrix.png
    ├── model_comparison.png
    ├── actual_vs_predicted.png
    └── residual_analysis.png
```

The three official dataset files are kept in `data/` exactly as
uploaded, unmodified — no rows or columns were altered in place; all
cleaning happens in code inside the notebook/app.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Notebook

```bash
jupyter notebook project_notebook.ipynb
```

Run all cells top to bottom. This regenerates `outputs/*.png`,
`model/trained_model.joblib`, `model/model_metadata.joblib`, and
`data/Sample_Submission_Predictions.csv`.

## Running the Streamlit App

```bash
streamlit run app.py
```

Three tabs:
- **Price Prediction** — sidebar inputs → real-time inference + ±5% valuation band
- **Batch Processing** — upload a `Cap_Test_Data_2025.csv`-shaped CSV → download `ID, Price` predictions
- **Visual Insights** — Age-vs-Price, Distance-vs-Price, and Maker-vs-Price charts from the actual (cleaned) training data

## Reproducing This Project

1. Place the three official CSV files in `data/` under their original
   filenames (already done here).
2. Run `project_notebook.ipynb` top to bottom — this performs all
   cleaning, EDA, model training/tuning, evaluation, and saves the
   final model + submission predictions.
3. Run `streamlit run app.py` to launch the deployed application.

## Notes

- All preprocessing (imputation, one-hot encoding) lives inside a
  single `sklearn.Pipeline` / `ColumnTransformer`, fit only on training
  data/folds — the same fitted pipeline object is reused for
  validation, official test-set prediction, and the Streamlit app, so
  there is no train/inference preprocessing mismatch.
- No technologies outside the specification (no Docker, databases,
  auth, cloud deployment, extra APIs) were introduced.
- This is a rebuild against the **official dataset only** — no
  synthetic or generated data is used anywhere in this version.
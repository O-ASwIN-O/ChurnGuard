# ChurnGuard — Customer Churn Prediction

## Project overview
ChurnGuard analyses FreshBasket Retail loyalty customers and generates risk rankings to support retention planning.

**Target:** `CHURNED = 1` means no purchase transactions during April–June 2024 despite earlier purchase history. Predictive activity inputs are intended to end before 1 April 2024.

## Repository structure
- `notebook/01_eda.ipynb` — data loading, quality checks and feature preparation
- `notebook/02_eda.ipynb` — churn EDA and customer-level exploratory analysis
- `notebook/03_model_training.ipynb` — exploratory random split, Logistic Regression and early ablation
- `src/run_analysis.py` — reproducible end-to-end modeling, validation and deliverable-generation script
- `requirements.txt` — Python dependencies
- `data/processed/model_eligible.csv` — initial customer-level dataset
- `data/processed/final_customer_modeling_dataset.csv` — final feature-complete modeling table including recency and recent-vs-prior trends
- `data/processed/cohort_model_comparison.csv` — baseline, Logistic Regression and Random Forest validation metrics
- `data/processed/ablation_results.csv` — profile-only, behaviour-only and all-feature comparisons
- `data/processed/feature_importance.csv` — Random Forest global feature importance
- `data/processed/segment_churn_rates.csv` — churn rates by tier, city and signup quarter
- `data/processed/customer_churn_predictions.csv` — customer-level scores, predicted label and local sensitivity notes
- `data/processed/validation_cohort_predictions.csv` — held-out cohort scores with observed labels
- `data/processed/retention_scenario.csv` — illustrative risk-reduction sensitivity assumptions
- `reports/FINAL_REPORT.md` — generated project report
- `presentation/ChurnGuard_Final_Presentation.pptx` — generated presentation
- `figures/` — analysis charts used in the report and deck

## Run the complete analysis
From the repository root:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python src/run_analysis.py
```

The script regenerates the cohort metrics, customer-level modeling table, predictions, driver analysis, ablation, segment tables, retention scenario, charts, report and PowerPoint. A GitHub Actions workflow also runs the script and uploads the generated files as the `churnguard-submission-artifacts` artifact.

## Verified validation results
The eligible dataset contains **2,225 customers**, with **662 labelled churned** (29.75%). A signup-cohort holdout uses 1,780 earlier-cohort customers for training and 445 later-signup-cohort customers for validation (cutoff 17 July 2023).

| Model | Accuracy | Churn precision | Churn recall | F1 | PR-AUC (average precision) |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.897 | 0.510 | 0.553 | 0.531 | 0.600 |
| Random Forest | 0.933 | 0.840 | 0.447 | 0.583 | 0.699 |

Random Forest has the higher PR-AUC and precision at threshold 0.50, but lower recall than Logistic Regression. Metric choice should reflect the relative cost of missed churners and unnecessary offers.

## Important limitations and assumptions
- All labels refer to the same April–June 2024 outcome window. This is a signup-cohort split, **not** a true future-outcome temporal test.
- The validation cohort has 10.56% observed churn, versus 34.55% in training. Selected-model mean predicted risk is 20.23%, so scores are **not calibrated probabilities** and should be treated primarily as risk rankings.
- Four recent-activity features are missing together for 449 customers; 442 of those have the churn label. The script encodes absent recent records as zero and adds a missing-activity flag. Confirm whether missing rows mean no activity or an upstream data issue.
- Retention scenarios apply assumed 5%, 10% and 15% relative risk reductions. They are sensitivity calculations, not measured causal effects or ROI.
- Driver importance and local sensitivity explain model behaviour; they do not prove causation.
- The all-customer prediction file is scored by a model refitted on all eligible records, so those scores are in-sample. Use the held-out prediction file for validation discussion.
- The source workbook includes a fictional/assessment dataset. Confirm assessment sharing terms before redistributing the raw workbook publicly.

## Deliverables checklist
The repository and generated artifact contain the EDA notebooks, feature-complete modelling table, cohort model metrics, prediction files, feature impact analysis, tier/city/signup-quarter comparisons, ablation analysis, retention scenario, report and 14-slide presentation. See `reports/FINAL_REPORT.md` for methodology and caveats.

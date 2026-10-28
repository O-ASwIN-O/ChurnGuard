# ChurnGuard — Customer Churn Prediction

ChurnGuard is a reproducible Python project that analyses FreshBasket Retail loyalty customers, compares churn classifiers, generates customer-level risk rankings, and prepares evidence-based retention recommendations.

## Business target

`CHURNED = 1` means a customer had zero purchase transactions during April–June 2024 despite having purchase history earlier in the observation window. Predictive activity features are built from information before 1 April 2024.

## Repository contents

### Source code
- `src/run_analysis.py` — end-to-end data preparation, feature engineering, cohort validation, model evaluation, driver analysis, segment analysis, retention scenario, charts and report generation.
- `src/score_customers.py` — score prepared customer CSV files with the saved notebook model and add a recommended action.
- `notebook/01_eda.ipynb` — data loading, quality checks and feature preparation.
- `notebook/02_eda.ipynb` — churn exploratory analysis.
- `notebook/03_model_training.ipynb` — exploratory training and ablation work.
- `requirements.txt` — Python dependencies.
- `.github/workflows/build-submission.yml` — runs the analysis and commits generated analysis outputs to this repository.

### Generated outputs

These files are produced by `src/run_analysis.py`. The GitHub Actions workflow publishes them to `main` after a successful run, so they are available as ordinary repository files, not only as temporary workflow artifacts.

**Customer-level datasets and predictions — `data/processed/`**
- `final_customer_modeling_dataset.csv` — final customer feature table, including recency and recent-versus-prior trends.
- `cohort_validation_design.json` — validation split details.
- `cohort_model_comparison.csv` — baseline, Logistic Regression and Random Forest metrics.
- `cohort_confusion_matrix.csv` — held-out confusion matrix.
- `validation_cohort_predictions.csv` — validation predictions and observed labels.
- `customer_churn_predictions.csv` — customer-level risk score, predicted label, recommended action and diagnostic driver notes. The action follows the pipeline's 0.50 classification threshold.
- `feature_importance.csv` — Random Forest global feature importance.
- `ablation_results.csv` — comparisons across feature groups.
- `segment_churn_rates.csv` — observed churn by membership tier, city and signup cohort.
- `retention_scenario.csv` — illustrative retention sensitivity assumptions.
- `analysis_summary.json` — machine-readable summary of the analysis.

**Charts — `figures/`**
- `churn_distribution.png`
- `model_comparison.png`
- `feature_importance.png`
- `churn_by_tier.png`
- `churn_by_city.png`
- `churn_by_signup_cohort.png`
- `feature_ablation.png` (created when the ablation output is available)

**Written results — `reports/`**
- `FINAL_REPORT.md` — methodology, results, interpretation, limitations and recommendations.
- `ANALYSIS_SUMMARY.json` — summary metrics in JSON format.

Presentation files are intentionally not committed to this repository. The repository focuses on code, datasets, charts, model outputs and the written analysis report.

## Run locally

From the repository root, using PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python src/run_analysis.py
```

The script regenerates the analysis tables, model predictions, feature impact and ablation results, segment tables, retention scenario, charts and report.

## Verified cohort-validation metrics

The eligible modelling data contains **2,225 customers**, including **662 labelled churned** (29.75%). The signup-cohort split uses 1,780 earlier-cohort customers for training and 445 later-signup-cohort customers for validation, with a cutoff of 17 July 2023.

| Model | Accuracy | Churn precision | Churn recall | F1 | PR-AUC (average precision) |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.897 | 0.510 | 0.553 | 0.531 | 0.600 |
| Random Forest | 0.933 | 0.840 | 0.447 | 0.583 | 0.699 |

At a threshold of 0.50, Random Forest has higher precision and PR-AUC but lower recall than Logistic Regression. Threshold choice should reflect the relative costs of missed churners and unnecessary outreach.

## Limitations and assumptions

- All labels use the same April–June 2024 outcome window. The signup-cohort split is **not** a true future-outcome temporal test.
- The validation cohort has 10.56% observed churn, versus 34.55% in training. The selected model's mean predicted risk is 20.23%; scores are **not calibrated probabilities** and should primarily be treated as risk rankings.
- Four recent-activity features are jointly missing for 449 customers; 442 of those have the churn label. The pipeline encodes absent recent records as zero and adds a missing-activity flag. This pattern needs investigation before operational use.
- Retention scenarios use assumed 5%, 10% and 15% relative risk reductions. They are sensitivity calculations, not measured treatment effects or ROI.
- Feature importance and local sensitivity notes explain model behaviour; they do not prove causation.
- `customer_churn_predictions.csv` is scored by a model refitted on all eligible records, so those scores are in-sample. Use `validation_cohort_predictions.csv` for evaluation discussion.
- Check the assessment's terms before redistributing the source workbook outside the project.

## GitHub Actions

On pushes to `main` that change source/project files, the workflow installs dependencies, runs the analysis, uploads a backup artifact and commits generated CSV/JSON outputs, PNG charts and report files to `main`. You can also start it manually from the repository's **Actions** tab. Generated output commits do not trigger an endless workflow loop.

## Notebook model and local customer scoring

The notebooks also contain a separate logistic-regression evaluation using a stratified random 60/20/20 train/validation/test split. Its provisional threshold is 0.399586, selected by maximum validation F1. This is distinct from the signup-cohort evaluation in `src/run_analysis.py`; the random split is not a future-period validation. The test-set metrics and capacity scenarios are saved in `data/processed/day4_holdout_metrics.csv` and `data/processed/day4_capacity_analysis.csv`.

Run all cells in `notebook/03_model_training.ipynb` to recreate the notebook model and its metadata. This writes `outputs/churnguard_model.joblib` and `outputs/churnguard_model_metadata.json`. The notebook also saves one example input/output and a full in-sample customer score file under `outputs/`; the in-sample scores are not evaluation results.

To score a new CSV, include the same model input columns as `data/processed/model_eligible.csv` (excluding the churn target and history flag), then run from the repository root:

```powershell
python src/score_customers.py --input path\to\new_customer_features.csv --output outputs\new_customer_scores.csv
```

The scorer adds a `RECOMMENDED_ACTION`: at or above its saved threshold it suggests **Send a personalized coupon**; below it suggests **Continue regular engagement**. This is a simple model-based recommendation, not an automatic coupon send or eligibility check. The notebook's risk scores are not independently validated for operational decisions; test on a later labeled period and run a controlled pilot before using them in a campaign.


## Missingness sensitivity audit

The standalone audit checks whether the saved Logistic Regression model relies heavily on absent January–March 2024 activity records. It compares three Logistic Regression variants using the same stratified 60/20/20 split, selects thresholds on validation data, and evaluates each variant once on the untouched test set.

- Script: `src/audit_missingness.py`
- Test metrics: `data/processed/missingness_audit_metrics.csv`
- Missingness profile: `data/processed/missingness_audit_missingness_profile.csv`
- Risk-ranking capacity: `data/processed/missingness_audit_capacity.csv`
- Comparison chart: `figures/missingness_audit_comparison.png`
- Report: `reports/MISSINGNESS_AUDIT.md`

Audit results: the full model records F1 0.875 and PR-AUC 0.955. Removing the explicit missingness flag reduces F1 to 0.669 and PR-AUC to 0.728; removing all recent-activity signals reduces them to 0.655 and 0.726. There are 449 customers with no Jan–Mar activity records, and 442 (98.44%) are labelled churned. This is a strong predictive-shortcut candidate, **not proof of leakage**; real prediction-time availability and future stability remain unverified. Reconstructing the documented churn rule from raw transactions also flags five label exceptions for review.

The GitHub Actions workflow runs this audit and commits the generated CSVs, chart and report to the repository. Keep this audit separate from the main signup-cohort model comparison and the notebook's random-split scorer evaluation.

# ChurnGuard — Customer Churn Prediction

ChurnGuard is a reproducible Python project that analyses FreshBasket Retail loyalty customers, compares churn classifiers, generates customer-level risk rankings, and prepares evidence-based retention recommendations.

## Business target

`CHURNED = 1` means a customer had zero purchase transactions during April–June 2024 despite having purchase history earlier in the observation window. Predictive activity features are built from information before 1 April 2024.

## Repository contents

### Source code
- `src/run_analysis.py` — end-to-end data preparation, feature engineering, cohort validation, model evaluation, driver analysis, segment analysis, retention scenario, charts and report generation.
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
- `customer_churn_predictions.csv` — customer-level risk score, predicted label and diagnostic driver notes.
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

# ChurnGuard — Customer Churn Prediction

## Project overview
ChurnGuard investigates customer churn in FreshBasket Retail's loyalty program. It uses customer profiles and monthly activity observed before the churn-label window to estimate which eligible customers are at risk of churn and to support targeted retention planning.

**Target definition:** `CHURNED = 1` means no purchase transactions during April–June 2024 despite earlier purchase history. The prediction features are intended to use data through March 2024 only.

## Repository structure
- `notebook/01_eda.ipynb` — data loading, quality checks and feature preparation
- `notebook/02_eda.ipynb` — exploratory analysis of the customer-level modelling dataset
- `notebook/03_model_training.ipynb` — baseline and Logistic Regression modelling, threshold comparison and an initial ablation
- `data/processed/model_eligible.csv` — customer-level modelling data
- `data/processed/eda_feature_summary.csv` — summary statistics used in EDA
- `data/processed/day3_model_comparison.csv` — initial Logistic Regression threshold results

## Current verified results
The eligible dataset contains 2,225 customers: 1,563 non-churned and 662 churned (29.75% churn rate).

An initial stratified random 80/20 split was used for exploration, giving 445 test examples. This is **not** a time-based or cohort-based validation and should not be presented as the final validation design.

On that exploratory split, Logistic Regression at threshold 0.50 achieved:
- Accuracy: 0.836
- Churn precision: 0.761
- Churn recall: 0.652
- Churn F1: 0.702

At threshold 0.40, the initial results were:
- Accuracy: 0.825
- Churn precision: 0.699
- Churn recall: 0.720
- Churn F1: 0.709

The threshold comparison was made on the same test set, so these metrics are exploratory and are not an unbiased final estimate after threshold selection. PR-AUC, Random Forest, cohort validation, final prediction outputs, driver attribution, segment comparisons and retention-scenario results must be generated and recorded before claiming those requirements are complete.

## Important data caveat
Four recent-activity features are missing together for 449 customers. In the current dataset, 442 of those customers are labelled churned and 7 are not. These customers have no recorded activity rows during January–March 2024 in the source table. Treating missing recent activity as zero is an explicit assumption—not proof that the customers had literally zero activity. This signal can dominate prediction and should be examined in sensitivity analysis.

The working dataset excludes insufficient-history customers and two additional customers without recorded pre-label activity. Document these population choices whenever reporting metrics.

## Reproducibility
Recommended environment: Python 3.11+ with pandas, numpy, matplotlib, seaborn, scikit-learn, openpyxl, jupyter and python-pptx (for presentation generation).

From the repository root, create/activate a virtual environment and install packages:
```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl jupyter python-pptx
```
Open the notebooks from the `notebook/` folder and run cells in order. Relative file paths assume that working directory. Check all displayed outputs after rerunning; notebook outputs committed in GitHub may be from an earlier run.

## Remaining deliverables to verify before submission
- Cohort-based validation, with limitations stated (the dataset contains one labelled outcome window)
- Logistic Regression vs a second ML classifier, including precision, recall, F1, accuracy and PR-AUC
- Feature importance/driver attribution
- Churn-rate comparisons by tier, city and signup cohort
- Demographics-only vs engagement/behaviour ablation
- Retention scenario with explicit assumptions and sensitivity caveats
- Customer prediction CSV with customer ID, churn probability, predicted label and key drivers
- Final report and 10–15 minute presentation

## Limitations
This is a retrospective analysis of a synthetic/assessment dataset and should not be treated as a deployed production churn system. A single label window limits robust temporal validation. Observational feature relationships do not prove causation, and an intervention scenario is not a measured causal treatment effect.

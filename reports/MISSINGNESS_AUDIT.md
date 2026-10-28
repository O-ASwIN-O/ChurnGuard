# Missingness audit — ChurnGuard

## Purpose
This controlled sensitivity audit checks whether the saved Logistic Regression workflow relies heavily on missing January–March activity data. Three variants use the same stratified random 60/20/20 split. Preprocessing is fitted on the training set; thresholds are selected on validation data by maximum F1; the untouched test set is evaluated once.

## Missing-activity pattern
- Eligible population: 2,225 customers.
- All four recent activity features missing: 449 customers.
- Churn-labelled in that group: 442 (98.44%).
- The source workbook has no January–March activity rows for this group. This predates the April–June target window in the constructed data, but the files cannot prove that the missingness indicator is available and reliable at real prediction time.

## Split and results
- Train / validation / test: 1,335 / 445 / 445.
- Test customers: 445; observed churn: 29.66%.
- Random seed: 42; stratified split; test set evaluated once per variant.

| Variant | Threshold | Accuracy | Precision | Recall | F1 | PR-AUC | ROC-AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A — Full model | 0.400 | 0.926 | 0.878 | 0.871 | 0.875 | 0.955 | 0.974 | 0.051 |
| B — Without explicit flag | 0.299 | 0.775 | 0.594 | 0.765 | 0.669 | 0.728 | 0.861 | 0.138 |
| C — Without recent signals | 0.351 | 0.775 | 0.601 | 0.720 | 0.655 | 0.726 | 0.847 | 0.140 |

## Effect of removing missingness features
- Without the explicit flag, F1 falls by 0.206 and PR-AUC falls by 0.227.
- Without recent activity fields and the flag, F1 falls by 0.219 and PR-AUC falls by 0.229.

**Conclusion:** missingness is a strong predictive-shortcut candidate in this dataset. This does not prove leakage, because the missingness appears in the constructed pre-label period. Production-time availability and future stability remain unverified; test on a later independently labelled period before operational use.

## Churn-label rule reconstruction
The documented churn rule was reconstructed from raw monthly transaction counts for eligible customers. It flagged **5 label exceptions** for review. This is diagnostic reconstruction, not a source-owner-approved correction, and it does not establish why any label differs.

## Recommended next steps
1. Confirm whether absent Jan–Mar records mean no shopping, missing telemetry, or an upstream data issue.
2. Verify that missingness is available at the real prediction timestamp.
3. Investigate the label-rule exceptions with the data owner.
4. Validate against a later labelled outcome period.
5. Run a controlled retention pilot; scenario reductions are assumptions, not measured coupon effects or ROI.

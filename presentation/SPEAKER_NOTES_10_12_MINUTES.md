# ChurnGuard — Presenter Notes (10–12 minutes)

Use this speaking guide for the 15-slide deck. The retention figures are scenario assumptions, not measured campaign effects.

## Slide 1 — ChurnGuard (20 seconds)
Good morning. Our project is ChurnGuard, a customer churn prediction and diagnosis solution for FreshBasket Retail's loyalty program. We focused not only on predicting churn but also on understanding signals, comparing segments, and translating results into practical retention recommendations.

## Slide 2 — Business problem (40 seconds)
FreshBasket currently sends offers broadly, which can waste budget on members who are unlikely to leave. We want to rank customers by churn risk so the retention team can prioritise outreach. The risk score is decision support, not an automatic decision or a guarantee that a customer will leave.

## Slide 3 — Target and data (45 seconds)
The workbook has customer profiles, monthly activity and churn labels. The target is one when a customer had no purchase transactions from April through June 2024 despite earlier purchase history. We have 2,225 eligible customers, 662 labelled churned, for an overall rate of about 29.75 percent. Predictive activity features use records before April 2024.

## Slide 4 — Data quality (55 seconds)
We checked exact duplicates, repeated customer-month records, inconsistent spend, negative spend, zero-spend anomalies, missing profile fields and inconsistent text categories. Two conflicting spend pairs had positive and negative versions with otherwise matching information; we retained the positive versions, and documented that as an inference, not a confirmed correction. One key finding is that 449 customers have no recorded activity rows in January through March; 442 of them are labelled churned. This may reflect real disengagement, the way the dataset was generated, or a data ingestion issue, so it is not treated as causal proof.

## Slide 5 — Leakage-safe features (45 seconds)
We built features from information available before April 2024: total and average transactions and spend, active months, app and email engagement, coupons, support tickets, recent January-to-March activity, tenure, and recent-versus-prior behaviour trends. We also derive time since the last recorded purchase when the source workbook is available. Missing values and categorical encoding are handled in the model pipeline.

## Slide 6 — Cohort validation (55 seconds)
The assessment says a random split must not be the primary validation. We train on earlier signup cohorts and hold out the latest signup-date cohort: 1,780 training customers and 445 validation customers, with a cutoff of July 17, 2023. All customers still share the same April-to-June 2024 label period, so this is signup-cohort validation, not a true test on a later outcome period. Churn also differs substantially: 34.55 percent in training and 10.56 percent in validation.

## Slide 7 — Model comparison (45 seconds)
We compare a majority-class baseline, Logistic Regression and Random Forest. The majority-class baseline gets high accuracy by predicting everyone stayed, but it detects no churners. That is why accuracy alone is not enough. We also report precision, recall, F1, PR-AUC or average precision, and ROC-AUC.

## Slide 8 — Model performance (60 seconds)
At a fixed threshold of 0.50, Logistic Regression has 89.7 percent accuracy, 51.0 percent churn precision, 55.3 percent recall, 0.531 F1 and 0.600 PR-AUC. Random Forest has 93.3 percent accuracy, 84.0 percent precision, 44.7 percent recall, 0.583 F1 and 0.699 PR-AUC. Random Forest is stronger for ranking and has fewer false positives at this threshold, but misses more labelled churners. The operating threshold should be selected based on the costs of missed churn versus unnecessary outreach.

## Slide 9 — Main churn drivers (45 seconds)
Random Forest feature importance places recent transactions, months since the last purchase, recent spend, recent app sessions and the indicator for missing recent-activity records among the strongest inputs. This tells us what the model relied on. It does not prove that changing a feature will cause churn to change. Customer-level sensitivity notes are diagnostics, not SHAP and not causal explanations.

## Slide 10 — Membership tier (30 seconds)
Observed churn rates are fairly close by tier: about 27.98 percent for Gold, 29.62 percent for Platinum and 30.76 percent for Silver. Tier alone does not separate churn strongly in this eligible dataset. We should combine tier with behaviour and engagement rather than target only by membership level.

## Slide 11 — City (35 seconds)
The highest observed rates are Austin at about 36.04 percent and Columbus at 34.08 percent, while Spokane is about 25.12 percent. These are descriptive differences. We should check group sizes, local conditions and data quality before treating a city as a cause of churn or making offers based only on geography.

## Slide 12 — Signup cohort (40 seconds)
The signup-quarter results vary widely. For example, 2022 Q2 has a high observed rate, while 2024 Q1 has a very low rate. These comparisons are affected by when customers joined relative to the fixed label window and by the amount of purchase history available. Treat them as cohort diagnostics, not stable rankings of customer types.

## Slide 13 — Retention scenario (50 seconds)
We model an illustrative scenario targeting the highest-risk 20 percent of the held-out validation cohort, 89 customers. If outreach lowered risk among targeted customers by an assumed relative 10 percent, the arithmetic gives around 4.3 expected churns reduced. This is not a measured treatment effect: we have no measured coupon effect or offer cost, and predicted risk is not calibrated to observed validation churn. The next step is a controlled pilot with a control group.

## Slide 14 — Recommendations and limitations (50 seconds)
First, run a controlled retention pilot before scaling offers. Second, investigate why recent activity records are missing for 449 customers. Third, select thresholds using precision, recall, contact capacity and cost together. Fourth, monitor results across city, tier and signup cohort. We need a later labelled period for genuine out-of-time validation and better probability calibration.

## Slide 15 — Conclusion (25 seconds)
ChurnGuard shows that behavioural and engagement signals carry more predictive information than demographics alone in this held-out cohort. Random Forest is strongest by PR-AUC, but recall and calibration still need improvement. The output is a practical starting point for prioritising retention outreach, with a pilot and future-period validation required before production use. Thank you.

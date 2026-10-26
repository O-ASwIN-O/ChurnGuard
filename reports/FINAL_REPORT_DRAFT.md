# ChurnGuard — Customer Churn Prediction for FreshBasket Retail
## Final Project Report (submission draft)

### Executive summary
This project builds an initial customer-level churn analysis using FreshBasket loyalty profiles and monthly engagement data. Churn is defined as no purchase transactions during April–June 2024, despite earlier purchase history. To limit leakage, the engineered activity features use records before April 2024.

The current eligible population contains 2,225 customers: 1,563 non-churned and 662 churned, a 29.75% churn rate. An exploratory Logistic Regression model reached 83.6% accuracy, 76.1% precision, 65.2% recall and 0.702 F1 for the churn class at a 0.50 decision threshold on a stratified random holdout. At a 0.40 threshold, recall rose to 72.0% and F1 to 0.709, while precision decreased to 69.9%. Because threshold choice was examined on the same test set, these values are exploratory rather than a final unbiased estimate.

### 1. Business problem
FreshBasket sends retention offers broadly. The business needs a way to identify at-risk loyalty members and understand whether engagement, shopping behaviour, support experience or customer segments may help target retention activity more efficiently.

### 2. Data and target
The workbook contains customer profiles, monthly activity, and churn labels. The target label corresponds to the final three months of the observation period, April–June 2024. Predictive features should be computed only from dates before April 2024.

The modelling table was constructed at customer level by joining the profile, engineered pre-period activity features, and churn labels. The eligible subset contains 2,225 customers after excluding customers flagged for insufficient history and two customers with no recorded pre-April activity.

### 3. Data quality and preprocessing
Quality checks found exact duplicate records, repeated customer-month groups, conflicting spend records, missing demographic fields, negative spend, zero spend despite transactions, and spend-consistency flags. Exact duplicates were removed. For two conflicting customer-month records, the negative-spend variant was excluded because the other fields matched positive-spend versions; this is an inference-based cleaning decision, not a verified correction from the data owner.

Missing age was filled with the profile median and missing categorical values were normalised (for example, Unknown). For model training, imputation and scaling/encoding are performed within a scikit-learn pipeline to reduce preprocessing leakage.

### 4. Feature engineering
The current table includes pre-label aggregates for transactions, spend, app sessions, email opens, coupon redemption and support tickets; average monthly behaviours; number of active months; and January–March 2024 recent activity. Tenure is computed relative to 1 April 2024.

### 5. Exploratory findings
Observed averages show lower historical engagement among churned customers:
- Average total spend before April 2024: $749.21 for churned customers vs $1,655.99 for non-churned customers.
- Average transactions before April 2024: 23.45 vs 51.17.
- Average app sessions per observed month: 1.22 vs 2.27.
- Average email opens per observed month: 0.77 vs 1.11.
- Average active months: 6.83 vs 10.56.

These are descriptive associations, not proof that low engagement causes churn. Initial tier rates were similar: Silver 30.76%, Gold 27.98%, and Platinum 29.62% churn.

A major caveat is that all four recent-activity features are missing for 449 customers; 442 of those are labelled churned and 7 are not. These customers have no recorded activity rows in January–March 2024. A zero-fill assumption may be useful operationally, but the missingness pattern itself could dominate predictions and may reflect the data-generation/label definition. This requires sensitivity analysis and transparent reporting.

### 6. Initial model results
An exploratory stratified random 80/20 split produced 1,780 training and 445 test records. This is an exploratory diagnostic only; the assessment asks for time- or cohort-based validation as the primary evaluation, and the data has only one labelled outcome period.

| Model / threshold | Accuracy | Churn precision | Churn recall | Churn F1 |
|---|---:|---:|---:|---:|
| Majority-class dummy baseline | 0.703 | 0.000 | 0.000 | 0.000 |
| Logistic Regression, 0.50 | 0.836 | 0.761 | 0.652 | 0.702 |
| Logistic Regression, 0.40 | 0.825 | 0.699 | 0.720 | 0.709 |

PR-AUC and a second ML classifier are not yet verified in the committed results and must be produced before claiming this requirement complete. Threshold comparison was performed on the same holdout, so final threshold selection should occur using a validation cohort rather than the final evaluation sample.

### 7. Feature-group ablation
An initial comparison removed four recent activity features from Logistic Regression. With recent features, the model classified 86 churners correctly at threshold 0.50 and generated 27 false positives; without those features, it classified 84 churners correctly and generated 23 false positives. These were comparisons on the same exploratory split. The required demographics-only versus behavioural/engagement feature-group ablation remains to be completed.

### 8. Validation, interpretation and retention scenario — outstanding completion
The submission needs a cohort-based validation design, Random Forest or another second classifier, PR-AUC, feature importance/driver attribution, segment comparisons by city and signup cohort, a customer-level prediction file, and a retention scenario. These outputs should be generated by rerunning the notebooks/code and should not be filled with assumed numbers.

For the scenario, state explicit assumptions such as targeting the highest-risk cohort with a coupon or proactive outreach. Any “risk reduction” should be labelled an illustrative sensitivity scenario—not a measured causal effect—unless a credible experimental effect size is available.

### 9. Recommendations (provisional)
1. Prioritise members whose recent purchase activity and engagement are declining, but use a carefully validated threshold that balances missed churners against the cost of unnecessary outreach.
2. Pilot retention outreach on a small cohort and measure incremental retention against a control group before scaling.
3. Review the data pipeline for missing Jan–Mar activity records and confirm whether absence means no transactions, missing telemetry or an upstream ingestion issue.
4. Track retention effectiveness by tier, city and signup cohort rather than sending the same offer to all customers.

### 10. Limitations
- One labelled outcome window constrains honest temporal validation.
- The random split results are exploratory, not the required primary cohort validation.
- The missing recent-activity pattern is strongly related to the target and could reflect a structural signal.
- Observational results are associative and do not establish causality.
- The intervention scenario must be treated as an assumption-based estimate, not a proven impact.
- More work is needed before deployment: out-of-time testing, probability calibration, monitoring, and business-cost evaluation.

### Submission checklist
Complete and verify: cohort validation, second classifier, PR-AUC, feature importance, segment analysis, demographic vs behaviour ablation, retention scenario, customer-level prediction CSV and presentation. Ensure the README accurately describes the final code and reproducible instructions.

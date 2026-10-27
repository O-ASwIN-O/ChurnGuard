# ChurnGuard — Final Project Report
## Customer Churn Prediction for FreshBasket Retail

### Executive summary
ChurnGuard uses pre-April 2024 customer activity to analyse customers labelled for the April–June 2024 churn window. The eligible population contains **2,225 customers**, of whom **662** are labelled churned (**29.75%**). The highest-PR-AUC classifier on the signup-cohort validation set was **Random Forest** (average precision / PR-AUC: **0.699**). At a fixed 0.50 threshold, its accuracy was 0.933, precision 0.840, recall 0.447, and F1 0.583.

### 1. Business objective
Identify customers who may churn and understand behavioural and engagement patterns that can guide focused retention outreach. Churn scores are intended to support human-led targeting, not automatic irreversible decisions.

### 2. Target and population
The label defines churn as zero purchase transactions in April–June 2024 despite prior purchase history. To avoid using label-window information, engineered activity inputs were cut off before 1 April 2024. The model-ready table contains 2,225 eligible customers and excludes insufficient-history records and two additional customers with no recorded pre-label activity.

### 3. Data-quality and preprocessing
The upstream EDA identified exact duplicates, repeated customer-month records, two conflicting spend pairs, missing demographic values, negative spend, zero spend alongside positive transactions, and spend-consistency review flags. Exact duplicates were removed. The negative-spend versions of two conflicting rows were excluded because their other fields matched their positive-spend counterparts; this is an inference-based decision, not confirmed source correction. Numeric and categorical preprocessing is fitted within scikit-learn pipelines.

Four January–March features are missing together for 449 customers. Of those, 442 are labelled churned. The records have no monthly activity entries in that recent period. This run encodes the four missing values as zero and adds a missing-activity flag. The association is strong and may reflect both real disengagement and data-generation/recording behaviour; it should not be interpreted as causal proof.

### 4. Features
Features include purchase recency derived from monthly activity before April 2024, pre-period totals/averages, recent January–March activity, recent-vs-prior monthly trend features for spend/transactions/app sessions/support tickets, profile/demographic/tier attributes, and tenure in days as of 1 April 2024. All feature engineering is intended to stop before the target period.

### 5. Validation design
Training uses earlier customer signup cohorts; validation uses the latest signup-date cohort. The cutoff was **2023-07-17**. Training contained 1,780 customers (observed churn rate 34.55%) and the validation cohort contained 445 customers (observed churn rate 10.56%).

**Important limitation:** every customer is labelled over the same April–June 2024 period. This is cohort-based validation by signup date, not a true later-outcome temporal test. Performance may differ when new customer cohorts and future label windows become available. The model with the highest validation PR-AUC is selected descriptively from this cohort comparison; the same holdout is used for comparison, so model-selection optimism is possible.

### 6. Model comparison
The majority-class baseline, Logistic Regression and Random Forest were evaluated on the same cohort holdout at a fixed 0.50 threshold. Precision, recall, F1, accuracy, PR-AUC (average precision) and ROC-AUC are provided.

| Model                   |   Decision threshold |   Customers |   Accuracy |   Churn precision |   Churn recall |   Churn F1 |   PR-AUC / average precision |   ROC-AUC |
|:------------------------|---------------------:|------------:|-----------:|------------------:|---------------:|-----------:|-----------------------------:|----------:|
| Majority-class baseline |                  0.5 |         445 |      0.894 |              0    |          0     |      0     |                        0.106 |     0.5   |
| Logistic Regression     |                  0.5 |         445 |      0.897 |              0.51 |          0.553 |      0.531 |                        0.6   |     0.9   |
| Random Forest           |                  0.5 |         445 |      0.933 |              0.84 |          0.447 |      0.583 |                        0.699 |     0.884 |

**Probability calibration caveat:** the selected model's average predicted validation risk is 20.23%, versus an observed validation churn rate of 10.56%. This discrepancy means the absolute probabilities should not yet be interpreted as calibrated real-world churn likelihoods. Use scores cautiously for ranking; recalibrate and validate on additional labelled periods before operational use.

### 7. Feature-group ablation
The ablation compares profile/demographics only, behaviour/engagement/support only, and all feature groups on the same signup-cohort holdout.

| Model                             |   Decision threshold |   Customers |   Accuracy |   Churn precision |   Churn recall |   Churn F1 |   PR-AUC / average precision |   ROC-AUC |
|:----------------------------------|---------------------:|------------:|-----------:|------------------:|---------------:|-----------:|-----------------------------:|----------:|
| Demographics/profile only         |                  0.5 |         445 |      0.883 |             0     |          0     |      0     |                        0.105 |     0.489 |
| Behaviour/engagement/support only |                  0.5 |         445 |      0.885 |             0.469 |          0.638 |      0.541 |                        0.566 |     0.898 |
| All feature groups                |                  0.5 |         445 |      0.897 |             0.51  |          0.553 |      0.531 |                        0.6   |     0.9   |

### 8. Driver attribution
Global feature importance from Random Forest is aggregated back to original input columns. The customer-level prediction file includes a **local sensitivity** explanation: for each row, selected important features are replaced one at a time with training-reference values and the change in model probability is recorded. This is a model sensitivity diagnostic, not SHAP and not a causal attribution.

| Input feature                      |   Global importance |
|:-----------------------------------|--------------------:|
| RECENT_TRANSACTIONS                |              0.1654 |
| MONTHS_SINCE_LAST_PURCHASE_PRE     |              0.1373 |
| RECENT_SPEND                       |              0.1252 |
| RECENT_APP_SESSIONS                |              0.1033 |
| NO_RECENT_ACTIVITY_RECORDS         |              0.0724 |
| SPEND_TREND_RECENT_VS_PRIOR        |              0.0518 |
| ACTIVE_MONTHS_PRE                  |              0.0476 |
| TRANSACTIONS_TREND_RECENT_VS_PRIOR |              0.047  |
| APP_SESSIONS_PRE                   |              0.0353 |
| APP_SESSIONS_TREND_RECENT_VS_PRIOR |              0.0341 |

### 9. Segment analysis
Observed historical churn rates are calculated by membership tier, city and signup quarter. These are descriptive comparisons and small group counts should be reviewed before taking action.

**By membership tier**

| Segment   |   Customers |   Churned |   Churn rate (%) |
|:----------|------------:|----------:|-----------------:|
| Gold      |         697 |       195 |            27.98 |
| Platinum  |         260 |        77 |            29.62 |
| Silver    |        1268 |       390 |            30.76 |

**By city**

| Segment   |   Customers |   Churned |   Churn rate (%) |
|:----------|------------:|----------:|-----------------:|
| Austin    |         222 |        80 |            36.04 |
| Columbus  |         223 |        76 |            34.08 |
| Nashville |         207 |        63 |            30.43 |
| Raleigh   |         254 |        77 |            30.31 |
| Boise     |         209 |        62 |            29.67 |
| Denver    |         226 |        65 |            28.76 |
| Tulsa     |         253 |        72 |            28.46 |
| Madison   |         219 |        61 |            27.85 |
| Portland  |         205 |        54 |            26.34 |
| Spokane   |         207 |        52 |            25.12 |

**By signup quarter**

| Segment   |   Customers |   Churned |   Churn rate (%) |
|:----------|------------:|----------:|-----------------:|
| 2021Q1    |          72 |        27 |            37.5  |
| 2021Q2    |         197 |        68 |            34.52 |
| 2021Q3    |         193 |        67 |            34.72 |
| 2021Q4    |         215 |        74 |            34.42 |
| 2022Q1    |         182 |        61 |            33.52 |
| 2022Q2    |         166 |        72 |            43.37 |
| 2022Q3    |         176 |        63 |            35.8  |
| 2022Q4    |         183 |        57 |            31.15 |
| 2023Q1    |         176 |        60 |            34.09 |
| 2023Q2    |         182 |        58 |            31.87 |
| 2023Q3    |         177 |        32 |            18.08 |
| 2023Q4    |         187 |        22 |            11.76 |
| 2024Q1    |         119 |         1 |             0.84 |

Full breakdown: `data/processed/segment_churn_rates.csv`.

### 10. Retention scenario
The scenario targets customers in the top-risk fifth of the held-out signup-cohort validation population and assumes a relative risk reduction of 5%, 10% or 15% among targeted customers. The scenario is computed from held-out model probabilities rather than in-sample predictions.

|   Targeted customers | Assumed relative risk reduction   |   Expected churn reduction (modelled) |   Expected churn after action (modelled) |   Mean predicted risk before (%) |   Mean predicted risk after (%) |
|---------------------:|:----------------------------------|--------------------------------------:|-----------------------------------------:|---------------------------------:|--------------------------------:|
|                   89 | 5%                                |                                 2.152 |                                   87.89  |                           20.234 |                          19.751 |
|                   89 | 10%                               |                                 4.305 |                                   85.738 |                           20.234 |                          19.267 |
|                   89 | 15%                               |                                 6.457 |                                   83.585 |                           20.234 |                          18.783 |

These figures are an illustrative sensitivity analysis applied to held-out signup-cohort probabilities, not a measured treatment effect. Because the selected model's mean validation risk (20.23%) differs from the observed validation churn rate (10.56%), the expected counts are not calibrated forecasts and should be treated only as scenario arithmetic until calibration is improved. The dataset provides no experimentally measured coupon effect or offer cost, so causal impact and ROI cannot be claimed. A controlled pilot is needed.

### 11. Recommendations
1. Pilot proactive outreach/coupons for the highest-risk customers and compare incremental retention with a randomised control group.
2. Investigate the missing January–March activity pattern and confirm what missing records mean operationally.
3. Use recall and precision, contact capacity and offer costs together when selecting a business threshold.
4. Monitor risk and pilot results by city, membership tier and signup cohort; avoid over-interpreting small segments.

### 12. Limitations
- A single label window prevents genuine evaluation against a future churn period.
- Signup-cohort validation is a useful cohort holdout but not equivalent to future-time validation.
- Missing recent-activity records are highly associated with the churn label and may dominate prediction.
- Correlation does not establish causation.
- The retention scenario assumes an intervention effect without experimental evidence.
- Customer probabilities in `customer_churn_predictions.csv` are generated by a model refitted on all eligible rows and are in-sample. Use `validation_cohort_predictions.csv` for held-out cohort scores.

### Reproducibility
From the repository root, run:
```bash
pip install -r requirements.txt
python src/run_analysis.py
```
The script regenerates cohort metrics, feature importance, ablation results, segment analyses, predictions, scenario tables, plots, this report and the PowerPoint deck.

# ChurnGuard — Customer Churn Prediction

## Day 1: Data Understanding, Quality Checks and Initial Feature Engineering

### 1. Project Overview

ChurnGuard is a customer churn prediction project for FreshBasket Retail's loyalty program. The objective is to identify customers who may have stopped purchasing and help the business develop targeted customer-retention strategies.

The dataset contains customer profiles, monthly shopping activity and churn labels.

### 2. Dataset Understanding

The Excel workbook `FreshBasket_Loyalty_Churn_Dataset.xlsx` contains the following relevant sheets:

- **Customer Profile:** Customer demographics, membership tier, marketing preferences and store preferences.
- **Monthly Activity:** Monthly transactions, spending, basket value, app sessions, email engagement, coupon redemption and support interactions.
- **Churn Label:** Customer churn status, observation period, last purchase date and history sufficiency flag.
- **Glossary and Data Dictionary:** Definitions of the dataset's fields and business terminology.

### 3. Data Quality Checks

The following checks were performed:

- Identified and removed exact duplicate records.
- Checked customer IDs and customer-month uniqueness.
- Investigated conflicting customer-month records.
- Examined missing values in demographic and churn-related fields.
- Checked for negative spending, zero spending despite positive transaction counts, and inconsistencies between total spend and average basket value.
- Checked numeric ranges for selected fields, including age and promotional transaction percentage.
- Identified customers without recorded pre-churn activity.

Two conflicting customer-month records were investigated. The negative-spend versions were excluded from the modelling activity table because their other recorded fields matched the positive-spend versions. This was a documented data-cleaning decision, not a confirmed correction from the original source.

Spending-quality flags were also created to identify records requiring further review. These flags do not establish that every flagged record is incorrect.

### 4. Feature Engineering

The churn prediction target is based on the absence of transactions during April–June 2024 despite earlier purchase history.

To avoid target-period data leakage, initial predictive activity features were constructed using records before April 2024.

The features include:

- Total transactions and total spending before April 2024.
- Average monthly transactions, spending and basket value.
- Average app sessions, email engagement and support tickets.
- Coupon redemptions and active months.
- Recent transactions, spending, app sessions and support tickets for January–March 2024.

The recent-period features still require appropriate missing-value handling and validation before model training.

### 5. Modelling Population

Customers flagged as having insufficient history were excluded from the primary modelling population. Two additional customers with sufficient-history flags but no recorded pre-April activity were also excluded from the current eligible dataset.

| Metric | Result |
|---|---:|
| Unique customer profiles after exact deduplication | 2,600 |
| Eligible customers after exclusions | 2,225 |
| Non-churned customers | 1,563 |
| Churned customers | 662 |
| Churn rate in eligible population | 29.75% |

### 6. Day 1 Outcome

Day 1 established the dataset structure, performed initial quality checks, documented data-cleaning decisions and created initial pre-period activity features.

### 7. Remaining Work

- Complete missing-value analysis and treatment.
- Perform exploratory data analysis and visualize churn patterns.
- Build baseline and machine-learning models.
- Evaluate models using appropriate classification metrics.
- Interpret the results and develop customer-retention recommendations.

**Project status:** Day 1 completed; exploratory data analysis is the next stage.
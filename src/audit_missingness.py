"""Controlled missingness sensitivity audit for ChurnGuard.

Run from repository root: python src/audit_missingness.py
Writes new audit-only outputs without replacing existing models or notebooks.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score, brier_score_loss,
    confusion_matrix, precision_recall_curve,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "model_eligible.csv"
RAW = ROOT / "data" / "raw" / "FreshBasket_Loyalty_Churn_Dataset.xlsx"
OUT = ROOT / "data" / "processed"
FIG = ROOT / "figures"
REPORTS = ROOT / "reports"
RECENT = ["RECENT_TRANSACTIONS", "RECENT_SPEND",
          "RECENT_APP_SESSIONS", "RECENT_SUPPORT_TICKETS"]
SEED = 42


def make_pipeline(X):
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    prep = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])
    return Pipeline([
        ("preprocess", prep),
        ("classifier", LogisticRegression(max_iter=2000)),
    ])


def count_label_rule_exceptions(eligible_ids):
    if not RAW.exists():
        return None
    labels = pd.read_excel(RAW, sheet_name="fb Churn Label", header=1).drop_duplicates()
    activity = pd.read_excel(RAW, sheet_name="fb Monthly Activity", header=1).drop_duplicates()
    activity["MONTH"] = pd.to_datetime(activity["MONTH"], errors="coerce")
    # Resolve two source pairs with matching fields but opposite spend signs as in the EDA.
    conflict_keys = [
        (501780, pd.Timestamp("2024-05-01")),
        (502065, pd.Timestamp("2023-04-01")),
    ]
    conflict = pd.MultiIndex.from_frame(activity[["CUSTOMER_ID", "MONTH"]]).isin(conflict_keys)
    activity = activity.loc[~(conflict & (activity["TOTAL_SPEND"] < 0))]
    prior = activity.loc[activity["MONTH"] < "2024-04-01"].groupby("CUSTOMER_ID")["TRANSACTIONS"].sum()
    target = activity.loc[
        (activity["MONTH"] >= "2024-04-01") &
        (activity["MONTH"] <= "2024-06-01")
    ].groupby("CUSTOMER_ID")["TRANSACTIONS"].sum()
    labels = labels.loc[labels["CUSTOMER_ID"].isin(eligible_ids)].copy()
    reconstructed = (
        (labels["CUSTOMER_ID"].map(prior).fillna(0) > 0) &
        (labels["CUSTOMER_ID"].map(target).fillna(0) == 0)
    ).astype(int)
    return int((labels["CHURNED"].astype(int) != reconstructed).sum())


def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Required input not found: {DATA}")
    for folder in (OUT, FIG, REPORTS):
        folder.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA)
    missing = df[RECENT].isna().all(axis=1)
    df["RECENT_ACTIVITY_MISSING"] = missing.astype(int)
    y = df["CHURNED"].astype(int)
    X = df.drop(columns=[
        "CUSTOMER_ID", "SIGNUP_DATE", "CHURNED", "INSUFFICIENT_HISTORY_FLAG"
    ])

    # Match notebook/03_model_training.ipynb: 80/20 split, then 75/25 of the 80%.
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.25, random_state=SEED, stratify=y_tv
    )

    variants = {
        "A - Full model": list(X.columns),
        "B - Without explicit flag": [
            col for col in X.columns if col != "RECENT_ACTIVITY_MISSING"
        ],
        "C - Without recent signals": [
            col for col in X.columns
            if col not in RECENT + ["RECENT_ACTIVITY_MISSING"]
        ],
    }
    metric_rows, test_scores = [], {}
    for name, cols in variants.items():
        model = make_pipeline(X_train[cols])
        model.fit(X_train[cols], y_train)
        val_scores = model.predict_proba(X_val[cols])[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_val, val_scores)
        val_f1 = (
            2 * precision[:-1] * recall[:-1] /
            (precision[:-1] + recall[:-1] + 1e-12)
        )
        best_f1 = float(val_f1.max())
        best_idx = np.flatnonzero(np.isclose(val_f1, best_f1))[-1]
        threshold = float(thresholds[best_idx])

        scores = model.predict_proba(X_test[cols])[:, 1]
        prediction = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, prediction, labels=[0, 1]).ravel()
        metric_rows.append({
            "Model": name,
            "Threshold": threshold,
            "Validation_F1": best_f1,
            "Test_Customers": int(len(y_test)),
            "Test_Observed_Churn_Rate": float(y_test.mean()),
            "Accuracy": accuracy_score(y_test, prediction),
            "Precision": precision_score(y_test, prediction, zero_division=0),
            "Recall": recall_score(y_test, prediction, zero_division=0),
            "F1": f1_score(y_test, prediction, zero_division=0),
            "PR_AUC": average_precision_score(y_test, scores),
            "ROC_AUC": roc_auc_score(y_test, scores),
            "Brier_Score": brier_score_loss(y_test, scores),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        })
        test_scores[name] = scores

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT / "missingness_audit_metrics.csv", index=False, float_format="%.10g")

    profile_rows = []
    for name, mask in [
        ("All eligible customers", pd.Series(True, index=df.index)),
        ("No recent activity records", missing),
        ("Recent activity records present", ~missing),
    ]:
        sub = df.loc[mask]
        profile_rows.append({
            "Group": name, "Customers": len(sub),
            "Stayed": int((sub["CHURNED"] == 0).sum()),
            "Churned": int((sub["CHURNED"] == 1).sum()),
            "Churn_Rate_Pct": float(sub["CHURNED"].mean() * 100),
        })
    pd.DataFrame(profile_rows).to_csv(
        OUT / "missingness_audit_missingness_profile.csv",
        index=False, float_format="%.10g"
    )

    capacity_rows = []
    for name, scores in test_scores.items():
        ranked = pd.DataFrame({"actual": y_test.to_numpy(), "score": scores})
        ranked = ranked.sort_values("score", ascending=False)
        total_churners = int(ranked["actual"].sum())
        for share in (0.10, 0.20, 0.30):
            count = int(np.ceil(len(ranked) * share))
            selected = ranked.head(count)
            captured = int(selected["actual"].sum())
            capacity_rows.append({
                "Model": name, "Targeted_Share": share,
                "Customers_Targeted": count, "Churners_Captured": captured,
                "Precision_At_Capacity": captured / count,
                "Churn_Recall_At_Capacity": captured / total_churners,
                "Test_Churners": total_churners,
            })
    pd.DataFrame(capacity_rows).to_csv(
        OUT / "missingness_audit_capacity.csv", index=False, float_format="%.10g"
    )

    plot_data = metrics.set_index("Model")[["F1", "PR_AUC"]].rename(
        columns={"PR_AUC": "PR-AUC"}
    )
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    plot_data.plot(kind="bar", ax=ax, color=["#209A99", "#F0A45D"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Test score")
    ax.set_xlabel("")
    ax.set_title("Missingness audit: test metrics across feature variants")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(loc="upper right")
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "missingness_audit_comparison.png", dpi=180)
    plt.close(fig)

    full, no_flag, no_recent = [metrics.iloc[i] for i in range(3)]
    exceptions = count_label_rule_exceptions(set(df["CUSTOMER_ID"].astype(int)))
    table = metrics[[
        "Model", "Threshold", "Validation_F1", "Accuracy", "Precision",
        "Recall", "F1", "PR_AUC", "ROC_AUC", "Brier_Score", "TN", "FP", "FN", "TP"
    ]].round(3).to_markdown(index=False)
    report = f"""# Missingness audit — ChurnGuard

## Purpose
This controlled sensitivity audit checks whether the saved Logistic Regression workflow relies heavily on missing January–March activity data. Three variants use the same stratified random 60/20/20 split. Preprocessing is fitted on the training set; thresholds are selected on validation data by maximum F1; the untouched test set is evaluated once.

## Missing-activity pattern
- Eligible population: {len(df):,} customers.
- All four recent activity features missing: {int(missing.sum()):,} customers.
- Churn-labelled in that group: {int(df.loc[missing, "CHURNED"].sum()):,} ({df.loc[missing, "CHURNED"].mean():.2%}).
- The source workbook has no January–March activity rows for this group. This predates the April–June target window in the constructed data, but the files cannot prove that the missingness indicator is available and reliable at real prediction time.

## Split and results
- Train / validation / test: {len(X_train):,} / {len(X_val):,} / {len(X_test):,}.
- Test customers: {len(y_test):,}; observed churn: {y_test.mean():.2%}.
- Random seed: {SEED}; stratified split; test set evaluated once per variant.

{table}

## Effect of removing missingness features
- Without the explicit flag, F1 falls by {full.F1-no_flag.F1:.3f} and PR-AUC falls by {full.PR_AUC-no_flag.PR_AUC:.3f}.
- Without recent activity fields and the flag, F1 falls by {full.F1-no_recent.F1:.3f} and PR-AUC falls by {full.PR_AUC-no_recent.PR_AUC:.3f}.

**Conclusion:** missingness is a strong predictive-shortcut candidate in this dataset. This does not prove leakage, because the missingness appears in the constructed pre-label period. Production-time availability and future stability remain unverified; test on a later independently labelled period before operational use.

## Churn-label rule reconstruction
The documented churn rule was reconstructed from raw monthly transaction counts for the eligible customers. It flagged **{exceptions if exceptions is not None else "an unverified number of"} label exceptions** for review. This is diagnostic reconstruction, not a source-owner-approved correction, and it does not establish why any label differs.

## Recommended next steps
1. Confirm whether absent Jan–Mar records mean no shopping, missing telemetry, or an upstream data issue.
2. Verify that missingness is available at the real prediction timestamp.
3. Investigate the label-rule exceptions with the data owner.
4. Validate against a later labelled outcome period.
5. Run a controlled retention pilot; scenario reductions are assumptions, not measured coupon effects or ROI.
"""
    (REPORTS / "MISSINGNESS_AUDIT.md").write_text(report, encoding="utf-8")
    print(metrics.round(4).to_string(index=False))
    print(f"Raw-rule label exceptions among eligible customers: {exceptions}")


if __name__ == "__main__":
    main()

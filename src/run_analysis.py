from __future__ import annotations

"""Run the ChurnGuard analyses and build submission artifacts.

Run from any directory:
    python src/run_analysis.py

The script uses the committed customer-level modelling CSV. Outputs are written
under data/processed, reports/, presentation/ and figures/.
"""

from pathlib import Path
import warnings
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score, confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "model_eligible.csv"
OUT = ROOT / "data" / "processed"
FIG = ROOT / "figures"
REPORTS = ROOT / "reports"
PRESENTATION = ROOT / "presentation"
LABEL_DATE = pd.Timestamp("2024-04-01")
RANDOM_STATE = 42

for folder in (OUT, FIG, REPORTS, PRESENTATION):
    folder.mkdir(parents=True, exist_ok=True)

def metric_row(name, y_true, probabilities, threshold=0.5):
    pred = (np.asarray(probabilities) >= threshold).astype(int)
    return {
        "Model": name,
        "Threshold": threshold,
        "N": int(len(y_true)),
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "PR_AUC_Average_Precision": average_precision_score(y_true, probabilities),
        "ROC_AUC": roc_auc_score(y_true, probabilities) if pd.Series(y_true).nunique() > 1 else np.nan,
    }

def build_pipeline(X_train, kind="logistic"):
    numeric = X_train.select_dtypes(include="number").columns.tolist()
    categorical = X_train.select_dtypes(exclude="number").columns.tolist()
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    prep = ColumnTransformer([
        ("numeric", num_pipe, numeric),
        ("categorical", cat_pipe, categorical),
    ])
    if kind == "logistic":
        classifier = LogisticRegression(max_iter=2000, class_weight=None, random_state=RANDOM_STATE)
    elif kind == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=400, min_samples_leaf=4, max_features="sqrt",
            class_weight="balanced_subsample", random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        raise ValueError(kind)
    return Pipeline([("preprocessor", prep), ("classifier", classifier)])

def transformed_importance(model, raw_columns):
    """Map one-hot/scaled model importances back to original input columns."""
    clf = model.named_steps["classifier"]
    prep = model.named_steps["preprocessor"]
    names = prep.get_feature_names_out()
    if hasattr(clf, "feature_importances_"):
        values = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        values = np.abs(clf.coef_[0])
    else:
        return pd.DataFrame(columns=["Feature", "Importance"])
    mapping = []
    ordered = sorted(raw_columns, key=len, reverse=True)
    for name in names:
        n = str(name)
        raw = n
        if n.startswith("numeric__"):
            raw = n[len("numeric__"):]
        elif n.startswith("categorical__"):
            raw = n[len("categorical__"):]
            for candidate in ordered:
                if raw == candidate or raw.startswith(candidate + "_"):
                    raw = candidate
                    break
        mapping.append(raw)
    imp = pd.DataFrame({"Feature": mapping, "Importance": values})
    return imp.groupby("Feature", as_index=False)["Importance"].sum().sort_values("Importance", ascending=False)

def group_metrics(name, feature_columns, X_train, X_test, y_train, y_test):
    model = build_pipeline(X_train[feature_columns], "logistic")
    model.fit(X_train[feature_columns], y_train)
    prob = model.predict_proba(X_test[feature_columns])[:, 1]
    row = metric_row(name, y_test, prob)
    return row

def segment_table(df, col, label):
    t = df.groupby(col, dropna=False).agg(
        Customers=("CHURNED", "size"),
        Churned=("CHURNED", "sum"),
        Churn_Rate=("CHURNED", "mean"),
    ).reset_index().rename(columns={col: "Segment"})
    t.insert(0, "Segment_Type", label)
    t["Churn_Rate_Pct"] = t["Churn_Rate"] * 100
    return t

def make_plot_files(df, metrics, importance, segments):
    # Churn balance
    counts = df["CHURNED"].value_counts().reindex([0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(["Stayed", "Churned"], counts.values)
    ax.set_title("Eligible customer outcomes")
    ax.set_ylabel("Customers")
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x()+bar.get_width()/2, value, f"{int(value):,}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(FIG / "churn_distribution.png", dpi=170)
    plt.close(fig)

    # Model metrics
    metric_names = ["Accuracy", "Precision", "Recall", "F1", "PR_AUC_Average_Precision"]
    plot_df = metrics.set_index("Model")[metric_names]
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df.plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Cohort-holdout model comparison")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "model_comparison.png", dpi=170)
    plt.close(fig)

    # Top drivers
    top = importance.head(12).sort_values("Importance")
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.barh(top["Feature"], top["Importance"])
    ax.set_title("Top model feature impacts (global importance)")
    ax.set_xlabel("Aggregated importance")
    fig.tight_layout()
    fig.savefig(FIG / "feature_importance.png", dpi=170)
    plt.close(fig)

    # Tier and city segment charts
    tier = segments[segments["Segment_Type"] == "Membership tier"].sort_values("Churn_Rate_Pct")
    city = segments[segments["Segment_Type"] == "City"].sort_values("Churn_Rate_Pct")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(tier["Segment"].astype(str), tier["Churn_Rate_Pct"])
    ax.set_title("Observed churn rate by membership tier")
    ax.set_ylabel("Churn rate (%)")
    fig.tight_layout()
    fig.savefig(FIG / "churn_by_tier.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.8, 4.5))
    ax.bar(city["Segment"].astype(str), city["Churn_Rate_Pct"])
    ax.set_title("Observed churn rate by city")
    ax.set_ylabel("Churn rate (%)")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(FIG / "churn_by_city.png", dpi=170)
    plt.close(fig)

    # Ablation results
    if (OUT / "ablation_results.csv").exists():
        abl = pd.read_csv(OUT / "ablation_results.csv").set_index("Model")
        fig, ax = plt.subplots(figsize=(8.2, 4.5))
        abl[["Accuracy", "Precision", "Recall", "F1", "PR_AUC_Average_Precision"]].plot(kind="bar", ax=ax)
        ax.set_ylim(0, 1)
        ax.set_title("Feature-group ablation — cohort holdout")
        ax.tick_params(axis="x", rotation=10)
        fig.tight_layout()
        fig.savefig(FIG / "feature_ablation.png", dpi=170)
        plt.close(fig)

def make_presentation(metrics, importance, segments, ablation, scenario, split_info, n_customers, n_churned, churn_rate):
    try:
        from pptx import Presentation
        from pptx.util import Inches, Pt
        from pptx.dml.color import RGBColor
    except ImportError:
        warnings.warn("python-pptx is not installed; skipped presentation generation.")
        return

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    navy = RGBColor(20, 34, 61)
    teal = RGBColor(29, 151, 160)
    grey = RGBColor(75, 86, 102)

    def add_bg(slide, color=navy):
        shape = slide.shapes.add_shape(1, 0, 0, prs.slide_width, prs.slide_height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = color
        shape.line.fill.background()
        # send background behind everything by moving to first in spTree
        sp = shape._element
        sp.getparent().remove(sp)
        slide.shapes._spTree.insert(2, sp)

    def textbox(slide, text, x, y, w, h, size=22, color=RGBColor(255,255,255), bold=False):
        box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = box.text_frame
        tf.word_wrap = True
        tf.clear()
        p = tf.paragraphs[0]
        p.text = str(text)
        p.font.size = Pt(size)
        p.font.bold = bold
        p.font.color.rgb = color
        return box

    def title(slide, heading, sub=None):
        textbox(slide, heading, 0.6, 0.45, 12, 0.6, 28, RGBColor(255,255,255), True)
        if sub:
            textbox(slide, sub, 0.65, 1.15, 12, 0.45, 13, RGBColor(180,210,224))

    def content_slide(heading, bullets, sub=None):
        s = prs.slides.add_slide(prs.slide_layouts[6])
        add_bg(s)
        title(s, heading, sub)
        y = 1.8
        for item in bullets:
            textbox(s, "• " + item, 0.85, y, 11.7, 0.78, 19, RGBColor(238,243,249))
            y += 1.03
        return s

    # 1
    s = prs.slides.add_slide(prs.slide_layouts[6]); add_bg(s)
    textbox(s, "CHURNGUARD", 0.8, 1.6, 11.7, 0.8, 34, RGBColor(255,255,255), True)
    textbox(s, "Customer Churn Prediction for FreshBasket Retail", 0.85, 2.6, 11.5, 0.8, 24, RGBColor(65,208,204), True)
    textbox(s, "Propel Assessment | Data quality • Behavioural signals • Risk scoring • Retention action", 0.85, 3.65, 11.4, 0.8, 17)
    textbox(s, "Prepared from the committed dataset and reproducible analysis outputs", 0.85, 6.45, 11.4, 0.4, 12, RGBColor(180,210,224))

    content_slide("1. Business problem", [
        "FreshBasket wants to identify loyalty members who may stop purchasing.",
        "Objective: estimate customer-level churn risk and focus retention outreach where it may be most useful.",
        "Prediction features must be available before the label window; business action needs a controlled pilot."
    ])
    content_slide("2. Target and data", [
        "CHURNED = 1: no purchase transactions in April–June 2024 despite earlier purchase history.",
        f"Eligible modelling population: {n_customers:,} customers; {n_churned:,} churned ({churn_rate:.1%}).",
        "Predictive activity features end in March 2024; insufficient-history customers are excluded."
    ])
    content_slide("3. Data-quality decisions", [
        "Removed exact duplicate rows and investigated repeated customer-months.",
        "Two conflicting spend pairs were handled by excluding the negative-spend version; this is an inference, not source-owner confirmation.",
        "449 customers have all four recent activity features missing; 442 are labelled churned. Missingness may strongly influence the model."
    ])
    content_slide("4. Leakage-safe feature groups", [
        "Behaviour: pre-period transactions, spend, basket value, promotions, coupons and active months.",
        "Engagement/support: app sessions, email opens and support tickets, including January–March recent activity.",
        "Profile: tenure, age, city, membership tier, marketing opt-in and preferences."
    ])
    s = prs.slides.add_slide(prs.slide_layouts[6]); add_bg(s); title(s, "5. Cohort-based validation", "Hold out customers from the latest signup-date cohort")
    textbox(s, f"Cut-off date: {split_info['cutoff']}", 0.85, 1.95, 11.4, 0.55, 22, RGBColor(65,208,204), True)
    textbox(s, f"Training cohort: {split_info['train_n']:,} customers ({split_info['train_churn_rate']:.1%} churn)  |  Validation: {split_info['test_n']:,} customers ({split_info['test_churn_rate']:.1%} churn)", 0.85, 2.65, 11.7, 0.6, 17)
    textbox(s, f"Calibration check: selected-model mean risk {holdout_prob.mean():.1%} vs observed validation churn {y_test.mean():.1%}.", 0.85, 3.25, 11.4, 0.45, 16, RGBColor(65,208,204), True)
    textbox(s, "Caveat: all customers share the same Apr–Jun 2024 label window. Signup-cohort validation is not the same as testing on a later outcome period; the score gap also means absolute probabilities need further calibration.", 0.85, 3.85, 11.4, 1.1, 17, RGBColor(240,215,165))
    content_slide("6. Model comparison", [
        f"Best cohort-holdout PR-AUC (average precision): {metrics.loc[metrics['PR_AUC_Average_Precision'].idxmax(), 'Model']} ({metrics['PR_AUC_Average_Precision'].max():.3f}).",
        f"Best cohort-holdout F1: {metrics.loc[metrics['F1'].idxmax(), 'Model']} ({metrics['F1'].max():.3f}).",
        "Accuracy alone is insufficient for the 29.75% churn class; precision, recall, F1 and PR-AUC are reported."
    ])
    # plot slide
    def image_slide(heading, file, note=None):
        s = prs.slides.add_slide(prs.slide_layouts[6]); add_bg(s); title(s, heading, note)
        p = FIG / file
        if p.exists():
            s.shapes.add_picture(str(p), Inches(1.0), Inches(1.65), width=Inches(11.3), height=Inches(5.25))
        return s
    image_slide("7. Model performance", "model_comparison.png", "Cohort-holdout metrics at a fixed 0.50 decision threshold")
    image_slide("8. Main churn drivers", "feature_importance.png", "Global model importance aggregated to original input features; not causal effects")
    image_slide("9. Segment risk", "churn_by_tier.png", "Observed historical churn rates by membership tier")
    # Retention scenario slide
    r10 = scenario[scenario["Assumed_Relative_Risk_Reduction"] == 0.10]
    if len(r10):
        rr = r10.iloc[0]
        bullets = [
            f"Target the top {rr['Targeted_Pct']:.0f}% by predicted risk ({int(rr['Targeted_Customers']):,} customers).",
            f"Under an assumed 10% relative reduction for contacted customers, modelled expected churn reduction is {rr['Expected_Churn_Reduction_Customers']:.1f} customers.",
            "This is an illustrative sensitivity scenario, not a causal effect estimate. Coupon cost and treatment effectiveness are not provided."
        ]
    else:
        bullets = ["Target high-risk customers with a controlled coupon/outreach pilot.", "Estimate impact under explicit assumptions.", "This is a sensitivity scenario, not a measured causal effect."]
    content_slide("10. Retention scenario", bullets)
    content_slide("11. Recommendations and limitations", [
        "Run a controlled retention pilot and compare outcomes against a holdout group before scaling.",
        "Investigate missing Jan–Mar activity records; confirm whether they mean no shopping or missing telemetry.",
        f"Validation calibration warning: mean selected-model risk {holdout_prob.mean():.1%} vs observed churn {y_test.mean():.1%}; use scores for ranking only until calibrated.",
        "One label window limits genuine out-of-time validation; observational patterns do not establish causality."
    ])
    content_slide("12. Conclusion and next steps", [
        "Use churn scores to prioritise outreach—not as a final automated decision.",
        "Monitor recall and precision alongside offer cost and incremental retention.",
        "Validate on a future outcome window when new labels become available."
    ])
    prs.save(PRESENTATION / "ChurnGuard_Final_Presentation.pptx")

def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Required dataset not found: {DATA}")
    df = pd.read_csv(DATA)
    if "CHURNED" not in df or "CUSTOMER_ID" not in df:
        raise ValueError("Expected CUSTOMER_ID and CHURNED in model_eligible.csv")
    df["SIGNUP_DATE"] = pd.to_datetime(df["SIGNUP_DATE"], errors="coerce")
    df["TENURE_DAYS"] = (LABEL_DATE - df["SIGNUP_DATE"]).dt.days

    # Derive purchase recency from monthly activity if the source workbook is present.
    raw_book = ROOT / "data" / "raw" / "FreshBasket_Loyalty_Churn_Dataset.xlsx"
    if raw_book.exists():
        try:
            activity_raw = pd.read_excel(
                raw_book, sheet_name="fb Monthly Activity", header=1,
                usecols=["CUSTOMER_ID", "MONTH", "TRANSACTIONS"],
            ).drop_duplicates()
            activity_raw["MONTH"] = pd.to_datetime(activity_raw["MONTH"], errors="coerce")
            pre_purchase = activity_raw[
                (activity_raw["MONTH"] < LABEL_DATE) &
                (activity_raw["TRANSACTIONS"] > 0)
            ]
            last_purchase = pre_purchase.groupby("CUSTOMER_ID")["MONTH"].max()
            last_purchase_map = last_purchase.to_dict()
            df["LAST_ACTIVE_MONTH_PRE"] = df["CUSTOMER_ID"].map(last_purchase_map)
            last_month = pd.to_datetime(df["LAST_ACTIVE_MONTH_PRE"], errors="coerce")
            df["MONTHS_SINCE_LAST_PURCHASE_PRE"] = (
                (LABEL_DATE.year - last_month.dt.year) * 12
                + (LABEL_DATE.month - last_month.dt.month)
            )
            df["NO_PRIOR_PURCHASE_RECORD_FLAG"] = last_month.isna().astype(int)
            df.drop(columns=["LAST_ACTIVE_MONTH_PRE"], inplace=True)
        except Exception as exc:
            warnings.warn(f"Could not derive recency from source workbook: {exc}")

    recent_cols = ["RECENT_TRANSACTIONS", "RECENT_SPEND", "RECENT_APP_SESSIONS", "RECENT_SUPPORT_TICKETS"]
    missing_recent = df[recent_cols].isna().all(axis=1)
    df["NO_RECENT_ACTIVITY_RECORDS"] = missing_recent.astype(int)
    # Explicit assumption for model features: no recorded recent activity is encoded as zero.
    # A separate flag preserves the fact that those source rows were absent.
    df[recent_cols] = df[recent_cols].fillna(0)

    # Compare monthly averages in Jan-Mar 2024 with the preceding 12 calendar months.
    trend_pairs = [
        ("TRANSACTIONS_PRE", "RECENT_TRANSACTIONS", "TRANSACTIONS_TREND_RECENT_VS_PRIOR"),
        ("TOTAL_SPEND_PRE", "RECENT_SPEND", "SPEND_TREND_RECENT_VS_PRIOR"),
        ("APP_SESSIONS_PRE", "RECENT_APP_SESSIONS", "APP_SESSIONS_TREND_RECENT_VS_PRIOR"),
        ("SUPPORT_TICKETS_PRE", "RECENT_SUPPORT_TICKETS", "SUPPORT_TICKETS_TREND_RECENT_VS_PRIOR"),
    ]
    for pre_total, recent_total, trend_name in trend_pairs:
        prior_twelve_months = (df[pre_total] - df[recent_total]).clip(lower=0)
        df[trend_name] = (df[recent_total] / 3.0) - (prior_twelve_months / 12.0)

    target = df["CHURNED"].astype(int)
    n = len(df)
    n_churned = int(target.sum())
    churn_rate = float(target.mean())

    # Cohort split: earlier signup dates train the model; latest signup-date cohort is held out.
    cutoff = df["SIGNUP_DATE"].quantile(0.80)
    is_test = df["SIGNUP_DATE"] >= cutoff
    if is_test.sum() < 50 or target[is_test].nunique() < 2 or target[~is_test].nunique() < 2:
        # Fallback to latest 20% rows after sorting by signup date, still in cohort order.
        ordered = df.sort_values("SIGNUP_DATE")
        boundary = max(1, int(len(ordered) * 0.8))
        test_ids = set(ordered.iloc[boundary:]["CUSTOMER_ID"])
        is_test = df["CUSTOMER_ID"].isin(test_ids)
        cutoff = ordered.iloc[boundary]["SIGNUP_DATE"]
    if target[is_test].nunique() < 2 or target[~is_test].nunique() < 2:
        raise ValueError("Cohort split did not contain both target classes in train and validation.")
    train_idx = np.flatnonzero(~is_test.to_numpy())
    test_idx = np.flatnonzero(is_test.to_numpy())

    exclude = {"CUSTOMER_ID", "SIGNUP_DATE", "CHURNED", "INSUFFICIENT_HISTORY_FLAG"}
    feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols].copy()
    y = target.copy()
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    split_info = {
        "cutoff": str(pd.Timestamp(cutoff).date()),
        "train_n": len(train_idx),
        "test_n": len(test_idx),
        "train_churn_rate": float(y_train.mean()),
        "test_churn_rate": float(y_test.mean()),
        "strategy": "Train on earlier signup cohorts; validate on latest signup-date cohort",
        "label_window": "April-June 2024",
        "note": "Cohort validation by signup date, not a later outcome window.",
    }
    (OUT / "cohort_validation_design.json").write_text(json.dumps(split_info, indent=2), encoding="utf-8")

    # Baseline + two classifiers on exactly the same cohort holdout.
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_prob = baseline.predict_proba(X_test)[:, 1]
    results = [metric_row("Majority-class baseline", y_test, baseline_prob)]
    models = {}
    for kind, name in [("logistic", "Logistic Regression"), ("random_forest", "Random Forest")]:
        model = build_pipeline(X_train, kind)
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        results.append(metric_row(name, y_test, prob))
        models[name] = model
    metrics = pd.DataFrame(results)
    metrics.to_csv(OUT / "cohort_model_comparison.csv", index=False)

    # Choose the classifier with highest average precision on this validation cohort.
    # This same holdout is used for comparison, so the winning metric is descriptive.
    candidates = metrics[metrics["Model"].isin(models.keys())].sort_values("PR_AUC_Average_Precision", ascending=False)
    best_name = str(candidates.iloc[0]["Model"])
    best_holdout_model = models[best_name]
    holdout_prob = best_holdout_model.predict_proba(X_test)[:, 1]
    holdout_pred = (holdout_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_test, holdout_pred, labels=[0, 1])
    pd.DataFrame(cm, index=["Actual_Stayed", "Actual_Churned"], columns=["Pred_Stayed", "Pred_Churned"]).to_csv(OUT / "cohort_confusion_matrix.csv")

    holdout_predictions = df.iloc[test_idx][["CUSTOMER_ID", "SIGNUP_DATE", "CHURNED", "MEMBERSHIP_TIER", "CITY"]].copy()
    holdout_predictions["CHURN_PROBABILITY"] = holdout_prob
    holdout_predictions["PREDICTED_CHURNED"] = holdout_pred
    holdout_predictions.to_csv(OUT / "validation_cohort_predictions.csv", index=False)

    # Ablation: demographics/profile only vs behaviours only vs all features.
    demographics = [c for c in [
        "AGE", "GENDER", "CITY", "MEMBERSHIP_TIER", "MARKETING_OPT_IN",
        "PREFERRED_STORE_ID", "PREFERRED_CATEGORY", "HOME_STORE_PRICE_TIER", "TENURE_DAYS"
    ] if c in feature_cols]
    behavioural = [c for c in feature_cols if c not in demographics]
    ablation_rows = []
    ablation_rows.append(group_metrics("Demographics/profile only", demographics, X_train, X_test, y_train, y_test))
    ablation_rows.append(group_metrics("Behaviour/engagement/support only", behavioural, X_train, X_test, y_train, y_test))
    ablation_rows.append(group_metrics("All feature groups", feature_cols, X_train, X_test, y_train, y_test))
    ablation = pd.DataFrame(ablation_rows)
    ablation.to_csv(OUT / "ablation_results.csv", index=False)

    # Global feature importance from Random Forest, grouped back to original feature columns.
    rf = models["Random Forest"]
    importance = transformed_importance(rf, feature_cols)
    importance = importance.sort_values("Importance", ascending=False)
    importance.to_csv(OUT / "feature_importance.csv", index=False)

    # Segment analysis using observed labels.
    df_segments = df.copy()
    df_segments["SIGNUP_COHORT"] = df_segments["SIGNUP_DATE"].dt.to_period("Q").astype(str)
    segments = pd.concat([
        segment_table(df_segments, "MEMBERSHIP_TIER", "Membership tier"),
        segment_table(df_segments, "CITY", "City"),
        segment_table(df_segments, "SIGNUP_COHORT", "Signup quarter"),
    ], ignore_index=True)
    segments.to_csv(OUT / "segment_churn_rates.csv", index=False)

    # Score all eligible customers with a model refit to all eligible rows.
    final_kind = "logistic" if best_name == "Logistic Regression" else "random_forest"
    final_model = build_pipeline(X, final_kind)
    final_model.fit(X, y)
    all_prob = final_model.predict_proba(X)[:, 1]
    all_pred = (all_prob >= 0.5).astype(int)

    # Local sensitivity explanations: replace each of the highest-ranked raw features
    # with its training reference value, then observe the probability change.
    top_raw = importance["Feature"].head(8).tolist()
    reference = {}
    for c in top_raw:
        if pd.api.types.is_numeric_dtype(X[c]):
            reference[c] = X_train[c].median()
        else:
            mode = X_train[c].mode(dropna=True)
            reference[c] = mode.iloc[0] if len(mode) else "Unknown"
    deltas = {c: np.zeros(n, dtype=float) for c in top_raw}
    for c in top_raw:
        changed = X.copy()
        changed[c] = reference[c]
        changed_prob = final_model.predict_proba(changed)[:, 1]
        deltas[c] = all_prob - changed_prob

    keys = []
    for i in range(n):
        candidates_local = sorted(
            [(c, float(deltas[c][i])) for c in top_raw],
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:3]
        labels = []
        for c, delta in candidates_local:
            direction = "raises" if delta > 1e-6 else ("lowers" if delta < -1e-6 else "little effect on")
            labels.append(f"{c} {direction} risk (sensitivity {delta:+.3f})")
        keys.append("; ".join(labels))
    prediction_output = df[["CUSTOMER_ID"]].copy()
    prediction_output["CHURN_PROBABILITY"] = all_prob
    prediction_output["PREDICTED_CHURNED"] = all_pred
    prediction_output["ACTUAL_CHURNED"] = y
    prediction_output["KEY_DRIVERS_SENSITIVITY"] = keys
    prediction_output["SCORING_NOTE"] = "In-sample scores from model refit on eligible population; use holdout file for validation metrics."
    prediction_output.to_csv(OUT / "customer_churn_predictions.csv", index=False)

    # Illustrative retention sensitivity scenarios on the held-out validation cohort.
    # Using held-out probabilities avoids basing the scenario on in-sample scores.
    validation_n = len(holdout_prob)
    cutoff_risk = float(np.quantile(holdout_prob, 0.80))
    targeted = holdout_prob >= cutoff_risk
    if not targeted.any():
        targeted[np.argmax(holdout_prob)] = True
    baseline_expected = float(holdout_prob.sum())
    scenario_rows = []
    for reduction in [0.05, 0.10, 0.15]:
        saved = float((holdout_prob[targeted] * reduction).sum())
        scenario_rows.append({
            "Scenario_Population": "Held-out signup-cohort validation population",
            "Scenario_Population_N": validation_n,
            "Targeting_Rule": "Top 20% predicted risk (quantile cutoff; ties included)",
            "Targeted_Pct": 100 * float(targeted.mean()),
            "Targeted_Customers": int(targeted.sum()),
            "Assumed_Relative_Risk_Reduction": reduction,
            "Baseline_Expected_Churn_Customers": baseline_expected,
            "Validation_Observed_Churn_Customers": int(y_test.sum()),
            "Expected_Churn_Reduction_Customers": saved,
            "Scenario_Expected_Churn_Customers": baseline_expected - saved,
            "Baseline_Avg_Risk_Pct": float(holdout_prob.mean() * 100),
            "Scenario_Avg_Risk_Pct": float((baseline_expected - saved) / validation_n * 100),
            "Risk_Reduction_Pct_Points": float(saved / validation_n * 100),
            "Assumption_Note": "Illustrative relative-risk reduction applied to held-out model probabilities; not a causal estimate or measured treatment effect.",
        })
    scenario = pd.DataFrame(scenario_rows)
    scenario.to_csv(OUT / "retention_scenario.csv", index=False)

    # A compact machine-readable run summary.
    summary = {
        "eligible_customers": n,
        "churned_customers": n_churned,
        "observed_churn_rate": churn_rate,
        "missing_recent_activity_rows": int(missing_recent.sum()),
        "churned_among_missing_recent_activity": int(y[missing_recent].sum()),
        "cohort_validation": split_info,
        "selected_model_by_holdout_pr_auc": best_name,
        "validation_observed_churn_rate": float(y_test.mean()),
        "validation_mean_predicted_risk": float(holdout_prob.mean()),
        "validation_calibration_gap_pp": float((holdout_prob.mean() - y_test.mean()) * 100),
        "metrics": metrics.to_dict(orient="records"),
        "ablation": ablation.to_dict(orient="records"),
        "top_global_drivers": importance.head(10).to_dict(orient="records"),
        "retention_scenario_10pct_assumption": scenario[scenario["Assumed_Relative_Risk_Reduction"] == 0.10].to_dict(orient="records"),
    }
    (OUT / "analysis_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    make_plot_files(df, metrics[metrics["Model"] != "Majority-class baseline"], importance, segments)
    # Add ablation plot after CSV is written.
    try:
        make_plot_files(df, metrics[metrics["Model"] != "Majority-class baseline"], importance, segments)
    except Exception as e:
        warnings.warn(f"Some plots could not be created: {e}")

    # Write report from actual generated outputs.
    best = metrics.loc[metrics["Model"] == best_name].iloc[0]
    metric_table = metrics.round(3).to_markdown(index=False)
    ablation_table = ablation.round(3).to_markdown(index=False)
    top_table = importance.head(10).round(4).to_markdown(index=False)
    segment_table_text = segments[segments["Segment_Type"] == "Membership tier"][["Segment", "Customers", "Churned", "Churn_Rate_Pct"]].round(2).to_markdown(index=False)
    scenario_text = scenario.round(3).to_markdown(index=False)
    report_text = f"""# ChurnGuard — Final Project Report
## Customer Churn Prediction for FreshBasket Retail

### Executive summary
ChurnGuard uses pre-April 2024 customer activity to analyse customers labelled for the April–June 2024 churn window. The eligible population contains **{n:,} customers**, of whom **{n_churned:,}** are labelled churned (**{churn_rate:.2%}**). The highest-PR-AUC classifier on the signup-cohort validation set was **{best_name}** (average precision / PR-AUC: **{best['PR_AUC_Average_Precision']:.3f}**). At a fixed 0.50 threshold, its accuracy was {best['Accuracy']:.3f}, precision {best['Precision']:.3f}, recall {best['Recall']:.3f}, and F1 {best['F1']:.3f}.

### 1. Business objective
Identify customers who may churn and understand behavioural and engagement patterns that can guide focused retention outreach. Churn scores are intended to support human-led targeting, not automatic irreversible decisions.

### 2. Target and population
The label defines churn as zero purchase transactions in April–June 2024 despite prior purchase history. To avoid using label-window information, engineered activity inputs were cut off before 1 April 2024. The model-ready table contains {n:,} eligible customers and excludes insufficient-history records and two additional customers with no recorded pre-label activity.

### 3. Data-quality and preprocessing
The upstream EDA identified exact duplicates, repeated customer-month records, two conflicting spend pairs, missing demographic values, negative spend, zero spend alongside positive transactions, and spend-consistency review flags. Exact duplicates were removed. The negative-spend versions of two conflicting rows were excluded because their other fields matched their positive-spend counterparts; this is an inference-based decision, not confirmed source correction. Numeric and categorical preprocessing is fitted within scikit-learn pipelines.

Four January–March features are missing together for {int(missing_recent.sum())} customers. Of those, {int(y[missing_recent].sum())} are labelled churned. The records have no monthly activity entries in that recent period. This run encodes the four missing values as zero and adds a missing-activity flag. The association is strong and may reflect both real disengagement and data-generation/recording behaviour; it should not be interpreted as causal proof.

### 4. Features
Features include purchase recency derived from monthly activity before April 2024, pre-period totals/averages, recent January–March activity, recent-vs-prior monthly trend features for spend/transactions/app sessions/support tickets, profile/demographic/tier attributes, and tenure in days as of 1 April 2024. All feature engineering is intended to stop before the target period.

### 5. Validation design
Training uses earlier customer signup cohorts; validation uses the latest signup-date cohort. The cutoff was **{split_info['cutoff']}**. Training contained {split_info['train_n']:,} customers (observed churn rate {split_info['train_churn_rate']:.2%}) and the validation cohort contained {split_info['test_n']:,} customers (observed churn rate {split_info['test_churn_rate']:.2%}).

**Important limitation:** every customer is labelled over the same April–June 2024 period. This is cohort-based validation by signup date, not a true later-outcome temporal test. Performance may differ when new customer cohorts and future label windows become available. The model with the highest validation PR-AUC is selected descriptively from this cohort comparison; the same holdout is used for comparison, so model-selection optimism is possible.

### 6. Model comparison
The majority-class baseline, Logistic Regression and Random Forest were evaluated on the same cohort holdout at a fixed 0.50 threshold. Precision, recall, F1, accuracy, PR-AUC (average precision) and ROC-AUC are provided.

{metric_table}

**Probability calibration caveat:** the selected model's average predicted validation risk is {float(holdout_prob.mean()):.2%}, versus an observed validation churn rate of {float(y_test.mean()):.2%}. This discrepancy means the absolute probabilities should not yet be interpreted as calibrated real-world churn likelihoods. Use scores cautiously for ranking; recalibrate and validate on additional labelled periods before operational use.

### 7. Feature-group ablation
The ablation compares profile/demographics only, behaviour/engagement/support only, and all feature groups on the same signup-cohort holdout.

{ablation_table}

### 8. Driver attribution
Global feature importance from Random Forest is aggregated back to original input columns. The customer-level prediction file includes a **local sensitivity** explanation: for each row, selected important features are replaced one at a time with training-reference values and the change in model probability is recorded. This is a model sensitivity diagnostic, not SHAP and not a causal attribution.

{top_table}

### 9. Segment analysis
Observed historical churn rates are calculated by membership tier, city and signup quarter. These are descriptive comparisons and small group counts should be reviewed before taking action.

{segment_table_text}

Full breakdown: \`data/processed/segment_churn_rates.csv\`.

### 10. Retention scenario
The scenario targets customers in the top-risk fifth of the held-out signup-cohort validation population and assumes a relative risk reduction of 5%, 10% or 15% among targeted customers. The scenario is computed from held-out model probabilities rather than in-sample predictions.

{scenario_text}

These figures are an illustrative sensitivity analysis applied to held-out signup-cohort probabilities, not a measured treatment effect. Because the selected model's mean validation risk ({float(holdout_prob.mean()):.2%}) differs from the observed validation churn rate ({float(y_test.mean()):.2%}), the expected counts are not calibrated forecasts and should be treated only as scenario arithmetic until calibration is improved. The dataset provides no experimentally measured coupon effect or offer cost, so causal impact and ROI cannot be claimed. A controlled pilot is needed.

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
- Customer probabilities in \`customer_churn_predictions.csv\` are generated by a model refitted on all eligible rows and are in-sample. Use \`validation_cohort_predictions.csv\` for held-out cohort scores.

### Reproducibility
From the repository root, run:
\`\`\`bash
pip install -r requirements.txt
python src/run_analysis.py
\`\`\`
The script regenerates cohort metrics, feature importance, ablation results, segment analyses, predictions, scenario tables, plots, this report and the PowerPoint deck.
"""
    (REPORTS / "FINAL_REPORT.md").write_text(report_text, encoding="utf-8")
    (REPORTS / "ANALYSIS_SUMMARY.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    make_presentation(metrics[metrics["Model"] != "Majority-class baseline"].reset_index(drop=True), importance, segments, ablation, scenario, split_info, n, n_churned, churn_rate)

    print("ChurnGuard analysis completed.")
    print(f"Population: {n} customers; churn rate: {churn_rate:.2%}")
    print(f"Cohort cutoff: {split_info['cutoff']}; train={split_info['train_n']}; validation={split_info['test_n']}")
    print(f"Selected model by validation PR-AUC: {best_name}")
    print(metrics.round(3).to_string(index=False))
    print("Outputs written to data/processed/, reports/, figures/ and presentation/")

if __name__ == "__main__":
    main()

"""Score prepared customer feature rows with the saved ChurnGuard model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "outputs" / "churnguard_model.joblib"
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "model_eligible.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "customer_churn_scores.csv"


def score_customers(
    input_path: Path,
    output_path: Path,
    model_path: Path = DEFAULT_MODEL,
) -> pd.DataFrame:
    """Score one row per customer from a prepared feature CSV."""
    bundle: dict[str, Any] = joblib.load(model_path)
    model = bundle["model"]
    threshold = float(bundle["threshold"])
    feature_columns: list[str] = bundle["feature_columns"]
    recent_columns: list[str] = bundle["recent_activity_columns"]
    positive_label = bundle["positive_label"]
    indicator_column = "RECENT_ACTIVITY_MISSING"
    input_columns = [
        column for column in feature_columns if column != indicator_column
    ]

    customers = pd.read_csv(input_path)
    missing_columns = sorted(set(input_columns) - set(customers.columns))
    if missing_columns:
        raise ValueError(
            "Input is missing model feature columns: "
            + ", ".join(missing_columns)
        )

    features = customers[input_columns].copy()
    features[indicator_column] = (
        features[recent_columns].isna().all(axis=1).astype(int)
    )
    features = features[feature_columns]
    probabilities = model.predict_proba(features)
    positive_column = list(model.classes_).index(positive_label)

    scores = pd.DataFrame(
        {
            "CUSTOMER_ID": (
                customers["CUSTOMER_ID"]
                if "CUSTOMER_ID" in customers.columns
                else pd.Series(range(len(customers)), name="CUSTOMER_ID")
            ),
            "CHURN_RISK_SCORE": probabilities[:, positive_column],
        }
    )
    scores["PREDICTED_CHURN"] = (
        scores["CHURN_RISK_SCORE"] >= threshold
    ).astype(int)
    scores["DECISION_THRESHOLD"] = threshold
    scores["RECOMMENDED_ACTION"] = scores["PREDICTED_CHURN"].map(
        {
            1: "Send a personalized coupon",
            0: "Continue regular engagement",
        }
    )
    scores = scores.sort_values(
        "CHURN_RISK_SCORE", ascending=False, ignore_index=True
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output_path, index=False)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Score prepared customer features with the locally trained "
            "ChurnGuard model."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()

    scores = score_customers(args.input, args.output, args.model)
    print(f"Scored {len(scores):,} customers.")
    print(f"Saved scores to {args.output.resolve()}")
    print(f"Predicted churners: {int(scores['PREDICTED_CHURN'].sum()):,}")


if __name__ == "__main__":
    main()

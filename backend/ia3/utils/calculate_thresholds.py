import pandas as pd
import json
from compute_derived_metrics import compute_derived_metrics_ia3


def compute_derived_for_dataset_ia3(input_csv: str, output_csv: str):
    """
    Compute IA3-derived metrics for the entire aggregated dataset.
    Appends only newly derived columns to avoid duplication.
    """
    df = pd.read_csv(input_csv)

    derived_rows = []
    for _, row in df.iterrows():
        derived_rows.append(
            compute_derived_metrics_ia3(row.to_dict())
        )

    derived_df = pd.DataFrame(derived_rows)

    df = df.loc[:, ~df.columns.duplicated()]
    derived_df = derived_df.loc[:, ~derived_df.columns.duplicated()]

    derived_only = derived_df[[c for c in derived_df.columns if c not in df.columns]]

    full_df = pd.concat(
        [df.reset_index(drop=True), derived_only.reset_index(drop=True)],
        axis=1
    )

    full_df.to_csv(output_csv, index=False)
    print(f"[IA3] Derived metrics computed and saved to {output_csv}")

    return full_df


def calculate_thresholds_ia3(df: pd.DataFrame):
    """
    Calculate quantile-based thresholds for IA3.

    Covers:
    - Stability metrics (*_stability)
    - Gap metrics (*_gap)
    - Core performance metrics
    - Overfitting & efficiency signals
    """

    thresholds = {}
    df = df.loc[:, ~df.columns.duplicated()].copy()

    numeric_cols = [
        "train_loss_last_mean",
        "eval_loss_mean",
        "quality_score_mean",
        "training_efficiency_mean",
        "overfit_flag_mean",
        "loss_diff_train_eval",
        "robustness_score"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    stability_cols = [c for c in df.columns if c.endswith("_stability")]
    gap_cols = [c for c in df.columns if c.endswith("_gap")]

    quantiles = [0.25, 0.50, 0.75, 0.90]

    for col in stability_cols:
        for q in quantiles:
            thresholds[f"{col}_q{int(q * 100)}"] = float(df[col].quantile(q))

    for col in gap_cols:
        for q in quantiles:
            thresholds[f"{col}_q{int(q * 100)}"] = float(df[col].quantile(q))

    if "training_efficiency_mean" in df.columns:
        for q in quantiles:
            thresholds[f"training_efficiency_mean_q{int(q * 100)}"] = float(
                df["training_efficiency_mean"].quantile(q)
            )

    if "overfit_flag_mean" in df.columns:
        for q in quantiles:
            thresholds[f"overfit_flag_mean_q{int(q * 100)}"] = float(
                df["overfit_flag_mean"].quantile(q)
            )

    if "eval_loss_mean" in df.columns:
        thresholds["eval_loss_mean_q25"] = float(df["eval_loss_mean"].quantile(0.25))
        thresholds["eval_loss_mean_q50"] = float(df["eval_loss_mean"].quantile(0.50))

    if "quality_score_mean" in df.columns:
        thresholds["quality_score_mean_q75"] = float(df["quality_score_mean"].quantile(0.75))
        thresholds["quality_score_mean_q90"] = float(df["quality_score_mean"].quantile(0.90))

    if "robustness_score" in df.columns:
        thresholds["robustness_score_q75"] = float(df["robustness_score"].quantile(0.75))
        thresholds["robustness_score_q90"] = float(df["robustness_score"].quantile(0.90))

    return thresholds


if __name__ == "__main__":
    input_csv = "aggregated_ia3_results.csv"
    derived_csv = "aggregated_ia3_with_derived.csv"
    thresholds_json = "ia3_recommendation_thresholds.json"

    full_df = compute_derived_for_dataset_ia3(input_csv, derived_csv)
    thresholds = calculate_thresholds_ia3(full_df)

    with open(thresholds_json, "w") as f:
        json.dump(thresholds, f, indent=4)

    print(f"[IA3] Thresholds saved to {thresholds_json}")

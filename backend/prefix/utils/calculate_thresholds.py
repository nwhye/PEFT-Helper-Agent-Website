import pandas as pd
import json
from compute_derived_metrics import compute_derived_metrics


def compute_derived_for_dataset(input_csv: str, output_csv: str):
    """
    Compute derived metrics for the entire dataset and save as a new CSV.
    Works for ANY PEFT type (LoRA, Prefix, Prompt Tuning…)
    and avoids duplicated original columns.
    """

    df = pd.read_csv(input_csv)

    derived_rows = []
    for _, row in df.iterrows():
        predicted_meta = row.to_dict()
        derived_meta = compute_derived_metrics(predicted_meta)
        derived_rows.append(derived_meta)

    derived_df = pd.DataFrame(derived_rows)

    df = df.loc[:, ~df.columns.duplicated()]
    derived_df = derived_df.loc[:, ~derived_df.columns.duplicated()]

    derived_only = derived_df[
        [c for c in derived_df.columns if c not in df.columns]
    ]

    full_df = pd.concat([df.reset_index(drop=True),
                         derived_only.reset_index(drop=True)], axis=1)

    full_df.to_csv(output_csv, index=False)
    print(f"Derived metrics computed and saved to {output_csv}")
    return full_df


def calculate_thresholds(
        df: pd.DataFrame,
        stability_cols=None,
        gap_cols=None,
        efficiency_col="training_efficiency_mean",
        overfit_col="overfit_flag_mean"):
    """
    Automatically calculates quantile thresholds for recommendation engine.
    Supports ANY PEFT method (Prefix, LoRA, Prompt…)
    because it discovers stability/gap columns dynamically.
    """
    thresholds = {}

    df = df.loc[:, ~df.columns.duplicated()]

    numeric_cols = [
        efficiency_col,
        overfit_col,
        "eval_loss_mean",
        "quality_score_mean"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if stability_cols is None:
        stability_cols = [c for c in df.columns if c.endswith("_stability")]

    if gap_cols is None:
        gap_cols = [c for c in df.columns if c.endswith("_gap")]

    quantiles = [0.25, 0.5, 0.75, 0.90]

    for col in stability_cols:
        for q in quantiles:
            thresholds[f"{col}_q{int(q * 100)}"] = float(df[col].quantile(q))

    for col in gap_cols:
        for q in quantiles:
            thresholds[f"{col}_q{int(q * 100)}"] = float(df[col].quantile(q))

    for q in quantiles:
        thresholds[f"{efficiency_col}_q{int(q * 100)}"] = float(df[efficiency_col].quantile(q))
        thresholds[f"{overfit_col}_q{int(q * 100)}"] = float(df[overfit_col].quantile(q))

    if "eval_loss_mean" in df:
        thresholds["eval_loss_mean_q75"] = float(df["eval_loss_mean"].quantile(0.75))

    if "quality_score_mean" in df:
        thresholds["quality_score_mean_q25"] = float(df["quality_score_mean"].quantile(0.25))

    return thresholds


if __name__ == "__main__":
    input_csv = "aggregated_prefix_results.csv"
    derived_csv = "aggregated_with_derived.csv"
    thresholds_json = "recommendation_thresholds.json"

    full_df = compute_derived_for_dataset(input_csv, derived_csv)

    thresholds = calculate_thresholds(full_df)

    with open(thresholds_json, "w") as f:
        json.dump(thresholds, f, indent=4)

    print(f"Thresholds saved to {thresholds_json}")

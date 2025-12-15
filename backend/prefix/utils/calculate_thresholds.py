import pandas as pd
import json
from compute_derived_metrics import compute_derived_metrics


def compute_derived_for_dataset(input_csv: str, output_csv: str):
    """
    Compute derived metrics for the entire Prefix Tuning dataset
    and save as a new CSV.
    """
    df = pd.read_csv(input_csv)

    derived_rows = []
    for _, row in df.iterrows():
        derived_rows.append(compute_derived_metrics(row.to_dict()))

    derived_df = pd.DataFrame(derived_rows)

    full_df = pd.concat(
        [df.reset_index(drop=True), derived_df.reset_index(drop=True)],
        axis=1
    )

    full_df.to_csv(output_csv, index=False)
    print(f"Derived metrics computed and saved to {output_csv}")
    return full_df


def calculate_thresholds(
    df: pd.DataFrame,
    speed_col="training_speed",
    loss_slope_col="loss_slope",
    grad_norm_col="gradient_norm",
):
    """
    Calculate quantile-based thresholds aligned with training dynamics:
    - training speed
    - loss slope
    - gradient norm
    - stability & gap metrics

    Mirrors LoRA threshold logic for cross-PEFT consistency.
    """

    thresholds = {}
    df = df.loc[:, ~df.columns.duplicated()].copy()

    main_cols = [speed_col, loss_slope_col, grad_norm_col]
    for col in main_cols:
        if col in df.columns:
            df.loc[:, col] = pd.to_numeric(df[col], errors="coerce")

    stability_cols = [c for c in df.columns if c.endswith("_stability")]
    gap_cols = [c for c in df.columns if c.endswith("_gap")]

    quantiles = [0.25, 0.5, 0.75, 0.9]

    for col in stability_cols:
        for q in quantiles:
            thresholds[f"{col}_q{int(q * 100)}"] = float(df[col].quantile(q))

    for col in gap_cols:
        for q in quantiles:
            thresholds[f"{col}_q{int(q * 100)}"] = float(df[col].quantile(q))

    if speed_col in df.columns:
        for q in quantiles:
            thresholds[f"{speed_col}_q{int(q * 100)}"] = float(
                df[speed_col].quantile(q)
            )

    if loss_slope_col in df.columns:
        thresholds[f"{loss_slope_col}_q10"] = float(df[loss_slope_col].quantile(0.10))
        thresholds[f"{loss_slope_col}_q25"] = float(df[loss_slope_col].quantile(0.25))
        thresholds[f"{loss_slope_col}_q50"] = float(df[loss_slope_col].quantile(0.50))

    if grad_norm_col in df.columns:
        thresholds[f"{grad_norm_col}_q75"] = float(df[grad_norm_col].quantile(0.75))
        thresholds[f"{grad_norm_col}_q90"] = float(df[grad_norm_col].quantile(0.90))

    return thresholds


if __name__ == "__main__":
    input_csv = "aggregated_prefix_results.csv"
    derived_csv = "aggregated_prefix_with_derived.csv"
    thresholds_json = "recommendation_thresholds.json"

    full_df = compute_derived_for_dataset(input_csv, derived_csv)
    thresholds = calculate_thresholds(full_df)

    with open(thresholds_json, "w") as f:
        json.dump(thresholds, f, indent=4)

    print(f"Derived dataset saved to {derived_csv}")
    print(f"Thresholds saved to {thresholds_json}")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import joblib


# Aggregation and computation


def aggregate_dataset(input_csv, output_csv=None, eps=1e-8):
    df = pd.read_csv(input_csv)
    df["seed"] = df["seed"].astype(str)

    hp_cols = [
        "model_name", "peft", "lora_r", "lora_alpha", "lora_dropout",
        "target_modules", "layers_tuned", "learning_rate", "batch_size", "epoch"
    ]

    metric_cols = [
        "train_loss_first", "train_loss_last", "loss_slope", "gradient_norm_mean",
        "learning_rate_final", "eval_loss", "rouge1", "rouge2", "rougeL", "bleu",
        "bert_score", "exact_match", "quality_score", "overfit_flag", "training_efficiency"
    ]

    # Aggregate across seeds
    agg_funcs = {m: ["mean", "std", "min", "max"] for m in metric_cols}
    aggregated = df.groupby(hp_cols).agg(agg_funcs)
    aggregated.columns = [f"{metric}_{stat}" for metric, stat in aggregated.columns]
    aggregated = aggregated.reset_index()

    # Derived metrics
    if output_csv:
        aggregated.to_csv(output_csv, index=False)
        print(f"Aggregated dataset saved to {output_csv}")

    return aggregated, hp_cols


# Training


def train_helper_model(aggregated_df, hp_cols, save_path="peft_helper_model.pkl", scaler_path="peft_scalers.pkl"):
    """
    Train a helper model to predict raw numeric metrics only.
    Derived metrics will be computed later.
    """
    X_cols = ["lora_r", "lora_alpha", "lora_dropout", "learning_rate", "batch_size"]

    exclude_derived = ["loss_stability", "grad_stability", "eval_stability", "robustness_score"]
    y_cols = [c for c in aggregated_df.columns if c not in hp_cols + exclude_derived and aggregated_df[c].dtype != 'object']

    X = aggregated_df[X_cols].copy()
    y = aggregated_df[y_cols].copy()

    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_scaled = X_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y)

    joblib.dump({"X_scaler": X_scaler, "y_scaler": y_scaler, "y_cols": y_cols}, scaler_path)
    print(f"Scalers and y_cols saved to {scaler_path}")

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    helper_model = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=42))
    helper_model.fit(X_train, y_train)

    r2 = helper_model.score(X_test, y_test)
    print(f"Helper model R² score: {r2:.4f}")

    joblib.dump(helper_model, save_path)
    print(f"Helper model saved to {save_path}")

    return helper_model, X_cols, y_cols


if __name__ == "__main__":
    input_csv = "./flan_lora_grid_with_epochs.csv"
    output_csv = "./aggregated_peft_results_exp.csv"

    aggregated_df, hp_cols = aggregate_dataset(input_csv, output_csv=output_csv)
    helper_model, X_cols, y_cols = train_helper_model(aggregated_df, hp_cols,
                                                      save_path="peft_helper_model.pkl",
                                                      scaler_path="peft_scalers.pkl")

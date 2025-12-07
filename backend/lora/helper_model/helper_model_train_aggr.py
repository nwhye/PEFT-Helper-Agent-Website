import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import joblib


def preprocess_categorical(df):
    df["target_modules"] = df["target_modules"].apply(eval)

    module_vocab = sorted({m for lst in df["target_modules"] for m in lst})
    for m in module_vocab:
        df[f"tm_{m}"] = df["target_modules"].apply(lambda lst: 1 if m in lst else 0)
    df["layers_tuned"] = df["layers_tuned"].astype("category").cat.codes

    return df, module_vocab


def aggregate_dataset(input_csv, output_csv=None):
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

    agg_funcs = {m: ["mean", "std", "min", "max"] for m in metric_cols}
    aggregated = df.groupby(hp_cols).agg(agg_funcs)
    aggregated.columns = [f"{metric}_{stat}" for metric, stat in aggregated.columns]
    aggregated = aggregated.reset_index()

    aggregated, module_vocab = preprocess_categorical(aggregated)

    if output_csv:
        aggregated.to_csv(output_csv, index=False)
        print(f"Aggregated dataset saved to {output_csv}")

    return aggregated, hp_cols, module_vocab


def train_helper_model(df, hp_cols, module_vocab,
                       save_path="peft_helper_model.pkl",
                       scaler_path="peft_scalers.pkl"):
    X_cols = (
        ["lora_r", "lora_alpha", "lora_dropout", "learning_rate", "batch_size", "epoch"] +
        ["layers_tuned"] +
        [f"tm_{m}" for m in module_vocab]
    )

    exclude_derived = ["loss_stability", "grad_stability", "eval_stability", "robustness_score"]
    y_cols = [c for c in df.columns if c not in hp_cols + exclude_derived and df[c].dtype != "object"]

    X = df[X_cols].copy()
    y = df[y_cols].copy()

    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_scaled = X_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y)

    joblib.dump({
        "X_scaler": X_scaler,
        "y_scaler": y_scaler,
        "y_cols": y_cols,
        "X_cols": X_cols,
        "module_vocab": module_vocab
    }, scaler_path)

    print(f"Scalers saved to {scaler_path}")

    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=42))
    model.fit(X_train, Y_train)

    print("Helper Model R²:", model.score(X_test, Y_test))

    joblib.dump(model, save_path)
    print(f"Model saved to {save_path}")

    return model, X_cols, y_cols


if __name__ == "__main__":
    input_csv = "./flan_lora_grid_with_epochs.csv"
    output_csv = "./aggregated_peft_results_exp.csv"
    aggregated_df, hp_cols, module_vocab = aggregate_dataset(input_csv, output_csv=output_csv)
    helper_model, X_cols, y_cols = train_helper_model(
        df=aggregated_df,
        hp_cols=hp_cols,
        module_vocab=module_vocab,
        save_path="peft_helper_model.pkl",
        scaler_path="peft_scalers.pkl"
    )


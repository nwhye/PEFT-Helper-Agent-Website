import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import joblib


def preprocess_categorical(df):
    # Encode layers_tuned
    df["layers_tuned"] = df["layers_tuned"].astype("category").cat.codes
    # Encode boolean prefix_projection
    df["prefix_projection"] = df["prefix_projection"].astype(int)
    return df


def aggregate_dataset(input_csv, output_csv=None):
    df = pd.read_csv(input_csv)
    df["seed"] = df["seed"].astype(str)

    df["prefix_hidden"] = df["prefix_hidden"].replace("", 0).fillna(0).astype(float)

    hp_cols = [
        "model_name", "peft", "prefix_length", "layers_tuned",
        "prefix_hidden", "prefix_projection", "learning_rate", "batch_size", "epoch"
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

    aggregated = preprocess_categorical(aggregated)

    if output_csv:
        aggregated.to_csv(output_csv, index=False)
        print(f"Aggregated dataset saved to {output_csv}")

    return aggregated, hp_cols


def train_helper_model(df, hp_cols,
                       save_path="prefix_helper_model.pkl",
                       scaler_path="prefix_scalers.pkl"):
    X_cols = [
        "prefix_length", "prefix_hidden", "prefix_projection",
        "learning_rate", "batch_size", "epoch", "layers_tuned"
    ]
    X = df[X_cols].copy()


    exclude_derived = ["seed", "model_name", "peft"]
    y_cols = [c for c in df.columns if c not in hp_cols + exclude_derived and df[c].dtype != "object"]
    y = df[y_cols].copy()

    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    X_scaled = X_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y)

    joblib.dump({
        "X_scaler": X_scaler,
        "y_scaler": y_scaler,
        "y_cols": y_cols,
        "X_cols": X_cols
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
    input_csv = "./flan_prefix_grid_with_seed.csv"
    output_csv = "./aggregated_prefix_results.csv"

    aggregated_df, hp_cols = aggregate_dataset(input_csv, output_csv=output_csv)
    helper_model, X_cols, y_cols = train_helper_model(
        df=aggregated_df,
        hp_cols=hp_cols,
        save_path="prefix_helper_model.pkl",
        scaler_path="prefix_scalers.pkl"
    )
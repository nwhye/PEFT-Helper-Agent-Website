import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import ast
import joblib


import re

MODEL_LAYERS = {
    "google/flan-t5-base": 12,
    "google/flan-t5-large": 24,
    "google/flan-t5-xl": 24,
    "google/flan-t5-xxl": 24,
}


def normalize_layers_tuned(df, model_col="model_name"):
    def resolve(row):
        val = str(row["layers_tuned"]).lower()
        if val == "all":
            return MODEL_LAYERS.get(row[model_col], 12)
        m = re.search(r"_(\d+)$", val)
        if m:
            return int(m.group(1))
        try:
            return int(val)
        except:
            return 1
    df["layers_tuned"] = df.apply(resolve, axis=1)
    return df



def expand_target_modules(df, col="target_modules"):
    df[col] = df[col].apply(ast.literal_eval)

    module_vocab = sorted({m for mods in df[col] for m in mods})
    for m in module_vocab:
        df[f"tm_{m}"] = df[col].apply(lambda mods: int(m in mods))

    return df, module_vocab


def aggregate_dataset(input_csv, output_csv=None):
    df = pd.read_csv(input_csv)
    df["seed"] = df["seed"].astype(str)

    df = normalize_layers_tuned(df)

    df, module_vocab = expand_target_modules(df)

    hp_cols = [
        "model_name",
        "peft",
        "layers_tuned",
        "learning_rate",
        "batch_size",
        "epoch",
    ] + [f"tm_{m}" for m in module_vocab]

    metric_cols = [
        "train_loss_first", "train_loss_last", "loss_slope",
        "gradient_norm_mean", "learning_rate_final", "eval_loss",
        "rouge1", "rouge2", "rougeL", "bleu", "bert_score",
        "exact_match", "quality_score", "overfit_flag", "training_efficiency"
    ]

    agg_funcs = {m: ["mean", "std", "min", "max"] for m in metric_cols}
    aggregated = df.groupby(hp_cols).agg(agg_funcs)
    aggregated.columns = [f"{m}_{s}" for m, s in aggregated.columns]
    aggregated = aggregated.reset_index()

    if output_csv:
        aggregated.to_csv(output_csv, index=False)
        print(f"Aggregated dataset saved to {output_csv}")

    return aggregated, hp_cols, module_vocab


def train_helper_model(
    df,
    hp_cols,
    module_vocab,
    save_path="ia3_helper_model.pkl",
    scaler_path="ia3_scalers.pkl"
):
    X_cols = (
        ["layers_tuned", "learning_rate", "batch_size", "epoch"]
        + [f"tm_{m}" for m in module_vocab]
    )

    X = df[X_cols].copy()

    exclude = ["seed", "model_name", "peft"]
    y_cols = [c for c in df.columns if c not in hp_cols + exclude]
    y = df[y_cols].copy()

    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_scaled = X_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y)

    joblib.dump(
        {
            "X_scaler": X_scaler,
            "y_scaler": y_scaler,
            "X_cols": X_cols,
            "y_cols": y_cols,
        },
        scaler_path,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )

    model = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=300, random_state=42)
    )
    model.fit(X_train, y_train)

    print("Helper model R²:", model.score(X_test, y_test))

    joblib.dump(model, save_path)
    print(f"Model saved to {save_path}")

    return model, X_cols, y_cols


if __name__ == "__main__":
    input_csv = "./flan_ia3_grid_with_seed.csv"
    output_csv = "./aggregated_ia3_results.csv"

    aggregated_df, hp_cols, module_vocab = aggregate_dataset(
        input_csv, output_csv=output_csv
    )

    helper_model, X_cols, y_cols = train_helper_model(
        aggregated_df,
        hp_cols,
        module_vocab,
        save_path="ia3_helper_model.pkl",
        scaler_path="ia3_scalers.pkl",
    )

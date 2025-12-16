from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
import pandas as pd
import joblib
from typing import Dict
from utils.compute_derived_metrics import compute_derived_metrics_ia3
from rules_engine import generate_ia3_recommendations


helper_model = joblib.load("helper_model/ia3_helper_model.pkl")
helper_scalers = joblib.load("helper_model/ia3_scalers.pkl")

X_scaler_helper = helper_scalers["X_scaler"]
y_scaler_helper = helper_scalers["y_scaler"]
helper_y_cols = helper_scalers["y_cols"]
helper_X_cols = helper_scalers["X_cols"]

X_scaler_main = joblib.load("ia3_input_scaler.pkl")
y_scaler_main = joblib.load("ia3_target_scaler.pkl")

input_dim = len(X_scaler_main.feature_names_in_)
output_dim = len(y_scaler_main.scale_)


class IA3RecommendationModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.net(x)

model = IA3RecommendationModel(input_dim=input_dim, output_dim=output_dim)
model.load_state_dict(torch.load("ia3_recommendation_model.pt"))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


app = FastAPI(title="IA3 Hyperparameter Recommendation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess_input(config: Dict, scaler_cols: list, encode_layers: bool = True):
    """Generic preprocessing for both helper and main models."""
    df = pd.DataFrame([config])

    if isinstance(df["target_modules"].iloc[0], str):
        df["target_modules"] = df["target_modules"].apply(eval)
    elif not isinstance(df["target_modules"].iloc[0], (list, tuple)):
        df["target_modules"] = df["target_modules"].apply(lambda x: [])

    ia3_modules = ["q", "k", "v", "o"]
    for m in ia3_modules:
        df[f"tm_{m}"] = df["target_modules"].apply(lambda lst: int(m in lst) if isinstance(lst, (list, tuple)) else 0)

    if encode_layers and "layers_tuned" in df.columns:
        layers_categories = {cat: idx for idx, cat in enumerate(sorted(df["layers_tuned"].unique()))}
        df["layers_tuned"] = df["layers_tuned"].map(layers_categories)

    for col in scaler_cols:
        if col not in df.columns:
            df[col] = 0

    df = df[scaler_cols].copy()
    return df

@app.post("/predict/")
def predict_ia3(config: Dict):

    helper_input = preprocess_input(config, helper_X_cols)
    X_scaled_helper = X_scaler_helper.transform(helper_input)
    y_pred_scaled_helper = helper_model.predict(X_scaled_helper)
    y_pred_helper = y_scaler_helper.inverse_transform(y_pred_scaled_helper)
    predicted_raw = {col: float(val) for col, val in zip(helper_y_cols, y_pred_helper[0])}

    derived_meta = compute_derived_metrics_ia3(predicted_raw)

    layers_map = {"all": 0, "encoder_last_3": 1, "decoder_last_3": 2}
    layers_tuned_val = config.get("layers_tuned", "all")
    derived_meta["layers_tuned"] = layers_map.get(layers_tuned_val, 0)

    full_input = {**config, **derived_meta}
    df_main = preprocess_input(full_input, X_scaler_main.feature_names_in_)
    X_main_scaled = X_scaler_main.transform(df_main)
    X_tensor = torch.tensor(X_main_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        preds_scaled = model(X_tensor).cpu().numpy()[0]

    preds = y_scaler_main.inverse_transform([preds_scaled])[0]
    predicted_metrics = {
        "training_speed": float(preds[0]),
        "loss_slope": float(preds[1]),
        "gradient_norm": float(preds[2])
    }

    full_meta = {**config, **derived_meta}
    recommendations = generate_ia3_recommendations(full_meta)

    return {
        "predicted_metrics": predicted_metrics,
        "derived_metrics": full_meta,
        "final_recommendations": recommendations,
    }

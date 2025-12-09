from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
import pandas as pd
import joblib
from typing import Dict

from flan_prefix_predict_model import RecommendationModel
from utils.compute_derived_metrics import compute_derived_metrics
from rules_engine import generate_prefix_recommendations


X_scaler_main = joblib.load("prefix_input_scaler.pkl")
y_scaler_main = joblib.load("prefix_target_scaler.pkl")

input_dim = len(X_scaler_main.feature_names_in_)
output_dim = 3  # overfit_mean, training_efficiency_mean, loss_difain_eval

model = RecommendationModel(input_dim=input_dim, output_dim=output_dim)
model.load_state_dict(torch.load("prefix_recommendation_model.pt"))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


helper_model = joblib.load("helper_model/prefix_helper_model.pkl")
helper_scalers = joblib.load("helper_model/prefix_scalers.pkl")

X_scaler_helper = helper_scalers["X_scaler"]
y_scaler_helper = helper_scalers["y_scaler"]
helper_y_cols = helper_scalers["y_cols"]
helper_X_cols = helper_scalers["X_cols"]
prefix_vocab = helper_scalers["prefix_vocab"]


def preprocess_helper_input(config: Dict):
    df = pd.DataFrame([config])

    for key, vocab in prefix_vocab.items():
        for v in vocab:
            df[f"{key}_{v}"] = (df[key] == v).astype(int)

    return df[helper_X_cols].copy()


app = FastAPI(title="Prefix Tuning Hyperparameter Recommendation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict/")
def predict_prefix(config: Dict):
    # Preprocess helper input and scale
    helper_input = preprocess_helper_input(config)
    X_scaled_helper = X_scaler_helper.transform(helper_input)

    y_pred_scaled_helper = helper_model.predict(X_scaled_helper)
    y_pred_helper = y_scaler_helper.inverse_transform(y_pred_scaled_helper)

    predicted_raw = {
        col: float(val) for col, val in zip(helper_y_cols, y_pred_helper[0])
    }

    full_meta = compute_derived_metrics(predicted_raw)

    full_input = {**config, **full_meta}
    df_main = pd.DataFrame([full_input])

    X_main_scaled = X_scaler_main.transform(df_main[X_scaler_main.feature_names_in_])
    X_tensor = torch.tensor(X_main_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        preds_scaled = model(X_tensor).cpu().numpy()[0]

    preds = y_scaler_main.inverse_transform([preds_scaled])[0]

    recommendations = generate_prefix_recommendations(full_meta, preds)

    return {
        "predicted_overfit": float(preds[0]),
        "predicted_efficiency": float(preds[1]) / 100000,
        "predicted_generalization_gap": float(preds[2]),
        "predicted_metrics": full_meta,
        "final_recommendations": recommendations
    }

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
output_dim = 3

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


app = FastAPI(title="Prefix Tuning Hyperparameter Recommendation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess_helper_input(config: Dict):
    df = pd.DataFrame([config])

    for col in helper_X_cols:
        if col not in df.columns:
            if col == "layers_tuned":
                df[col] = 0
            elif col == "prefix_hidden":
                df[col] = 64
            elif col == "prefix_projection":
                df[col] = 1
            else:
                df[col] = 0

    return df[helper_X_cols].astype(float)


@app.post("/predict/")
def predict_prefix(config: Dict):

    helper_input = preprocess_helper_input(config)
    X_scaled_helper = X_scaler_helper.transform(helper_input)

    y_pred_scaled_helper = helper_model.predict(X_scaled_helper)
    y_pred_helper = y_scaler_helper.inverse_transform(y_pred_scaled_helper)

    predicted_raw = {
        col: float(val)
        for col, val in zip(helper_y_cols, y_pred_helper[0])
    }

    derived_meta = compute_derived_metrics(predicted_raw)

    full_input = {**config, **derived_meta}
    df_main = pd.DataFrame([full_input])

    X_main_scaled = X_scaler_main.transform(
        df_main[X_scaler_main.feature_names_in_]
    )

    X_tensor = torch.tensor(X_main_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        preds_scaled = model(X_tensor).cpu().numpy()[0]

    preds = y_scaler_main.inverse_transform([preds_scaled])[0]

    training_speed, loss_slope, gradient_norm = preds

    full_meta = {**config, **derived_meta}

    recommendations = generate_prefix_recommendations(
        full_meta,
        {
            "training_speed": training_speed,
            "loss_slope": loss_slope,
            "gradient_norm": gradient_norm,
        }
    )

    return {
        "training_speed": float(training_speed),
        "loss_slope": float(loss_slope),
        "gradient_norm": float(gradient_norm),
        "derived_metrics": full_meta,
        "final_recommendations": recommendations,
    }

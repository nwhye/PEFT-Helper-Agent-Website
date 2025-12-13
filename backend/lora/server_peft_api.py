from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
import pandas as pd
import joblib
from typing import Dict
from flan_lora_predict_model import RecommendationModel
from utils.compute_derived_metrics import compute_derived_metrics
from rules_engine import generate_full_recommendations

# -----------------------------
# Load scalers and model
# -----------------------------
X_scaler_main = joblib.load("peft_input_scaler.pkl")
y_scaler_main = joblib.load("peft_target_scaler.pkl")

input_dim = len(X_scaler_main.feature_names_in_)
output_dim = 3

model = RecommendationModel(input_dim=input_dim, output_dim=output_dim)
model.load_state_dict(torch.load("peft_recommendation_model.pt"))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -----------------------------
# Helper model for derived metrics
# -----------------------------
helper_model = joblib.load("helper_model/peft_helper_model.pkl")
helper_scalers = joblib.load("helper_model/peft_scalers.pkl")

X_scaler_helper = helper_scalers["X_scaler"]
y_scaler_helper = helper_scalers["y_scaler"]
helper_y_cols = helper_scalers["y_cols"]
helper_X_cols = helper_scalers["X_cols"]
module_vocab = helper_scalers["module_vocab"]

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="PEFT Hyperparameter Recommendation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Preprocess user input
# -----------------------------
def preprocess_helper_input(config: Dict):
    df = pd.DataFrame([config])

    # Convert target_modules from string if needed
    if isinstance(df["target_modules"].iloc[0], str):
        df["target_modules"] = df["target_modules"].apply(eval)

    # Encode target_modules as binary features
    for m in module_vocab:
        df[f"tm_{m}"] = df["target_modules"].apply(lambda lst: int(m in lst))

    # Encode layers_tuned
    layers_categories = {cat: idx for idx, cat in enumerate(sorted(df["layers_tuned"].unique()))}
    df["layers_tuned"] = df["layers_tuned"].map(layers_categories)

    return df[helper_X_cols].copy()

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict/")
def predict_peft(config: Dict):

    # 1️⃣ Prepare helper input to compute derived metrics
    helper_input = preprocess_helper_input(config)
    X_scaled_helper = X_scaler_helper.transform(helper_input)

    # 2️⃣ Predict raw helper metrics
    y_pred_scaled_helper = helper_model.predict(X_scaled_helper)
    y_pred_helper = y_scaler_helper.inverse_transform(y_pred_scaled_helper)

    predicted_raw = {col: float(val) for col, val in zip(helper_y_cols, y_pred_helper[0])}

    # 3️⃣ Compute derived metrics (training_speed, loss_slope, gradient_norm, etc.)
    derived_meta = compute_derived_metrics(predicted_raw)

    # 4️⃣ Combine user hyperparameters and derived metrics for main model input
    full_input = {**config, **derived_meta}
    df_main = pd.DataFrame([full_input])
    X_main_scaled = X_scaler_main.transform(df_main[X_scaler_main.feature_names_in_])
    X_tensor = torch.tensor(X_main_scaled, dtype=torch.float32).to(device)

    # 5️⃣ Predict main targets
    with torch.no_grad():
        preds_scaled = model(X_tensor).cpu().numpy()[0]

    preds = y_scaler_main.inverse_transform([preds_scaled])[0]
    training_speed, loss_slope, gradient_norm = preds

    # 6️⃣ Generate final recommendations
    recommendations = generate_full_recommendations(
        derived_meta,
        {
            "training_speed": training_speed,
            "loss_slope": loss_slope,
            "gradient_norm": gradient_norm,
        }
    )

    # 7️⃣ Return response
    return {
        "training_speed": float(training_speed),
        "loss_slope": float(loss_slope),
        "gradient_norm": float(gradient_norm),
        "derived_metrics": derived_meta,
        "final_recommendations": recommendations,
    }

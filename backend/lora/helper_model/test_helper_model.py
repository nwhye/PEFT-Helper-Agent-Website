import pandas as pd
import joblib
import numpy as np

# -------------------------------
# Load the helper model and scalers
# -------------------------------
helper_model = joblib.load("peft_helper_model.pkl")
scalers = joblib.load("peft_scalers.pkl")
X_scaler = scalers["X_scaler"]
y_scaler = scalers["y_scaler"]
y_cols = scalers["y_cols"]

print("Helper model and scalers loaded successfully.")

# -------------------------------
# Define sample hyperparameter configurations
# -------------------------------
sample_inputs = [
    {"lora_r": 4, "lora_alpha": 8, "lora_dropout": 0.05, "learning_rate": 1e-6, "batch_size": 4},
    {"lora_r": 8, "lora_alpha": 16, "lora_dropout": 0.1, "learning_rate": 5e-6, "batch_size": 8},
    {"lora_r": 2, "lora_alpha": 4, "lora_dropout": 0.0, "learning_rate": 1e-5, "batch_size": 2}
]

# Convert to DataFrame
X_test = pd.DataFrame(sample_inputs)

# -------------------------------
# Scale inputs before prediction
# -------------------------------
X_test_scaled = X_scaler.transform(X_test)

# -------------------------------
# Predict derived numeric metrics (scaled)
# -------------------------------
predicted_scaled = helper_model.predict(X_test_scaled)

# -------------------------------
# Inverse transform to original ranges
# -------------------------------
predicted_metrics = y_scaler.inverse_transform(predicted_scaled)

# -------------------------------
# Print results mapped to metric names
# -------------------------------
for i, config in enumerate(sample_inputs):
    print(f"\n--- Hyperparameters set {i+1} ---")
    for k, v in config.items():
        print(f"{k}: {v}")
    print("\nPredicted derived metrics:")
    for col_name, value in zip(y_cols, predicted_metrics[i]):
        # Clip negatives to zero for metrics that can't be negative
        if any(substr in col_name for substr in ["loss", "training_efficiency", "gradient", "dropout"]):
            value = max(value, 0)
        print(f"{col_name}: {value:.6f}")

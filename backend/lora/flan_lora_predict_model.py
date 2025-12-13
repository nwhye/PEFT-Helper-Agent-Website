import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib

from utils.compute_derived_metrics import compute_derived_metrics


class PEFTDataset(Dataset):
    def __init__(self, df, input_cols, target_cols):
        self.X = df[input_cols].values.astype(np.float32)
        self.y = df[target_cols].values.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class RecommendationModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(256, 128)):
        super().__init__()
        layers = []
        last_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h

        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


df = pd.read_csv("helper_model/aggregated_peft_results_exp.csv")
derived_rows = [compute_derived_metrics(row) for _, row in df.iterrows()]
df_derived = pd.DataFrame(derived_rows)


target_cols = ["training_speed", "loss_slope", "gradient_norm"]
df_derived["training_speed"] = df_derived["training_efficiency_mean"]
df_derived["loss_slope"] = df_derived["loss_slope_mean"]
df_derived["gradient_norm"] = df_derived["gradient_norm_mean_mean"]


hyperparameter_cols = ["batch_size", "learning_rate", "lora_r", "lora_alpha", "lora_dropout", "layers_tuned"]


for col in hyperparameter_cols:
    if col not in df_derived.columns:
        df_derived[col] = 0


if df_derived["layers_tuned"].dtype == object:
    layers_map = {cat: idx for idx, cat in enumerate(sorted(df_derived["layers_tuned"].unique()))}
    df_derived["layers_tuned"] = df_derived["layers_tuned"].map(layers_map)


module_vocab = ["q", "v", "o"]
if "target_modules" in df_derived.columns:
    df_derived["target_modules"] = df_derived["target_modules"].apply(eval)
    for m in module_vocab:
        df_derived[f"tm_{m}"] = df_derived["target_modules"].apply(lambda lst: 1 if m in lst else 0)


input_cols = [
    c for c in df_derived.select_dtypes(include=[float, int]).columns
    if c not in target_cols
]

df_derived = df_derived.dropna(subset=input_cols + target_cols)


scaler_X = StandardScaler()
df_derived[input_cols] = scaler_X.fit_transform(df_derived[input_cols])
joblib.dump(scaler_X, "peft_input_scaler.pkl")

scaler_y = StandardScaler()
df_derived[target_cols] = scaler_y.fit_transform(df_derived[target_cols])
joblib.dump(scaler_y, "peft_target_scaler.pkl")


dataset = PEFTDataset(df_derived, input_cols, target_cols)
loader = DataLoader(dataset, batch_size=32, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RecommendationModel(input_dim=len(input_cols), output_dim=len(target_cols)).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
epochs = 50
max_grad_norm = 1.0


model.train()
for epoch in range(epochs):
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)

    avg_loss = total_loss / len(loader.dataset)
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")


torch.save(model.state_dict(), "peft_recommendation_model.pt")
print("Model trained with new targets: training_speed, loss_slope, gradient_norm")
print("User hyperparameters now included in input features: batch_size, learning_rate, lora_r, lora_alpha, lora_dropout, layers_tuned, target_modules")

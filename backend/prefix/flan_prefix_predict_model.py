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
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 128]):
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


df = pd.read_csv("helper_model/aggregated_prefix_results.csv")
df_derived = pd.DataFrame([compute_derived_metrics(row) for _, row in df.iterrows()])

target_cols = ["overfit_mean", "training_efficiency_mean", "loss_diff_train_eval"]
input_cols = [c for c in df_derived.select_dtypes(include=[float, int]).columns if c not in target_cols]


constant_targets = [c for c in target_cols if df_derived[c].std() == 0]
target_cols = [c for c in target_cols if c not in constant_targets]


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
optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)
max_grad_norm = 1.0
epochs = 50

model.train()
for epoch in range(epochs):
    total_loss = 0
    batch_count = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        batch_count += X_batch.size(0)

    avg_loss = total_loss / batch_count
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

torch.save(model.state_dict(), "./peft_recommendation_model.pt")
print("Pre-trained model and scalers saved.")

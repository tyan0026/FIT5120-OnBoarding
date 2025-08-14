# train_model.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.neighbors import BallTree
import joblib

# ==== 1. 数据集类 ====
class ParkingDataset(Dataset):
    def __init__(self, df, label_encoder, scaler):
        self.df = df.copy()

        self.df["Status_Timestamp"] = pd.to_datetime(self.df["Status_Timestamp"], errors='coerce')
        self.df["hour"] = self.df["Status_Timestamp"].dt.hour
        self.df["dayofweek"] = self.df["Status_Timestamp"].dt.dayofweek

        self.df[["lat", "lon"]] = self.df["Location"].str.split(",", expand=True).astype(float)

        self.df["Zone_Number"] = self.df["Zone_Number"].fillna("unknown").astype(str)
        self.df["Zone_Number_enc"] = label_encoder.transform(self.df["Zone_Number"])

        self.df["is_free"] = (self.df["Status_Description"].str.lower() == "unoccupied").astype(int)

        features = ["hour", "dayofweek", "Zone_Number_enc", "lat", "lon"]
        self.X = scaler.transform(self.df[features])
        self.y = self.df["is_free"].values.astype(np.float32)

        self.kerbside_id = self.df["KerbsideID"].values
        self.latlon = self.df[["lat", "lon"]].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)


# ==== 2. 模型 ====
class ParkingModel(nn.Module):
    def __init__(self, input_dim):
        super(ParkingModel, self).__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = self.backbone(x)
        return torch.sigmoid(self.fc(x))


# ==== 3. 读取和预处理数据 ====
print("Loading dataset...")
df = pd.read_csv("on-street-parking-bay-sensors.csv")

df["Status_Timestamp"] = df["Status_Timestamp"].astype(str).str.strip()
df["Status_Timestamp"] = pd.to_datetime(
    df["Status_Timestamp"], errors="coerce", utc=True).dt.tz_convert(None)

df["hour"] = df["Status_Timestamp"].dt.hour
df["dayofweek"] = df["Status_Timestamp"].dt.dayofweek

df[["lat", "lon"]] = df["Location"].str.split(",", expand=True).astype(float)
df["Zone_Number"] = df["Zone_Number"].fillna("unknown").astype(str)

def clean_zone_number(z):
    try:
        f = float(z)
        if f.is_integer():
            return str(int(f))
        else:
            return str(f)
    except:
        return z

df["Zone_Number"] = df["Zone_Number"].apply(clean_zone_number)

label_encoder = LabelEncoder()
df["Zone_Number_enc"] = label_encoder.fit_transform(df["Zone_Number"])

features = ["hour", "dayofweek", "Zone_Number_enc", "lat", "lon"]
scaler = StandardScaler()
scaler.fit(df[features])

# 4. split for pretrain & finetune
recent_cutoff = df["Status_Timestamp"].max() - pd.DateOffset(months=2)
recent_df = df[df["Status_Timestamp"] >= recent_cutoff].copy()
pretrain_df = df.copy()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ParkingModel(input_dim=len(features)).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).unsqueeze(1)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()
    return val_loss / len(loader)

def compute_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).unsqueeze(1)
            outputs = model(X_batch)
            preds = (outputs >= 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
    return correct / total

# Pretrain
print("Starting pretraining...")
train_df_pre, val_df_pre = train_test_split(pretrain_df, test_size=0.2, random_state=42)
train_dataset_pre = ParkingDataset(train_df_pre, label_encoder, scaler)
val_dataset_pre = ParkingDataset(val_df_pre, label_encoder, scaler)
train_loader_pre = DataLoader(train_dataset_pre, batch_size=64, shuffle=True)
val_loader_pre = DataLoader(val_dataset_pre, batch_size=64, shuffle=False)

pretrain_epochs = 500
for epoch in range(pretrain_epochs):
    train_loss = train_one_epoch(model, train_loader_pre, criterion, optimizer, device)
    val_loss = eval_one_epoch(model, val_loader_pre, criterion, device)
    if (epoch + 1) % 50 == 0:
        print(f"Pretrain Epoch {epoch+1}/{pretrain_epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

# Fine-tune
print("Starting fine-tuning on recent data...")
train_df_fine, val_df_fine = train_test_split(recent_df, test_size=0.2, random_state=42)
train_dataset_fine = ParkingDataset(train_df_fine, label_encoder, scaler)
val_dataset_fine = ParkingDataset(val_df_fine, label_encoder, scaler)
train_loader_fine = DataLoader(train_dataset_fine, batch_size=64, shuffle=True)
val_loader_fine = DataLoader(val_dataset_fine, batch_size=64, shuffle=False)

finetune_epochs = 10
for epoch in range(finetune_epochs):
    train_loss = train_one_epoch(model, train_loader_fine, criterion, optimizer, device)
    val_loss = eval_one_epoch(model, val_loader_fine, criterion, device)
    val_acc = compute_accuracy(model, val_loader_fine, device)
    print(f"Finetune {epoch+1}/{finetune_epochs} loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

# Save artifacts
print("Saving model and preprocessors...")
torch.save(model.state_dict(), "parking_model_finetuned.pth")
joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")
print("All done.")

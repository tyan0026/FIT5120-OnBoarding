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

# ==== 4. 分割数据用于预训练和微调 ====
recent_cutoff = df["Status_Timestamp"].max() - pd.DateOffset(months=2)
recent_df = df[df["Status_Timestamp"] >= recent_cutoff].copy()
pretrain_df = df.copy()

# ==== 5. 设备和模型初始化 ====
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


# ==== 6. 预训练阶段 ====
train_df_pre, val_df_pre = train_test_split(pretrain_df, test_size=0.2, random_state=42)
train_dataset_pre = ParkingDataset(train_df_pre, label_encoder, scaler)
val_dataset_pre = ParkingDataset(val_df_pre, label_encoder, scaler)
train_loader_pre = DataLoader(train_dataset_pre, batch_size=64, shuffle=True)
val_loader_pre = DataLoader(val_dataset_pre, batch_size=64, shuffle=False)

# print("=== Pretraining on all historical data ===")
pretrain_epochs = 500
for epoch in range(pretrain_epochs):
    train_loss = train_one_epoch(model, train_loader_pre, criterion, optimizer, device)
    val_loss = eval_one_epoch(model, val_loader_pre, criterion, device)
    # print(f"Pretrain Epoch [{epoch+1}/{pretrain_epochs}] Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")

# ==== 7. 微调阶段 ====
train_df_fine, val_df_fine = train_test_split(recent_df, test_size=0.2, random_state=42)
train_dataset_fine = ParkingDataset(train_df_fine, label_encoder, scaler)
val_dataset_fine = ParkingDataset(val_df_fine, label_encoder, scaler)
train_loader_fine = DataLoader(train_dataset_fine, batch_size=64, shuffle=True)
val_loader_fine = DataLoader(val_dataset_fine, batch_size=64, shuffle=False)

# print("=== Fine-tuning on recent 2 months data ===")
finetune_epochs = 10
for epoch in range(finetune_epochs):
    train_loss = train_one_epoch(model, train_loader_fine, criterion, optimizer, device)
    val_loss = eval_one_epoch(model, val_loader_fine, criterion, device)
    val_acc = compute_accuracy(model, val_loader_fine, device)
    # print(f"Finetune Epoch [{epoch+1}/{finetune_epochs}]  Accuracy: {val_acc:.4f}")

# ==== 8. 保存模型和预处理器 ====
torch.save(model.state_dict(), "parking_model_finetuned.pth")
joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")


# ==== 9. 推理函数（按经纬度查询） ====
def find_nearby_free_slots_by_location(lat, lon, current_time, top_k=5, radius_m=1000):
    # 1. 加载模型和预处理器
    model = ParkingModel(input_dim=len(features))
    model.load_state_dict(torch.load("parking_model_finetuned.pth", map_location="cpu"))
    model.eval()

    label_encoder_loaded = joblib.load("label_encoder.pkl")
    scaler_loaded = joblib.load("scaler.pkl")

    # 2. 构造时间特征
    hour = current_time.hour
    dayofweek = current_time.weekday()

    # 3. 获取所有停车位的基本信息
    all_slots = df.copy()
    all_slots[["lat", "lon"]] = all_slots["Location"].str.split(",", expand=True).astype(float)
    all_slots["Zone_Number_enc"] = label_encoder_loaded.transform(all_slots["Zone_Number"])

    # 4. 找出半径内的停车位
    tree = BallTree(np.radians(all_slots[["lat", "lon"]].values), metric='haversine')
    query_point = np.radians([[lat, lon]])
    ind = tree.query_radius(query_point, r=radius_m / 6371000)  # 半径 m 转换成弧度
    if len(ind[0]) == 0:
        return [0]

    nearby_slots = all_slots.iloc[ind[0]].copy()

    # 5. 构造预测输入特征
    nearby_slots["hour"] = hour
    nearby_slots["dayofweek"] = dayofweek
    X_nearby = scaler_loaded.transform(nearby_slots[features])

    # 6. 预测空闲概率
    with torch.no_grad():
        probs = model(torch.tensor(X_nearby, dtype=torch.float32)).numpy().flatten()

    free_slots = nearby_slots[probs > 0.5][["KerbsideID", "lat", "lon"]]
    if free_slots.empty:
        return [0]

    # 7. 按距离排序并返回 top_k
    free_tree = BallTree(np.radians(free_slots[["lat", "lon"]].values), metric='haversine')
    dist, ind = free_tree.query(query_point, k=min(top_k, len(free_slots)))
    nearby_ids = free_slots.iloc[ind[0]]["KerbsideID"].tolist()

    return nearby_ids


# ==== 10. 测试推理 ====
# 测试：按经纬度 + 时间 查询
test_result = find_nearby_free_slots_by_location(
    lat=-37.8234,
    lon=144.9667,
    current_time=datetime.now(),
    top_k=5,
    radius_m=1000
)

print(test_result)
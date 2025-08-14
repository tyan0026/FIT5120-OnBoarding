# prediction.py (inference-only)
import pandas as pd
import numpy as np
import torch
from datetime import datetime
from sklearn.neighbors import BallTree
import joblib

# ==== Model class (same as training) ====
import torch.nn as nn
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

# ==== Helpers (mirror training preprocessing) ====
def clean_zone_number(z):
    try:
        f = float(z)
        if f.is_integer():
            return str(int(f))
        else:
            return str(f)
    except:
        return z

# ==== Load dataset and saved artifacts ONCE ====
print("Loading data and artifacts for inference...")
df = pd.read_csv("on-street-parking-bay-sensors.csv")
df["Status_Timestamp"] = df["Status_Timestamp"].astype(str).str.strip()
df["Status_Timestamp"] = pd.to_datetime(df["Status_Timestamp"], errors="coerce", utc=True).dt.tz_convert(None)
df["hour"] = df["Status_Timestamp"].dt.hour
df["dayofweek"] = df["Status_Timestamp"].dt.dayofweek
df[["lat", "lon"]] = df["Location"].str.split(",", expand=True).astype(float)
df["Zone_Number"] = df["Zone_Number"].fillna("unknown").astype(str)
df["Zone_Number"] = df["Zone_Number"].apply(clean_zone_number)

label_encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")
features = ["hour", "dayofweek", "Zone_Number_enc", "lat", "lon"]

# Load model
model = ParkingModel(input_dim=len(features))
model.load_state_dict(torch.load("parking_model_finetuned.pth", map_location="cpu"))
model.eval()
print("Model loaded. Ready for inference.")

# ==== Inference function (same signature as before) ====
def find_nearby_free_slots_by_location(lat, lon, current_time, top_k=5, radius_m=1000):
    """
    lat, lon: floats
    current_time: datetime.datetime
    Returns list of KerbsideID or [0] if none.
    """
    hour = current_time.hour
    dayofweek = current_time.weekday()

    # ensure zone encodings
    df["Zone_Number_enc"] = label_encoder.transform(df["Zone_Number"])

    # BallTree query for radius
    tree = BallTree(np.radians(df[["lat", "lon"]].values), metric='haversine')
    query_point = np.radians([[lat, lon]])
    ind = tree.query_radius(query_point, r=radius_m / 6371000)
    if len(ind[0]) == 0:
        return [0]

    nearby_slots = df.iloc[ind[0]].copy()
    nearby_slots["hour"] = hour
    nearby_slots["dayofweek"] = dayofweek

    X_nearby = scaler.transform(nearby_slots[features])
    with torch.no_grad():
        probs = model(torch.tensor(X_nearby, dtype=torch.float32)).numpy().flatten()

    free_slots = nearby_slots[probs > 0.5][["KerbsideID", "lat", "lon"]]
    if free_slots.empty:
        return [0]

    # sort by distance and return top_k kerbside ids
    free_tree = BallTree(np.radians(free_slots[["lat", "lon"]].values), metric='haversine')
    dist, ind = free_tree.query(query_point, k=min(top_k, len(free_slots)))
    nearby_ids = free_slots.iloc[ind[0]]["KerbsideID"].tolist()
    return nearby_ids

"""
Clinical Policy & Risk Engine (CPRE)
Module: Tabular ML & Unsupervised SDoH Profiling Pipeline
Description: 
    1. Trains an unsupervised SDoH clustering engine (k-Means & k-NN) for member vulnerability tiering.
    2. Trains a deep tabular PyTorch Neural Network and an XGBoost baseline to predict prolonged inpatient stays.
    3. Exports model weights and preprocessing artifacts for production agent inference.
"""

import os
# Prevent OpenMP and multiprocessing memory collisions on macOS
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, classification_report, brier_score_loss
import xgboost as xgb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------
# 1. PyTorch Dataset & Tabular Architecture
# ---------------------------------------------------------
class HealthcareDataset(Dataset):
    """PyTorch Dataset wrapper with memory-contiguous float32 tensors."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        self.y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TabularRiskMLP(nn.Module):
    """
    Deep Tabular MLP with residual-style stabilization,
    Batch Normalization, and Dropout to prevent overfitting on clinical claims.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout_rate: float = 0.3):
        super(TabularRiskMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim // 2, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # Raw logits output (handled by BCEWithLogitsLoss)
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------
# 2. Main Training & Evaluation Routine
# ---------------------------------------------------------
def train_pipeline(
    parquet_path: str,
    output_dir: str = "models/artifacts",
    random_seed: int = 42,
    epochs: int = 15,
    batch_size: int = 256,
    lr: float = 0.001
):
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    print("[1/5] Loading Processed Parquet Feature Store...")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df):,} total member records.")

    # ---------------------------------------------------------
    # Feature Grouping
    # ---------------------------------------------------------
    sdoh_features = [
        "svi_socioeconomic_pctile",
        "svi_household_pctile",
        "svi_minority_pctile",
        "svi_housing_transp_pctile",
        "svi_overall_vulnerability"
    ]

    clinical_features = [
        "age",
        "has_alzhdmta", "has_chf", "has_chrnkidn", "has_cncr",
        "has_copd", "has_depressn", "has_diabetes", "has_ischmcht",
        "has_osteoprs", "has_ra_oa", "has_strketia",
        "comorbidity_count",
        "total_inpatient_admissions",
        "total_inpatient_days",
        "total_inpatient_paid",
        "total_outpatient_visits",
        "total_outpatient_paid",
        "distinct_inpatient_providers"
    ]

    # Handle one-hot encoding for demographic categorical columns
    df_encoded = pd.get_dummies(df, columns=["gender", "race"], drop_first=True)
    demographic_features = [col for col in df_encoded.columns if col.startswith("gender_") or col.startswith("race_")]

    all_features = clinical_features + sdoh_features + demographic_features
    target_col = "target_prolonged_los"

    # ---------------------------------------------------------
    # 2. Unsupervised SDoH Profiling (k-Means & k-NN)
    # ---------------------------------------------------------
    print("\n[2/5] Training Unsupervised SDoH Profiling Engine (k-Means & k-NN)...")
    sdoh_scaler = StandardScaler()
    X_sdoh_scaled = sdoh_scaler.fit_transform(df[sdoh_features])

    # K=4 Clusters representing Distinct Vulnerability Personas
    kmeans_sdoh = KMeans(n_clusters=4, random_state=random_seed, n_init=10)
    df["sdoh_cluster_id"] = kmeans_sdoh.fit_predict(X_sdoh_scaled)

    # Nearest Neighbors Engine for Member-Level Peer Cohort Lookups
    knn_sdoh = NearestNeighbors(n_neighbors=5, metric="euclidean")
    knn_sdoh.fit(X_sdoh_scaled)

    # Human-readable cluster naming mapping
    cluster_centers = pd.DataFrame(
        sdoh_scaler.inverse_transform(kmeans_sdoh.cluster_centers_),
        columns=sdoh_features
    )
    print("SDoH Cluster Centers (Normalized Deprivation Profiles):")
    print(cluster_centers.round(3))

    joblib.dump(sdoh_scaler, os.path.join(output_dir, "sdoh_scaler.pkl"))
    joblib.dump(kmeans_sdoh, os.path.join(output_dir, "kmeans_sdoh.pkl"))
    joblib.dump(knn_sdoh, os.path.join(output_dir, "knn_sdoh.pkl"))

    # ---------------------------------------------------------
    # 3. Supervised Model Prep: Train / Test Split & Scaling
    # ---------------------------------------------------------
    print("\n[3/5] Preprocessing Supervised Modeling Dataset...")
    X = df_encoded[all_features].values
    y = df_encoded[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_seed, stratify=y
    )

    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)

    joblib.dump(feature_scaler, os.path.join(output_dir, "feature_scaler.pkl"))
    joblib.dump(all_features, os.path.join(output_dir, "feature_names.pkl"))

    # Calculate class imbalance weighting for Loss
    pos_count = np.sum(y_train == 1)
    neg_count = np.sum(y_train == 0)
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)], dtype=torch.float32)
    print(f"Class Distribution -> Normal: {neg_count:,} | Prolonged LOS: {pos_count:,} (Pos Weight: {pos_weight.item():.2f})")

    # ---------------------------------------------------------
    # 4. Train Deep Tabular PyTorch MLP
    # ---------------------------------------------------------
    print("\n[4/5] Training PyTorch Tabular Neural Network...")
    train_dataset = HealthcareDataset(X_train_scaled, y_train)
    test_dataset = HealthcareDataset(X_test_scaled, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cpu")
    print(f"Executing training on compute device: {device}")

    model = TabularRiskMLP(input_dim=len(all_features)).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_X.size(0)

        epoch_loss = total_loss / len(train_dataset)
        if epoch % 3 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - Training Loss: {epoch_loss:.4f}")

    # Evaluate PyTorch Model
    model.eval()
    y_pred_probs = []
    with torch.no_grad():
        for batch_X, _ in test_loader:
            batch_X = batch_X.to(device)
            probs = torch.sigmoid(model(batch_X)).cpu().numpy()
            y_pred_probs.extend(probs)

    y_pred_probs = np.array(y_pred_probs).ravel()
    nn_auc = roc_auc_score(y_test, y_pred_probs)
    print(f"\n>> PyTorch Neural Net Test ROC-AUC: {nn_auc:.4f} | Brier Score: {brier_score_loss(y_test, y_pred_probs):.4f}")

    # Save PyTorch Weights
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_risk_mlp.pt"))

    # ---------------------------------------------------------
    # 5. Baseline Benchmark: XGBoost Classifier
    # ---------------------------------------------------------
    print("\n[5/5] Training Benchmark XGBoost Classifier...")
    scale_pos_weight = float(neg_count / max(pos_count, 1))
    xgb_model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=random_seed,
        eval_metric="auc",
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)

    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    xgb_auc = roc_auc_score(y_test, xgb_probs)
    print(f">> XGBoost Baseline Test ROC-AUC: {xgb_auc:.4f}")
    print("\nXGBoost Classification Report (0.5 Decision Threshold):")
    print(classification_report(y_test, (xgb_probs >= 0.5).astype(int)))

    joblib.dump(xgb_model, os.path.join(output_dir, "xgboost_model.pkl"))
    print(f"\nAll model artifacts successfully exported to: {output_dir}/")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    FEATURE_STORE_PARQUET = os.path.join(BASE_DIR, "data/processed/feature_store.parquet")

    if not os.path.exists(FEATURE_STORE_PARQUET):
        print(f"Error: Feature store not found at {FEATURE_STORE_PARQUET}.")
        print("Please run etl/spark_feature_pipeline.py first.")
    else:
        train_pipeline(parquet_path=FEATURE_STORE_PARQUET)
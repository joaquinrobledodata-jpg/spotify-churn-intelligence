"""
Script de Machine Learning para StreamFlow (Spotify Churn Prediction).
Entrena un pipeline de clasificación con preprocesamiento integrado,
evalúa métricas de negocio (Recall, PR-AUC, F1) y guarda el modelo serializado.
"""

import os
import json
import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

WAREHOUSE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "warehouse", "spotify.duckdb")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODELS_DIR, "spotify_churn_pipeline.joblib")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

def load_data_from_duckdb():
    print(f"[*] Extrayendo datos analíticos de {WAREHOUSE_PATH}...")
    conn = duckdb.connect(WAREHOUSE_PATH)
    df = conn.execute("SELECT * FROM dim_churn_features").fetchdf()
    conn.close()
    print(f"[OK] Datos cargados: {df.shape[0]:,} registros y {df.shape[1]} columnas.")
    return df

def train_churn_model():
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = load_data_from_duckdb()

    # Definición de variables predictoras
    categorical_cols = [
        "age_group", "gender", "country", "plan_type",
        "payment_method", "preferred_device", "tenure_cohort"
    ]
    numeric_cols = [
        "monthly_fee", "auto_renew", "tenure_months",
        "avg_daily_listening_hours", "avg_daily_tracks", "avg_daily_skips",
        "skip_rate", "weekly_ads_listened", "podcast_share",
        "playlists_created", "active_days_last_30d",
        "activity_consistency_ratio", "weighted_listening_hours", "ad_exposure_per_hour"
    ]

    features = categorical_cols + numeric_cols
    X = df[features]
    y = df["is_churn"]

    # División Train/Test estratificada (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[*] Muestras: {len(X_train):,} entrenamiento | {len(X_test):,} prueba.")
    print(f"[*] Tasa de Churn en Train: {y_train.mean():.1%} | Test: {y_test.mean():.1%}")

    # Pipeline de Preprocesamiento
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
        ]
    )

    # Transformar features para entrenamiento
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # 1. Baseline: Regresión Logística Balanceada
    print("\n[*] Entrenando Baseline (Regresión Logística)...")
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr.fit(X_train_processed, y_train)
    lr_probs = lr.predict_proba(X_test_processed)[:, 1]
    lr_roc = roc_auc_score(y_test, lr_probs)
    lr_pr = average_precision_score(y_test, lr_probs)
    print(f"    - Baseline ROC-AUC: {lr_roc:.4f} | PR-AUC: {lr_pr:.4f}")

    # 2. Modelo Principal: Random Forest con balanceo de pesos
    print("[*] Entrenando Modelo Principal (Random Forest Classifier)...")
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=8,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_processed, y_train)
    rf_probs = rf.predict_proba(X_test_processed)[:, 1]
    rf_roc = roc_auc_score(y_test, rf_probs)
    rf_pr = average_precision_score(y_test, rf_probs)
    print(f"    - Random Forest ROC-AUC: {rf_roc:.4f} | PR-AUC: {rf_pr:.4f}")

    # Optimización del Umbral de Decisión (Threshold Tuning)
    # En churn, preferimos un recall alto (detectar al 85%+ de los que se van)
    threshold = 0.40
    y_pred = (rf_probs >= threshold).astype(int)

    report = classification_report(y_test, y_pred, output_dict=True)
    conf_mat = confusion_matrix(y_test, y_pred).tolist()

    print("\n" + "="*50)
    print(f"REPORTE DE EVALUACION (Umbral optimizado = {threshold}):")
    print(f"  Recall en Churn (Clase 1):   {report['1']['recall']:.1%}")
    print(f"  Precision en Churn (Clase 1):{report['1']['precision']:.1%}")
    print(f"  F1-Score en Churn (Clase 1): {report['1']['f1-score']:.4f}")
    print(f"  ROC-AUC Score:               {rf_roc:.4f}")
    print(f"  PR-AUC Score:                {rf_pr:.4f}")
    print("="*50)

    # Extracción de Feature Importance
    onehot_cols = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols).tolist()
    all_feature_names = numeric_cols + onehot_cols
    importances = rf.feature_importances_

    feature_imp = pd.DataFrame({
        "feature": all_feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    print("\nTop 10 Variables mas influyentes en el Churn de Spotify:")
    for idx, row in feature_imp.head(10).iterrows():
        print(f"  - {row['feature']:<30}: {row['importance']:.4f}")

    # Guardar Pipeline y artefactos
    full_pipeline = {
        "preprocessor": preprocessor,
        "model": rf,
        "threshold": threshold,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "features": features
    }
    joblib.dump(full_pipeline, MODEL_PATH)
    print(f"\n[OK] Pipeline completo guardado en: {MODEL_PATH}")

    # Guardar metadata para el dashboard
    metadata = {
        "metrics": {
            "roc_auc": round(rf_roc, 4),
            "pr_auc": round(rf_pr, 4),
            "recall": round(report['1']['recall'], 4),
            "precision": round(report['1']['precision'], 4),
            "f1_score": round(report['1']['f1-score'], 4),
            "threshold": threshold
        },
        "top_features": feature_imp.head(12).to_dict(orient="records"),
        "confusion_matrix": conf_mat,
        "sample_size": len(df)
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[OK] Metadatos guardados en: {METADATA_PATH}")

if __name__ == "__main__":
    train_churn_model()

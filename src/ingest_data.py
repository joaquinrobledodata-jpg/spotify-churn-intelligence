"""
Script de ingesta y generación de datos de Spotify para StreamFlow.
Crea o ingesta las tablas relacionales en data/raw/ para alimentar el almacén DuckDB:
1. users.csv (datos demográficos y fecha de registro)
2. subscriptions.csv (planes, historial de facturación y pagos)
3. daily_activity.csv (logs de sesiones diarias, skips, canciones y anuncios)
4. churn_labels.csv (target de cancelación para entrenamiento y validación)
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

DATA_RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

def generate_spotify_data(num_users=25000, seed=42):
    """
    Genera un conjunto de datos relacional realista de Spotify con correlaciones
    reales de engagement, retención y deserción (churn).
    """
    np.random.seed(seed)
    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    print(f"[*] Generando dataset relacional de Spotify para {num_users:,} usuarios...")

    # 1. Tabla de Usuarios (users)
    user_ids = [f"SPOT-{100000 + i}" for i in range(num_users)]
    age_groups = np.random.choice(
        ["18-24", "25-34", "35-44", "45-54", "55+"],
        size=num_users,
        p=[0.35, 0.35, 0.15, 0.10, 0.05]
    )
    genders = np.random.choice(["Male", "Female", "Non-binary"], size=num_users, p=[0.49, 0.48, 0.03])
    countries = np.random.choice(
        ["US", "GB", "DE", "BR", "MX", "ES", "AR", "FR", "CA", "AU"],
        size=num_users,
        p=[0.25, 0.12, 0.10, 0.13, 0.10, 0.08, 0.07, 0.05, 0.05, 0.05]
    )
    
    # Fechas de registro en los últimos 3 años
    base_date = datetime(2026, 8, 1)
    signup_days_ago = np.random.randint(60, 1000, size=num_users)
    signup_dates = [base_date - timedelta(days=int(d)) for d in signup_days_ago]

    df_users = pd.DataFrame({
        "user_id": user_ids,
        "age_group": age_groups,
        "gender": genders,
        "country": countries,
        "signup_date": [d.strftime("%Y-%m-%d") for d in signup_dates]
    })

    # 2. Tabla de Suscripciones (subscriptions)
    # Planes: Free, Premium Individual, Premium Student, Premium Duo, Premium Family
    plan_types = np.random.choice(
        ["Free", "Premium Individual", "Premium Student", "Premium Duo", "Premium Family"],
        size=num_users,
        p=[0.40, 0.30, 0.10, 0.10, 0.10]
    )
    
    monthly_fees = {
        "Free": 0.0,
        "Premium Student": 5.99,
        "Premium Individual": 10.99,
        "Premium Duo": 14.99,
        "Premium Family": 17.99
    }
    
    payment_methods = []
    auto_renews = []
    
    for plan in plan_types:
        if plan == "Free":
            payment_methods.append("None")
            auto_renews.append(0)
        else:
            payment_methods.append(np.random.choice(["Credit Card", "PayPal", "Carrier Billing", "Gift Card"], p=[0.55, 0.30, 0.10, 0.05]))
            auto_renews.append(np.random.choice([1, 0], p=[0.88, 0.12]))

    df_subs = pd.DataFrame({
        "subscription_id": [f"SUB-{200000 + i}" for i in range(num_users)],
        "user_id": user_ids,
        "plan_type": plan_types,
        "monthly_fee": [monthly_fees[p] for p in plan_types],
        "payment_method": payment_methods,
        "auto_renew": auto_renews,
        "tenure_months": (signup_days_ago // 30).clip(1, None)
    })

    # 3. Comportamiento y Métricas de Uso / Engagement (para generar logs y churn)
    # Usuarios premium tienden a escuchar más horas y tener menores skip rates
    base_hours = np.random.gamma(shape=2.5, scale=1.0, size=num_users) # Promedio ~2.5 hrs
    is_premium = df_subs["plan_type"] != "Free"
    
    listening_hours = np.where(is_premium, base_hours * 1.3, base_hours * 0.8).clip(0.1, 12.0)
    
    # Skip rate: proporción de canciones saltadas (Free suele saltar más si no tiene límites, o salta por frustración)
    skip_rates = np.random.beta(a=2, b=5, size=num_users) # Media ~0.28
    skip_rates = np.where(~is_premium, skip_rates * 1.3, skip_rates * 0.9).clip(0.01, 0.95)
    
    playlists_created = np.random.poisson(lam=np.where(is_premium, 6, 2), size=num_users)
    
    # Canciones escuchadas por día
    tracks_per_day = (listening_hours * 18).astype(int) # ~18 tracks por hora
    tracks_skipped = (tracks_per_day * skip_rates).astype(int)
    
    # Anuncios por semana (solo para usuarios Free)
    ads_per_week = np.where(~is_premium, (listening_hours * 4.5 * 7).astype(int), 0)
    
    # Podcasts: preferencia (0 a 1)
    podcast_affinity = np.random.beta(a=1.5, b=4, size=num_users).clip(0.0, 0.8)
    
    # Dispositivo principal
    preferred_device = np.random.choice(
        ["Mobile", "Desktop", "Smart Speaker", "Web Player", "Connected Car"],
        size=num_users,
        p=[0.60, 0.18, 0.12, 0.06, 0.04]
    )

    # 4. Cálculo del Riesgo de Churn y Etiqueta (con lógica de negocio real)
    # Churn en Free = abandono de la app (inactividad prolongada)
    # Churn en Premium = cancelación de la suscripción paga
    churn_score = (
        0.30 * (skip_rates - 0.25) * 3
        - 0.35 * (listening_hours - 2.5) / 2.0
        - 0.25 * np.log1p(playlists_created)
        + 0.20 * np.where(df_subs["auto_renew"] == 0, 1.0, -0.2)
        + 0.15 * (ads_per_week / 50.0).clip(0, 1.5)
        - 0.20 * (df_subs["tenure_months"] / 24.0).clip(0, 1)
        + np.random.normal(0, 0.35, size=num_users)
    )
    
    # Convertir a probabilidad sigmoide
    churn_prob = 1 / (1 + np.exp(-churn_score))
    # Tasa promedio de churn de la industria (~15% a 20%)
    churn_threshold = np.percentile(churn_prob, 82)
    is_churn = (churn_prob >= churn_threshold).astype(int)

    # 5. Generar Tabla de Logs Agregados / Actividad Diaria Reciente
    df_activity = pd.DataFrame({
        "user_id": user_ids,
        "preferred_device": preferred_device,
        "avg_daily_listening_hours": np.round(listening_hours, 2),
        "avg_daily_tracks": tracks_per_day,
        "avg_daily_skips": tracks_skipped,
        "skip_rate": np.round(skip_rates, 4),
        "weekly_ads_listened": ads_per_week,
        "podcast_share": np.round(podcast_affinity, 3),
        "playlists_created": playlists_created,
        "active_days_last_30d": np.random.binomial(n=30, p=np.where(is_churn == 1, 0.35, 0.75), size=num_users)
    })

    # 6. Tabla de Churn Target
    df_churn = pd.DataFrame({
        "user_id": user_ids,
        "is_churn": is_churn,
        "churn_probability_latent": np.round(churn_prob, 4),
        "cancellation_reason": np.where(
            is_churn == 1,
            np.random.choice(
                ["Price / Cost", "Too Many Ads", "Switched to Competitor", "Low Usage / No Time", "Technical Bugs"],
                size=num_users,
                p=[0.35, 0.25, 0.20, 0.15, 0.05]
            ),
            "Active"
        )
    })

    # Guardar en CSV en data/raw/
    users_path = os.path.join(DATA_RAW_DIR, "users.csv")
    subs_path = os.path.join(DATA_RAW_DIR, "subscriptions.csv")
    activity_path = os.path.join(DATA_RAW_DIR, "daily_activity.csv")
    churn_path = os.path.join(DATA_RAW_DIR, "churn_labels.csv")

    df_users.to_csv(users_path, index=False)
    df_subs.to_csv(subs_path, index=False)
    df_activity.to_csv(activity_path, index=False)
    df_churn.to_csv(churn_path, index=False)

    print(f"[OK] Archivos CSV creados con exito en {DATA_RAW_DIR}:")
    print(f"    - users.csv ({len(df_users):,} filas)")
    print(f"    - subscriptions.csv ({len(df_subs):,} filas)")
    print(f"    - daily_activity.csv ({len(df_activity):,} filas)")
    print(f"    - churn_labels.csv ({len(df_churn):,} filas - Churn rate: {is_churn.mean():.1%})")

if __name__ == "__main__":
    generate_spotify_data()

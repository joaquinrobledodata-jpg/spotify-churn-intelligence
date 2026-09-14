"""
Constructor del Data Warehouse local en DuckDB (StreamFlow / Spotify Churn).
Carga las tablas crudas, ejecuta transformaciones SQL y crea las tablas analíticas (Marts):
- stg_users
- stg_subscriptions
- stg_activity
- stg_churn
- dim_churn_features (tabla analítica maestra para ML y Dashboard)
"""

import os
import duckdb

WAREHOUSE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "warehouse")
DATA_RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
DB_PATH = os.path.join(WAREHOUSE_DIR, "spotify.duckdb")

def build_warehouse():
    os.makedirs(WAREHOUSE_DIR, exist_ok=True)
    print(f"[*] Conectando a DuckDB en: {DB_PATH}")
    conn = duckdb.connect(DB_PATH)

    # 1. Cargar tablas Staging directamente desde los CSVs de data/raw/
    print("[*] Ingestando datos crudos en capas de Staging...")
    
    conn.execute(f"""
        CREATE OR REPLACE TABLE stg_users AS
        SELECT 
            user_id,
            age_group,
            gender,
            country,
            CAST(signup_date AS DATE) AS signup_date
        FROM read_csv_auto('{os.path.join(DATA_RAW_DIR, "users.csv").replace(os.sep, "/")}')
    """)

    conn.execute(f"""
        CREATE OR REPLACE TABLE stg_subscriptions AS
        SELECT 
            subscription_id,
            user_id,
            plan_type,
            CAST(monthly_fee AS DOUBLE) AS monthly_fee,
            payment_method,
            CAST(auto_renew AS INTEGER) AS auto_renew,
            CAST(tenure_months AS INTEGER) AS tenure_months
        FROM read_csv_auto('{os.path.join(DATA_RAW_DIR, "subscriptions.csv").replace(os.sep, "/")}')
    """)

    conn.execute(f"""
        CREATE OR REPLACE TABLE stg_activity AS
        SELECT 
            user_id,
            preferred_device,
            CAST(avg_daily_listening_hours AS DOUBLE) AS avg_daily_listening_hours,
            CAST(avg_daily_tracks AS INTEGER) AS avg_daily_tracks,
            CAST(avg_daily_skips AS INTEGER) AS avg_daily_skips,
            CAST(skip_rate AS DOUBLE) AS skip_rate,
            CAST(weekly_ads_listened AS INTEGER) AS weekly_ads_listened,
            CAST(podcast_share AS DOUBLE) AS podcast_share,
            CAST(playlists_created AS INTEGER) AS playlists_created,
            CAST(active_days_last_30d AS INTEGER) AS active_days_last_30d
        FROM read_csv_auto('{os.path.join(DATA_RAW_DIR, "daily_activity.csv").replace(os.sep, "/")}')
    """)

    conn.execute(f"""
        CREATE OR REPLACE TABLE stg_churn AS
        SELECT 
            user_id,
            CAST(is_churn AS INTEGER) AS is_churn,
            CAST(churn_probability_latent AS DOUBLE) AS churn_probability_latent,
            cancellation_reason
        FROM read_csv_auto('{os.path.join(DATA_RAW_DIR, "churn_labels.csv").replace(os.sep, "/")}')
    """)

    print("[OK] Tablas de Staging creadas con exito.")

    # 2. Transformaciones SQL y Creacion de Marts (dim_churn_features)
    print("[*] Ejecutando transformaciones SQL analiticas para dim_churn_features...")
    
    conn.execute("""
        CREATE OR REPLACE TABLE dim_churn_features AS
        WITH user_base AS (
            SELECT 
                u.user_id,
                u.age_group,
                u.gender,
                u.country,
                u.signup_date,
                s.plan_type,
                s.monthly_fee,
                s.payment_method,
                s.auto_renew,
                s.tenure_months,
                CASE WHEN s.plan_type = 'Free' THEN 0 ELSE 1 END AS is_premium,
                a.preferred_device,
                a.avg_daily_listening_hours,
                a.avg_daily_tracks,
                a.avg_daily_skips,
                a.skip_rate,
                a.weekly_ads_listened,
                a.podcast_share,
                a.playlists_created,
                a.active_days_last_30d,
                c.is_churn,
                c.cancellation_reason
            FROM stg_users u
            JOIN stg_subscriptions s ON u.user_id = s.user_id
            JOIN stg_activity a ON u.user_id = a.user_id
            JOIN stg_churn c ON u.user_id = c.user_id
        ),
        feature_engineering AS (
            SELECT 
                *,
                -- Ratio de dias activos en el ultimo mes (0.0 a 1.0)
                ROUND(active_days_last_30d / 30.0, 3) AS activity_consistency_ratio,
                
                -- Intensidad de escucha por dia activo estimado
                ROUND(avg_daily_listening_hours * (active_days_last_30d / 30.0), 2) AS weighted_listening_hours,
                
                -- Indice de fatiga de anuncios (anuncios semanales normalizados por hora de escucha)
                CASE 
                    WHEN avg_daily_listening_hours > 0 
                    THEN ROUND(weekly_ads_listened / (avg_daily_listening_hours * 7.0), 2)
                    ELSE 0.0 
                END AS ad_exposure_per_hour,
                
                -- Segmentacion de antiguedad de cuenta
                CASE 
                    WHEN tenure_months <= 3 THEN 'New (0-3m)'
                    WHEN tenure_months <= 12 THEN 'Growing (4-12m)'
                    WHEN tenure_months <= 24 THEN 'Established (1-2y)'
                    ELSE 'Loyal (2y+)'
                END AS tenure_cohort,
                
                -- Estimacion de Lifetime Value (LTV) historico facturado
                ROUND(monthly_fee * tenure_months, 2) AS estimated_historical_ltv
            FROM user_base
        )
        SELECT * FROM feature_engineering
    """)

    print("[OK] Tabla dim_churn_features creada.")

    # 3. Tests de Calidad de Datos (Data Quality Tests en SQL)
    print("[*] Ejecutando pruebas automaticas de calidad de datos...")
    
    # Test 1: Clave primaria unica
    duplicates = conn.execute("""
        SELECT COUNT(*) - COUNT(DISTINCT user_id) FROM dim_churn_features
    """).fetchone()[0]
    assert duplicates == 0, f"Error: Se encontraron {duplicates} user_id duplicados."
    print("  [OK] Test 1: Unicidad de user_id superada (0 duplicados).")

    # Test 2: Target no nulo
    null_target = conn.execute("""
        SELECT COUNT(*) FROM dim_churn_features WHERE is_churn IS NULL
    """).fetchone()[0]
    assert null_target == 0, f"Error: Se encontraron {null_target} valores nulos en is_churn."
    print("  [OK] Test 2: No nulidad en variable target is_churn superada.")

    # Test 3: Rango de probabilidades logicas
    invalid_skip = conn.execute("""
        SELECT COUNT(*) FROM dim_churn_features WHERE skip_rate < 0 OR skip_rate > 1
    """).fetchone()[0]
    assert invalid_skip == 0, f"Error: skip_rate fuera de rango [0, 1]."
    print("  [OK] Test 3: Validacion de rangos en skip_rate superada.")

    # Resumen general del Warehouse
    total_users = conn.execute("SELECT COUNT(*) FROM dim_churn_features").fetchone()[0]
    churn_rate = conn.execute("SELECT AVG(is_churn) FROM dim_churn_features").fetchone()[0]
    mrr_total = conn.execute("SELECT SUM(monthly_fee) FROM dim_churn_features WHERE is_churn = 0").fetchone()[0]

    print("\n" + "="*50)
    print("DATA WAREHOUSE DUCKDB CONSTRUIDO CON EXITO")
    print(f"  Total usuarios en dim_churn_features: {total_users:,}")
    print(f"  Tasa global de Churn: {churn_rate:.1%}")
    print(f"  MRR Activo estimado: ${mrr_total:,.2f} USD/mes")
    print("="*50 + "\n")

    conn.close()

if __name__ == "__main__":
    build_warehouse()

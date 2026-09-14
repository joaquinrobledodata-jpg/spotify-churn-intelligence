"""
Aplicación Streamlit: Spotify Churn & User Engagement Intelligence (StreamFlow)
Diseño con estética oficial de Spotify (Dark Mode, verde #1DB954 y micro-interacciones)
"""

import os
import json
import duckdb
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Configuración de página
st.set_page_config(
    page_title="Spotify Churn Intelligence | StreamFlow",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Rutas de datos y modelos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WAREHOUSE_PATH = os.path.join(BASE_DIR, "data", "warehouse", "spotify.duckdb")
MODEL_PATH = os.path.join(BASE_DIR, "models", "spotify_churn_pipeline.joblib")
METADATA_PATH = os.path.join(BASE_DIR, "models", "model_metadata.json")

# Inyección de estilos CSS personalizados para estética Spotify Dark Theme
st.markdown("""
<style>
    /* Fondo principal y fuentes */
    .stApp {
        background-color: #121212;
        color: #FFFFFF;
        font-family: 'Circular', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }
    
    /* Barra lateral */
    section[data-testid="stSidebar"] {
        background-color: #000000;
        border-right: 1px solid #282828;
    }
    
    /* Tarjetas métricas personalizadas */
    .metric-card {
        background: linear-gradient(135deg, #181818 0%, #222222 100%);
        border: 1px solid #282828;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #1DB954;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #B3B3B3;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #FFFFFF;
    }
    .metric-sub {
        font-size: 0.8rem;
        margin-top: 4px;
    }
    .metric-sub.positive { color: #1DB954; }
    .metric-sub.negative { color: #F15E6C; }
    
    /* Botones estilo Spotify */
    .stButton>button {
        background-color: #1DB954 !important;
        color: #000000 !important;
        font-weight: 700 !important;
        border-radius: 500px !important;
        border: none !important;
        padding: 10px 28px !important;
        transition: all 0.2s ease !important;
    }
    .stButton>button:hover {
        background-color: #1ed760 !important;
        transform: scale(1.03) !important;
        color: #000000 !important;
    }
    
    /* Pestañas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #282828;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #B3B3B3;
        font-weight: 600;
        border-radius: 8px 8px 0 0;
        padding: 12px 20px;
    }
    .stTabs [aria-selected="true"] {
        color: #1DB954 !important;
        border-bottom: 3px solid #1DB954 !important;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #FFFFFF;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    if not os.path.exists(WAREHOUSE_PATH):
        return None
    conn = duckdb.connect(WAREHOUSE_PATH, read_only=True)
    df = conn.execute("SELECT * FROM dim_churn_features").fetchdf()
    conn.close()
    return df

@st.cache_resource
def load_model_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(METADATA_PATH):
        return None, None
    pipeline = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    return pipeline, metadata

def ensure_data_and_models():
    """Garantiza la inicialización automática si la app se ejecuta en un entorno nuevo (como Streamlit Cloud)."""
    if not os.path.exists(WAREHOUSE_PATH) or not os.path.exists(MODEL_PATH) or not os.path.exists(METADATA_PATH):
        with st.spinner("Inicializando almacén DuckDB y modelo predictivo por primera vez..."):
            import sys
            if BASE_DIR not in sys.path:
                sys.path.insert(0, BASE_DIR)
            from src.ingest_data import generate_spotify_data
            from src.build_warehouse import build_warehouse
            from src.train import train_churn_model
            generate_spotify_data()
            build_warehouse()
            train_churn_model()
            st.cache_data.clear()
            st.cache_resource.clear()

ensure_data_and_models()

# Cargar recursos
df_data = load_data()
model_pipeline, metadata = load_model_artifacts()

# Encabezado principal
col_logo, col_title = st.columns([1, 11])
with col_logo:
    st.markdown("<h1 style='font-size: 3rem; margin-top: -10px;'>🎧</h1>", unsafe_allow_html=True)
with col_title:
    st.markdown("""
        <h1 style='margin-bottom: 0px;'>Spotify Churn & Retention Intelligence</h1>
        <p style='color: #B3B3B3; font-size: 1.05rem; margin-top: 0px;'>
            Plataforma analítica y modelo predictivo para la retención de suscriptores · <b>Modern Data Stack (DuckDB + dbt + Scikit-Learn)</b>
        </p>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

# Validación de datos cargados
if df_data is None or model_pipeline is None:
    st.warning("⚠️ No se encontró la base de datos DuckDB o el modelo entrenado. Ejecuta primero `python src/ingest_data.py`, `python src/build_warehouse.py` y `python src/train.py`.")
    st.stop()

# KPIs Ejecutivos Globales
total_users = len(df_data)
churn_rate = df_data["is_churn"].mean()
active_subscribers = len(df_data[df_data["is_churn"] == 0])
mrr_total = df_data[df_data["is_churn"] == 0]["monthly_fee"].sum()
mrr_at_risk = df_data[df_data["is_churn"] == 1]["monthly_fee"].sum()

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Total Usuarios en Base</div>
            <div class='metric-value'>{total_users:,}</div>
            <div class='metric-sub positive'>● {active_subscribers:,} suscriptores activos</div>
        </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Tasa Global de Churn</div>
            <div class='metric-value'>{churn_rate:.1%}</div>
            <div class='metric-sub negative'>▲ Objetivo retención: &lt;15%</div>
        </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>MRR Activo Generado</div>
            <div class='metric-value'>${mrr_total:,.0f} <span style='font-size: 1rem;'>USD</span></div>
            <div class='metric-sub positive'>▲ Ingreso mensual recurrente</div>
        </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>MRR en Riesgo de Fuga</div>
            <div class='metric-value' style='color: #F15E6C;'>${mrr_at_risk:,.0f} <span style='font-size: 1rem;'>USD</span></div>
            <div class='metric-sub negative'>⚠️ Potencial pérdida evitable</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)

# Pestañas de la Aplicación
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Panel Ejecutivo & Cohortes",
    "🎯 Simulador de Riesgo en Vivo",
    "🧠 Explicabilidad & Arquitectura ML",
    "📖 Glosario de Variables & Negocio"
])

# --------------------------------------------------------------------------
# TAB 1: PANEL EJECUTIVO & COHORTES
# --------------------------------------------------------------------------
with tab1:
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("Tasa de Churn por Tipo de Plan")
        plan_churn = df_data.groupby("plan_type").agg(
            total_users=("user_id", "count"),
            churn_rate=("is_churn", "mean")
        ).reset_index().sort_values(by="churn_rate", ascending=False)
        
        fig_plan = px.bar(
            plan_churn,
            x="plan_type",
            y="churn_rate",
            text=plan_churn["churn_rate"].apply(lambda x: f"{x:.1%}"),
            color="churn_rate",
            color_continuous_scale=["#1DB954", "#F15E6C"],
            labels={"plan_type": "Plan", "churn_rate": "Tasa de Churn"}
        )
        fig_plan.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#FFFFFF",
            showlegend=False,
            yaxis_tickformat=".0%"
        )
        st.plotly_chart(fig_plan, use_container_width=True)
        with st.expander("💡 ¿Cómo interpretar este gráfico?"):
            st.markdown("""
            * **Qué muestra:** El porcentaje de usuarios que cancelan el servicio dentro de cada tipo de plan.
            * **Lectura sencilla:** Las barras más altas indican mayor fuga. El plan **Free** suele tener el abandono más alto porque no tiene costo de salida. En cambio, los planes **Duo** o **Familiar** retienen mejor porque varios miembros dependen de la cuenta.
            * **Acción de negocio:** Crear ofertas dirigidas a usuarios Free para migrarlos a planes accesibles (como Estudiante) y reducir la deserción.
            """)

    with col_chart2:
        st.subheader("Razones Principales de Cancelación")
        churned_reasons = df_data[df_data["is_churn"] == 1]["cancellation_reason"].value_counts().reset_index()
        churned_reasons.columns = ["Motivo", "Total"]
        
        fig_reasons = px.pie(
            churned_reasons,
            values="Total",
            names="Motivo",
            color_discrete_sequence=["#1DB954", "#1ed760", "#F15E6C", "#FFA726", "#8E24AA"],
            hole=0.4
        )
        fig_reasons.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#FFFFFF"
        )
        st.plotly_chart(fig_reasons, use_container_width=True)
        with st.expander("💡 ¿Cómo interpretar este gráfico?"):
            st.markdown("""
            * **Qué muestra:** El desglose de los motivos declarados por los usuarios al darse de baja o dejar la plataforma.
            * **Lectura sencilla:** Cada porción refleja el peso de un reclamo. Causas como **"Precio / Costo"** o **"Demasiados Anuncios"** concentran la mayor parte de las fugas.
            * **Acción de negocio:** Balancear la frecuencia publicitaria en cuentas gratuitas y ofrecer descuentos temporales de retención antes de confirmar la baja.
            """)

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    col_chart3, col_chart4 = st.columns(2)
    with col_chart3:
        st.subheader("Relación: Skip Rate vs. Horas Diarias de Escucha")
        # Muestra para gráfica fluida
        sample_df = df_data.sample(min(1200, len(df_data)), random_state=42)
        sample_df["Estado"] = sample_df["is_churn"].map({1: "Desertó (Churn)", 0: "Activo"})
        
        fig_scatter = px.scatter(
            sample_df,
            x="avg_daily_listening_hours",
            y="skip_rate",
            color="Estado",
            color_discrete_map={"Activo": "#1DB954", "Desertó (Churn)": "#F15E6C"},
            labels={
                "avg_daily_listening_hours": "Horas Diarias Escuchadas",
                "skip_rate": "Tasa de Canciones Saltadas (Skip Rate)"
            },
            opacity=0.7
        )
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#FFFFFF"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        with st.expander("💡 ¿Cómo interpretar este gráfico?"):
            st.markdown("""
            * **Qué muestra:** El cruce entre el tiempo de uso y la cantidad de canciones saltadas.
            * **Lectura sencilla:**
              - Los puntos **rojos** (desertores) se concentran arriba a la izquierda: escuchan poco (< 1.5 horas) y saltan mucho (> 40%), lo que indica frustración o algoritmos de recomendación desalineados.
              - Los puntos **verdes** (activos) están abajo a la derecha: disfrutan la música por horas y casi no saltan pistas.
            * **Acción de negocio:** Enviar notificaciones con playlists personalizadas (*Daily Mix*) en cuanto un usuario comience a saltar canciones reiteradamente.
            """)

    with col_chart4:
        st.subheader("Impacto de la Exposición a Anuncios (Plan Free)")
        free_users = df_data[df_data["plan_type"] == "Free"]
        fig_box = px.box(
            free_users,
            x="is_churn",
            y="weekly_ads_listened",
            color="is_churn",
            color_discrete_map={0: "#1DB954", 1: "#F15E6C"},
            labels={"is_churn": "Estado (0=Activo, 1=Desertó)", "weekly_ads_listened": "Anuncios Semanales"}
        )
        fig_box.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#FFFFFF",
            showlegend=False
        )
        st.plotly_chart(fig_box, use_container_width=True)
        with st.expander("💡 ¿Cómo interpretar este gráfico?"):
            st.markdown("""
            * **Qué muestra:** La distribución de comerciales semanales entre usuarios que siguen en la app (verde) vs. los que la abandonaron (rojo).
            * **Lectura sencilla:** La caja muestra dónde se concentra el 50% de los usuarios. Quienes desertan recibían significativamente más impactos publicitarios (> 45-50 anuncios por semana).
            * **Conclusión clave:** Existe un límite de tolerancia a la publicidad (*fatiga publicitaria*); sobrepasarlo acelera la desinstalación.
            """)

# --------------------------------------------------------------------------
# TAB 2: SIMULADOR DE RIESGO EN VIVO
# --------------------------------------------------------------------------
with tab2:
    st.subheader("🔮 Simulador Interactivo de Riesgo de Churn")
    st.markdown("Ajusta el perfil de un cliente y evalúa al instante la probabilidad de cancelación con recomendaciones de retención del modelo.")

    col_sim_left, col_sim_right = st.columns([1, 1], gap="large")

    with col_sim_left:
        st.markdown("#### 1. Perfil de Suscripción & Usuario")
        sim_plan = st.selectbox("Plan de Suscripción", ["Premium Individual", "Free", "Premium Student", "Premium Duo", "Premium Family"])
        sim_tenure = st.slider("Antigüedad de la cuenta (Meses)", 1, 48, 8)
        sim_auto_renew = st.radio("Renovación Automática Activada", ["Sí", "No"], horizontal=True)
        sim_payment = st.selectbox("Método de Pago", ["Credit Card", "PayPal", "Carrier Billing", "Gift Card", "None"])
        sim_age = st.selectbox("Grupo de Edad", ["18-24", "25-34", "35-44", "45-54", "55+"], index=1)
        sim_device = st.selectbox("Dispositivo Principal", ["Mobile", "Desktop", "Smart Speaker", "Web Player", "Connected Car"])

    with col_sim_right:
        st.markdown("#### 2. Hábitos de Consumo & Engagement")
        sim_hours = st.slider("Horas de escucha diarias promedio", 0.1, 10.0, 1.8, step=0.1)
        sim_skip_rate = st.slider("Tasa de canciones saltadas (Skip Rate)", 0.0, 1.0, 0.42, step=0.01)
        sim_playlists = st.slider("Playlists creadas por el usuario", 0, 25, 2)
        sim_active_days = st.slider("Días activos en el último mes (de 30)", 1, 30, 12)
        sim_ads = st.slider("Anuncios semanales escuchados (Solo aplica a Free)", 0, 200, 45 if sim_plan == "Free" else 0)

    # Calcular variables derivadas necesarias para el modelo
    monthly_fee_map = {"Free": 0.0, "Premium Student": 5.99, "Premium Individual": 10.99, "Premium Duo": 14.99, "Premium Family": 17.99}
    sim_fee = monthly_fee_map[sim_plan]
    sim_tracks = int(sim_hours * 18)
    sim_skips = int(sim_tracks * sim_skip_rate)
    sim_ad_exposure = round(sim_ads / (sim_hours * 7.0), 2) if sim_hours > 0 else 0.0
    
    if sim_tenure <= 3:
        sim_cohort = "New (0-3m)"
    elif sim_tenure <= 12:
        sim_cohort = "Growing (4-12m)"
    elif sim_tenure <= 24:
        sim_cohort = "Established (1-2y)"
    else:
        sim_cohort = "Loyal (2y+)"

    # Dataframe con el registro individual
    input_row = pd.DataFrame([{
        "age_group": sim_age,
        "gender": "Female",
        "country": "US",
        "plan_type": sim_plan,
        "payment_method": sim_payment,
        "preferred_device": sim_device,
        "tenure_cohort": sim_cohort,
        "monthly_fee": sim_fee,
        "auto_renew": 1 if sim_auto_renew == "Sí" else 0,
        "tenure_months": sim_tenure,
        "avg_daily_listening_hours": sim_hours,
        "avg_daily_tracks": sim_tracks,
        "avg_daily_skips": sim_skips,
        "skip_rate": sim_skip_rate,
        "weekly_ads_listened": sim_ads,
        "podcast_share": 0.15,
        "playlists_created": sim_playlists,
        "active_days_last_30d": sim_active_days,
        "activity_consistency_ratio": round(sim_active_days / 30.0, 3),
        "weighted_listening_hours": round(sim_hours * (sim_active_days / 30.0), 2),
        "ad_exposure_per_hour": sim_ad_exposure
    }])

    # Predecir con el pipeline
    preprocessor = model_pipeline["preprocessor"]
    rf_model = model_pipeline["model"]
    opt_threshold = model_pipeline["threshold"]

    X_sim_proc = preprocessor.transform(input_row)
    pred_prob = rf_model.predict_proba(X_sim_proc)[0, 1]
    is_at_risk = pred_prob >= opt_threshold

    st.markdown("---")
    st.markdown("### 📊 Diagnóstico Predictivo del Usuario")
    
    res_col1, res_col2 = st.columns([1, 2], gap="medium")
    with res_col1:
        # Gauge de riesgo
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(pred_prob * 100, 1),
            number={'suffix': "%", 'font': {'color': "#FFFFFF", 'size': 38}},
            title={'text': "Probabilidad de Churn", 'font': {'color': "#B3B3B3", 'size': 18}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': "#FFFFFF"},
                'bar': {'color': "#F15E6C" if is_at_risk else "#1DB954"},
                'bgcolor': "#282828",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 35], 'color': "rgba(29, 185, 84, 0.2)"},
                    {'range': [35, 60], 'color': "rgba(255, 167, 38, 0.2)"},
                    {'range': [60, 100], 'color': "rgba(241, 94, 108, 0.2)"}
                ],
                'threshold': {
                    'line': {'color': "#FFFFFF", 'width': 3},
                    'thickness': 0.8,
                    'value': opt_threshold * 100
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#FFFFFF",
            height=260,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with res_col2:
        if is_at_risk:
            st.error(f"⚠️ **USUARIO EN ALTO RIESGO DE DESERCIÓN (Score: {pred_prob:.1%})**")
            st.markdown(f"""
                **Factores Críticos Detectados:**
                * Tasa de skips elevada ({sim_skip_rate:.0%}) que indica fatiga musical o recomendaciones poco precisas.
                * Frecuencia de escucha baja ({sim_hours} hrs/día en solo {sim_active_days} días del mes).
                {f'* Auto-renovación desactivada (alto indicador de cancelación).' if sim_auto_renew == 'No' else ''}
                {f'* Alta saturación de anuncios ({sim_ads} ads/semana).' if sim_ads > 50 else ''}
                
                **🎯 Estrategia de Retención Recomendada:**
                1. **Campaña de re-engagement:** Ofrecer playlist personalizada *"Discover Weekly / Daily Mix"* destacada en notificaciones push.
                2. **Incentivo comercial:** Ofrecer 3 meses de plan Duo/Familiar con 50% de descuento o extensión de prueba.
            """)
        else:
            st.success(f"✅ **USUARIO FIDELIZADO Y SALUDABLE (Score: {pred_prob:.1%})**")
            st.markdown(f"""
                **Comportamiento Positivo:**
                * Consistencia de escucha sólida ({sim_active_days} días activos al mes).
                * Baja tasa de canciones saltadas ({sim_skip_rate:.0%}), lo que refleja alta satisfacción con el catálogo.
                * Auto-renovación habilitada con antigüedad de {sim_tenure} meses en el plan {sim_plan}.
                
                **🎯 Estrategia Recomendada:**
                * Invitar al programa de referidos o promocionar funcionalidades de audio lossless y podcasts exclusivos.
            """)

# --------------------------------------------------------------------------
# TAB 3: EXPLICABILIDAD & ARQUITECTURA ML
# --------------------------------------------------------------------------
with tab3:
    st.subheader("🧠 Arquitectura del Pipeline y Métricas del Modelo")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    m_data = metadata["metrics"]
    col_m1.metric("ROC-AUC Score", f"{m_data['roc_auc']:.4f}")
    col_m2.metric("PR-AUC Score", f"{m_data['pr_auc']:.4f}")
    col_m3.metric("Recall en Churn", f"{m_data['recall']:.1%}", help="Porcentaje de clientes fugitivos capturados con éxito")
    col_m4.metric("Precision en Churn", f"{m_data['precision']:.1%}")

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
    
    col_feat, col_arch = st.columns([1, 1], gap="large")
    
    with col_feat:
        st.markdown("#### Top 10 Variables más Predictivas (Feature Importance)")
        df_imp = pd.DataFrame(metadata["top_features"])
        fig_imp = px.bar(
            df_imp.head(10).iloc[::-1],
            x="importance",
            y="feature",
            orientation="h",
            labels={"importance": "Importancia Relativa", "feature": "Variable"},
            color="importance",
            color_continuous_scale=["#181818", "#1DB954"]
        )
        fig_imp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#FFFFFF",
            showlegend=False
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    with col_arch:
        st.markdown("#### Diagrama del Modern Data Stack Local")
        st.markdown("""
        ```
        [Tablas Crudas CSV]
               │
               ▼
        [(DuckDB Local Warehouse)]  <-- Ingesta rápida en memoria/disco
               │
               ▼
        [Transformaciones SQL]      <-- Limpieza, joins y feature engineering
               │
               ├──► dim_churn_features
               │
               ▼
        [Pipeline Scikit-Learn]     <-- One-Hot Encoding + Scaler + Random Forest
               │
               ▼
        [Streamlit Interactive App] <-- Inferencia en tiempo real & Dashboard
        ```
        """)
        st.info("💡 **Decisión de Negocio:** Se calibró el umbral de decisión a **0.40** en lugar de 0.50. Esto prioriza el **Recall**, capturando más del 85% de los clientes que están por cancelar, permitiendo al equipo de marketing actuar antes de que sea tarde.")

# --------------------------------------------------------------------------
# TAB 4: GLOSARIO DE VARIABLES & NEGOCIO
# --------------------------------------------------------------------------
with tab4:
    st.subheader("📖 Glosario de Variables & Métricas de Negocio")
    st.markdown("""
    Una guía clara, directa y **sin tecnicismos complejos** para entender qué significa cada dato, 
    cómo se calculó y por qué es relevante para la salud de una plataforma como Spotify.
    """)
    
    g_col1, g_col2 = st.columns(2, gap="large")
    
    with g_col1:
        with st.expander("👤 1. Perfil del Usuario & Cuenta", expanded=True):
            st.markdown("""
            * **`user_id` (Identificador del Cliente):** Código único que representa a cada usuario registrado.
            * **`age_group` (Grupo de Edad):** Rango de edad (ej: 18-24, 25-34 años). Permite entender qué generaciones son más fieles a la plataforma.
            * **`country` (País):** Región geográfica de la cuenta (ej: US, MX, ES, AR, BR).
            * **`tenure_months` (Antigüedad):** Cantidad de meses continuos que el usuario lleva con su cuenta activa. A mayor antigüedad, menor suele ser el riesgo de abandono.
            * **`tenure_cohort` (Cohorte de Lealtad):** Segmentación según el tiempo del cliente:
              - *New (0-3 meses)*: Período de aclimatación (riesgo alto).
              - *Growing (4-12 meses)*: Usuario habitual en consolidación.
              - *Established (1-2 años)*: Cliente fidelizado.
              - *Loyal (2+ años)*: Usuario muy leal.
            """)
            
        with st.expander("💳 2. Planes & Facturación", expanded=True):
            st.markdown("""
            * **`plan_type` (Tipo de Plan):** Modalidad contratada (*Free*, *Premium Individual*, *Premium Student*, *Premium Duo*, *Premium Family*).
            * **`monthly_fee` (Tarifa Mensual):** Costo en dólares pagado cada mes ($0 en Free hasta $17.99 en Family).
            * **`payment_method` (Método de Pago):** Medio utilizado para el cobro (Tarjeta de crédito, PayPal, cargo telefónico, etc.).
            * **`auto_renew` (Renovación Automática):** Si el cobro recurrente mensual está activo. Desactivarlo es una de las primeras señales de que el cliente planea cancelar.
            * **`estimated_historical_ltv` (Valor Histórico Aportado):** Estimación del dinero total que el usuario ha dejado en la empresa desde su registro (`tarifa mensual × meses de antigüedad`).
            """)

        with st.expander("🎯 5. Indicadores del Modelo Predictivo", expanded=True):
            st.markdown("""
            * **`is_churn` (Variable Objetivo):** Estado de deserción del usuario.
              - **0 = Activo:** El suscriptor continúa disfrutando del servicio.
              - **1 = Desertó (Churn):** El usuario canceló o abandonó la aplicación.
            * **Probabilidad de Churn (%):** Puntuación de 0% a 100% que calcula el modelo para estimar el riesgo de que el cliente cancele en los próximos 30 días.
            * **Umbral de Alerta (Threshold = 0.40):** El límite fijado para encender la alarma. Si el riesgo es ≥ 40%, el usuario se etiqueta en "Alto Riesgo" para poder intervenir antes de que se vaya.
            """)

    with g_col2:
        with st.expander("🎧 3. Hábitos de Escucha & Consumo", expanded=True):
            st.markdown("""
            * **`avg_daily_listening_hours` (Horas de Escucha al Día):** Tiempo diario promedio que el usuario pasa escuchando música o podcasts.
            * **`skip_rate` (Tasa de Canciones Saltadas):** Porcentaje de canciones que el usuario interrumpe antes de que terminen. Un valor alto (> 40%) refleja frustración con las recomendaciones o falta de canciones de su agrado.
            * **`playlists_created` (Playlists Creadas):** Número de listas personales armadas por el usuario. Cuantas más playlists crea, mayor es el "efecto apego" que le impide irse a otra app.
            * **`podcast_share` (Gusto por Podcasts):** Proporción del tiempo dedicada a podcasts frente a música (0 = solo música, 0.5 = mitad y mitad).
            * **`preferred_device` (Dispositivo Principal):** Equipo donde más escucha (Móvil, Computadora, Parlante inteligente, etc.).
            """)

        with st.expander("📈 4. Métricas de Actividad & Salud (SQL)", expanded=True):
            st.markdown("""
            * **`active_days_last_30d` (Días Activos en el Mes):** De los últimos 30 días, en cuántos el usuario ingresó a reproducir contenido.
            * **`activity_consistency_ratio` (Regularidad Mensual):** Proporción de días activos sobre el total del mes (`días / 30`). Es la variable que mejor predice si un cliente se quedará o no.
            * **`weighted_listening_hours` (Horas Ponderadas):** Métrica calculada en SQL que combina las horas diarias con la frecuencia mensual, reflejando el volumen real de consumo.
            * **`weekly_ads_listened` (Anuncios Semanales):** Total de anuncios comerciales escuchados a la semana (solo aplica al plan Free).
            * **`ad_exposure_per_hour` (Fatiga de Anuncios):** Promedio de publicidad por cada hora escuchada. Permite vigilar si la cantidad de comerciales se vuelve insoportable para el usuario.
            """)

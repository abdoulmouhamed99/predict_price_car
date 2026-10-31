# ============================================================
# 1. IMPORTATION DES BIBLIOTHÈQUES
# ============================================================

import streamlit as st
import joblib
import pandas as pd
import numpy as np


# ============================================================
# 2. CONFIGURATION DE LA PAGE
# ============================================================

st.set_page_config(
    page_title="AutoPrix | Estimation de voitures",
    page_icon="🚗",
    layout="wide"
)


# ============================================================
# 3. DESIGN (CSS)
# ============================================================

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Playfair+Display:wght@600;700&display=swap');

:root {
    --gold: #D4AF37;
    --gold-light: #F1D98A;
    --bg: #0B0D12;
    --card: rgba(255, 255, 255, 0.04);
    --border: rgba(212, 175, 55, 0.22);
    --muted: #9AA0AE;
}

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif;
    color: #ECEEF3;
}

.stApp {
    background:
        radial-gradient(900px 500px at 85% -10%, rgba(212,175,55,0.14), transparent 60%),
        radial-gradient(700px 400px at -10% 30%, rgba(90,110,255,0.10), transparent 60%),
        var(--bg);
}

#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { padding-top: 2rem; max-width: 1150px; }

/* ---------- Bandeau d'accueil ---------- */
.hero {
    padding: 2.6rem 2.4rem;
    border-radius: 24px;
    border: 1px solid var(--border);
    background: linear-gradient(135deg, rgba(212,175,55,0.12), rgba(255,255,255,0.02));
    margin-bottom: 1.6rem;
}
.hero .badge {
    display: inline-block;
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--gold);
    border: 1px solid var(--border);
    padding: 0.3rem 0.8rem;
    border-radius: 999px;
    margin-bottom: 1rem;
}
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    line-height: 1.15;
    margin: 0 0 0.6rem 0;
    color: #fff;
}
.hero h1 span {
    background: linear-gradient(90deg, var(--gold), var(--gold-light));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p { color: var(--muted); font-size: 1.02rem; max-width: 680px; margin: 0; }

.chips { margin-top: 1.3rem; display: flex; flex-wrap: wrap; gap: 0.5rem; }
.chip {
    font-size: 0.8rem;
    padding: 0.35rem 0.8rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.09);
    color: #CBD0DB;
}

/* ---------- Onglets ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 0.5rem; border-bottom: 1px solid rgba(255,255,255,0.08); }
.stTabs [data-baseweb="tab"] {
    height: 46px;
    padding: 0 1.2rem;
    border-radius: 12px 12px 0 0;
    color: var(--muted);
    font-weight: 500;
}
.stTabs [aria-selected="true"] { color: var(--gold) !important; }
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--gold) !important; }

/* ---------- Titres de section ---------- */
h2, h3, [data-testid="stSubheader"] {
    font-family: 'Playfair Display', serif !important;
    color: #fff !important;
}

/* ---------- Champs ---------- */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div,
div[data-baseweb="base-input"] {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 12px !important;
}
div[data-baseweb="input"] > div:focus-within,
div[data-baseweb="select"] > div:focus-within {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 1px var(--gold) !important;
}
label, [data-testid="stWidgetLabel"] p {
    color: #CBD0DB !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

/* ---------- Boutons ---------- */
.stButton > button, .stDownloadButton > button {
    border-radius: 12px;
    padding: 0.7rem 1.6rem;
    font-weight: 600;
    border: 1px solid var(--border);
    transition: all 0.2s ease;
}
.stButton > button[kind="primary"], .stDownloadButton > button {
    background: linear-gradient(90deg, var(--gold), var(--gold-light));
    color: #14110A;
    border: none;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 28px rgba(212,175,55,0.28);
}

/* ---------- Carte de résultat ---------- */
.result-card {
    margin-top: 1.4rem;
    padding: 2rem;
    border-radius: 22px;
    text-align: center;
    border: 1px solid var(--border);
    background: linear-gradient(160deg, rgba(212,175,55,0.16), rgba(255,255,255,0.03));
}
.result-card .label {
    font-size: 0.78rem; letter-spacing: 0.2em; text-transform: uppercase; color: var(--muted);
}
.result-card .price {
    font-family: 'Playfair Display', serif;
    font-size: 3.6rem; font-weight: 700; margin: 0.4rem 0;
    background: linear-gradient(90deg, var(--gold), var(--gold-light));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.result-card .note { color: var(--muted); font-size: 0.85rem; }

/* ---------- Indicateurs ---------- */
[data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricLabel"] p { color: var(--muted) !important; }
[data-testid="stMetricValue"] { color: var(--gold-light); font-family: 'Playfair Display', serif; }

/* ---------- Zone d'import ---------- */
[data-testid="stFileUploaderDropzone"] {
    background: var(--card);
    border: 1.5px dashed var(--border);
    border-radius: 18px;
    padding: 1.6rem;
}

/* ---------- Tableaux et alertes ---------- */
[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; border: 1px solid rgba(255,255,255,0.08); }
[data-testid="stAlert"] { border-radius: 14px; }

/* ---------- Pied de page ---------- */
.footer {
    text-align: center; color: var(--muted); font-size: 0.8rem;
    margin-top: 3rem; padding-top: 1.4rem;
    border-top: 1px solid rgba(255,255,255,0.07);
}
.footer b { color: var(--gold); font-weight: 500; }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# ============================================================
# 4. CHARGEMENT DES OBJETS SAUVEGARDÉS
# ============================================================

@st.cache_resource
def load_objects():

    encoders = joblib.load("encoders.joblib")
    from xgboost import XGBRegressor
    xgb_model = XGBRegressor()
    xgb_model.load_model("xgb2_model.json")
    uniques = joblib.load("uniques.joblib")
    scaler = joblib.load("scaler.joblib")

    return encoders, xgb_model, uniques, scaler


encoders, xgb_model, uniques, scaler = load_objects()


# ============================================================
# 5. FONCTIONS DE PRÉDICTION
# ============================================================

def Pred_func(Kms_Driven, Present_Price, Fuel_Type,
              Seller_Type, Transmission, Age):

    Fuel_Type = encoders[0].transform([Fuel_Type])[0]
    Seller_Type = encoders[1].transform([Seller_Type])[0]
    Transmission = encoders[2].transform([Transmission])[0]

    x_new = np.array([
        Kms_Driven,
        Present_Price,
        Fuel_Type,
        Seller_Type,
        Transmission,
        Age
    ])

    x_new = x_new.reshape(1, -1)
    x_new = scaler.transform(x_new)
    y_pred = xgb_model.predict(x_new)

    return round(float(y_pred[0]), 2)


def Pred_func_df(df):

    df = df.copy()

    colonnes = [
        "Kms_Driven",
        "Present_Price",
        "Fuel_Type",
        "Seller_Type",
        "Transmission",
        "Age"
    ]

    if not all(col in df.columns for col in colonnes):
        raise ValueError(
            "Le fichier CSV doit contenir : " + ", ".join(colonnes)
        )

    predictions = []

    for _, row in df.iterrows():
        y_pred = Pred_func(
            row["Kms_Driven"],
            row["Present_Price"],
            row["Fuel_Type"],
            row["Seller_Type"],
            row["Transmission"],
            row["Age"]
        )
        predictions.append(y_pred)

    df["Selling_Price"] = predictions

    return df


# ============================================================
# 6. BANDEAU D'ACCUEIL
# ============================================================

st.markdown("""
<div class="hero">
    <div class="badge">Intelligence artificielle · XGBoost</div>
    <h1>Estimez le juste prix de <span>votre voiture</span></h1>
    <p>Renseignez les caractéristiques du véhicule et obtenez en un clic
    une estimation de son prix de vente, calculée par un modèle de Machine Learning.</p>
    <div class="chips">
        <span class="chip">🛣️ Kilométrage</span>
        <span class="chip">💰 Prix actuel</span>
        <span class="chip">⛽ Carburant</span>
        <span class="chip">🧑‍💼 Vendeur</span>
        <span class="chip">⚙️ Transmission</span>
        <span class="chip">📅 Âge</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# 7. ONGLETS
# ============================================================

tab1, tab2 = st.tabs([
    "🚘  Prédiction simple",
    "📁  Prédiction multiple"
])


# ============================================================
# 8. ONGLET 1 : PRÉDICTION SIMPLE
# ============================================================

with tab1:

    st.subheader("Caractéristiques du véhicule")

    col1, col2, col3 = st.columns(3)

    with col1:
        Kms_Driven = st.number_input(
            "Kilométrage parcouru (km)",
            min_value=0.0, value=50000.0, step=1000.0
        )
        Fuel_Type = st.selectbox("Type de carburant", options=uniques[0])

    with col2:
        Present_Price = st.number_input(
            "Prix actuel du véhicule",
            min_value=0.0, value=5.59, step=0.1
        )
        Seller_Type = st.selectbox("Type de vendeur", options=uniques[1])

    with col3:
        Age = st.number_input(
            "Âge du véhicule (années)",
            min_value=0, max_value=99, value=5, step=1
        )
        Transmission = st.selectbox("Type de transmission", options=uniques[2])

    st.write("")

    if st.button("✨ Estimer le prix de vente", type="primary"):

        try:
            prediction = Pred_func(
                Kms_Driven, Present_Price, Fuel_Type,
                Seller_Type, Transmission, Age
            )

            st.markdown(f"""
            <div class="result-card">
                <div class="label">Prix de vente estimé</div>
                <div class="price">{prediction:.2f} $</div>
                <div class="note">Estimation fournie par le modèle XGBoost</div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Erreur de prédiction : {e}")


# ============================================================
# 9. ONGLET 2 : PRÉDICTION MULTIPLE
# ============================================================

with tab2:

    st.subheader("Estimer plusieurs voitures d'un coup")

    st.markdown(
        "Importez un fichier CSV contenant les colonnes "
        "`Kms_Driven`, `Present_Price`, `Fuel_Type`, `Seller_Type`, "
        "`Transmission` et `Age` (ou `Year`)."
    )

    uploaded_file = st.file_uploader(
        "Importer votre fichier CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:
            df = pd.read_csv(uploaded_file, sep=None, engine="python")

            if "Age" not in df.columns and "Year" in df.columns:
                df["Age"] = 2026 - df["Year"]

            st.markdown("##### Aperçu des données importées")
            st.dataframe(df.head(), use_container_width=True)

            if st.button("🚀 Lancer les prédictions", type="primary"):

                resultats = Pred_func_df(df)

                st.success("Prédictions réalisées avec succès !")

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Voitures", f"{len(resultats)}")
                k2.metric("Prix moyen", f"{resultats['Selling_Price'].mean():.2f} $")
                k3.metric("Prix minimum", f"{resultats['Selling_Price'].min():.2f} $")
                k4.metric("Prix maximum", f"{resultats['Selling_Price'].max():.2f} $")

                st.markdown("##### Résultats détaillés")
                st.dataframe(resultats, use_container_width=True)

                csv = resultats.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="📥 Télécharger les prédictions",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Erreur : {e}")


# ============================================================
# 10. PIED DE PAGE
# ============================================================

st.markdown(
    '<div class="footer">AutoPrix · Prédiction du prix de vente des voitures '
    '· Modèle <b>XGBoost</b></div>',
    unsafe_allow_html=True
)

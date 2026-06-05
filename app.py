import pandas as pd
import streamlit as st
from pathlib import Path
from fraud_detection import detect_fraud, load_transactions

# Configuration globale
SAMPLE_CSV = Path(__file__).parent / "data" / "sample_transactions.csv"
CONVERSION_RATE = 655  # 1 EUR approx 655 FCFA

def apply_custom_style():
    st.markdown("""
        <style>
        .main {
            background-color: #0e1117;
        }
        .stMetric {
            background-color: #1e2130;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #3e445e;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #1e2130;
            border-radius: 4px 4px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #2e344e;
            border-bottom: 2px solid #00d4ff;
        }
        h1 {
            color: #00d4ff;
            font-family: 'Inter', sans-serif;
            font-weight: 800;
        }
        .ai-badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            background: linear-gradient(90deg, #00d4ff 0%, #0055ff 100%);
            color: white;
            font-size: 10px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

def render_interface(transactions: list[dict], results: list[dict]) -> None:
    apply_custom_style()
    
    st.markdown('<div class="ai-badge">NEURAL ENGINE ACTIVE</div>', unsafe_allow_html=True)
    st.title("🛡️ SENTINEL AI : Cyber-Surveillance")
    st.caption("Système autonome de détection de fraudes bancaires en temps réel")

    # Calcul des statistiques avancées en FCFA
    total_tx = len(results)
    suspicious_tx = [r for r in results if r["is_suspicious"]]
    nb_alerts = len(suspicious_tx)
    total_amount_raw = sum(tx["amount"] for tx in transactions if tx["amount"] is not None)
    total_amount_fcfa = total_amount_raw * CONVERSION_RATE
    alert_rate = (nb_alerts / total_tx) * 100 if total_tx > 0 else 0
    avg_score = sum(r["fraud_score"] for r in results) / total_tx if total_tx > 0 else 0

    # Dashboard Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("FLUX ANALYSÉS", f"{total_tx}", delta="Live Feed")
    with col2:
        st.metric("ANOMALIES IA", nb_alerts, f"{alert_rate:.1f}%", delta_color="inverse")
    with col3:
        st.metric("VOLUME SÉCURISÉ", f"{total_amount_fcfa:,.0f} FCFA")
    with col4:
        st.metric("INDICE DE CONFIANCE", f"{100 - (avg_score * 100):.1f}%", delta="Optimisé")

    st.write("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Centre d'Intervention", 
        "🌐 Matrice Géospatiale", 
        "📊 Intelligence Marché",
        "⚙️ Noyau de Configuration"
    ])

    with tab1:
        st.subheader("🚨 Alertes Détectées par le Moteur Neural")
        if not suspicious_tx:
            st.success("Aucune menace détectée. Le réseau est sain.")
        else:
            df_alerts = []
            for r in suspicious_tx:
                tx = next(t for t in transactions if t["transaction_id"] == r["transaction_id"])
                amount_fcfa = (tx['amount'] or 0) * CONVERSION_RATE
                df_alerts.append({
                    "Priorité": "🔴 CRITIQUE" if r["fraud_score"] > 0.85 else "🟠 HAUTE",
                    "ID": r["transaction_id"],
                    "Client": tx["user_id"],
                    "Montant": f"{amount_fcfa:,.0f} FCFA",
                    "Score IA": r["fraud_score"],
                    "Analyse de l'IA": ", ".join(r.get("risk_factors", [r["reason"]]))
                })
            
            st.dataframe(
                df_alerts,
                column_config={
                    "Score IA": st.column_config.ProgressColumn(
                        "Probabilité de Fraude",
                        format="%.2f",
                        min_value=0, max_value=1,
                        color="red"
                    ),
                    "Analyse de l'IA": st.column_config.TextColumn("Raisonnement du Modèle", width="large")
                },
                hide_index=True,
                use_container_width=True
            )

            # Espace d'investigation
            st.markdown("### 🔍 Investigation Approfondie")
            selected = st.selectbox("Sélectionner une cible :", [d["ID"] for d in df_alerts])
            if selected:
                res = next(r for r in results if r["transaction_id"] == selected)
                tx = next(t for t in transactions if t["transaction_id"] == selected)
                amount_fcfa = (tx['amount'] or 0) * CONVERSION_RATE
                
                c_info, c_reason = st.columns([1, 2])
                with c_info:
                    st.markdown("**Métadonnées Transactionnelles**")
                    display_tx = tx.copy()
                    display_tx["amount_fcfa"] = f"{amount_fcfa:,.0f} FCFA"
                    st.json(display_tx)
                with c_reason:
                    st.markdown(f"**Verdict Sentinel AI :** `{res['fraud_score']*100:.1f}% de risque`")
                    st.info(f"**Justification Neuronale :** {res['reason']}")
                    st.markdown("""
                    **Contre-mesures automatiques disponibles :**
                    - 🔒 Geler le compte immédiatement
                    - 📱 Exiger une authentification biométrique
                    - 📞 Escalader à l'unité d'intervention humaine
                    """)

    with tab2:
        st.subheader("📡 Visualisation des Flux Mondiaux")
        coords = {
            "FR": [46.2, 2.2], "JP": [36.2, 138.3], "US": [37.1, -95.7], "TG": [8.6, 0.8],
            "BJ": [9.3, 2.3], "CI": [7.5, -5.5], "SN": [14.5, -14.5], "NG": [9.1, 8.7],
            "ML": [17.5, -4.0], "NE": [17.6, 8.0], "BF": [12.2, -1.6], "GN": [10.0, -11.0]
        }
        map_data = []
        for r in results:
            tx = next(t for t in transactions if t["transaction_id"] == r["transaction_id"])
            c = coords.get(tx["country"])
            if c:
                map_data.append({
                    "lat": c[0], "lon": c[1], 
                    "status": "Alerte" if r["is_suspicious"] else "Normal",
                    "color": "#FF0000" if r["is_suspicious"] else "#00FF00",
                    "size": ((tx["amount"] or 0) * CONVERSION_RATE) / 500 + 100
                })
        
        if map_data:
            df_map = pd.DataFrame(map_data)
            st.map(df_map, color="color", size="size")
            st.caption("Carte thermique des transactions : le rayon indique le volume en FCFA, la couleur l'intégrité.")
        else:
            st.info("Attente de données géospatiales...")

    with tab3:
        st.subheader("🏢 Analyse des Vecteurs Marchands")
        m_list = []
        for tx in transactions:
            res = next(r for r in results if r["transaction_id"] == tx["transaction_id"])
            m_list.append({
                "Marchand": tx["merchant"], 
                "Fraude": 1 if res["is_suspicious"] else 0, 
                "Volume (FCFA)": (tx["amount"] or 0) * CONVERSION_RATE
            })
        
        df_m = pd.DataFrame(m_list)
        m_agg = df_m.groupby("Marchand").agg({
            "Fraude": "sum", 
            "Volume (FCFA)": "sum", 
            "Marchand": "count"
        }).rename(columns={"Marchand": "Total"})
        m_agg["Risque %"] = (m_agg["Fraude"] / m_agg["Total"]) * 100
        
        st.dataframe(m_agg.sort_values("Fraude", ascending=False), use_container_width=True)

    with tab4:
        st.subheader("🧠 Noyau Neural : Configuration")
        st.write("Ajustez les paramètres de l'IA en temps réel.")
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.slider("Seuil de Détection (Sensibilité)", 0.0, 1.0, 0.7)
            st.select_slider("Mode d'Apprentissage", options=["Conservateur", "Équilibré", "Agressif"])
        with col_s2:
            st.toggle("Analyse Comportementale", value=True)
            st.toggle("Détection d'Incohérence IP", value=False)
            st.toggle("Réseau de Neurones Profonds (DNN)", value=True)

def main() -> None:
    st.set_page_config(
        page_title="Sentinel AI | Fraud Defense",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    with st.sidebar:
        st.markdown("# 🛡️ SENTINEL AI")
        st.markdown("---")
        st.markdown("### 📥 Injection de Données")
        use_sample = st.toggle("Utiliser le Flux de Démo", value=True)
        
        transactions = []
        if use_sample:
            transactions = load_transactions(str(SAMPLE_CSV))
            st.success("Flux de démo connecté")
        else:
            uploaded = st.file_uploader("Importer Data Ledger (CSV)", type=["csv"])
            if uploaded:
                tmp = Path(".tmp_ledger.csv")
                tmp.write_bytes(uploaded.getvalue())
                transactions = load_transactions(str(tmp))
                tmp.unlink(missing_ok=True)
                st.success("Données injectées")

        st.markdown("---")
        st.markdown("### 🚦 État du Système")
        st.info("IA : Active\n\nNoyau : Stable\n\nFlux : Temps Réel")

    if not transactions:
        st.warning("Veuillez injecter des données transactionnelles pour activer Sentinel AI.")
        return

    # Auto-run analysis for better UX
    try:
        results = detect_fraud(transactions)
        render_interface(transactions, results)
    except Exception as exc:
        st.error(f"Erreur de traitement du noyau neural : {exc}")

if __name__ == "__main__":
    main()

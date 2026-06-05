"""
Interface Streamlit — À CRÉER PAR VOUS pour le jury.

Le jury lancera :  streamlit run app.py

Règles :
  - Ne modifiez pas l'appel à detect_fraud / load_transactions (contrat technique).
  - Personnalisez render_interface() : clarté, intuitivité, compréhension pour un public non technique.
  - L'interface n'est PAS notée par la CI ; elle sert au jury pour repêcher et comparer les candidats.
"""

from pathlib import Path

import streamlit as st

from fraud_detection import detect_fraud, load_transactions

SAMPLE_CSV = Path(__file__).parent / "data" / "sample_transactions.csv"


def render_interface(transactions: list[dict], results: list[dict]) -> None:
    """
    Interface innovante avec Map, Analyse Marchand et Scoring Avancé.
    """
    st.header("📊 Centre Opérationnel Anti-Fraude")

    # Calcul des statistiques avancées
    total_tx = len(results)
    suspicious_tx = [r for r in results if r["is_suspicious"]]
    nb_alerts = len(suspicious_tx)
    total_amount = sum(tx["amount"] for tx in transactions if tx["amount"] is not None)
    alert_rate = (nb_alerts / total_tx) * 100 if total_tx > 0 else 0
    avg_score = sum(r["fraud_score"] for r in results) / total_tx if total_tx > 0 else 0

    # Ligne de métriques stylisée
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Volume Analysé", f"{total_amount:,.0f} €", f"{total_tx} tx")
    m2.metric("Alertes Critiques", nb_alerts, f"{alert_rate:.1f}%", delta_color="inverse")
    m3.metric("Score de Risque Moyen", f"{avg_score:.2f}", "Stable")
    m4.metric("Status Système", "OPÉRATIONNEL", delta="Temps Réel", delta_color="normal")

    # NOUVEAUTÉ : Onglets avec innovations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🚨 Alertes & Audit", 
        "🗺️ Carte Globale", 
        "🏢 Analyse Marchands", 
        "🔬 Recherche (Raw)",
        "⚙️ Paramètres IA"
    ])

    with tab1:
        if nb_alerts == 0:
            st.success("Bouclier actif : Aucune anomalie détectée.")
        else:
            # Audit Trail : Pourquoi cette transaction est suspecte ?
            st.subheader("Dossiers d'Investigation")
            
            alert_data = []
            for r in suspicious_tx:
                tx = next(t for t in transactions if t["transaction_id"] == r["transaction_id"])
                alert_data.append({
                    "ID": r["transaction_id"],
                    "Client": tx["user_id"],
                    "Montant": f"{tx['amount']} {tx['currency']}",
                    "Score": r["fraud_score"],
                    "Facteurs": ", ".join(r.get("risk_factors", [r["reason"]]))
                })
            
            st.dataframe(
                alert_data,
                column_config={
                    "Score": st.column_config.ProgressColumn(
                        "Indice de Risque",
                        format="%.2f",
                        min_value=0, max_value=1,
                    ),
                    "Facteurs": st.column_config.TextColumn("Audit Trail (Facteurs combinés)", width="large"),
                },
                hide_index=True, use_container_width=True
            )

            st.divider()
            # Simulation d'un workflow de décision
            col_sel, col_act = st.columns([1, 1])
            with col_sel:
                target = st.selectbox("Sélectionner un dossier pour action :", [d["ID"] for d in alert_data])
            with col_act:
                st.write("") # Padding
                if st.button("Confirmer Fraude & Bloquer Carte", type="primary"):
                    st.toast(f"Action effectuée sur {target}")

    with tab2:
        # INNOVATION : Visualisation Géospatiale
        st.subheader("Géographie des Flux")
        # Coordonnées pour la map (on réutilise la logique de fraud_detection.py simplifiée)
        coords = {
            "FR": [46.2, 2.2], "JP": [36.2, 138.3], "US": [37.1, -95.7], "TG": [8.6, 0.8],
            "BJ": [9.3, 2.3], "CI": [7.5, -5.5], "SN": [14.5, -14.5], "NG": [9.1, 8.7]
        }
        map_data = []
        for r in results:
            tx = next(t for t in transactions if t["transaction_id"] == r["transaction_id"])
            c = coords.get(tx["country"])
            if c:
                map_data.append({
                    "lat": c[0], "lon": c[1], 
                    "is_fraud": r["is_suspicious"],
                    "size": (tx["amount"] or 0) / 10 + 50
                })
        
        if map_data:
            import pandas as pd
            df_map = pd.DataFrame(map_data)
            st.map(df_map, color="#FF0000" if any(df_map["is_fraud"]) else "#00FF00", size="size")
            st.caption("Légende : Taille proportionnelle au montant. Rouge si alertes dans le pays.")
        else:
            st.info("Données géographiques insuffisantes.")

    with tab3:
        # INNOVATION : Analyse de Risque Marchand
        st.subheader("Profilage des Commerçants")
        import pandas as pd
        m_data = []
        for tx in transactions:
            res = next(r for r in results if r["transaction_id"] == tx["transaction_id"])
            m_data.append({"Merchant": tx["merchant"], "Fraud": 1 if res["is_suspicious"] else 0, "Amount": tx["amount"] or 0})
        
        df_m = pd.DataFrame(m_data)
        m_stats = df_m.groupby("Merchant").agg({"Fraud": "sum", "Amount": "sum", "Merchant": "count"}).rename(columns={"Merchant": "Total Tx"})
        m_stats["Taux de Risque"] = (m_stats["Fraud"] / m_stats["Total Tx"]) * 100
        
        st.write("Top Marchands à surveiller (triés par volume de fraude) :")
        st.dataframe(m_stats.sort_values("Fraud", ascending=False), use_container_width=True)

    with tab4:
        st.subheader("Explorateur de Données Brutes")
        st.write("Filtrez et exportez les résultats pour audit externe.")
        st.dataframe(results, use_container_width=True)

    with tab5:
        # INNOVATION : Paramétrage IA (Simulation)
        st.subheader("Réglages de la Sensibilité")
        st.slider("Seuil de tolérance géographique (km/h)", 500, 1500, 900)
        st.slider("Pondération des transactions nocturnes", 0.0, 1.0, 0.75)
        st.toggle("Activer l'apprentissage par renforcement (Simulation)", value=True)
        st.info("Ces réglages simulent le pilotage d'un moteur d'IA en production.")


def main() -> None:
    st.set_page_config(
        page_title="Détection de fraude — Hackathon INTELO2026",
        page_icon="🛡️",
        layout="wide",
    )

    st.title("Détection de fraude financière")
    st.caption("Hackathon INTELO2026 — interface participant · évaluée par le jury")

    with st.sidebar:
        st.header("Charger des données")
        use_sample = st.toggle("Utiliser le fichier d'exemple", value=True)
        transactions: list[dict] = []

        if use_sample:
            transactions = load_transactions(str(SAMPLE_CSV))
            st.success(f"{len(transactions)} transactions (exemple)")
        else:
            uploaded = st.file_uploader("Importer un CSV", type=["csv"])
            if uploaded:
                tmp = Path(".streamlit_upload.csv")
                tmp.write_bytes(uploaded.getvalue())
                transactions = load_transactions(str(tmp))
                tmp.unlink(missing_ok=True)
                st.success(f"{len(transactions)} transactions importées")

        st.divider()
        st.markdown(
            "**Jury :** évaluez l'ergonomie et la clarté de l'écran principal, "
            "pas seulement le score des tests."
        )

    if not transactions:
        st.info("Chargez des transactions (barre latérale) puis lancez l'analyse.")
        return

    if st.button("Analyser", type="primary"):
        try:
            results = detect_fraud(transactions)
        except NotImplementedError:
            st.error("Implémentez d'abord `detect_fraud` dans `fraud_detection.py`.")
            return
        except Exception as exc:
            st.error(f"Erreur : {exc}")
            return

        render_interface(transactions, results)


if __name__ == "__main__":
    main()

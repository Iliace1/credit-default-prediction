import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

# === Mise en page ===
st.set_page_config(page_title="Prédiction de défaut de crédit", page_icon="💳")
st.title("💳 Application de prédiction de défaut de paiement")
st.markdown("Remplis les informations du client pour prédire s'il risque de faire défaut.")

# === Formulaire utilisateur ===
credit_lines = st.number_input("📊 Lignes de crédit actives", min_value=0, max_value=10, value=1)
loan_amt = st.number_input("💰 Montant du prêt (USD)", min_value=0.0, max_value=15000.0, value=3000.0)
total_debt = st.number_input("📉 Dette totale (USD)", min_value=0.0, max_value=50000.0, value=10000.0)
income = st.number_input("📈 Revenu annuel (USD)", min_value=0.0, max_value=200000.0, value=70000.0)
years_emp = st.slider("👨‍💼 Années d'emploi", 0, 10, 5)
fico = st.slider("📊 FICO Score", 300, 850, 650)

# === Préparation des features ===
features = pd.DataFrame(np.array([[credit_lines, loan_amt, total_debt, income, years_emp, fico]]),
                        columns=[
                            'credit_lines_outstanding',
                            'loan_amt_outstanding',
                            'total_debt_outstanding',
                            'income',
                            'years_employed',
                            'fico_score'
                        ])

# === Chargement du modèle ===
st.markdown("---")
st.subheader("📦 Chargement du modèle...")

try:
    run_id = "da8faa09e4c74a92be91aa449685e294"  # ← Ton run ID ici
    experiment_id = "330662523080338550"
    model_uri = f"mlruns/{experiment_id}/{run_id}/artifacts/random_forest_model"

    model = mlflow.sklearn.load_model(model_uri)
    st.success("Modèle chargé avec succès ✅")

    # === Prédiction ===
    if st.button("🔍 Prédire le risque"):
        prediction = model.predict(features)[0]
        probas = model.predict_proba(features)[0]
        probas = probas[1] if len(probas) > 1 else 0.0

        if prediction == 1:
            st.error(f"⚠️ Risque ÉLEVÉ de défaut de paiement ({probas:.2%})")
        else:
            st.success(f"✅ Client fiable ({1 - probas:.2%} de chance de remboursement)")

        # === Logger les inputs + résultats dans MLflow ===
        with mlflow.start_run(run_name="prediction_run"):
            mlflow.log_params({
                "credit_lines_outstanding": credit_lines,
                "loan_amt_outstanding": loan_amt,
                "total_debt_outstanding": total_debt,
                "income": income,
                "years_employed": years_emp,
                "fico_score": fico,
                "prediction": int(prediction),
            })
            mlflow.log_metric("default_probability", float(probas))

        # === Info explicative ===
        st.markdown("---")
        st.subheader("🧠 Explication de la prédiction (SHAP)")
        st.info("⚠️ L'explication automatique n’est pas disponible actuellement.\nLa prédiction reste fiable grâce au modèle RandomForest. ✅")

except Exception as e:
    st.error(f"❌ Erreur lors du chargement du modèle ou de la prédiction : {str(e)}")

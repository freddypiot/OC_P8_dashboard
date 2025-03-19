
# programme python permettant d'afficher les prédictions de refus de crédit et les graphiques SHAP

import pandas as pd
import joblib
import re
import json
from flask import Flask, request, jsonify, send_file
import shap
import matplotlib.pyplot as plt
import io

# Charger les données
df_read = pd.read_csv('export_base_credit_1000.csv', sep="!")
df_moyennes = pd.read_csv('base_credit_moyennes.csv', sep="!")
df_read = df_read.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
seuil = 0.1  # Seuil de classification

# Charger le modèle
model = joblib.load("saved_model.pkl")
X = df_read.drop('TARGET', axis=1)
y = df_read['TARGET']
df_read = df_read.drop('TARGET', axis=1)

# Fonction pour afficher les graphiques SHAP
explainer = shap.Explainer(model, X)
shap_values = explainer(X, check_additivity=False)

app = Flask(__name__)

@app.route('/test', methods=['POST'])
def test():
    result = {
        "id_client": 1000
    }
    return jsonify(result)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    id_client = data.get('id_client')

    if id_client is None:
        return jsonify({"error": "ID client manquant"}), 400

    input_df = df_read[df_read['SK_ID_CURR'] == id_client]

    if input_df.empty:
        return jsonify({"error": "Client inconnu"}), 404

    probabilities = model.predict_proba(input_df)
    probabilite = probabilities[0][1]
    prediction = int(probabilite >= seuil)

    # Données descriptive du client
    age = int(input_df['DAYS_BIRTH'] / -365)
    revenu_annuel = int(input_df['AMT_INCOME_TOTAL'])
    montant_mensualite = int(input_df['AMT_ANNUITY'])
    montant_credit = int(input_df['AMT_CREDIT'])
    duree_credit = int(input_df['AMT_CREDIT'] / input_df['AMT_ANNUITY'])

    # Données moyennes tous clients confondus
    age_moyen = int(df_moyennes[df_moyennes['FEATURE'] == 'DAYS_BIRTH']['MOYENNE'].values[0] / -365)
    revenu_annuel_moyen = int(df_moyennes[df_moyennes['FEATURE'] == 'AMT_INCOME_TOTAL']['MOYENNE'].values[0])
    montant_mensualite_moyen = int(df_moyennes[df_moyennes['FEATURE'] == 'AMT_ANNUITY']['MOYENNE'].values[0])
    montant_credit_moyen = int(df_moyennes[df_moyennes['FEATURE'] == 'AMT_CREDIT']['MOYENNE'].values[0])
    duree_credit_moyen = int(df_moyennes[df_moyennes['FEATURE'] == 'DUREE_CREDIT']['MOYENNE'].values[0])

    result = {
        "id_client": id_client,
        "refus_credit": prediction,
        "probabilite": round(probabilite, 4),
        "age": age,
        "revenu_annuel": revenu_annuel,
        "montant_mensualite": montant_mensualite,
        "montant_credit": montant_credit,
        "duree_credit": duree_credit,
        "age_moyen": age_moyen,
        "revenu_annuel_moyen": revenu_annuel_moyen,
        "montant_mensualite_moyen": montant_mensualite_moyen,
        "montant_credit_moyen": montant_credit_moyen,
        "duree_credit_moyen": duree_credit_moyen
    }
    return jsonify(result)


@app.route('/shap_feat_imp_globale', methods=['POST'])
def shap_feat_imp_globale():
    # Générer le plot SHAP
    plt.figure()
    shap.summary_plot(shap_values, X)

    # Sauvegarder l'image dans un buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    #return send_file(buf, mimetype='image/png', as_attachment=True, attachment_filename='shap_summary.png')
    return send_file(buf, mimetype='image/png')

@app.route('/shap_feat_imp_locale', methods=['POST'])
def shap_feat_imp_locale():
    data = request.get_json()
    id_client = data.get('id_client')

    if id_client is None:
        return jsonify({"error": "ID client manquant"}), 400

    input_df = df_read[df_read['SK_ID_CURR'] == id_client]
    ligne_numero = input_df.index.tolist()[0]

    if input_df.empty:
        return jsonify({"error": "Client inconnu"}), 404

    plt.figure()
    shap.plots.waterfall(shap_values[ligne_numero])

    # Sauvegarder l'image dans un buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7000)

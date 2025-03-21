import streamlit as st
import joblib
import requests
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df_read = pd.read_csv('export_base_credit_1000.csv', sep="!")
df_moyennes = pd.read_csv('base_credit_moyennes.csv', sep="!")

# Charger le modèle
model = joblib.load("saved_model.pkl")
X = df_read.drop('TARGET', axis=1)
y = df_read['TARGET']

model.fit(X, y)

# Initialisation des variables dans l'état de session
if 'variable_x' not in st.session_state:
    st.session_state['variable_x'] = df_read.columns[0]
if 'variable_y' not in st.session_state:
    st.session_state['variable_y'] = df_read.columns[1]

# Titre de l'application
st.title("Prédiction de Crédit")

# Entrée utilisateur pour l'ID client
id_client = st.number_input("Entrez l'ID client", min_value=0, step=1, value=100002)

# Bouton pour soumettre l'ID client
if st.button("Prédire"):
    # URL de l'API
    url_predict = "https://oc-p8-dashboard-freddy-piot.onrender.com/predict"
    url_shap_globale = "https://oc-p8-dashboard-freddy-piot.onrender.com/shap_feat_imp_globale"
    url_shap_locale = "https://oc-p8-dashboard-freddy-piot.onrender.com/shap_feat_imp_locale"
    payload = {"id_client": id_client}
    headers = {"content-type": "application/json"}

    # Appel à l'API
    response_predict = requests.post(url_predict, json=payload, headers=headers, verify=False)
    response_shap_globale = requests.post(url_shap_globale, verify=False)
    response_shap_locale = requests.post(url_shap_locale, json=payload, verify=False)


    if response_predict.status_code == 200 & response_shap_globale.status_code == 200 & response_shap_locale.status_code == 200:
        # Afficher les résultats formatés
        result = response_predict.json()
        st.write("### Résultats de la Prédiction")
        st.write(f"**ID Client**: {result['id_client']}")
        st.write(f"**Refus de Crédit**: {'Oui' if result['refus_credit'] else 'Non'}")
        st.write(f"**Probabilité**: {result['probabilite']}")

        # Visualiser la probabilité par rapport au seuil
        seuil = 0.1
        fig, ax = plt.subplots(figsize=(6, 1))  # Set the figsize to limit the height
        if result['probabilite'] >= seuil:
            color_graph = 'red'
        else:
            color_graph = 'green'
        ax.barh(['Probabilité'], [result['probabilite']], color=color_graph)
        ax.axvline(x=seuil, color='blue', linestyle='--')
        ax.set_xlim(0, 1)
        ax.set_xlabel('Probabilité')
        st.pyplot(fig)

        # Interprétation du score
        if result['probabilite'] >= seuil:
            st.write("Le score est supérieur au seuil, donc le crédit est refusé.")
        else:
            st.write("Le score est inférieur au seuil, donc le crédit est accordé.")

        # Affichage des données descriptives du client comparées à la moyenne
        client_info = {
            "Age": str(result['age']) + ' (moyenne : ' + str(result['age_moyen']) + ')',
            "Revenu annuel": str(result['revenu_annuel']) + ' (moyenne : ' + str(result['revenu_annuel_moyen']) + ')',
            "Montant du crédit": str(result['montant_credit']) + ' (moyenne : ' + str(result['montant_credit_moyen']) + ')',
            "Durée du crédit": str(result['duree_credit']) + ' (moyenne : ' + str(result['duree_credit_moyen']) + ')'
        }

        st.write("### Informations Descriptives du Client")
        for key, value in client_info.items():
            st.write(f"**{key}**: {value}")

        # Création du scatter plot
        fig, ax = plt.subplots()
        sns.scatterplot(data=df_read, x=st.session_state['variable_x'], y=st.session_state['variable_y'], ax=ax)
        client_data = df_read[df_read['SK_ID_CURR'] == id_client]
        if not client_data.empty:
            ax.scatter(client_data[st.session_state['variable_x']], client_data[st.session_state['variable_y']], color='red', s=100, label='ID Client')
            ax.legend()
        st.pyplot(fig)      

        # Affichage des graphiques SHAP
        st.write("### Graphiques SHAP")
        st.write("#### Feature importance globale")
        response_shap_globale = requests.post(url_shap_globale, verify=False)
        st.image(response_shap_globale.content, use_container_width=True)

        st.write("#### Feature importance locale")
        response_shap_locale = requests.post(url_shap_locale, json=payload, verify=False)
        st.image(response_shap_locale.content, use_container_width=True)


    else:
        st.error("Erreur lors de la prédiction. Veuillez vérifier l'ID client et réessayer.")

# Listes des variables avec feature importance élevée
variables_liste_deroulante = [
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "PAYMENT_RATE",
    "CODE_GENDER",
    "INSTAL_DPD_MEAN",
    "AMT_ANNUITY",
    "APPROVED_CNT_PAYMENT_MEAN",
    "DAYS_EMPLOYED",
    "AMT_GOODS_PRICE",
    "INSTAL_AMT_PAYMENT_SUM",
    "NAME_EDUCATION_TYPE_Highereducation",
    "BURO_AMT_CREDIT_SUM_DEBT_MEAN",
    "NAME_FAMILY_STATUS_Married",
    "PREV_CNT_PAYMENT_MEAN"
]
# Création de la barre latérale

st.sidebar.image('logo_pret_a_depenser.png', width=300, caption='Prêt à Dépenser')

# Sélection des variables pour le scatter plot
st.sidebar.title("Sélection des Variables")
st.session_state['variable_x'] = st.sidebar.selectbox("Choisissez la variable pour l'axe X", variables_liste_deroulante)
st.session_state['variable_y'] = st.sidebar.selectbox("Choisissez la variable pour l'axe Y", variables_liste_deroulante)

st.sidebar.write("### Description des variables")


st.sidebar.write("**EXT_SOURCE_1** - Score externe 1")
st.sidebar.write("**EXT_SOURCE_2** - Score externe 2")
st.sidebar.write("**EXT_SOURCE_3** - Score externe 3")
st.sidebar.write("**PAYMENT_RATE** - Taux de paiement (mensualité / total)")
st.sidebar.write("**CODE_GENDER** - Genre du client")
st.sidebar.write("**INSTAL_DPD_MEAN** - Retard de paiement (jours)")
st.sidebar.write("**AMT_ANNUITY** - Montant de la mensualité")
st.sidebar.write("**APPROVED_CNT_PAYMENT_MEAN** - Moyenne des paiements approuvés")
st.sidebar.write("**DAYS_EMPLOYED** - Ancienneté emploi (jours)")
st.sidebar.write("**AMT_GOODS_PRICE** - Prix des biens")
st.sidebar.write("**INSTAL_AMT_PAYMENT_SUM** - Montants de paiement précédents prêts internes")
st.sidebar.write("**NAME_EDUCATION_TYPE_Highereducation** - Enseignement supérieur")
st.sidebar.write("**BURO_AMT_CREDIT_SUM_DEBT_MEAN** - Moyenne des dettes de crédit externes")
st.sidebar.write("**NAME_FAMILY_STATUS_Married** - Statut familial - Marié")
st.sidebar.write("**PREV_CNT_PAYMENT_MEAN** - Moyenne des paiements précédents")



import streamlit as st
import joblib
import requests
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import altair as alt
import plotly.express as px
import plotly.graph_objects as go

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
simu_retard = st.number_input("Simulation : Retard de paiement", min_value=0, max_value=80, step=5, value=None, placeholder="Entrez le retard de paiement à simuler entre 0 et 80 jours")
simu_annuite = st.number_input("Simulation : montant de l'annuité", min_value=0, max_value=100000, step=5000, value=None, placeholder="Entrez l'annuité à simuler entre 0 et 100 K€")
simu_anciennete = st.number_input("Simulation : ancienneté emploi (années)", min_value=0, max_value=40,step=1, value=None, placeholder="Entrez l'ancienneté à simuler entre 0 et 40 ans")

# Bouton pour soumettre l'ID client
if st.button("Prédire"):
    # URL de l'API
    url_predict = "https://oc-p8-dashboard-freddy-piot.onrender.com/predict"
    url_shap_globale = "https://oc-p8-dashboard-freddy-piot.onrender.com/shap_feat_imp_globale"
    url_shap_locale = "https://oc-p8-dashboard-freddy-piot.onrender.com/shap_feat_imp_locale"
    payload = {
        "id_client": id_client,
        "simu_retard": simu_retard,
        "simu_annuite": simu_annuite,
        "simu_anciennete": simu_anciennete
        }
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
            "Durée du crédit": str(result['duree_credit']) + ' (moyenne : ' + str(result['duree_credit_moyen']) + ')',
            "Retard de paiement": str(result['retard']),
            "Annuité": str(result['annuite']),
            "Ancienneté": str(int(-result['anciennete'] / 365.25))
        }

        st.write("### Informations Descriptives du Client")
        for key, value in client_info.items():
            st.write(f"**{key}**: {value}")

        # Création du scatter plot
        fig, ax = plt.subplots()
        sns.scatterplot(data=df_read, x=st.session_state['variable_x'], y=st.session_state['variable_y'], ax=ax)
        client_data = df_read[df_read['SK_ID_CURR'] == id_client]
        if not client_data.empty:
            ax.scatter(client_data[st.session_state['variable_x']], client_data[st.session_state['variable_y']], color='red', marker='x', s=100, label='ID Client')
            ax.legend()
        st.pyplot(fig)

        st.write("### Croisement de l'annuité et de l'ancienneté ")

        # Création d'un second graphique interactif
        fig = px.scatter(df_read, x='AMT_ANNUITY', y='DAYS_EMPLOYED')        
        fig.add_trace(
            go.Scatter(
            x=client_data['AMT_ANNUITY'],
            y=client_data['DAYS_EMPLOYED'],
            mode='markers',
            marker=dict(color='red', size=10, symbol='x'),
            name='ID Client'
            )
        )

        # Ajouter un texte de remplacement au graphique
        fig.add_annotation(
            x=client_data['AMT_ANNUITY'],
            y=client_data['DAYS_EMPLOYED'],
            text="ID Client",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40
        )

        # Afficher le graphique dans Streamlit
        st.plotly_chart(fig)

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
st.sidebar.write("**AMT_ANNUITY** - Montant des annuités")
st.sidebar.write("**APPROVED_CNT_PAYMENT_MEAN** - Moyenne des paiements approuvés")
st.sidebar.write("**DAYS_EMPLOYED** - Ancienneté emploi (jours)")
st.sidebar.write("**AMT_GOODS_PRICE** - Prix des biens")
st.sidebar.write("**INSTAL_AMT_PAYMENT_SUM** - Montants de paiement précédents prêts internes")
st.sidebar.write("**NAME_EDUCATION_TYPE_Highereducation** - Enseignement supérieur")
st.sidebar.write("**BURO_AMT_CREDIT_SUM_DEBT_MEAN** - Moyenne des dettes de crédit externes")
st.sidebar.write("**NAME_FAMILY_STATUS_Married** - Statut familial - Marié")
st.sidebar.write("**PREV_CNT_PAYMENT_MEAN** - Moyenne des paiements précédents")

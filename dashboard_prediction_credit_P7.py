import streamlit as st
import requests
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df_read = pd.read_csv('export_base_credit_1000.csv', sep="!")
df_moyennes = pd.read_csv('base_credit_moyennes.csv', sep="!")

# Titre de l'application
st.title("Prédiction de Crédit")

# Entrée utilisateur pour l'ID client
id_client = st.number_input("Entrez l'ID client", min_value=0, step=1, value=100002)

# Bouton pour soumettre l'ID client
if st.button("Prédire"):
    # URL de l'API
    url = "https://oc-p7-scoring-freddy-piot.onrender.com/predict"
    payload = {"id_client": id_client}
    headers = {"content-type": "application/json"}

    # Appel à l'API
    response = requests.post(url, json=payload, headers=headers, verify=False)

    if response.status_code == 200:
        # Afficher les résultats formatés
        result = response.json()
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




    else:
        st.error("Erreur lors de la prédiction. Veuillez vérifier l'ID client et réessayer.")

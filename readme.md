# Projet de Scoring Crédit

## Contexte
L’entreprise "Prêt à dépenser" souhaite mettre en œuvre un outil de scoring crédit pour calculer la probabilité qu’un client rembourse son crédit, puis classifier la demande en crédit accordé ou refusé. Elle souhaite donc développer un algorithme de classification en s’appuyant sur des sources de données variées (données comportementales, données provenant d'autres institutions financières, etc.).

## Objectifs
- **Construire un modèle de scoring** qui donnera une prédiction sur la probabilité de faillite d'un client de façon automatique.
- **Analyser les features** qui contribuent le plus au modèle, d’une manière générale (feature importance globale) et au niveau d’un client (feature importance locale), afin de permettre à un chargé d’études de mieux comprendre le score attribué par le modèle.
- **Mettre en production** le modèle de scoring de prédiction à l’aide d’une API et réaliser une interface de test de cette API.
- **Mettre en œuvre une approche globale MLOps** de bout en bout, du tracking des expérimentations à l’analyse en production du data drift.

## Structure du projet
1. **Préparation des données** : Nettoyage, encodage et création de nouvelles features.
2. **Modélisation** : Entraînement et évaluation du modèle de scoring.
3. **Analyse des features** : Importance globale et locale des features.
4. **Déploiement** : Création et déploiement de l'API de scoring.
5. **MLOps** : Suivi des expérimentations, gestion du cycle de vie du modèle, et détection du data drift en production.

## Fichiers : 
- **Bases de données**
    - export_base_credit_1000.csv : base de données
    - base_credit_moyennes.csv : moyennes utiles pour le dashboard
- **Exécutables**
    - prediction_credit_flask.py : API
    - dashboard_prediction_credit_P7.py : dashboard
- **Paramètres**
    - requirements.txt : pré-requis pour le déploiment de l'API
    - saved_model.pkl : modèle de prédiction sauvegardé
- **Tests unitaires**
    - test_prediction_credit.py : jeu de tests unitaires joués lors du déploiement
    - test_prediction_credit_failed.py : variante permettant de tester le bon fonctionnement des tests unitaires en cas d'erreur

## Versioning des Librairies Utilisées
Pour assurer la compatibilité et la stabilité de l'API, les versions spécifiques des librairies suivantes ont été utilisées :

- **Flask** : `flask==3.1.0`
- **Pandas** : `pandas==2.2.3`
- **Joblib** : `joblib==1.4.2`
- **Scikit-learn** : `scikit-learn==1.6.1`
- **Gunicorn** : `gunicorn==23.0.0`
- **Pytest** : `pytest==8.3.5`
- **Streamlit** : `streamlit==1.43.2`
- **Requests** : `requests==2.32.3`
- **Matplotlib** : `matplotlib==3.10.1`
- **Seaborn** : `seaborn==0.13.2`

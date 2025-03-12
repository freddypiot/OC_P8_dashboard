import pytest
from flask import Flask
import json
from prediction_credit_flask import app, df_read, model

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_predict_valid_client(client):
    # Test avec un ID client valide
    valid_id = df_read['SK_ID_CURR'].iloc[0]  # Utiliser un ID client existant dans les données
    response = client.post('/predict', json={"id_client": valid_id})
    data = json.loads(response.data)

    assert response.status_code == 200
    assert "id_client" in data
    assert "refus_credit" in data
    assert "probabilite" in data

def test_predict_invalid_client(client):
    # Test avec un ID client invalide
    invalid_id = 999999999  # Utiliser un ID client qui n'existe pas dans les données
    response = client.post('/predict', json={"id_client": invalid_id})
    data = json.loads(response.data)

    assert response.status_code == 404
    assert data["error"] == "Client inconnu"

def test_predict_missing_id(client):
    # Test avec un ID client manquant
    response = client.post('/predict', json={})
    data = json.loads(response.data)

    assert response.status_code == 400
    assert data["error"] == "ID client manquant"

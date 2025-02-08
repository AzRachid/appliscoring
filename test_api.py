import pytest
from fastapi.testclient import TestClient
from scoring import app 

client = TestClient(app)

# Test de la route racine
def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API Scoring Crédit - Prêt à dépenser"}

# Test de la route pour obtenir les IDs des clients
def test_get_clients():
    response = client.get("/clients")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

# Test pour un client existant (100001)
def test_get_client_data_valid():
    client_id = 100001 
    response = client.get(f"/client/{client_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["client_id"] == client_id
    assert "age" in data
    assert "income" in data
    assert "employment_length" in data
    assert "credit_amount" in data
    assert "score" in data
    assert "decision" in data

# Test pour un client inexistant
def test_get_client_data_invalid():
    client_id = 999999  # ID qui n'existe pas
    response = client.get(f"/client/{client_id}")
    assert response.status_code == 404
    assert response.json()["detail"] == "Client non trouvé"

# Test de l'analyse des features pour un client existant
def test_get_feature_importance_valid():
    client_id = 100001  
    response = client.get(f"/analyze/{client_id}")
    assert response.status_code == 200
    data = response.json()
    assert "global_importance_names" in data
    assert "global_importance_values" in data
    assert "local_importance_names" in data
    assert "local_importance_values" in data

# Test de l'analyse des features pour un client inexistant
def test_get_feature_importance_invalid():
    client_id = 999999  
    response = client.get(f"/analyze/{client_id}")
    assert response.status_code == 404
    assert response.json()["detail"] == "Client non trouvé"

if __name__ == "__main__":
    pytest.main()


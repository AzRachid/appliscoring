import pytest
from fastapi.testclient import TestClient
from app import app  # Importation de l'API depuis app.py

# Cr�ation d'un client de test pour interagir avec l'API
client = TestClient(app)

# Test de la route d'accueil "/"
def test_read_root():
    """
    V�rifie que l'API r�pond correctement sur la page d'accueil.
    """
    response = client.get("/")
    assert response.status_code == 200  # V�rifie que la requ�te r�ussit
    assert response.json() == {"message": "API Scoring Cr�dit - Pr�t � d�penser"}

# Test pour r�cup�rer les informations d'un client existant
def test_get_client_data_valid():
    """
    V�rifie que l'API retourne les bonnes informations pour un client existant.
    """
    client_id = 279252  # ID valide 
    response = client.get(f"/client/{client_id}")

    assert response.status_code == 200  # V�rifie que le client existe bien
    data = response.json()

    # V�rification des informations principales du client
    assert data["client_id"] == client_id
    assert "age" in data
    assert "income" in data
    assert "employment_length" in data
    assert "credit_amount" in data
    assert "score" in data
    assert "decision" in data

    # V�rification des importances globales
    assert "global_importance_names" in data
    assert "global_importance_values" in data
    assert isinstance(data["global_importance_names"], list)
    assert isinstance(data["global_importance_values"], list)

    # V�rification des importances locales
    assert "local_importance_names" in data
    assert "local_importance_values" in data
    assert isinstance(data["local_importance_names"], list)
    assert isinstance(data["local_importance_values"], list)

    # V�rification des valeurs importantes pour le client
    assert "client_important_values" in data
    assert isinstance(data["client_important_values"], dict)

# Test pour un client inexistant
def test_get_client_data_invalid():
    """
    V�rifie que l'API renvoie une erreur 404 lorsqu'un client n'existe pas.
    """
    client_id = 999999  # ID inexistant
    response = client.get(f"/client/{client_id}")

    assert response.status_code == 404  # V�rifie que le client est bien absent
    assert response.json()["detail"] == "Client non trouv�"

if __name__ == "__main__":
    pytest.main()



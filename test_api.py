import pytest
from fastapi.testclient import TestClient
from app import app  # Importation de l'API depuis app.py

# Création d'un client de test pour interagir avec l'API
client = TestClient(app)

# Test de la route d'accueil "/"
def test_read_root():
    """
    Vérifie que l'API répond correctement sur la page d'accueil.
    """
    response = client.get("/")
    assert response.status_code == 200  # Vérifie que la requête réussit
    assert response.json() == {"message": "API Scoring Crédit - Prêt à dépenser"}

# Test pour récupérer les informations d'un client existant
def test_get_client_data_valid():
    """
    Vérifie que l'API retourne les bonnes informations pour un client existant.
    """
    client_id = 279252  # ID valide 
    response = client.get(f"/client/{client_id}")

    assert response.status_code == 200  # Vérifie que le client existe bien
    data = response.json()

    # Vérification des informations principales du client
    assert data["client_id"] == client_id
    assert "age" in data
    assert "income" in data
    assert "employment_length" in data
    assert "credit_amount" in data
    assert "score" in data
    assert "decision" in data

    # Vérification des importances globales
    assert "global_importance_names" in data
    assert "global_importance_values" in data
    assert isinstance(data["global_importance_names"], list)
    assert isinstance(data["global_importance_values"], list)

    # Vérification des importances locales
    assert "local_importance_names" in data
    assert "local_importance_values" in data
    assert isinstance(data["local_importance_names"], list)
    assert isinstance(data["local_importance_values"], list)

    # Vérification des valeurs importantes pour le client
    assert "client_important_values" in data
    assert isinstance(data["client_important_values"], dict)

# Test pour un client inexistant
def test_get_client_data_invalid():
    """
    Vérifie que l'API renvoie une erreur 404 lorsqu'un client n'existe pas.
    """
    client_id = 999999  # ID inexistant
    response = client.get(f"/client/{client_id}")

    assert response.status_code == 404  # Vérifie que le client est bien absent
    assert response.json()["detail"] == "Client non trouvé"

if __name__ == "__main__":
    pytest.main()



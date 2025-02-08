 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import re
import joblib
import lime
import lime.lime_tabular
from typing import List, Dict
import uvicorn

# Initialisation de l'API
app = FastAPI(
    title="API Scoring Crédit",
    description="API pour le dashboard de scoring crédit",
    version="1.0.0"
)

# Chargement des données
data = pd.read_csv("./test_data.csv", encoding="utf-8")

# Chargement du modèle
model = joblib.load("./pipeline_production.joblib")

# Récupération des features sélectionnées
selected_features_indices = model.named_steps['feature_selection'].get_support(indices=True)
selected_features = data.drop(columns=['SK_ID_CURR']).columns[selected_features_indices]
data_selected = data[selected_features]

# Seuil de décision
THRESHOLD = 0.51

# Modèles Pydantic pour la validation des données
class ClientData(BaseModel):
    client_id: int
    age: float
    income: float
    employment_length: float
    credit_amount: float
    score: float
    decision: str

class FeatureImportance(BaseModel):
    global_importance_names: List[str]
    global_importance_values: List[float]
    local_importance_names: List[str]
    local_importance_values: List[float]

class Distribution(BaseModel):
    accepted_values: List[float]
    rejected_values: List[float]
    accepted_mean: float
    rejected_mean: float
    client_value: float

# Routes API

@app.get("/")
def read_root():
    """Page d'accueil de l'API"""
    return {"message": "API Scoring Crédit - Prêt à dépenser"}

@app.get("/clients", response_model=List[int])
def get_clients():
    """Renvoie la liste des IDs clients disponibles"""
    return data['SK_ID_CURR'].tolist()

@app.get("/client/{client_id}", response_model=ClientData)
def get_client_data(client_id: int):
    """Renvoie les informations d'un client et son score"""
    # Vérification de l'existence du client
    if client_id not in data['SK_ID_CURR'].values:
        raise HTTPException(status_code=404, detail="Client non trouvé")

    # Récupération des données du client
    row = data[data['SK_ID_CURR'] == client_id].drop(columns=['SK_ID_CURR'])

    # Calcul du score
    probability = model.predict_proba(row)[:, 1][0]
    decision = "Refusé" if probability > THRESHOLD else "Accepté"

    return {
        'client_id': client_id,
        'age': abs(row['DAYS_BIRTH'].values[0] / 365),  # Conversion en années
        'income': row['AMT_INCOME_TOTAL'].values[0],
        'employment_length': abs(row['DAYS_EMPLOYED'].values[0] / 365),  # Conversion en années
        'credit_amount': row['AMT_CREDIT'].values[0],
        'score': probability,
        'decision': decision
    }


@app.get("/analyze/{client_id}", response_model=FeatureImportance)
def get_feature_importance(client_id: int):
    """Renvoie les importances globales et locales des features"""

    if client_id not in data['SK_ID_CURR'].values:
        raise HTTPException(status_code=404, detail="Client non trouvé")

    data_transformed = model.named_steps['preprocessor'].transform(data.drop(columns=['SK_ID_CURR']))
    data_selected = data_transformed[:, selected_features_indices] 

    row = data[data['SK_ID_CURR'] == client_id].drop(columns=['SK_ID_CURR'])
    row_transformed = model.named_steps['preprocessor'].transform(row)
    row_selected = row_transformed[:, selected_features_indices] 

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=data_selected,  
        feature_names=selected_features,
        mode="classification"
    )

    final_estimator = model.named_steps['classifier']  

    exp = explainer.explain_instance(
        row_selected[0],
        lambda X: final_estimator.predict_proba(X)
    ) 
    local_importance = pd.DataFrame(exp.as_list(), columns=["Feature", "Impact"]).head(10)
 

    coefficients = model.named_steps['classifier'].coef_[0]
    global_importance = pd.DataFrame({
        'Feature': selected_features,
        'Importance': coefficients
    }).sort_values('Importance', key=abs, ascending=False).head(10) 

    return {
        'global_importance_names': global_importance['Feature'].tolist(),
        'global_importance_values': global_importance['Importance'].tolist(),
        'local_importance_names': local_importance['Feature'].tolist(),
        'local_importance_values': local_importance['Impact'].tolist(),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

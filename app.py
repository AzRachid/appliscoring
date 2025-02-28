from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import joblib
import lime
import lime.lime_tabular
from typing import List, Dict, Optional
import uvicorn

# Initialisation de l'API
app = FastAPI(
    title="API Scoring Credit",
    description="API pour le dashboard de scoring credit",
    version="1.0.0"
)

# Chargement des donnees
data = pd.read_csv("./test_data.csv", encoding="utf-8")

# Chargement du modèle
model = joblib.load("./pipeline_production.joblib")

# Recuperation des features selectionnees
selected_features_indices = model.named_steps['feature_selection'].get_support(indices=True)
selected_features = data.drop(columns=['SK_ID_CURR']).columns[selected_features_indices]

# Seuil de decision
THRESHOLD = 0.46

# Modèle pour la reponse
class ClientData(BaseModel):
    client_id: int
    age: Optional[float]
    income: Optional[float]
    employment_length: Optional[float]
    credit_amount: Optional[float]
    score: float
    decision: str
    global_importance_names: List[str]
    global_importance_values: List[float]
    local_importance_names: List[str]
    local_importance_values: List[float]
    client_important_values: Dict[str, Optional[float]]

# Fonction pour nettoyer les valeurs incompatibles avec JSON
def clean_json(obj):
    """Remplace les valeurs non compatibles avec JSON (NaN, Inf)"""
    if isinstance(obj, float) and (pd.isna(obj) or np.isinf(obj)):
        return None
    elif isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json(v) for v in obj]
    return obj

# Routes API
@app.get("/")
def read_root():
    """Page d'accueil de l'API"""
    return {"message": "API Scoring Credit - Pret a depenser"}

@app.get("/client/{client_id}", response_model=ClientData)
def get_client_data(client_id: int):
    """Renvoie toutes les informations d'un client et les importances des features"""
    
    # Verification de l'existence du client
    if client_id not in data['SK_ID_CURR'].values:
        raise HTTPException(status_code=404, detail="Client non trouve")

    # Recuperation des donnees du client
    client_row = data[data['SK_ID_CURR'] == client_id].drop(columns=['SK_ID_CURR'])

    # Gestion des valeurs optionnelles
    employment_length = None if pd.isna(client_row['DAYS_EMPLOYED'].values[0]) else abs(client_row['DAYS_EMPLOYED'].values[0] / 365)
    age = None if pd.isna(client_row['DAYS_BIRTH'].values[0]) else abs(client_row['DAYS_BIRTH'].values[0] / 365)
    income = None if pd.isna(client_row['AMT_INCOME_TOTAL'].values[0]) else client_row['AMT_INCOME_TOTAL'].values[0]
    credit_amount = None if pd.isna(client_row['AMT_CREDIT'].values[0]) else client_row['AMT_CREDIT'].values[0]

    client_info = {
        'client_id': client_id,
        'age': age,
        'income': income,
        'employment_length': employment_length,
        'credit_amount': credit_amount,
    }

    # Calcul du score
    probability = model.predict_proba(client_row)[:, 1][0]
    decision = "Refuse" if probability > THRESHOLD else "Accepte"

    score_info = {
        'score': probability,
        'decision': decision
    }

    # Transformation des donnees pour le calcul des importances
    data_transformed = model.named_steps['preprocessor'].transform(data.drop(columns=['SK_ID_CURR']))
    data_selected = data_transformed[:, selected_features_indices]

    row_transformed = model.named_steps['preprocessor'].transform(client_row)
    row_selected = row_transformed[:, selected_features_indices]

    # Creation de l'explainer LIME
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=data_selected,
        feature_names=selected_features,
        mode="classification"
    )

    # Extraction du classifieur final du pipeline
    final_estimator = model.named_steps['classifier']

    # Generation des explications locales
    exp = explainer.explain_instance(
        row_selected[0],
        lambda X: final_estimator.predict_proba(X)
    )
    local_importance = pd.DataFrame(exp.as_list(), columns=["Feature", "Impact"]).head(10)

    # Calcul des importances globales
    coefficients = model.named_steps['classifier'].coef_[0]
    global_importance = pd.DataFrame({
        'Feature': selected_features,
        'Importance': coefficients
    }).sort_values('Importance', key=abs, ascending=False).head(10)

    # Creation du dictionnaire d'importances des features
    feature_importance = {
        'global_importance_names': global_importance['Feature'].tolist(),
        'global_importance_values': global_importance['Importance'].tolist(),
        'local_importance_names': local_importance['Feature'].tolist(),
        'local_importance_values': local_importance['Impact'].tolist()
    }

    # Recuperation des valeurs du client pour les variables importantes
    important_features = global_importance['Feature'].tolist()
    client_values = client_row[important_features].iloc[0].to_dict()
    client_values = {k: (None if pd.isna(v) else float(v)) for k, v in client_values.items()}

    client_important_values = {'client_important_values': client_values}

    # Combinaison de toutes les informations
    result = {**client_info, **score_info, **feature_importance, **client_important_values}

    # Nettoyage final des NaN et Inf avant de retourner la reponse
    return clean_json(result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

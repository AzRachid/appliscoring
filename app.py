from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
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

# Seuil de décision
THRESHOLD = 0.46

# Modèle pour la réponse
class ClientData(BaseModel):
    # Informations client
    client_id: int
    age: float
    income: float
    employment_length: float
    credit_amount: float
    score: float
    decision: str
    # Importances des features
    global_importance_names: List[str]
    global_importance_values: List[float]
    local_importance_names: List[str]
    local_importance_values: List[float]
    # Valeurs du client pour les variables importantes
    client_important_values: Dict[str, float]


# Routes API
@app.get("/")
def read_root():
    """Page d'accueil de l'API"""
    return {"message": "API Scoring Crédit - Prêt à dépenser"}

@app.get("/client/{client_id}", response_model=ClientData)
def get_client_data(client_id: int):
    """Renvoie toutes les informations d'un client et les importances des features"""
    # Vérification de l'existence du client
    if client_id not in data['SK_ID_CURR'].values:
        raise HTTPException(status_code=404, detail="Client non trouvé")

    # Récupération des données du client
    client_row = data[data['SK_ID_CURR'] == client_id].drop(columns=['SK_ID_CURR'])

    # Informations client
    client_info = {
        'client_id': client_id,
        'age': abs(client_row['DAYS_BIRTH'].values[0] / 365),  # Conversion en années
        'income': client_row['AMT_INCOME_TOTAL'].values[0],
        'employment_length': abs(client_row['DAYS_EMPLOYED'].values[0] / 365),  # Conversion en années
        'credit_amount': client_row['AMT_CREDIT'].values[0],
    }

    # Calcul du score
    probability = model.predict_proba(client_row)[:, 1][0]
    decision = "Refusé" if probability > THRESHOLD else "Accepté"
    
    score_info = {
        'score': probability,
        'decision': decision
    }

    # Transformation des données pour le calcul des importances
    data_transformed = model.named_steps['preprocessor'].transform(data.drop(columns=['SK_ID_CURR']))
    data_selected = data_transformed[:, selected_features_indices] 

    row_transformed = model.named_steps['preprocessor'].transform(client_row)
    row_selected = row_transformed[:, selected_features_indices] 

    # Création de l'explainer LIME
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=data_selected,  
        feature_names=selected_features,
        mode="classification"
    )

    # Extraction du classifieur final du pipeline
    final_estimator = model.named_steps['classifier']  

    # Génération des explications locales
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



    # Création du dictionnaire d'importances des features
    feature_importance = {
        'global_importance_names': global_importance['Feature'].tolist(),
        'global_importance_values': global_importance['Importance'].tolist(),
        'local_importance_names': local_importance['Feature'].tolist(),
        'local_importance_values': local_importance['Impact'].tolist()
    }
    
    # Récupération des valeurs du client pour les variables importantes
    important_features = global_importance['Feature'].tolist()
    client_values = client_row[important_features].iloc[0].to_dict()
    client_values = {k: float(v) for k, v in client_values.items()}

    client_important_values = {'client_important_values': client_values}

    # Combinaison de toutes les informations
    result = {**client_info, **score_info, **feature_importance, **client_important_values}
    
    return result

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

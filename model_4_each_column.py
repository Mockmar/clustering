import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Charger et préparer les données
data = pd.read_csv('votre_fichier.csv')

# Encodage des variables catégorielles
label_encoders = {}
for column in ['type_achat', 'magasin', 'ville']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Fonction pour entraîner un modèle pour chaque colonne cible
def train_model_for_column(target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    model = RandomForestRegressor()
    model.fit(X, y)
    return model

# Entraîner des modèles pour chaque colonne
models = {}
for target in ['type_achat', 'quantité', 'magasin', 'ville', 'prix']:
    models[target] = train_model_for_column(target)

# Prédiction pour un formulaire partiellement rempli
def predict_missing_fields(partially_filled_form):
    input_data = pd.DataFrame([partially_filled_form])
    predictions = {}
    
    # Remplir les champs manquants
    for column in ['type_achat', 'quantité', 'magasin', 'ville', 'prix']:
        if pd.isna(input_data[column][0]):
            X_input = input_data.drop(columns=[column])
            predictions[column] = models[column].predict(X_input)[0]
        else:
            predictions[column] = input_data[column][0]
    
    # Décodage des champs catégoriels
    for column in ['type_achat', 'magasin', 'ville']:
        predictions[column] = label_encoders[column].inverse_transform([int(predictions[column])])[0]
    
    return predictions

# Exemple d'un formulaire partiellement rempli
partially_filled_form = {
    'client_id': 1,
    'type_achat': np.nan,  # Non rempli
    'quantité': 15,        # Rempli
    'magasin': np.nan,     # Non rempli
    'ville': 2,            # Rempli
    'prix': np.nan         # Non rempli
}

predicted_values = predict_missing_fields(partially_filled_form)

print(f"Valeurs prédites pour les champs non remplis : {predicted_values}")

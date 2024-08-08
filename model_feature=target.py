import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

# Charger les données
data = pd.DataFrame({
    'client_id': [1, 2, 1],
    'type_achat': ['carotte', 'pomme', 'fraise'],
    'quantité': [40, 10, 20],
    'magasin': ['Auchan', 'Leclerc', 'Casino'],
    'ville': ['paris', 'Lyon', 'pau'],
    'prix': [30, 40, 20]
})

# Encodage des variables catégorielles
data_encoded = pd.get_dummies(data, columns=['type_achat', 'magasin', 'ville'])

# Préparation des données avec les NaN
X = data_encoded.copy()

# Modèle unique pour toutes les colonnes
model = RandomForestRegressor()

# Imputation simple des valeurs manquantes pour l'entraînement
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Entraîner le modèle
model.fit(X_imputed, X_imputed)  # Le modèle est entraîné pour prédire les valeurs d'entrée elles-mêmes

# Fonction de prédiction pour remplir les valeurs manquantes
def predict_missing_values(partially_filled_form):
    # Encodage des données d'entrée partiellement remplies
    input_data = pd.DataFrame([partially_filled_form])
    input_data_encoded = pd.get_dummies(input_data, columns=['type_achat', 'magasin', 'ville'])
    input_data_encoded = input_data_encoded.reindex(columns=X.columns, fill_value=0)
    
    # Imputation des NaN dans les features
    input_data_imputed = imputer.transform(input_data_encoded)
    
    # Prédiction avec le modèle
    predicted_values = model.predict(input_data_imputed)
    
    # Remplir les valeurs manquantes dans le formulaire
    filled_form = input_data_encoded.copy()
    filled_form.loc[0] = predicted_values[0]
    
    # Décodage des valeurs prédictes
    for col in ['type_achat', 'magasin', 'ville']:
        filled_form[col] = input_data[col]
    
    return filled_form

# Exemple d'utilisation avec des valeurs manquantes
partially_filled_form = {
    'client_id': 1,
    'type_achat': np.nan,  # Non rempli
    'quantité': 40,        # Rempli
    'magasin': 'Auchan',   # Rempli
    'ville': 'paris',      # Rempli
    'prix': np.nan         # Non rempli
}

predicted_form = predict_missing_values(partially_filled_form)
print(predicted_form)

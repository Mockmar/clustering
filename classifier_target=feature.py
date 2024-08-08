import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Exemple de données
data = pd.DataFrame({
    'client_id': [1, 2, 1],
    'type_achat': ['carotte', 'pomme', 'fraise'],
    'quantité': [40, 10, 20],
    'magasin': ['Auchan', 'Leclerc', 'Casino'],
    'ville': ['paris', 'Lyon', 'pau'],
    'prix': [30, 40, 20]
})

# Encodage des variables catégorielles
label_encoders = {}
for col in ['type_achat', 'magasin', 'ville']:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])

# Préparation des données avec les NaN
X = data.copy()

# Modèle de Random Forest Classifier
model = RandomForestClassifier()

# Entraînement du modèle pour prédire les NaN
def train_and_predict_missing_values(data, col_to_predict):
    # Diviser les données en entrées (features) et sorties (labels)
    X_train = data.drop(columns=col_to_predict)
    y_train = data[col_to_predict]

    # Entraîner le modèle
    model.fit(X_train, y_train)

    # Prédiction des valeurs manquantes
    X_missing = data[data[col_to_predict].isnull()].drop(columns=col_to_predict)
    if not X_missing.empty:
        y_missing = model.predict(X_missing)
        data.loc[data[col_to_predict].isnull(), col_to_predict] = y_missing

    return data

# Exemple d'utilisation avec des valeurs manquantes
partially_filled_form = {
    'client_id': 1,
    'type_achat': np.nan,  # Non rempli
    'quantité': 40,        # Rempli
    'magasin': 'Auchan',   # Rempli
    'ville': 'paris',      # Rempli
    'prix': np.nan         # Non rempli
}

# Convertir les variables catégorielles en valeurs numériques
for col in ['type_achat', 'magasin', 'ville']:
    if col in partially_filled_form and pd.isna(partially_filled_form[col]):
        partially_filled_form[col] = label_encoders[col].transform([partially_filled_form[col]])[0]

# Prédire les valeurs manquantes avec le modèle
predicted_form = train_and_predict_missing_values(pd.DataFrame([partially_filled_form]), 'prix')

# Convertir les variables catégorielles en chaînes de caractères
for col in ['type_achat', 'magasin', 'ville']:
    if col in predicted_form.columns:
        predicted_form[col] = label_encoders[col].inverse_transform(predicted_form[col])

print(predicted_form)

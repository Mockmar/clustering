import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

# Données initiales
data = {
    'client1': [
        {'produit': 'carotte', 'beneficiaires': ['jean', 'lea'], 'payeur': ['lea']},
        {'produit': 'tomate', 'beneficiaires': ['lea', 'antoine'], 'payeur': ['jean', 'ali']},
        {'produit': 'oignon', 'beneficiaires': ['jean'], 'payeur': ['jean']}
    ],
    'client2': [
        {'produit': 'tomate', 'beneficiaires': ['jean', 'antoine'], 'payeur': ['jean', 'ali']},
        {'produit': 'poivron', 'beneficiaires': ['lea'], 'payeur': ['jean']}
    ],
    'client3': [
        {'produit': 'carotte', 'beneficiaires': ['antoine'], 'payeur': ['lea']},
        {'produit': 'tomate', 'beneficiaires': ['jean', 'lea'], 'payeur': ['lea']},
        {'produit': 'oignon', 'beneficiaires': ['lea'], 'payeur': ['jean']},
        {'produit': 'poivron', 'beneficiaires': ['antoine'], 'payeur': ['ali']}
    ],
    'client4': [
        {'produit': 'tomate', 'beneficiaires': ['jean'], 'payeur': ['antoine']},
        {'produit': 'poivron', 'beneficiaires': ['lea', 'antoine'], 'payeur': ['ali']},
        {'produit': 'banane', 'beneficiaires': ['antoine'], 'payeur': ['jean']}
    ],
    'client5': [
        {'produit': 'carotte', 'beneficiaires': ['lea'], 'payeur': ['jean']},
        {'produit': 'pomme', 'beneficiaires': ['antoine'], 'payeur': ['ali']},
        {'produit': 'tomate', 'beneficiaires': ['jean'], 'payeur': ['lea']},
        {'produit': 'oignon', 'beneficiaires': ['lea'], 'payeur': ['antoine']}
    ],
    'client6': [
        {'produit': 'banane', 'beneficiaires': ['jean', 'antoine'], 'payeur': ['lea']},
        {'produit': 'pomme', 'beneficiaires': ['antoine'], 'payeur': ['jean']},
        {'produit': 'poivron', 'beneficiaires': ['lea'], 'payeur': ['ali']}
    ],
    'client7': [
        {'produit': 'carotte', 'beneficiaires': ['jean', 'lea'], 'payeur': ['antoine']},
        {'produit': 'tomate', 'beneficiaires': ['lea'], 'payeur': ['antoine']},
        {'produit': 'pomme', 'beneficiaires': ['antoine'], 'payeur': ['lea']},
        {'produit': 'banane', 'beneficiaires': ['jean'], 'payeur': ['antoine']}
    ],
    'client8': [
        {'produit': 'poivron', 'beneficiaires': ['lea', 'antoine'], 'payeur': ['ali']},
        {'produit': 'oignon', 'beneficiaires': ['antoine'], 'payeur': ['jean']},
        {'produit': 'tomate', 'beneficiaires': ['jean'], 'payeur': ['antoine']}
    ],
    'client9': [
        {'produit': 'pomme', 'beneficiaires': ['antoine'], 'payeur': ['ali']},
        {'produit': 'carotte', 'beneficiaires': ['jean', 'lea'], 'payeur': ['jean']},
        {'produit': 'poivron', 'beneficiaires': ['lea'], 'payeur': ['antoine']}
    ],
    'client10': [
        {'produit': 'banane', 'beneficiaires': ['jean', 'antoine'], 'payeur': ['ali']},
        {'produit': 'pomme', 'beneficiaires': ['lea'], 'payeur': ['jean']},
        {'produit': 'tomate', 'beneficiaires': ['antoine'], 'payeur': ['lea']},
        {'produit': 'poivron', 'beneficiaires': ['jean'], 'payeur': ['jean']}
    ]
}

# Préparation des listes pour encodage
produits = []
beneficiaires = []
payeurs = []

for client, transactions in data.items():
    client_produits = []
    client_beneficiaires = []
    client_payeurs = []
    for transaction in transactions:
        client_produits.append(transaction['produit'])
        client_beneficiaires.extend(transaction['beneficiaires'])
        client_payeurs.extend(transaction['payeur'])
    produits.append(client_produits)
    beneficiaires.append(client_beneficiaires)
    payeurs.append(client_payeurs)

# Encodage One-Hot
mlb_produit = MultiLabelBinarizer()
mlb_beneficiaires = MultiLabelBinarizer()
mlb_payeurs = MultiLabelBinarizer()

encoded_produits = mlb_produit.fit_transform(produits)
encoded_beneficiaires = mlb_beneficiaires.fit_transform(beneficiaires)
encoded_payeurs = mlb_payeurs.fit_transform(payeurs)

# Combinaison des vecteurs encodés
encoded_data = pd.concat([
    pd.DataFrame(encoded_produits, columns=mlb_produit.classes_),
    pd.DataFrame(encoded_beneficiaires, columns=mlb_beneficiaires.classes_),
    pd.DataFrame(encoded_payeurs, columns=mlb_payeurs.classes_)
], axis=1)

print(encoded_data)

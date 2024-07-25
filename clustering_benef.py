import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Exemple de données avec plusieurs bénéficiaires par produit
data = {
    'client1': [
        {'produit': 'carotte', 'montant': 50, 'beneficiaires': ['benef1', 'benef2']},
        {'produit': 'tomate', 'montant': 30, 'beneficiaires': ['benef2', 'benef3']},
        {'produit': 'oignon', 'montant': 10, 'beneficiaires': ['benef1']}
    ],
    'client2': [
        {'produit': 'tomate', 'montant': 10, 'beneficiaires': ['benef1', 'benef3']},
        {'produit': 'poivron', 'montant': 20, 'beneficiaires': ['benef2']}
    ],
    'client3': [
        {'produit': 'carotte', 'montant': 20, 'beneficiaires': ['benef3']},
        {'produit': 'tomate', 'montant': 30, 'beneficiaires': ['benef1', 'benef2']},
        {'produit': 'oignon', 'montant': 15, 'beneficiaires': ['benef2']},
        {'produit': 'poivron', 'montant': 10, 'beneficiaires': ['benef3']}
    ],
    'client4': [
        {'produit': 'tomate', 'montant': 5, 'beneficiaires': ['benef1']},
        {'produit': 'poivron', 'montant': 25, 'beneficiaires': ['benef2', 'benef3']},
        {'produit': 'banane', 'montant': 15, 'beneficiaires': ['benef3']}
    ],
    'client5': [
        {'produit': 'carotte', 'montant': 10, 'beneficiaires': ['benef2']},
        {'produit': 'pomme', 'montant': 30, 'beneficiaires': ['benef3']},
        {'produit': 'tomate', 'montant': 15, 'beneficiaires': ['benef1']},
        {'produit': 'oignon', 'montant': 5, 'beneficiaires': ['benef2']}
    ],
    'client6': [
        {'produit': 'banane', 'montant': 20, 'beneficiaires': ['benef1', 'benef3']},
        {'produit': 'pomme', 'montant': 25, 'beneficiaires': ['benef3']},
        {'produit': 'poivron', 'montant': 10, 'beneficiaires': ['benef2']}
    ],
    'client7': [
        {'produit': 'carotte', 'montant': 30, 'beneficiaires': ['benef1', 'benef2']},
        {'produit': 'tomate', 'montant': 20, 'beneficiaires': ['benef2']},
        {'produit': 'pomme', 'montant': 10, 'beneficiaires': ['benef3']},
        {'produit': 'banane', 'montant': 15, 'beneficiaires': ['benef1']}
    ],
    'client8': [
        {'produit': 'poivron', 'montant': 30, 'beneficiaires': ['benef2', 'benef3']},
        {'produit': 'oignon', 'montant': 5, 'beneficiaires': ['benef3']},
        {'produit': 'tomate', 'montant': 15, 'beneficiaires': ['benef1']}
    ],
    'client9': [
        {'produit': 'pomme', 'montant': 20, 'beneficiaires': ['benef3']},
        {'produit': 'carotte', 'montant': 10, 'beneficiaires': ['benef1', 'benef2']},
        {'produit': 'poivron', 'montant': 5, 'beneficiaires': ['benef2']}
    ],
    'client10': [
        {'produit': 'banane', 'montant': 5, 'beneficiaires': ['benef1', 'benef3']},
        {'produit': 'pomme', 'montant': 30, 'beneficiaires': ['benef2']},
        {'produit': 'tomate', 'montant': 10, 'beneficiaires': ['benef3']},
        {'produit': 'poivron', 'montant': 15, 'beneficiaires': ['benef1']}
    ]
}

# Transformation des données en DataFrame
clients = []
produits = []
montants = []
beneficiaires = []

for client, achats in data.items():
    for achat in achats:
        for beneficiaire in achat['beneficiaires']:
            clients.append(client)
            produits.append(achat['produit'])
            montants.append(achat['montant'])
            beneficiaires.append(beneficiaire)

df = pd.DataFrame({
    'client': clients,
    'produit': produits,
    'montant': montants,
    'beneficiaire': beneficiaires
})

# Pivot de la DataFrame pour créer une matrice client-produit
df_pivot = df.pivot_table(index='client', columns='produit', values='montant', aggfunc='sum', fill_value=0)

# Suite du code reste inchangée


# Standardisation des données
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_pivot)

# Détermination du nombre optimal de clusters
inertia = []
for n in range(1, 10):
    kmeans = KMeans(n_clusters=n, random_state=0).fit(df_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 10), inertia)
plt.xlabel('Nombre de clusters')
plt.ylabel('Inertie')
plt.title('Méthode du coude pour déterminer le nombre optimal de clusters')
plt.show()

# Application de K-means avec le nombre optimal de clusters (par exemple, 3)
optimal_clusters = 3
kmeans = KMeans(n_clusters=optimal_clusters, random_state=0).fit(df_scaled)
df_pivot['cluster'] = kmeans.labels_

# Évaluation des clusters
score = silhouette_score(df_scaled, kmeans.labels_)
print(f'Silhouette Score: {score}')

# Réduction de dimensions avec t-SNE
tsne = TSNE(n_components=2, random_state=0)
df_tsne = tsne.fit_transform(df_scaled)

# Réduction de dimensions avec PCA (alternative)
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

# Visualisation des clusters
plt.figure(figsize=(14, 7))

# t-SNE plot
plt.subplot(1, 2, 1)
scatter = plt.scatter(df_tsne[:, 0], df_tsne[:, 1], c=df_pivot['cluster'], cmap='viridis', marker='o')
plt.title('t-SNE')
plt.colorbar(scatter, label='Cluster')
for i, txt in enumerate(df_pivot.index):
    plt.annotate(txt, (df_tsne[i, 0], df_tsne[i, 1]))

# PCA plot
plt.subplot(1, 2, 2)
scatter = plt.scatter(df_pca[:, 0], df_pca[:, 1], c=df_pivot['cluster'], cmap='viridis', marker='o')
plt.title('PCA')
plt.colorbar(scatter, label='Cluster')
for i, txt in enumerate(df_pivot.index):
    plt.annotate(txt, (df_pca[i, 0], df_pca[i, 1]))

plt.tight_layout()
plt.show()


#--------------------------------------------------------------------------------------------------------------#

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Exemple de données avec bénéficiaires et payeurs sans montants
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

# Transformation des données en DataFrame avec produits spécifiques aux bénéficiaires et payeurs
clients = []
produits_beneficiaires = []
produits_payeurs = []

for client, achats in data.items():
    for achat in achats:
        for beneficiaire in achat['beneficiaires']:
            clients.append(client)
            produits_beneficiaires.append(f"{achat['produit']}_benef_{beneficiaire}")
        for payeur in achat['payeur']:
            clients.append(client)
            produits_payeurs.append(f"{achat['produit']}_payeur_{payeur}")

# Création de DataFrames pour les bénéficiaires et les payeurs
df_beneficiaires = pd.DataFrame({
    'client': clients[:len(produits_beneficiaires)],
    'produit_beneficiaire': produits_beneficiaires,
    'valeur': 1
})

df_payeurs = pd.DataFrame({
    'client': clients[len(produits_beneficiaires):],
    'produit_payeur': produits_payeurs,
    'valeur': 1
})

# Pivot de la DataFrame pour créer des matrices client-produit_beneficiaire et client-produit_payeur
df_pivot_beneficiaires = df_beneficiaires.pivot_table(index='client', columns='produit_beneficiaire', values='valeur', aggfunc='sum', fill_value=0)
df_pivot_payeurs = df_payeurs.pivot_table(index='client', columns='produit_payeur', values='valeur', aggfunc='sum', fill_value=0)

# Fusion des matrices bénéficiaires et payeurs
df_pivot = pd.concat([df_pivot_beneficiaires, df_pivot_payeurs], axis=1)

# Standardisation des données
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_pivot)

# Détermination du nombre optimal de clusters
inertia = []
for n in range(1, 10):
    kmeans = KMeans(n_clusters=n, random_state=0).fit(df_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 10), inertia)
plt.xlabel('Nombre de clusters')
plt.ylabel('Inertie')
plt.title('Méthode du coude pour déterminer le nombre optimal de clusters')
plt.show()

# Application de K-means avec le nombre optimal de clusters (par exemple, 3)
optimal_clusters = 3
kmeans = KMeans(n_clusters=optimal_clusters, random_state=0).fit(df_scaled)
df_pivot['cluster'] = kmeans.labels_

# Évaluation des clusters
score = silhouette_score(df_scaled, kmeans.labels_)
print(f'Silhouette Score: {score}')

# Réduction de dimensions avec t-SNE
tsne = TSNE(n_components=2, random_state=0)
df_tsne = tsne.fit_transform(df_scaled)

# Réduction de dimensions avec PCA (alternative)
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

# Visualisation des clusters
plt.figure(figsize=(14, 7))

# t-SNE plot
plt.subplot(1, 2, 1)
scatter = plt.scatter(df_tsne[:, 0], df_tsne[:, 1], c=df_pivot['cluster'], cmap='viridis', marker='o')
plt.title('t-SNE')
plt.colorbar(scatter, label='Cluster')
for i, txt in enumerate(df_pivot.index):
    plt.annotate(txt, (df_tsne[i, 0], df_tsne[i, 1]))

# PCA plot
plt.subplot(1, 2, 2)
scatter = plt.scatter(df_pca[:, 0], df_pca[:, 1], c=df_pivot['cluster'], cmap='viridis', marker='o')
plt.title('PCA')
plt.colorbar(scatter, label='Cluster')
for i, txt in enumerate(df_pivot.index):
    plt.annotate(txt, (df_pca[i, 0], df_pca[i, 1]))

plt.tight_layout()
plt.show()

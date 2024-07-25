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

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Exemple de données
data = {
    'client1': {'carotte': 50, 'tomate': 30, 'oignon': 10},
    'client2': {'tomate': 10, 'poivron': 20},
    'client3': {'carotte': 20, 'tomate': 30, 'oignon': 15, 'poivron': 10}
}

# Convertir les données en DataFrame
df = pd.DataFrame(data).fillna(0).T

# Standardisation des données
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Détermination du nombre optimal de clusters
inertia = []
for n in range(1, 10):
    kmeans = KMeans(n_clusters=n, random_state=0).fit(df_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 10), inertia)
plt.xlabel('Nombre de clusters')
plt.ylabel('Inertie')
plt.show()

# Application de K-means avec le nombre optimal de clusters (par exemple, 3)
kmeans = KMeans(n_clusters=3, random_state=0).fit(df_scaled)
df['cluster'] = kmeans.labels_
print(df)

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
plt.figure(figsize=(12, 6))

# t-SNE plot
plt.subplot(1, 2, 1)
plt.scatter(df_tsne[:, 0], df_tsne[:, 1], c=df['cluster'], cmap='viridis', marker='o')
for i, txt in enumerate(df.index):
    plt.annotate(txt, (df_tsne[i, 0], df_tsne[i, 1]))
plt.title('t-SNE')

# PCA plot
plt.subplot(1, 2, 2)
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=df['cluster'], cmap='viridis', marker='o')
for i, txt in enumerate(df.index):
    plt.annotate(txt, (df_pca[i, 0], df_pca[i, 1]))
plt.title('PCA')

plt.show()
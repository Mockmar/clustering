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

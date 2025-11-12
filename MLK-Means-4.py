import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


data = pd.read_csv('sales_data_sample.csv', encoding='ISO-8859-1')
print(data.head())


features = data.select_dtypes(include=[np.number]).copy()
features.fillna(features.mean(), inplace=True)


scaler = StandardScaler()
scaled = scaler.fit_transform(features)


inertia = []
K = range(1, 11)
for k in K:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(list(K), inertia, marker='o')
plt.title('Elbow Method for optimal k')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.xticks(list(K))
plt.grid(True)
plt.show()


optimal_k = 3


kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
data['Cluster'] = kmeans.fit_predict(scaled)


pca = PCA(n_components=2, random_state=42)
proj = pca.fit_transform(scaled)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(proj[:, 0], proj[:, 1], c=data['Cluster'], cmap='viridis', s=30)
plt.title(f'K-Means (k={optimal_k}) — PCA projection')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar(scatter, label='Cluster')
plt.grid(True)
plt.show()


print(data['Cluster'].value_counts())

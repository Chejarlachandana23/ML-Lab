import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_iris 
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA
iris = load_iris() 
X = iris.data       
y = iris.target 

df = pd.DataFrame(X, columns=iris.feature_names) 
print("Original Dataset shapes:", df.shape) 

scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X) 

pca = PCA(n_components=2) 
X_pca = pca.fit_transform(X_scaled) 

print("Reduced Dataset Shape(after PCA):", X_pca.shape) 
print("\nExplained variance Ratio:", pca.explained_variance_ratio_) 

plt.figure(figsize=(8, 6)) 
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", edgecolor="k", s=80) 
plt.xlabel("Principal component 1") 
plt.ylabel("Principal component 2") 
plt.title("PCA - Iris Dataset (Dimensionality Reduction)") 
plt.colorbar(label="Target classes") 
plt.show()

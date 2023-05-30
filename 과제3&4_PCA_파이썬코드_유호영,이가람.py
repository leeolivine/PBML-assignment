'''
# 과제 3&4 PCA, 18011152 유호영 / 21011215 이가람

"Generating Datasets using Scikit-learn"

Assignment 3&4 - PCA

@author: You Ho Yeong, Lee Ga Ram
'''

#%% import libraries
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn import decomposition

center_list=[2,5,10]
std_list=[1,2,3]
sample_list=[100,500,1000]
#%% PCA(center) - Generating Datasets
for center in center_list:
    X, y,centers = make_blobs(n_samples=500, n_features=2,return_centers=True, centers=center, cluster_std=1, random_state=100)

    # PCA - Data Preprocessing
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # PCA - Fitting
    pca=decomposition.PCA()
    X_pca=pca.fit_transform(X)
    
    # PCA - Model Evaluation
    kmeans = KMeans(n_clusters=len(centers), random_state=100)
    kmeans.fit(X_pca)
    cluster_labels = kmeans.labels_

    # PCA - Plotting Result
    plt.figure(figsize=(10,7))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette=sns.hls_palette(len(centers)), legend='full')
    plt.title(f'PCA : centers={center}')
    plt.tight_layout()
    out_path = f'/Users/leeol/OneDrive - Sejong University/바탕 화면/세종대학교/3학년 1학기/파이썬기반기계학습/과제3,4/pca_cen{center}'
    plt.savefig(out_path, dpi=600)
    
    # silhouette score
    silhouette = silhouette_score(X_pca, cluster_labels)
    print("Silhouette score: {:.2f}".format(silhouette))
#%% PCA(std) - Generating Datasets
for std in std_list:
    X, y,centers = make_blobs(n_samples=500, n_features=2,return_centers=True, centers=5, cluster_std=std, random_state=100)

    mglearn.discrete_scatter(X[:,0],X[:,1],y)

    # PCA - Data Preprocessing
    scaler = StandardScaler()
    data_1 = scaler.fit_transform(X)

    # PCA - Fitting
    pca=decomposition.PCA()
    X_pca=pca.fit_transform(X)
    
    # PCA - Model Evaluation
    kmeans = KMeans(n_clusters=len(centers), random_state=100)
    kmeans.fit(X_pca)
    cluster_labels = kmeans.labels_

    # PCA - Plotting Result
    plt.figure(figsize=(10,7))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette=sns.hls_palette(len(centers)), legend='full')
    plt.title(f'PCA : cluster_std={std}')
    plt.tight_layout()
    out_path = f'/Users/leeol/OneDrive - Sejong University/바탕 화면/세종대학교/3학년 1학기/파이썬기반기계학습/과제3,4/pca_std{std}'
    plt.savefig(out_path, dpi=600)
    
    # silhouette score
    silhouette = silhouette_score(X_pca, cluster_labels)
    print("Silhouette score: {:.2f}".format(silhouette))
#%% PCA(sample) - Generating Datasets
for sample in sample_list:
    X, y,centers = make_blobs(n_samples=sample, n_features=2,return_centers=True, centers=5, cluster_std=1, random_state=100)

    # PCA - Data Preprocessing
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # PCA - Fitting
    pca=decomposition.PCA()
    X_pca=pca.fit_transform(X)
    
    # PCA - Model Evaluation
    kmeans = KMeans(n_clusters=len(centers), random_state=100)
    kmeans.fit(X_pca)
    cluster_labels = kmeans.labels_

    # PCA - Plotting Result
    plt.figure(figsize=(10,7))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette=sns.hls_palette(len(centers)), legend='full')
    plt.title(f'PCA : n_samples={sample}')
    plt.tight_layout()
    out_path = f'/Users/leeol/OneDrive - Sejong University/바탕 화면/세종대학교/3학년 1학기/파이썬기반기계학습/과제3,4/pca_sam{sample}'
    plt.savefig(out_path, dpi=600)
    
    # silhouette score
    silhouette = silhouette_score(X_pca, cluster_labels)
    print("Silhouette score: {:.2f}".format(silhouette))

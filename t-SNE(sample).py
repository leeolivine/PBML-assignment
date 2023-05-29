#%% import libraries
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import seaborn as sns
from sklearn.cluster import KMeans

sample_list=[100,500,1000]
#%% t-SNE - Generating Datasets
for sample in sample_list:
    X, y,centers = make_blobs(n_samples=sample, n_features=2,return_centers=True, centers=5, cluster_std=1, random_state=100)

    # t-SNE - Data Preprocessing
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # t-SNE - Fitting
    tsne=TSNE(n_components=2,random_state=100)
    X_tsne=tsne.fit_transform(X)
    
    # t-SNE - Model Evaluation
    kmeans = KMeans(n_clusters=len(centers), random_state=100)
    kmeans.fit(X_tsne)
    cluster_labels = kmeans.labels_

    # t-SNE - Plotting Result
    plt.figure(figsize=(10,7))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette=sns.hls_palette(len(centers)), legend='full')
    plt.title(f't-SNE : n_Samples={sample}')
    plt.tight_layout()
    out_path = f'/Users/leeol/OneDrive - Sejong University/바탕 화면/세종대학교/3학년 1학기/파이썬기반기계학습/과제3,4/tsne_samp{sample}'
    plt.savefig(out_path, dpi=600)
    
    # silhouette score
    silhouette = silhouette_score(X_tsne, cluster_labels)
    print("Silhouette score: {:.2f}".format(silhouette))
# %%

import numpy as np
import pandas as pd
from sklearn import metrics
import scanpy as sc
import ot
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


def clustering(adata, n_clusters=7,  key='emb'):
    pca = PCA(n_components=20, random_state=42)
    embedding = pca.fit_transform(adata.obsm[key].copy())
    adata.obsm['emb_pca'] = embedding
    adata = Kmeans_cluster(adata, num_cluster=n_clusters, used_obsm='emb_pca')
    adata.obs['scTACL'] = adata.obs['kmeans']



def Kmeans_cluster(adata, num_cluster, used_obsm='emb_pca', key_added_pred="kmeans", random_seed=2024):
    np.random.seed(random_seed)
    cluster_model = KMeans(n_clusters=num_cluster, init='k-means++', n_init=100, max_iter=1000, tol=1e-6)
    cluster_labels = cluster_model.fit_predict(adata.obsm[used_obsm])
    adata.obs[key_added_pred] = cluster_labels
    adata.obs[key_added_pred] = adata.obs[key_added_pred].astype('int')
    adata.obs[key_added_pred] = adata.obs[key_added_pred].astype('category')
    return adata
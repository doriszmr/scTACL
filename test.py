from scTACL import scTACL
import os
import torch
from sklearn import metrics
import numpy as np
from preprocess import *
import scanpy as sc

path = '/home/luxin1/scRNA-DATA/Quake_10x_Limb_Muscle/'
adata = sc.read_h5ad(path + 'data.h5ad')
adata.var_names_make_unique()

unique_categories = adata.obs['cell_type1'].unique()  # 获取唯一值
num_unique_categories = len(unique_categories)
n_cluster = num_unique_categories

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model = scTACL(adata,
              n_top_genes=3000,
              epochs = 500,
              dim_output=96,
              n_neighbor=5,
              device=device,
              alpha = 5,
              beta = 0.6,
              gama = 0.6)
adata = model.train()

from utils import clustering
clustering(adata, n_clusters = n_cluster)

cell_name = np.array(adata.obs["cell_type1"])
cell_type, cell_label = np.unique(cell_name, return_inverse=True)
ARI = metrics.adjusted_rand_score(adata.obs['scTACL'], cell_label)
NMI = metrics.normalized_mutual_info_score(adata.obs['scTACL'], cell_label)
print(f"{ARI:.4f}, {NMI:.4f}")
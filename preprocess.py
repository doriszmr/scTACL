import h5py
import anndata as ad
import scipy.sparse as sp
import pandas as pd
import os
import ot
import torch
import random
import numpy as np
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from torch.backends import cudnn
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
from torch import nn
from sklearn.decomposition import PCA

def sc2h5(path):
    f = h5py.File(path + '/data.h5', 'r')
    f.keys()  # 可以查看所有的主键

    data = f['exprs']
    expr_data = sp.csr_matrix((data['data'], data['indices'], data['indptr'])).toarray()

    if 'obs' in f.keys():
        key_list = list(f['obs'].keys())
        obs = pd.DataFrame(index=f['obs_names'][:], columns=key_list)
        for name in key_list:
            obs[name] = f['obs'][name][:]

    if 'var' in f.keys():
        temp_var = pd.DataFrame(index=f['var_names'][:])
        if len(temp_var) == expr_data.shape[1]:
            var = temp_var
        else:
            var = pd.DataFrame(index=list(range(1, expr_data.shape[1] + 1)))

    used_gene = {}
    if 'uns' in f.keys():
        key_list = list(f['uns'].keys())
        for i, values in enumerate(key_list):
            used_gene[i] = f['uns'][values][:]

    adata = ad.AnnData(expr_data, obs=obs, var=var)
    label_list = adata.obs['cell_type1'].unique()
    label = adata.obs['cell_type1'][:].copy()
    label_mapping = {value: i for i, value in enumerate(label_list)}
    label = label.map(label_mapping)

    df_to_save = pd.DataFrame({
        'sequence': adata.obs.index,  # 序列（假设是索引）
        'label': label
    })
    df_to_save.to_csv(path + '/label.csv', index=False, header=False)
    results_path = path + '/data.h5ad'
    adata.write_h5ad(results_path)
    return adata


# def sc2h5(path):
#     f = h5py.File(path + '/data.h5', 'r')
#     f.keys()  # 可以查看所有的主键
#
#     data = f['exprs']
#     expr_data = sp.csr_matrix((data['data'], data['indices'], data['indptr'])).toarray()
#
#     # 处理 'obs' 数据
#     if 'obs' in f.keys():
#         key_list = list(f['obs'].keys())
#         obs = pd.DataFrame(index=f['obs_names'][:], columns=key_list)
#         for name in key_list:
#             obs[name] = f['obs'][name][:]
#
#     # 处理 'var' 数据
#     if 'var' in f.keys():
#         temp_var = pd.DataFrame(index=f['var_names'][:])
#         if len(temp_var) == expr_data.shape[1]:
#             var = temp_var
#         else:
#             var = pd.DataFrame(index=list(range(1, expr_data.shape[1] + 1)))
#
#     # 处理 'uns' 数据
#     used_gene = {}
#     if 'uns' in f.keys():
#         key_list = list(f['uns'].keys())
#         for i, values in enumerate(key_list):
#             used_gene[i] = f['uns'][values][:]
#
#     # 创建 AnnData 对象
#     adata = ad.AnnData(expr_data, obs=obs, var=var)
#
#     # 将基因表达数据保存为 CSV 文件
#     expr_data_df = pd.DataFrame(expr_data, index=f['obs_names'][:], columns=f['var_names'][:])
#     expr_data_df.to_csv(path + '/gene_expression.csv')
#
#     sc.pp.highly_variable_genes(adata)
#
#     # 获取高变基因的表达数据
#     high_var_genes = adata.var[adata.var['highly_variable']]
#     high_var_expr_data = adata[:, high_var_genes.index].to_df()
#
#     # 保存高变基因的表达数据为 CSV 文件
#     high_var_expr_data.to_csv(path + '/high_variable_genes.csv')
#
#     # 保存细胞标签
#     label_list = adata.obs['cell_type1'].unique()
#     label = adata.obs['cell_type1'][:].copy()
#     label_mapping = {value: i for i, value in enumerate(label_list)}
#     label = label.map(label_mapping)
#
#     # 创建标签 CSV
#     df_to_save = pd.DataFrame({
#         'sequence': adata.obs.index,  # 序列（假设是索引）
#         'label': label
#     })
#     df_to_save.to_csv(path + '/label.csv', index=False, header=False)
#
#     # 保存 AnnData 对象为 h5ad 格式
#     results_path = path + '/data.h5ad'
#     adata.write_h5ad(results_path)
#
#     return adata

def permutation(feature):
    # fix_seed(FLAGS.random_seed)
    ids = np.arange(feature.shape[0])
    ids = np.random.permutation(ids)
    feature_permutated = feature[ids]

    return feature_permutated

def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x


def construct_interaction(adata,n_neighbors = 3):
    sc.pp.pca(adata, n_comps=30,random_state=42)
    pca_data = adata.obsm['X_pca']
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean').fit(pca_data)
    distances, indices = knn.kneighbors(pca_data)

    n_spot = pca_data.shape[0]
    adj = np.zeros([n_spot, n_spot])
    for i in range(n_spot):
        for j in indices[i]:
            if i != j:
                adj[i, j] = 1  # 添加边

        # 对称化邻接矩阵
    adj = adj + adj.T
    adj = np.where(adj > 1, 1, adj)  # 确保值为 0 或 1

    # 将邻接矩阵保存到 adata
    adata.obsm['adj'] = adj
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import scanpy as sc


def construct_cosine_graph_from_pca(adata, n_neighbors=3):
    # 确保已进行 PCA
    if 'X_pca' not in adata.obsm:
        sc.pp.pca(adata, n_comps=30, random_state=42)

    pca_data = adata.obsm['X_pca']  # shape: [n_spots, n_pcs]

    # 计算余弦相似度矩阵
    sim_matrix = cosine_similarity(pca_data)

    n_spot = pca_data.shape[0]
    adj = np.zeros((n_spot, n_spot))

    for i in range(n_spot):
        sim_scores = sim_matrix[i].copy()
        sim_scores[i] = -1  # 排除自己
        neighbors = np.argsort(sim_scores)[-n_neighbors:]  # 最大的 n 个相似度邻居
        for j in neighbors:
            adj[i, j] = 1

    # 对称化邻接矩阵（构建无向图）
    adj = adj + adj.T
    adj = np.where(adj > 1, 1, adj)

    # 保存到 AnnData 对象
    adata.obsm['adj'] = adj


import numpy as np
from scipy.stats import pearsonr
import scanpy as sc


def construct_pearson_graph_from_pca(adata, n_neighbors=3):
    # 确保已进行 PCA
    if 'X_pca' not in adata.obsm:
        sc.pp.pca(adata, n_comps=30, random_state=42)

    pca_data = adata.obsm['X_pca']  # shape: [n_spots, n_pcs]
    n_spot = pca_data.shape[0]

    # 初始化相关系数矩阵
    corr_matrix = np.zeros((n_spot, n_spot))

    # 计算每对节点之间的皮尔森相关系数
    for i in range(n_spot):
        for j in range(i + 1, n_spot):
            corr, _ = pearsonr(pca_data[i], pca_data[j])
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr  # 对称填充

    # 构建邻接矩阵：每个节点选择相关性最高的 k 个邻居
    adj = np.zeros((n_spot, n_spot))
    for i in range(n_spot):
        corr_scores = corr_matrix[i].copy()
        corr_scores[i] = -2  # 排除自身
        neighbors = np.argsort(corr_scores)[-n_neighbors:]  # 取最大相关性
        for j in neighbors:
            adj[i, j] = 1

    # 对称化邻接矩阵
    adj = adj + adj.T
    adj = np.where(adj > 1, 1, adj)

    # 保存到 AnnData 对象中
    adata.obsm['adj'] = adj

def construct_adj_snn(adata, n_neighbors=6, use_rep='X_pca'):
    sc.pp.pca(adata, n_comps=30, random_state=42)
    X = adata.obsm[use_rep] if use_rep in adata.obsm else adata.X.toarray()

    nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    nn.fit(X)
    knn_graph = nn.kneighbors_graph(X, mode='connectivity')  # sparse matrix

    snn_matrix = knn_graph @ knn_graph.T
    snn_matrix.setdiag(0)
    snn_adj = snn_matrix.sign().toarray()

    adata.obsm['adj'] = snn_adj
# def mask_feature(x, feat_mask_rate=0.3,use_token = True):
#     num_nodes = x.shape[0]
#     perm = torch.randperm(num_nodes, device=x.device)
#     # random masking
#     num_mask_nodes = int(feat_mask_rate * num_nodes)
#     mask_nodes = perm[: num_mask_nodes]
#     keep_nodes = perm[num_mask_nodes:]
#     out_x = x.clone()
#     if use_token:
#         out_x[mask_nodes] += nn.Parameter(torch.zeros(1, x.shape[1]))
#     else:
#         out_x[mask_nodes] = 0.0
#     return out_x, mask_nodes

def preprocess(adata,n_top_genes):
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top_genes)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)
    adata.raw = adata

def get_feature(adata):

    adata_Vars = adata[:, adata.var['highly_variable']]

    if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
        feat = adata_Vars.X.toarray()[:, ]
    else:
        feat = adata_Vars.X[:, ]

        # data augmentation
    # feat_a = permutation(feat)

    adata.obsm['feat'] = feat

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj) + np.eye(adj.shape[0])
    return adj_normalized



def add_contrastive_label(adata):
    # contrastive label
    n_spot = adata.n_obs
    one_matrix = np.ones([n_spot, 1])
    zero_matrix = np.zeros([n_spot, 1])
    label_CSL = np.concatenate([one_matrix, zero_matrix], axis=1)
    adata.obsm['label_CSL'] = label_CSL


def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'






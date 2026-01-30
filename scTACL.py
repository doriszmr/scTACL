import torch
from preprocess import *
import sys
from model import Encoder
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
MAX_LOGSTD = 20

class scTACL(torch.nn.Module):
    def __init__(self,
                 adata,
                 device=torch.device('cuda'),
                 learning_rate=0.0008,
                 learning_rate_sc=0.01,
                 weight_decay=0.00,
                 epochs=500,
                 n_top_genes=5000,
                 dim_output=256,
                 random_seed=41,
                 alpha=1.0,
                 beta=1,
                 gama=0.5,
                 n_neighbor=5,
                 ):
        super(scTACL, self).__init__()
        self.adata = adata.copy()
        self.device = device
        self.learning_rate = learning_rate
        self.learning_rate_sc = learning_rate_sc
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.n_top_genes = n_top_genes
        self.random_seed = random_seed
        self.alpha = alpha
        self.beta = beta
        self.gama = gama
        self.n_neighbor = n_neighbor
        fix_seed(self.random_seed)

        if 'highly_variable' not in adata.var.keys():
            preprocess(self.adata, self.n_top_genes)

        if 'feat' not in adata.obsm.keys():
            get_feature(self.adata)

        if 'adj' not in adata.obsm.keys():
            construct_interaction(self.adata,self.n_neighbor)

        self.features = torch.FloatTensor(self.adata.obsm['feat'].copy()).to(self.device)
        self.adj = self.adata.obsm['adj']
        self.dim_input = self.features.shape[1]
        self.dim_output = dim_output
        self.adata.obsm['adj'] = preprocess_adj(self.adj)
        self.adj = torch.FloatTensor(self.adata.obsm['adj'].copy()).to(self.device)
        self.model = Encoder(self.dim_input, self.dim_output, self.adj).to(self.device)


    def train(self):

        self.model = Encoder(self.dim_input, self.dim_output, self.adj).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, weight_decay=self.weight_decay)

        print('Begin to train SC data...')
        self.model.train()
        loss_list = []
        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            self.features_a = permutation(self.features)
            self.hidden_feat,self.z_a,self.emb, self.emb_a,self.rec, self.A_rec, zinb_loss, meanbatch, dispbatch, pibatch= self.model(
                self.features,
                self.features_a,
                self.adj)

            self.zinb_loss = zinb_loss(self.features, meanbatch, dispbatch, pibatch, device=self.device)


            feat_loss = F.mse_loss(self.features,self.rec)
            adj_loss = F.mse_loss(self.adj,self.A_rec)
            con_loss = F.mse_loss(self.cross_correlation(self.hidden_feat,self.z_a),self.adj)
            loss = self.alpha*(feat_loss + 0.6 * adj_loss)  +  self.gama * con_loss + self.beta * self.zinb_loss
            loss_list.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        plt.figure(figsize=(10, 5))
        plt.plot(loss_list, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()
        plt.show()
        print("Optimization finished for SC data!")
        with torch.no_grad():
            self.model.eval()
            self.emb_rec = self.model(self.features,self.features_a,self.adj)[4]
            self.adata.obsm['emb'] = F.normalize(self.emb_rec, p=2, dim=1).detach().cpu().numpy()
            #
            # self.emb_rec = self.model(self.features, self.features_a, self.adj)[4].detach().cpu().numpy()
            # self.adata.obsm['emb'] = self.emb_rec

            return self.adata

    def cross_correlation(self, Z_v1, Z_v2):
        """
        calculate the cross-view correlation matrix S
        Args:
            Z_v1: the first view embedding
            Z_v2: the second view embedding
        Returns: S
        """
        return torch.mm(F.normalize(Z_v1, dim=1), F.normalize(Z_v2, dim=1).t())


    def cross_correlation_2(self, Z_v1, Z_v2):
        """
        calculate the cross-view correlation matrix S
        Args:
            Z_v1: the first view embedding
            Z_v2: the second view embedding
        Returns: S
        """
        return torch.mm(Z_v1, Z_v2.t())











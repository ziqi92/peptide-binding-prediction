# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 17:50:01 2019

@author: ziqi
"""

import torch
import torch.nn as nn
import math
import numpy as np
from config import device

class PositionEmbedding(nn.Module):
    def __init__(self,
                 num_embeddings,
                 embedding_dim, embed_type, seed=100):
        super(PositionEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        self.embed_type = embed_type
        if embed_type == "Random":
            self.weight = nn.Embedding(num_embeddings+1, embedding_dim, padding_idx=0)
        elif embed_type == "Sinusoid":
            self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
            self.get_sinusoid_encoding_table(num_embeddings, embedding_dim)

    def reset_parameters(self, seed):
        if device == 'cpu':
            torch.manual_seed(seed)
        else:
            torch.cuda.manual_seed(seed)
        torch.nn.init.xavier_normal_(self.weight)

    def get_sinusoid_encoding_table(self, n_position, d_hid, padding_idx=None):
        ''' Sinusoid position encoding table '''
    
        def cal_angle(position, hid_idx):
            return position / math.pow(10000, 2 * (hid_idx // 2) / d_hid)
    
        def get_posi_angle_vec(position):
            return [cal_angle(position, hid_j) for hid_j in range(d_hid)]
    
        sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        
        if padding_idx is not None:
            # zero vector for padding dimension
            sinusoid_table[padding_idx] = 0.
        
        self.weight.data = torch.FloatTensor(sinusoid_table).to(device)
    
    def forward(self, x, seq_len):
        if self.embed_type == "Random":
            batch_size = x.size()[0]
            len_q = x.size()[2]
            
            idx = torch.arange(1,len_q+1).view(1,-1).expand((x.size()[0],x.size()[2])).to(device)
            len_expanded = seq_len.view(-1,1).expand((x.size()[0],x.size()[2])).to(device)
            mask = idx <= len_expanded
                
            pos_idx = torch.zeros(idx.size(),dtype=torch.long).to(device)
            pos_idx[mask] = idx[mask]
            inv_idx = len_expanded - idx + 1
            inv_idx[~mask] = 0
            
            embeddings = torch.transpose(torch.cat((self.weight(pos_idx), self.weight(inv_idx)), dim=2), 1, 2)
            return torch.cat((x, embeddings), dim=1)
        elif self.embed_type == "Sinusoid":
            batch_size = x.size()[0]
            seq_len = x.size()[2]
            idx = [i for i in range(seq_len-1, -1, -1)]
            pos_embeddings = self.weight[:seq_len,:]
            return x+pos_embeddings
            

    def extra_repr(self):
        return 'num_embeddings={}, embedding_dim={}, mode={}'.format(
            self.num_embeddings, self.embedding_dim, self.mode,
        )

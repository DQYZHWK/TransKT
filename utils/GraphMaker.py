import numpy as np
import random
import scipy.sparse as sp
import torch
import codecs
import json
import copy

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class GraphMaker(object):
    def __init__(self, opt, filename,kc_mat_path):
        self.opt = opt

        self.adj, self.adj_single = self.preprocess(filename, opt)

    
    def preprocess(self,data,opt):
        
        VV_edges_npy='./dataset/'+opt['dataset']+'/adj-fu.npy'
        VV_edges_single_npy='./dataset/'+opt['dataset']+'/adj_single-fu.npy'
        VV_edges = np.load(VV_edges_npy)
        VV_edges_single = np.load(VV_edges_single_npy)
        adj = sp.coo_matrix((np.ones(VV_edges.shape[0]), (VV_edges[:, 0], VV_edges[:, 1])),
                               shape=(opt["nitem"], opt["nitem"]),
                               dtype=np.float32)  
        adj_single = sp.coo_matrix((np.ones(VV_edges_single.shape[0]), (VV_edges_single[:, 0], VV_edges_single[:, 1])),shape=(opt["nitem"], opt["nitem"]),dtype=np.float32)

        adj = normalize(adj)
        adj_single = normalize(adj_single)
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        adj_single = sparse_mx_to_torch_sparse_tensor(adj_single)

        print("real graph loaded!")
        return adj, adj_single





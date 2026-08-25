import os
import torch
import deepdish as dd
import numpy as np
import torch.nn as nn
import scipy.io as sio
import scipy.sparse as sp
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch import Tensor
from typing import Optional, Tuple, Union
import torch as th
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset,WikiCSDataset
from dgl.data import AmazonCoBuyPhotoDataset, AmazonCoBuyComputerDataset
from dgl.data import CoauthorCSDataset, CoauthorPhysicsDataset, CoraFullDataset

def Dataset_Preprocess(root_path, data_path):
    temp = dd.io.load(root_path + data_path)
    timeseries = torch.tensor(temp['timeseires'])
    timeseries = (torch.max(timeseries) - timeseries) / (torch.max(timeseries) - torch.min(timeseries))
    adj = torch.tensor(temp['corr'])
    a, _ = torch.topk(adj, 20, dim=1, largest=True, sorted=True)
    a_min = torch.min(a, dim=-1).values
    ge = torch.ge(adj, a_min)
    zero = torch.zeros_like(adj)
    adj = torch.where(ge, adj, zero)
    adj = adj.repeat(4, 1)
    adj = np.reshape(adj,[4,82,82])
    his = torch.zeros(4, 82, 800)
    pre = torch.zeros(4, 82, 800)
    for i in range(4):

        h = timeseries[:, 100 * i : 800 + 100 * i]
        p = timeseries[:, 100 * (i + 1) : 800 + 100 * (i + 1)]
        his[i,:,:] = h
        pre[i,:,:] = p

    return adj, his, pre

def load_h5(path):
    cat_data_fined = np.load(path, allow_pickle=True)
    adjs = cat_data_fined['adj']
    features = cat_data_fined['x']
    labels = cat_data_fined['y']

    return adjs, features, labels



def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()


def normalize_graph(A):
    eps = 2.2204e-16
    deg_inv_sqrt = (A.sum(dim=-1).clamp(min=0.) + eps).pow(-0.5)
    if A.size()[0] != A.size()[1]:
        A = deg_inv_sqrt.unsqueeze(-1) * (deg_inv_sqrt.unsqueeze(-1) * A)
    else:
        A = deg_inv_sqrt.unsqueeze(-1) * A * deg_inv_sqrt.unsqueeze(-2)
    return A


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def add_random_edge(edge_index, p: float, force_undirected: bool = False,
                    num_nodes: Optional[Union[Tuple[int], int]] = None,
                    training: bool = True):
    if p < 0. or p > 1.:
        raise ValueError(f'Ratio of added edges has to be between 0 and 1 '
                         f'(got {p}')
    if force_undirected and isinstance(num_nodes, (tuple, list)):
        raise RuntimeError('`force_undirected` is not supported for'
                           ' heterogeneous graphs')

    device = edge_index.device
    if not training or p == 0.0:
        edge_index_to_add = torch.tensor([[], []], device=device)
        return edge_index, edge_index_to_add

    if not isinstance(num_nodes, (tuple, list)):
        num_nodes = (num_nodes, num_nodes)
    num_src_nodes = maybe_num_nodes(edge_index, num_nodes[0])
    num_dst_nodes = maybe_num_nodes(edge_index, num_nodes[1])

    num_edges_to_add = round(edge_index.size(1) * p)
    row = torch.randint(0, num_src_nodes, size=(num_edges_to_add, ))
    col = torch.randint(0, num_dst_nodes, size=(num_edges_to_add, ))

    if force_undirected:
        mask = row < col
        row, col = row[mask], col[mask]
        row, col = torch.cat([row, col]), torch.cat([col, row])
    edge_index_to_add = torch.stack([row, col], dim=0).to(device)
    edge_index = torch.cat([edge_index, edge_index_to_add], dim=1)
    return edge_index, edge_index_to_add

def dropout_edge(edge_index: Tensor, p: float = 0.5,
                 force_undirected: bool = False,
                 training: bool = True) -> Tuple[Tensor, Tensor]:

    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    if not training or p == 0.0:
        edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
        return edge_index, edge_mask

    row, col = edge_index

    edge_mask = torch.rand(row.size(0), device=edge_index.device) >= p

    if force_undirected:
        edge_mask[row > col] = False

    edge_index = edge_index[:, edge_mask]

    if force_undirected:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_mask = edge_mask.nonzero().repeat((2, 1)).squeeze()

    return edge_index, edge_mask

def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1),),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0
    return x

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, drop_prob, isBias=False):
        super(GCN, self).__init__()

        self.fc_1 = nn.Linear(in_ft, out_ft, bias=False)
        if act == 'prelu':
            self.act = nn.PReLU()
        elif act == 'relu':
            self.act = nn.ReLU()
        if isBias:
            self.bias_1 = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias_1.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)
        for m in self.modules():
            self.weights_init(m)
        self.drop_prob = drop_prob
        self.isBias = isBias


    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=False):
        seq = F.dropout(seq, self.drop_prob, training=self.training)
        seq_raw = self.fc_1(seq)
        if sparse:
            seq = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_raw, 0)), 0)
        else:
            seq = torch.mm(adj, seq_raw)

        if self.isBias:
            seq += self.bias_1

        return self.act(seq)

def update_S(model, features, adj_list, c_dim, batch_size):
    model.eval()
    FF = []
    with torch.no_grad():
            # Forward
        _, common, _ = model.encode(features, adj_list)
        FF.append(torch.cat(common, 1))
        FF = torch.cat(FF, 0)

        # The projection step, i.e., subtract the mean
        FF = FF - torch.mean(FF, 0, True)
        h=[]
        for i in range(int(batch_size)):
            h.append(FF[:,i*c_dim:(i+1)*c_dim])

        FF = torch.stack(h, dim=2)

        # The SVD step
        U, _, T = torch.svd(torch.sum(FF, dim=2))
        S = torch.mm(U, T.t())
        S = S*(FF.shape[0])**0.5
    return S

class Linearlayer(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(Linearlayer, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)




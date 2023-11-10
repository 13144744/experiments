import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import global_add_pool, GCNConv, GINConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from hgcn.layers.hyp_layers import GCN
from hgcn.layers.hyplayers import HgpslPool
from hgcn.layers.layers import Linear
from hgcn.layers import hyp_layers, hyplayers
from layers import HGPSLPool
from hgcn.manifolds.poincare import PoincareBall


def edge_to_adj(edge_index, x):
    row, col = edge_index
    xrow, xcol = x[row], x[col]
    cat = torch.cat([xrow, xcol], dim=1).sum(dim=-1).div(2)
    weights = (torch.cat([x[row], x[col]], dim=1)).sum(dim=-1).div(2)
    adj = torch.zeros((x.size(0), x.size(0)), dtype=torch.float, device=x.device)
    adj[row, col] = weights
    adj_cpu = adj
    # narray = adj_cpu.cpu().detach().numpy()
    return adj


class hyp_GCN(nn.Module):
    def __init__(self, manifold, nfeat, nhid, nclass, nlayer, dropout, args):
        super(hyp_GCN, self).__init__()
        self.num_features = nfeat
        self.nhid = nhid
        self.args = args
        self.c = args.c
        self.manifold = PoincareBall()
        self.use_bias = args.use_bias  # 使用偏移量
        self.act = torch.nn.ReLU()
        self.pooling_ratio = 0.5
        self.sample = True
        self.sparse = True
        self.sl = True
        self.lamb = 1.0

        self.hgcn1 = nn.Sequential(
            hyp_layers.HyperbolicGraphConvolution(
                self.manifold, nfeat, self.nhid, self.c, self.c, args.dropout, self.act, args.bias, args.use_att
            )
        )
        self.hgcn2 = nn.Sequential(
            hyp_layers.HyperbolicGraphConvolution(
                self.manifold, self.nhid, self.nhid, self.c, self.c, args.dropout, self.act, args.bias, args.use_att
            )
        )
        self.hgcn3 = nn.Sequential(
            hyp_layers.HyperbolicGraphConvolution(
                self.manifold, self.nhid, self.nhid, self.c, self.c, args.dropout, self.act, args.bias, args.use_att
            )
        )

        self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin3 = torch.nn.Linear(self.nhid // 2, nclass)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # hyperbolic embedding
        x = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(x, self.c), self.c), self.c)
        x, _ = self.hgcn1((x, edge_index))
        x, _ = self.hgcn2((x, edge_index))
        x, _ = self.hgcn3((x, edge_index))
        x = self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.c), c=self.c)

        x = global_add_pool(x, batch)
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x

    def loss(self, pred, label):
        return F.nll_loss(pred, label)


class hyp_GCN2(nn.Module):
    def __init__(self, manifold, nfeat, nhid, nclass, nlayer, dropout, args):
        super(hyp_GCN2, self).__init__()
        self.num_features = nfeat
        self.nhid = nhid
        self.args = args
        self.c = args.c
        self.manifold = PoincareBall()
        self.use_bias = args.use_bias  # 使用偏移量
        self.dropout = args.dropout
        self.act = torch.nn.ReLU()
        self.pooling_ratio = 0.5
        self.sample = True
        self.sparse = True
        self.sl = True
        self.lamb = 1.0

        self.hgcn1 = nn.Sequential(
            hyp_layers.HyperbolicGraphConvolution(
                self.manifold, nfeat, self.nhid, self.c, self.c, args.dropout, self.act, args.bias, args.use_att
            )
        )
        self.hgcn2 = nn.Sequential(
            hyp_layers.HyperbolicGraphConvolution(
                self.manifold, self.nhid, self.nhid, self.c, self.c, args.dropout, self.act, args.bias, args.use_att
            )
        )
        self.hgcn3 = nn.Sequential(
            hyp_layers.HyperbolicGraphConvolution(
                self.manifold, self.nhid, self.nhid, self.c, self.c, args.dropout, self.act, args.bias, args.use_att
            )
        )

        self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin3 = torch.nn.Linear(self.nhid // 2, nclass)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # hyperbolic embedding
        x = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(x, self.c), self.c), self.c)
        x = self.hgcn1((x, edge_index))
        x = self.hgcn2((x, edge_index))
        x = self.hgcn3((x, edge_index))
        x = self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.c), c=self.c)

        x = global_add_pool(x, batch)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x

    def loss(self, pred, label):
        return F.nll_loss(pred, label)

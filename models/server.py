import random
from copy import deepcopy

from dtaidistance import dtw
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import AgglomerativeClustering
from torch import nn

from hgcn.manifolds import PoincareBall
from utils.data_utils import *
from utils.util import *


class Server_Net(nn.Module):
    def __init__(self, model, device, args):
        super().__init__()
        self.model = model.to(device)
        self.c = args.c
        self.manifold = PoincareBall()
        self.W = {key: real for key, real in self.model.named_parameters()}
        self.dW = {key: real for key, real in self.model.named_parameters()}
        self.model_cache = []
        self.args = args

    def randomSample_clients(self, all_clients, frac):
        return random.sample(all_clients, int(len(all_clients) * frac))

    def aggregate_weights(self, selected_clients):
        # pass train_size, and weighted aggregate
        total_size = 0
        for client in selected_clients:
            total_size += client.train_size
        for k in self.W.keys():
            self.W[k].data = torch.div(
                torch.sum(
                    torch.stack([torch.mul(client.W[k].data, client.train_size) for client in selected_clients]),
                    dim=0), total_size).clone()

    def aggregate_hyperbolic_weights(self, selected_clients):
        # pass train_size, and weighted aggregate
        total_size = 0
        for client in selected_clients:
            total_size += client.train_size
        for k in self.W.keys():
            if "hyperbolic11212" in k:
                # hyp_W = None
                # for client in selected_clients:
                #     if hyp_W == None:
                #         hyp_W = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(client.W[k].data, self.c), self.c), self.c)
                #     else:
                #         hyp_W = self.manifold.mobius_add(hyp_W, client.W[k].data, c=self.c)
                # for client in selected_clients:
                #     X1 = client.W[k].data
                #     OneTen = 0.1 * torch.ones(X1.)
                #     W = self.manifold.logmap0(X1 * 0.1, self.c)
                #
                #     X2 = self.manifold.expmap0(W, c=self.c) * 10
                #     A = 0
                hyp_W = [torch.mul(self.manifold.proj_tan0(self.manifold.logmap0(client.W[k].data, self.c), self.c),
                                   client.train_size) for client in selected_clients]
                # hyp_W = [torch.mul(self.manifold.proj_tan0(self.manifold.logmap0(client.W[k].data * 0.1, self.c), self.c), client.train_size) for client in selected_clients]
                # hyp_W = [torch.mul(client.W[k].data * 0.1, client.train_size) for client in selected_clients]
                agg_hyp_W = torch.div(torch.sum(torch.stack(hyp_W), dim=0), total_size)
                # agg_hyp_W = self.manifold.mobius_matvec(n, hyp_W, self.c)
                # self.W[k].data = agg_hyp_W
                # self.W[k].data = self.manifold.proj(self.manifold.logmap0(agg_hyp_W, c=self.c), c=self.c)
                # self.W[k].data = (self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(agg_hyp_W, c=self.c), c=self.c), c=self.c) * 10).clone()
                self.W[k].data = (
                    self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(agg_hyp_W, c=self.c), c=self.c),
                                       c=self.c)).clone()

            else:
                self.W[k].data = torch.div(
                    torch.sum(
                        torch.stack([torch.mul(client.W[k].data, client.train_size) for client in selected_clients]),
                        dim=0), total_size).clone()

    def aggregate_with_personlize_lambda(self, selected_clients, personlize_lambda):
        a = 0
        if personlize_lambda == []:
        # if a == 0:
            total_size = 0
            for client in selected_clients:
                total_size += client.train_size
            for k in self.W.keys():
                self.W[k].data = torch.div(
                    torch.sum(
                        torch.stack([torch.mul(client.W[k].data, client.train_size) for client in selected_clients]),
                        dim=0), total_size).clone()
        else:
            total_size = 0
            for client in selected_clients:
                total_size += client.train_size
            l = 0
            for i in range(len(personlize_lambda)):
                l += personlize_lambda[i]
            for k in self.W.keys():
                self.W[k].data = torch.div(torch.sum(torch.stack(
                        [torch.mul(client.W[k].data, personlize_lambda[client.id - 1]) for client in selected_clients]),
                        dim=0), l).clone()
        A = 0

    def aggregate_hyp_weights(self, selected_clients, personlize_lambda):
        total_size = 0
        original_weight = self.W
        Ws, dWs, grads = [], [], []  # 定义权重、权重变值、梯度
        for client in selected_clients:
            W, dW, grad = {}, {}, {}  # 定义权重、权重变值、梯度
            # 获取dW
            for k in self.W.keys():
                W[k] = client.W[k]
                dW[k] = client.dW[k]
                grad[k] = client.W[k].grad
            Ws.append(W)
            grads.append((grad, client.train_size))
            dWs.append((dW, client.train_size))
            total_size += client.train_size
        # calculate_hypweights(Weights=Ws, delta_Weights=dWs, Grads=grads, total_size=total_size)
        calculate_klein_midpoint(self, Weights=Ws, delta_Weights=dWs, Grads=grads, total_size=total_size)
        A = 0

    def compute_pairwise_similarities(self, clients):
        client_dWs = []
        for client in clients:
            dW = {}
            for k in self.W.keys():
                dW[k] = client.dW[k]
            client_dWs.append(dW)
        return pairwise_angles(client_dWs)

    def compute_pairwise_cluster_similarities(self, clusters_dWs):
        return pairwise_angles(clusters_dWs)

    def compute_pairwise_distances(self, seqs, standardize=False):
        """ computes DTW distances """
        if standardize:
            # standardize to only focus on the trends
            seqs = np.array(seqs)
            seqs = seqs / seqs.std(axis=1).reshape(-1, 1)
            distances = dtw.distance_matrix(seqs)
        else:
            distances = dtw.distance_matrix(seqs)
        return distances

    def min_cut(self, similarity, idc):
        g = nx.Graph()
        for i in range(len(similarity)):
            for j in range(len(similarity)):
                g.add_edge(i, j, weight=similarity[i][j])
        cut, partition = nx.stoer_wagner(g)
        c1 = np.array([idc[x] for x in partition[0]])
        c2 = np.array([idc[x] for x in partition[1]])
        return c1, c2

    # def max_choose(self, similarity, idc):
    #     g = nx.Graph()
    #     for i in range(len(similarity)):
    #         for j in range(len(similarity)):
    #             g.add_edge(i, j, weight=similarity[i][j])
    #     cut, partition = nx.stoer_wagner(g)
    #     c1 = np.array([idc[x] for x in partition[0]])
    #     c2 = np.array([idc[x] for x in partition[1]])
    #     return c1, c2

    def get_clusters_with_alg(linkage_matrix, n_sampled, weights):
        """Algorithm 2"""
        epsilon = int(10 ** 10)

        # associate each client to a cluster
        link_matrix_p = deepcopy(linkage_matrix)
        augmented_weights = deepcopy(weights)

        for i in range(len(link_matrix_p)):
            idx_1, idx_2 = int(link_matrix_p[i, 0]), int(link_matrix_p[i, 1])

            new_weight = np.array(
                [augmented_weights[idx_1] + augmented_weights[idx_2]]
            )
            augmented_weights = np.concatenate((augmented_weights, new_weight))
            link_matrix_p[i, 2] = int(new_weight * epsilon)

        clusters = fcluster(
            link_matrix_p, int(epsilon / n_sampled), criterion="distance"
        )

        n_clients, n_clusters = len(clusters), len(set(clusters))

        # Associate each cluster to its number of clients in the cluster
        pop_clusters = np.zeros((n_clusters, 2)).astype(int)
        for i in range(n_clusters):
            pop_clusters[i, 0] = i + 1
            for client in np.where(clusters == i + 1)[0]:
                pop_clusters[i, 1] += int(weights[client] * epsilon * n_sampled)

        pop_clusters = pop_clusters[pop_clusters[:, 1].argsort()]

        distri_clusters = np.zeros((n_sampled, n_clients)).astype(int)

        # n_sampled biggest clusters that will remain unchanged
        kept_clusters = pop_clusters[n_clusters - n_sampled:, 0]

        for idx, cluster in enumerate(kept_clusters):
            for client in np.where(clusters == cluster)[0]:
                distri_clusters[idx, client] = int(
                    weights[client] * n_sampled * epsilon
                )

        k = 0
        for j in pop_clusters[: n_clusters - n_sampled, 0]:

            clients_in_j = np.where(clusters == j)[0]
            np.random.shuffle(clients_in_j)

            for client in clients_in_j:

                weight_client = int(weights[client] * epsilon * n_sampled)

                while weight_client > 0:

                    sum_proba_in_k = np.sum(distri_clusters[k])

                    u_i = min(epsilon - sum_proba_in_k, weight_client)

                    distri_clusters[k, client] = u_i
                    weight_client += -u_i

                    sum_proba_in_k = np.sum(distri_clusters[k])
                    if sum_proba_in_k == 1 * epsilon:
                        k += 1

        distri_clusters = distri_clusters.astype(float)
        for l in range(n_sampled):
            distri_clusters[l] /= np.sum(distri_clusters[l])

        return distri_clusters

    def aggregate_clusterwise(self, client_clusters):
        for cluster in client_clusters:
            targs = []
            sours = []
            total_size = 0
            for client in cluster:
                W = {}
                dW = {}
                for k in self.W.keys():
                    W[k] = client.W[k]
                    dW[k] = client.dW[k]
                targs.append(W)
                sours.append((dW, client.train_size))
                total_size += client.train_size
            # pass train_size, and weighted aggregate
            reduce_add_average(targets=targs, sources=sours, total_size=total_size)

    def compute_max_update_norm(self, cluster):
        max_dW = -np.inf
        for client in cluster:
            dW = {}
            for k in self.W.keys():
                dW[k] = client.dW[k]
            update_norm = torch.norm(flatten(dW)).item()
            if update_norm > max_dW:
                max_dW = update_norm
        return max_dW
        # return np.max([torch.norm(flatten(client.dW)).item() for client in cluster])

    def compute_mean_update_norm(self, cluster):
        cluster_dWs = []
        for client in cluster:
            dW = {}
            for k in self.W.keys():
                dW[k] = client.dW[k]
            cluster_dWs.append(flatten(dW))

        return torch.norm(torch.mean(torch.stack(cluster_dWs), dim=0)).item()

    def cache_model(self, idcs, params, accuracies):
        self.model_cache += [(idcs,
                              {name: params[name].data.clone() for name in params},
                              [accuracies[i] for i in idcs])]

    def reset_cache_model(self):
        self.model_cache = []

    def evaluate(self):
        return eval_server(self.model, self.dataLoader['test'], self.args.device)


def flatten(source):
    return torch.cat([value.flatten() for value in source.values()])


def pairwise_angles(sources):
    angles = torch.zeros([len(sources), len(sources)])
    for i, source1 in enumerate(sources):
        for j, source2 in enumerate(sources):
            s1 = flatten(source1)
            s2 = flatten(source2)
            angles[i, j] = torch.true_divide(torch.sum(s1 * s2), max(torch.norm(s1) * torch.norm(s2), 1e-16)) + 1
    return angles.numpy()


def reduce_add_average(targets, sources, total_size):
    for target in targets:
        for name in target:
            tmp = torch.div(
                torch.sum(torch.stack([torch.mul(source[0][name].data, source[1]) for source in sources]), dim=0),
                total_size).clone()
            target[name].data += tmp


def calculate_hypweights(Weights, delta_Weights, Grads, total_size):
    for Weight in Weights:
        for name in Weight:
            delta = torch.div(
                torch.sum(torch.stack(
                    [torch.mul(delta_Weight[0][name].data, delta_Weight[1]) for delta_Weight in delta_Weights]), dim=0),
                total_size).clone()
            Weight[name].data += delta
    a = 0
    return Weights


def calculate_klein_midpoint(self, Weights, delta_Weights, Grads, total_size):
    a = 1
    if a==0:
        # 假设我们的gradient是在欧几里得的，直接用FedAvg对gradient聚合
        for Weight in Weights:
            for name in Weight:
                delta = torch.div(
                    torch.sum(torch.stack([torch.mul(Grad[0][name].data * 0.01, Grad[1]) for Grad in Grads]), dim=0),
                    total_size).clone()
                Weight[name].data += delta
        return Weights
    elif a == 1:
        # 假设我们的gradient是在欧几里得的，我们将其映射到庞加莱球中再用klein midpoint计算
        for Weight in Weights:
            for name in Weight:
                sum_al = 0
                als = []
                for delta_Weight in delta_Weights:
                    norm = torch.norm(delta_Weight[0][name].data)
                    l = 2 / ((1+self.c*norm) ** 2)
                    l1 = l
                    l2 = l-1
                    als.append((delta_Weight[1] / total_size * l1, delta_Weight[0]))
                    sum_al += delta_Weight[1] / total_size * l2
                delta = torch.sum(torch.stack([torch.mul(al[0]/sum_al, self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(al[1][name].data, self.c), self.c), self.c)) for al in als]), dim=0).clone()
                delta = self.manifold.proj_tan0(self.manifold.logmap0(0.5 * delta, c=self.c), c=self.c)
                Weight[name].data += delta
        return Weights
    elif a == 2:
        # 假设我们的gradient是在庞加莱球模型中的用klein midpoint
        for Weight in Weights:
            for name in Weight:
                sum_al = 0
                als = []
                for Grad in Grads:
                    norm = torch.norm(Grad[0][name].data)
                    l = 2 / ((norm) ** 2)
                    l1 = l
                    l2 = l-1
                    als.append((Grad[1] / total_size * l1, Grad[0]))
                    sum_al += Grad[1] / total_size * l2
                delta = torch.sum(torch.stack([torch.mul(al[0]/sum_al, al[1][name].data) for al in als]), dim=0).clone()
                Weight[name].data += 0.5 * delta * 0.01
        return Weights


def eval_server(model, test_loader, device):
    model.eval()
    total_loss = 0.
    acc_sum = 0.
    ngraphs = 0
    for databatch in test_loader:
        databatch.to(device)
        # print("eval_local")
        adj = reset_batch_adj(databatch)
        pred = model(databatch, adj)
        label = databatch.y
        loss = model.loss(pred, label)
        total_loss += loss.item() * databatch.num_graphs
        acc_sum += pred.max(dim=1)[1].eq(label).sum().item()
        ngraphs += databatch.num_graphs
    return total_loss / ngraphs, acc_sum / ngraphs


def test_in_server(model, dataloaders, device):
    test_loader = dataloaders
    return eval_server(model, test_loader, device)
    # losses_test.append(loss_tt)
    # accs_test.append(acc_tt)

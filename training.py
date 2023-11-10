import math

import pandas as pd
import numpy as np
import time
import copy
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from torch import relu
from torch.distributed import reduce


def run_selftrain(clients, server, COMMUNICATION_ROUNDS, local_epoch=1, samp=None, frac=1.0):
    # all clients are initialized with the same weights
    client_number = len(clients)
    for client in clients:
        client.download_from_server(server)
    frame1, frame2 = pd.DataFrame(), pd.DataFrame()
    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        avg_loss = 0.0
        avg_acc = 0.0
        i = 0
        avg_train_loss = 0.0
        for client in clients:
            train_loss = client.local_train(local_epoch)
            avg_train_loss = avg_train_loss + train_loss
        loss_train = avg_train_loss / client_number

        for client in clients:
            i = i + 1
            loss, acc = client.evaluate()
            frame2.loc[client.name, 'test_acc'] = acc
            avg_loss = avg_loss + loss
            avg_acc = avg_acc + acc

        avg_loss = avg_loss / client_number
        avg_acc = avg_acc / client_number
        frame1.loc[str(c_round), 'avg_loss'] = avg_loss
        frame1.loc[str(c_round), 'avg_acc'] = avg_acc

        print('Iteration: {:04d}'.format(c_round),
              'loss_train: {:.6f}'.format(loss_train),
              'loss_test: {:.6f}'.format(avg_loss),
              'acc_test: {:.6f}'.format(avg_acc))
    return frame1, frame2


def run_fedavg(clients, server, COMMUNICATION_ROUNDS, local_epoch=10, samp=None, frac=1.0):

    client_number = len(clients)
    for client in clients:
        client.download_from_server(server)
    if samp is None:
        sampling_fn = server.randomSample_clients
        frac = 1.0

    frame1, frame2 = pd.DataFrame(), pd.DataFrame()
    personlize_lambda = []
    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        start = time.perf_counter()
        loss_locals = []
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")

        if c_round == 1:
            selected_clients = clients
        else:
            selected_clients = sampling_fn(clients, frac)

        avg_train_loss = 0.0
        for client in selected_clients:
            # only get weights of graphconv layers
            # train_loss = client.local_train(local_epoch)
            train_loss = client.local_train(local_epoch)
            avg_train_loss = avg_train_loss + train_loss
            loss_locals.append(copy.deepcopy(train_loss))
        loss_train = avg_train_loss / client_number
        server.aggregate_weights(selected_clients)  # server做聚合
        # server.aggregate_hyperbolic_weights(selected_clients) # server的双曲做聚合
        # server.aggregate_loss_weights(selected_clients, personlize_lambda)
        for client in selected_clients:
            client.download_from_server(server)

        avg_loss = 0.0
        avg_acc = 0.0
        i = 0
        for client in clients:
            i = i + 1
            loss_t, acc_t = client.evaluate()
            frame2.loc[client.name, 'test_acc'] = acc_t
            avg_loss = avg_loss + loss_t
            avg_acc = avg_acc + acc_t
            # acc_t, loss_t = test_img(net_glob, dataset_test, args)
            # diff_loss = [abs(loss_t - i) for i in loss_locals]
            # avg_loss_diff = abs(loss_t - np.mean(loss_locals))  # 76.00
            # avg_loss_diff_list = [i / avg_loss_diff for i in diff_loss]
            # diff_list = [(i - min(avg_loss_diff_list)) / (max(avg_loss_diff_list) - min(avg_loss_diff_list)) for i in
            #              avg_loss_diff_list]
            # personlize_lambda = [1 - i for i in diff_loss]
            # personlize_lambda = [i / avg_loss_diff for i in diff_loss]
            # if i == 1:
            #     one_loss, one_acc = loss, acc
        avg_loss = avg_loss / client_number
        avg_acc = avg_acc / client_number
        costtime = time.perf_counter()-start
        frame1.loc[str(c_round), 'avg_loss'] = avg_loss
        frame1.loc[str(c_round), 'avg_acc'] = avg_acc
        frame1.loc[str(c_round), 'time'] = costtime

        print('Iteration: {:04d}'.format(c_round),
              'loss_train: {:.6f}'.format(loss_train),
              'loss_test: {:.6f}'.format(avg_loss),
              'acc_test: {:.6f}'.format(avg_acc),
              'time: {:.6f}'.format(costtime))

        # acc_t, loss_t = test_img(net_glob, dataset_test, args)
        # diff_loss = [abs(loss_t - i) for i in loss_locals]
        # avg_loss_diff = abs(loss_t - np.mean(loss_locals))  # 76.00
        # avg_loss_diff_list = [i / avg_loss_diff for i in diff_loss]
        # diff_list = [(i - min(avg_loss_diff_list)) / (max(avg_loss_diff_list) - min(avg_loss_diff_list)) for i in
        #              avg_loss_diff_list]
        # args.personlize_lambda = [1 - i for i in diff_loss]
        # print('Epoch: {:04d}'.format(c_round + 1), 'loss_train: {:.6f}'.format(loss_train),
        #       'acc_train: {:.6f}'.format(acc_train), 'loss_val: {:.6f}'.format(loss_val),
        #       'acc_val: {:.6f}'.format(acc_val), 'time: {:.6f}s'.format(time.time() - t))

    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: yellow' if v else '' for v in is_max]

    fs = frame2.style.apply(highlight_max).data
    print(fs)
    return frame1, frame2

def run_personlize_lambda(clients, server, num_list,  COMMUNICATION_ROUNDS, local_epoch=10, samp=None, frac=1.0):
    client_number = len(clients)
    for client in clients:
        client.download_from_server(server)
    if samp is None:
        sampling_fn = server.randomSample_clients
        frac = 1.0

    frame1, frame2 = pd.DataFrame(), pd.DataFrame()
    personlize_lambda = []
    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        start = time.perf_counter()
        acc_locals = []
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")

        if c_round == 1:
            selected_clients = clients
        else:
            selected_clients = sampling_fn(clients, frac)

        avg_train_loss = 0.0
        for client in selected_clients:
            # only get weights of graphconv layers
            # train_loss = client.local_train(local_epoch)
            train_acc, train_loss = client.compute_weight_update(local_epoch)
            avg_train_loss = avg_train_loss + train_loss
            acc_locals.append(copy.deepcopy(train_acc))
        loss_train = avg_train_loss / client_number
        # server.aggregate_weights(selected_clients)  # server做聚合
        server.aggregate_with_personlize_lambda(selected_clients, personlize_lambda)
        # server.aggregate_hyp_weights(selected_clients, personlize_lambda)
        for client in selected_clients:
            client.download_from_server(server)

        avg_loss = 0.0
        avg_acc = 0.0
        i = 0
        for client in clients:
            i = i + 1
            loss_t, acc_t = client.evaluate()
            frame2.loc[client.name, 'test_acc'] = acc_t
            avg_loss = avg_loss + loss_t
            avg_acc = avg_acc + acc_t
            # acc_t, loss_t = test_img(net_glob, dataset_test, args)
            # diff_loss = [abs(loss_t - i) for i in loss_locals]
            # avg_loss_diff = abs(loss_t - np.mean(loss_locals))  # 76.00
            # avg_loss_diff_list = [i / avg_loss_diff for i in diff_loss]
            # diff_list = [(i - min(avg_loss_diff_list)) / (max(avg_loss_diff_list) - min(avg_loss_diff_list)) for i in
            #              avg_loss_diff_list]
            # personlize_lambda = [i / np.mean(acc_locals) for i in acc_locals] # acc based

            acc_rito_z_score = [(i - np.mean(acc_locals))/ np.std(acc_locals) for i in acc_locals]
            size_rito = [i / np.mean(num_list) for i in num_list]
            personlize_lambda = np.sum([acc_rito_z_score,size_rito],axis=0).tolist()

            # personlize_lambda = [i / avg_loss_diff for i in diff_loss]
            # if i == 1:
            #     one_loss, one_acc = loss, acc
        avg_loss = avg_loss / client_number
        avg_acc = avg_acc / client_number
        costtime = time.perf_counter()-start
        frame1.loc[str(c_round), 'avg_loss'] = avg_loss
        frame1.loc[str(c_round), 'avg_acc'] = avg_acc
        frame1.loc[str(c_round), 'time'] = costtime

        print('Iteration: {:04d}'.format(c_round),
              'loss_train: {:.6f}'.format(loss_train),
              'loss_test: {:.6f}'.format(avg_loss),
              'acc_test: {:.6f}'.format(avg_acc),
              'time: {:.6f}'.format(costtime))

    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: yellow' if v else '' for v in is_max]

    fs = frame2.style.apply(highlight_max).data
    print(fs)
    return frame1, frame2

def run_fedavg_noised(clients, server, COMMUNICATION_ROUNDS, local_epoch=10, samp=None, frac=1.0, noise=None):
    client_number = 0
    noise = noise
    for client in clients:
        client.download_from_server(server)
        client_number = client_number + 1
    if samp is None:
        sampling_fn = server.randomSample_clients
        frac = 1.0

    frame1, frame2 = pd.DataFrame(), pd.DataFrame()

    for c_round in range(1, COMMUNICATION_ROUNDS + 1):

        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")

        if c_round == 1:
            selected_clients = clients
        else:
            selected_clients = sampling_fn(clients, frac)
        i = 0
        for client in selected_clients:
            need_noise = noise[i]
            # only get weights of graphconv layers
            client.local_train(local_epoch, need_noise)
            i = i + 1

        server.aggregate_weights(selected_clients)  # server做聚合

        for client in selected_clients:
            client.download_from_server(server)

        avg_loss = 0.0
        avg_acc = 0.0
        i = 0
        for client in clients:
            i = i + 1
            loss, acc = client.evaluate()
            frame2.loc[client.name, 'test_acc'] = acc
            avg_loss = avg_loss + loss
            avg_acc = avg_acc + acc
        avg_loss = avg_loss / client_number
        avg_acc = avg_acc / client_number
        frame1.loc[str(c_round), 'avg_loss'] = avg_loss
        frame1.loc[str(c_round), 'avg_acc'] = avg_acc

        print('Iteration: {:04d}'.format(c_round),
              'loss_test: {:.6f}'.format(avg_loss),
              'acc_test: {:.6f}'.format(avg_acc))

        # print('Epoch: {:04d}'.format(c_round + 1), 'loss_train: {:.6f}'.format(loss_train),
        #       'acc_train: {:.6f}'.format(acc_train), 'loss_val: {:.6f}'.format(loss_val),
        #       'acc_val: {:.6f}'.format(acc_val), 'time: {:.6f}s'.format(time.time() - t))

    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: yellow' if v else '' for v in is_max]

    fs = frame2.style.apply(highlight_max).data
    print(fs)
    return frame1, frame2

def run_klein_midpoint(clients, server, COMMUNICATION_ROUNDS, local_epoch=10, samp=None, frac=1.0):
    client_number = len(clients)
    for client in clients:
        client.download_from_server(server)
    if samp is None:
        sampling_fn = server.randomSample_clients
        frac = 1.0

    frame1, frame2 = pd.DataFrame(), pd.DataFrame()
    personlize_lambda = []
    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        start = time.perf_counter()
        loss_locals = []
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")

        if c_round == 1:
            for client in clients:
                client.download_from_server(server)
            selected_clients = clients
        else:
            selected_clients = sampling_fn(clients, frac)

        avg_train_loss = 0.0
        for client in selected_clients:
            # only get weights of graphconv layers
            # train_loss = client.local_train(local_epoch)
            train_acc, train_loss = client.compute_weight_update(local_epoch)
            avg_train_loss = avg_train_loss + train_loss
            loss_locals.append(copy.deepcopy(train_loss))
        loss_train = avg_train_loss / client_number
        # server.aggregate_weights(selected_clients)  # server做聚合
        # server.aggregate_with_personlize_lambda(selected_clients, personlize_lambda)
        server.aggregate_hyp_weights(selected_clients, personlize_lambda)

        # for client in selected_clients:
        #     client.download_from_server(server)

        avg_loss = 0.0
        avg_acc = 0.0
        i = 0
        for client in clients:
            i = i + 1
            loss_t, acc_t = client.evaluate()
            frame2.loc[client.name, 'test_acc'] = acc_t
            avg_loss = avg_loss + loss_t
            avg_acc = avg_acc + acc_t
            # acc_t, loss_t = test_img(net_glob, dataset_test, args)
            diff_loss = [abs(loss_t - i) for i in loss_locals]
            # avg_loss_diff = abs(loss_t - np.mean(loss_locals))
            # avg_loss_diff_list = [i / avg_loss_diff for i in diff_loss]
            # diff_list = [(i - min(avg_loss_diff_list)) / (max(avg_loss_diff_list) - min(avg_loss_diff_list)) for i in
            #              avg_loss_diff_list]
            personlize_lambda = [1 - i for i in diff_loss]
            # personlize_lambda = [i / avg_loss_diff for i in diff_loss]
            # if i == 1:
            #     one_loss, one_acc = loss, acc
        avg_loss = avg_loss / client_number
        avg_acc = avg_acc / client_number
        costtime = time.perf_counter()-start
        frame1.loc[str(c_round), 'avg_loss'] = avg_loss
        frame1.loc[str(c_round), 'avg_acc'] = avg_acc
        frame1.loc[str(c_round), 'time'] = costtime

        print('Iteration: {:04d}'.format(c_round),
              'loss_train: {:.6f}'.format(loss_train),
              'loss_test: {:.6f}'.format(avg_loss),
              'acc_test: {:.6f}'.format(avg_acc),
              'time: {:.6f}'.format(costtime))

    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: yellow' if v else '' for v in is_max]

    fs = frame2.style.apply(highlight_max).data
    print(fs)
    return frame1, frame2


def run_fedgraphcluster(clients, server, COMMUNICATION_ROUNDS, local_epoch=10, samp=None, frac=1.0):
    client_number = len(clients)
    cluster_indices = [np.arange(len(clients)).astype("int")]
    client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]
    # seqs_grads = {c.id: [] for c in clients}

    frame1, frame2 = pd.DataFrame(), pd.DataFrame()

    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")

        if c_round == 1:
            for client in clients:
                client.download_from_server(server)
            cluster_indices_new = []
            for idcs in cluster_indices:
                for i in idcs:
                    cluster_indices_new += [[i]]
            cluster_indices = cluster_indices_new

        selected_clients = server.randomSample_clients(clients, frac=1.0)

        avg_train_loss = 0.0
        for client in selected_clients:
            train_loss = client.compute_weight_update(local_epoch)
            avg_train_loss = avg_train_loss + train_loss
            client.reset()
            # seqs_grads[client.id].append(client.convDWsNorm)
        loss_train = avg_train_loss / client_number

        server.aggregate_weights(selected_clients)  # server做聚合
        similarities = server.compute_pairwise_similarities(clients)
        # cluster_indices_new = []
        # for idc in cluster_indices:
        #     if len(idc) > 2 and c_round > 20 and all(len(value) >= 5 for value in seqs_grads.values()):
        #
        #         server.cache_model(idc, clients[idc[0]].W, acc_clients)
        #
        #         tmp = [seqs_grads[id][-5:] for id in idc]
        #         dtw_distances = server.compute_pairwise_distances(tmp, False)
        #         c1, c2 = server.min_cut(np.max(dtw_distances) - dtw_distances, idc)
        #         cluster_indices_new += [c1, c2]
        #
        #         seqs_grads = {c.id: [] for c in clients}
        #     else:
        #         cluster_indices_new += [idc]
        cluster_indices_new = []
        if c_round >= 2 and c_round % 5 == 2 and len(cluster_indices) > 2:
            server.reset_cache_model()
            clusters, weightclusters = [], []
            for idc in cluster_indices:
                server.cache_model(idc, clients[idc[0]].W, acc_clients)
                clusters += [{name: clients[idc[0]].W[name].data.clone() for name in clients[idc[0]].W}]
                similarities = server.compute_pairwise_cluster_similarities(clusters)
            n_clusters_ = round((len(cluster_indices)) / 2)
            ac = AgglomerativeClustering(linkage="ward", n_clusters=n_clusters_)
            ac.fit(cluster_indices)
            labels = ac.labels_
            for i in range(n_clusters_):
                clusteri = []
                for j in range(len(labels)):
                    if labels[j] == i:
                        add = cluster_indices[j]
                        clusteri.append(add)
                clusteri = np.hstack(clusteri)
                cluster_indices_new += [clusteri]
            A = 0
        else:
            for idc in cluster_indices:
                cluster_indices_new += [idc]

        cluster_indices = cluster_indices_new
        client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]  # initial: [[0, 1, ...]]

        server.aggregate_clusterwise(client_clusters)
        acc_clients = [client.evaluate()[1] for client in clients]

        i, avg_loss, avg_acc = 0, 0.0, 0.0
        for client in clients:
            i = i + 1
            loss, acc = client.evaluate()
            frame2.loc[client.name, 'test_acc'] = acc
            avg_loss = avg_loss + loss
            avg_acc = avg_acc + acc
            # if i == 1:
            #     one_loss, one_acc = loss, acc
        avg_loss = avg_loss / client_number
        avg_acc = avg_acc / client_number
        frame1.loc[str(c_round), 'avg_loss'] = avg_loss
        frame1.loc[str(c_round), 'avg_acc'] = avg_acc

        print('Iteration: {:04d}'.format(c_round),
              'loss_train: {:.6f}'.format(loss_train),
              'loss_test: {:.6f}'.format(avg_loss),
              'acc_test: {:.6f}'.format(avg_acc))

    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: yellow' if v else '' for v in is_max]

    fs = frame2.style.apply(highlight_max).data
    print(fs)
    return frame1, frame2


def run_gcfl(clients, server, COMMUNICATION_ROUNDS, local_epoch=10, samp=None, frac=1.0):
    client_number = len(clients)
    cluster_indices = [np.arange(len(clients)).astype("int")]
    client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]
    seqs_grads = {c.id: [] for c in clients}

    frame1, frame2 = pd.DataFrame(), pd.DataFrame()

    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        # if (c_round) % 1 == 0:
        #     print(f"  > round {c_round}")
        if c_round == 1:
            for client in clients:
                client.download_from_server(server)

        participating_clients = server.randomSample_clients(clients, frac=1.0)
        avg_train_loss = 0.0
        for client in participating_clients:
            train_loss = client.compute_weight_update(local_epoch)
            avg_train_loss = avg_train_loss + train_loss
            client.reset()
            seqs_grads[client.id].append(client.convDWsNorm)
        loss_train = avg_train_loss / client_number
        # similarities = server.compute_pairwise_similarities(clients)

        cluster_indices_new = []
        for idc in cluster_indices:
            max_norm = server.compute_max_update_norm([clients[i] for i in idc])
            mean_norm = server.compute_mean_update_norm([clients[i] for i in idc])
            if mean_norm < 0.05 and max_norm > 0.1 and len(idc) > 2 and c_round > 20 \
                    and all(len(value) >= 5 for value in seqs_grads.values()):

                server.cache_model(idc, clients[idc[0]].W, acc_clients)

                tmp = [seqs_grads[id][-5:] for id in idc]
                dtw_distances = server.compute_pairwise_distances(tmp, False)
                c1, c2 = server.min_cut(np.max(dtw_distances) - dtw_distances, idc)
                cluster_indices_new += [c1, c2]

                seqs_grads = {c.id: [] for c in clients}
            else:
                cluster_indices_new += [idc]

        cluster_indices = cluster_indices_new
        client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]  # initial: [[0, 1, ...]]

        server.aggregate_clusterwise(client_clusters)

        acc_clients = [client.evaluate()[1] for client in clients]
        i, avg_loss, avg_acc = 0, 0.0, 0.0
        for client in clients:
            i = i + 1
            loss, acc = client.evaluate()
            frame2.loc[client.name, 'test_acc'] = acc
            avg_loss = avg_loss + loss
            avg_acc = avg_acc + acc
            # if i == 1:
            #     one_loss, one_acc = loss, acc
        avg_loss = avg_loss / client_number
        avg_acc = avg_acc / client_number
        frame1.loc[str(c_round), 'avg_loss'] = avg_loss
        frame1.loc[str(c_round), 'avg_acc'] = avg_acc

        print('Iteration: {:04d}'.format(c_round),
              'loss_train: {:.6f}'.format(loss_train),
              'loss_test: {:.6f}'.format(avg_loss),
              'acc_test: {:.6f}'.format(avg_acc))

    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: yellow' if v else '' for v in is_max]

    fs = frame2.style.apply(highlight_max).data
    print(fs)
    return frame1, frame2

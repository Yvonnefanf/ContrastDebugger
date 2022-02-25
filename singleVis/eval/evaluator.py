import os
import json
import gc

import numpy as np

from singleVis.eval.evaluate import *
from singleVis.backend import *

class Evaluator:
    def __init__(self, data_provider, trainer, verbose=1):
        self.data_provider = data_provider
        self.trainer = trainer
        self.verbose = verbose

    ####################################### ATOM #############################################

    def eval_nn_train(self, epoch, n_neighbors):
        train_data = self.data_provider.train_representation(epoch)
        self.trainer.model.eval()
        embedding = self.trainer.model.encoder(
            torch.from_numpy(train_data).to(dtype=torch.float32, device=self.trainer.DEVICE)).cpu().detach().numpy()
        val = evaluate_proj_nn_perseverance_knn(train_data, embedding, n_neighbors=n_neighbors, metric="euclidean")
        if self.verbose:
            print("#train# nn preserving: {:.2f}/{:d} in epoch {:d}".format(val, n_neighbors, epoch))
        return val

    def eval_nn_test(self, epoch, n_neighbors):
        train_data = self.data_provider.train_representation(epoch)
        test_data = self.data_provider.test_representation(epoch)
        fitting_data = np.concatenate((train_data, test_data), axis=0)
        self.trainer.model.eval()
        embedding = self.trainer.model.encoder(
            torch.from_numpy(fitting_data).to(dtype=torch.float32, device=self.trainer.DEVICE)).cpu().detach().numpy()
        val = evaluate_proj_nn_perseverance_knn(fitting_data, embedding, n_neighbors=n_neighbors, metric="euclidean")
        if self.verbose:
            print("#test# nn preserving : {:.2f}/{:d} in epoch {:d}".format(val, n_neighbors, epoch))
        return val

    def eval_b_train(self, epoch, n_neighbors):
        self.trainer.model.eval()
        train_data = self.data_provider.train_representation(epoch)
        border_centers = self.data_provider.border_representation(epoch)

        low_center = self.trainer.model.encoder(
            torch.from_numpy(border_centers).to(dtype=torch.float32, device=self.trainer.DEVICE)).cpu().detach().numpy()
        low_train = self.trainer.model.encoder(
            torch.from_numpy(train_data).to(dtype=torch.float32, device=self.trainer.DEVICE)).cpu().detach().numpy()

        val = evaluate_proj_boundary_perseverance_knn(train_data,
                                                      low_train,
                                                      border_centers,
                                                      low_center,
                                                      n_neighbors=n_neighbors)
        if self.verbose:
            print("#train# boundary preserving: {:.2f}/{:d} in epoch {:d}".format(val, n_neighbors, epoch))
        return val

    def eval_b_test(self, epoch, n_neighbors):
        self.trainer.model.eval()
        test_data = self.data_provider.test_representation(epoch)
        border_centers = self.data_provider.border_representation(epoch)

        low_center = self.trainer.model.encoder(
            torch.from_numpy(border_centers).to(dtype=torch.float32, device=self.trainer.DEVICE)).cpu().detach().numpy()
        low_test = self.trainer.model.encoder(
            torch.from_numpy(test_data).to(dtype=torch.float32, device=self.trainer.DEVICE)).cpu().detach().numpy()

        val = evaluate_proj_boundary_perseverance_knn(test_data,
                                                      low_test,
                                                      border_centers,
                                                      low_center,
                                                      n_neighbors=n_neighbors)
        if self.verbose:
            print("#test# boundary preserving: {:.2f}/{:d} in epoch {:d}".format(val, n_neighbors, epoch))
        return val

    def eval_inv_train(self, epoch):
        train_data = self.data_provider.train_representation(epoch)
        embedding = self.trainer.model.encoder(
            torch.from_numpy(train_data).to(dtype=torch.float32, device=self.trainer.DEVICE)).cpu().detach().numpy()
        inv_data = self.trainer.model.decoder(
            torch.from_numpy(embedding).to(dtype=torch.float32, device=self.trainer.DEVICE)).cpu().detach().numpy()

        pred = self.data_provider.get_pred(epoch, train_data).argmax(axis=1)
        new_pred = self.data_provider.get_pred(epoch, inv_data).argmax(axis=1)

        val = evaluate_inv_accu(pred, new_pred)
        if self.verbose:
            print("#train# PPR: {:.2f} in epoch {:d}".format(val, epoch))
        return val

    def eval_inv_test(self, epoch):
        test_data = self.data_provider.test_representation(epoch)
        embedding = self.trainer.model.encoder(
            torch.from_numpy(test_data).to(dtype=torch.float32, device=self.trainer.DEVICE)).cpu().detach().numpy()
        inv_data = self.trainer.model.decoder(
            torch.from_numpy(embedding).to(dtype=torch.float32, device=self.trainer.DEVICE)).cpu().detach().numpy()

        pred = self.data_provider.get_pred(epoch, test_data).argmax(axis=1)
        new_pred = self.data_provider.get_pred(epoch, inv_data).argmax(axis=1)

        val = evaluate_inv_accu(pred, new_pred)
        if self.verbose:
            print("#test# PPR: {:.2f} in epoch {:d}".format(val, epoch))
        return val

    def eval_temporal_train(self, n_neighbors):
        eval_num = (self.data_provider.e - self.data_provider.s) // self.data_provider.p
        l = self.data_provider.train_num

        alpha = np.zeros((eval_num, l))
        delta_x = np.zeros((eval_num, l))

        self.trainer.model.eval()
        for t in range(eval_num):
            prev_data = self.data_provider.train_representation(t * self.data_provider.p + self.data_provider.s)
            prev_embedding = self.trainer.model.encoder(
                torch.from_numpy(prev_data).to(dtype=torch.float32, device=self.trainer.DEVICE)).cpu().detach().numpy()

            curr_data = self.data_provider.train_representation((t+1) * self.data_provider.p + self.data_provider.s)
            curr_embedding = self.trainer.model.encoder(
                torch.from_numpy(curr_data).to(dtype=torch.float32, device=self.trainer.DEVICE)).cpu().detach().numpy()

            alpha_ = find_neighbor_preserving_rate(prev_data, curr_data, n_neighbors=n_neighbors)
            delta_x_ = np.linalg.norm(prev_embedding - curr_embedding, axis=1)

            alpha[t] = alpha_
            delta_x[t] = delta_x_

        val_corr, corr_std = evaluate_proj_temporal_perseverance_corr(alpha, delta_x)
        if self.verbose:
            print("Temporal preserving (train): {:.3f}\t std :{:.3f}".format(val_corr, corr_std))
        return val_corr, corr_std

    def eval_temporal_test(self, n_neighbors):
        eval_num = (self.data_provider.e - self.data_provider.s) // self.data_provider.p
        l = self.data_provider.train_num + self.data_provider.test_num

        alpha = np.zeros((eval_num, l))
        delta_x = np.zeros((eval_num, l))
        for t in range(eval_num):
            prev_data_test = self.data_provider.test_representation(t * self.data_provider.p + self.data_provider.s)
            prev_data_train = self.data_provider.train_representation(t * self.data_provider.p + self.data_provider.s)
            prev_data = np.concatenate((prev_data_train, prev_data_test), axis=0)
            prev_embedding = self.trainer.model.encoder(
                torch.from_numpy(prev_data).to(dtype=torch.float32, device=self.trainer.DEVICE)).cpu().detach().numpy()

            curr_data_test = self.data_provider.test_representation((t+1) * self.data_provider.p + self.data_provider.s)
            curr_data_train = self.data_provider.train_representation((t+1) * self.data_provider.p + self.data_provider.s)
            curr_data = np.concatenate((curr_data_train, curr_data_test), axis=0)
            curr_embedding = self.trainer.model.encoder(
                torch.from_numpy(curr_data).to(dtype=torch.float32, device=self.trainer.DEVICE)).cpu().detach().numpy()

            alpha_ = find_neighbor_preserving_rate(prev_data, curr_data, n_neighbors=n_neighbors)
            delta_x_ = np.linalg.norm(prev_embedding - curr_embedding, axis=1)

            alpha[t] = alpha_
            delta_x[t] = delta_x_

        val_corr, corr_std = evaluate_proj_temporal_perseverance_corr(alpha, delta_x)
        if self.verbose:
            print("Temporal preserving (test): {:.3f}\t std:{:.3f}".format(val_corr, corr_std))
        return val_corr, corr_std

    def eval_temporal_nn_train(self, epoch, n_neighbors):
        epoch_num = (self.data_provider.e - self.data_provider.s) // self.data_provider.p + 1
        l = self.data_provider.train_num
        high_dists = np.zeros((l, epoch_num))
        low_dists = np.zeros((l, epoch_num))

        self.trainer.model.eval()

        curr_data = self.data_provider.train_representation(epoch)
        curr_embedding = self.trainer.model.encoder(
            torch.from_numpy(curr_data).to(dtype=torch.float32, device=self.trainer.DEVICE)).cpu().detach().numpy()
        
        for t in range(epoch_num):
            data = self.data_provider.train_representation(t * self.data_provider.p + self.data_provider.s)
            embedding = self.trainer.model.encoder(
                torch.from_numpy(data).to(dtype=torch.float32, device=self.trainer.DEVICE)).cpu().detach().numpy()

            high_dist = np.linalg.norm(curr_data - data, axis=1)
            low_dist = np.linalg.norm(curr_embedding - embedding, axis=1)
            high_dists[:, t] = high_dist
            low_dists[:, t] = low_dist
        
        # find the index of top k dists
        high_orders = np.argsort(high_dists, axis=1)
        low_orders = np.argsort(low_dists, axis=1)
        high_rankings = np.zeros((high_orders.shape[0], n_neighbors), dtype=int)
        low_rankings = np.zeros((high_orders.shape[0], n_neighbors), dtype=int)
        for i in range(len(high_orders)):
            high_rankings[i] = np.argwhere(high_orders[i]<n_neighbors).squeeze()
            low_rankings[i] = np.argwhere(low_orders[i]<n_neighbors).squeeze()
        
        corr = np.zeros(len(high_dists))
        for i in range(len(data)):
            corr[i] = len(np.intersect1d(high_rankings[i], low_rankings[i]))

        if self.verbose:
            print("Temporal temporal neighbor preserving (train) for {}-th epoch {}: {:.3f}\t std :{:.3f}".format(epoch, n_neighbors, corr.mean(), corr.std()))
        return float(corr.mean())

    def eval_temporal_nn_test(self, epoch, n_neighbors):
        epoch_num = (self.data_provider.e - self.data_provider.s) // self.data_provider.p + 1
        l = self.data_provider.test_num
        high_dists = np.zeros((l, epoch_num))
        low_dists = np.zeros((l, epoch_num))

        self.trainer.model.eval()

        curr_data = self.data_provider.test_representation(epoch)
        curr_embedding = self.trainer.model.encoder(
            torch.from_numpy(curr_data).to(dtype=torch.float32, device=self.trainer.DEVICE)).cpu().detach().numpy()
        for t in range(epoch_num):
            data = self.data_provider.test_representation(t * self.data_provider.p + self.data_provider.s)
            embedding = self.trainer.model.encoder(
                torch.from_numpy(data).to(dtype=torch.float32, device=self.trainer.DEVICE)).cpu().detach().numpy()

            high_dist = np.linalg.norm(curr_data - data, axis=1)
            low_dist = np.linalg.norm(curr_embedding - embedding, axis=1)
            high_dists[:, t] = high_dist
            low_dists[:,t] = low_dist
        
        # find the index of top k dists
        high_orders = np.argsort(high_dists, axis=1)
        low_orders = np.argsort(low_dists, axis=1)
        high_rankings = np.zeros((high_orders.shape[0], n_neighbors), dtype=int)
        low_rankings = np.zeros((low_orders.shape[0], n_neighbors), dtype=int)
        for i in range(len(high_orders)):
            high_rankings[i] = np.argwhere(high_orders[i]<n_neighbors).squeeze()
            low_rankings[i] = np.argwhere(low_orders[i]<n_neighbors).squeeze()
        
        # high_rankings = np.argsort(high_dists, axis=1)[:, :n_neighbors]
        # low_rankings = np.argsort(low_dists, axis=1)[:, :n_neighbors]
        corr = np.zeros(len(high_dists))
        for i in range(len(data)):
            corr[i] = len(np.intersect1d(high_rankings[i], low_rankings[i]))

        if self.verbose:
            print("Temporal ranking preserving (test) for {}-th epoch {}: {:.3f}\t std:{:.3f}".format(epoch, n_neighbors, corr.mean(), corr.std()))
        return float(corr.mean())

    def eval_spatial_temporal_nn_train(self, epoch, n_neighbors):
        """
            evaluate whether vis model can preserve the ranking of close spatial and temporal neighbors
        """
        # find n temporal neighbors
        epoch_num = (self.data_provider.e - self.data_provider.s) // self.data_provider.p + 1
        l = self.data_provider.train_num
        high_dists = np.zeros((l, epoch_num))
        low_dists = np.zeros((l, epoch_num))

        self.trainer.model.eval()

        curr_data = self.data_provider.train_representation(epoch)
        curr_embedding = self.trainer.model.encoder(
            torch.from_numpy(curr_data).to(dtype=torch.float32, device=self.trainer.DEVICE)).cpu().detach().numpy()
        
        for t in range(epoch_num):
            data = self.data_provider.train_representation(t * self.data_provider.p + self.data_provider.s)
            embedding = self.trainer.model.encoder(
                torch.from_numpy(data).to(dtype=torch.float32, device=self.trainer.DEVICE)).cpu().detach().numpy()

            high_dist = np.linalg.norm(curr_data - data, axis=1)
            low_dist = np.linalg.norm(curr_embedding - embedding, axis=1)
            high_dists[:, t] = high_dist
            low_dists[:, t] = low_dist
        
        # retrive top n temporal dists
        # high_top_n_arg = np.argsort(high_dists, axis=1)[:, :n_neighbors].squeeze()
        high_orders = np.argsort(high_dists, axis=1)
        high_top_n_arg = np.zeros((high_orders.shape[0], n_neighbors), dtype=int)
        for i in range(len(high_orders)):
            high_top_n_arg[i] = np.argwhere(high_orders[i]<n_neighbors).squeeze()

        high_top_n_dists_t = np.zeros(high_top_n_arg.shape)
        low_top_n_dists_t = np.zeros(high_top_n_arg.shape)
        for t in range(len(high_top_n_arg)):
            high_top_n_dists_t[t] = high_dists[t, high_top_n_arg[t]]
            low_top_n_dists_t[t] = low_dists[t, high_top_n_arg[t]]
        
        # find n spatial neighbors
        metric = "euclidean"
        data = self.data_provider.train_representation(epoch)
        self.trainer.model.eval()
        embedding = self.trainer.model.encoder(
            torch.from_numpy(data).to(dtype=torch.float32, device=self.trainer.DEVICE)).cpu().detach().numpy()
        n_trees = 5 + int(round((data.shape[0]) ** 0.5 / 20.0))
        n_iters = max(5, int(round(np.log2(data.shape[0]))))
        # get nearest neighbors
        nnd = NNDescent(
            data,
            n_neighbors=n_neighbors,
            metric=metric,
            n_trees=n_trees,
            n_iters=n_iters,
            max_candidates=60,
            verbose=True
        )
        _, high_dists = nnd.neighbor_graph
        nnd = NNDescent(
            embedding,
            n_neighbors=n_neighbors,
            metric=metric,
            n_trees=n_trees,
            n_iters=n_iters,
            max_candidates=60,
            verbose=True
        )
        _, low_dists = nnd.neighbor_graph

        # # retrive top n spatial dists
        # high_top_n_arg = np.argsort(high_dists, axis=1).squeeze()
        # high_top_n_dists_s = np.zeros(high_top_n_arg.shape)
        # low_top_n_dists_s = np.zeros(high_top_n_arg.shape)
        # for t in range(len(high_top_n_arg)):
        #     high_top_n_dists_s[t] = high_dists[t, high_top_n_arg[t]]
        #     low_top_n_dists_s[t] = low_dists[t, high_top_n_arg[t]]
        high_top_n_dists_s = np.copy(high_dists)
        low_top_n_dists_s = np.copy(low_dists)
        
        high_top_n_dists = np.concatenate((high_top_n_dists_t, high_top_n_dists_s), axis=1)
        low_top_n_dists = np.concatenate((low_top_n_dists_t, low_top_n_dists_s), axis=1)
        high_ranks = np.argsort(high_top_n_dists)
        low_ranks = np.argsort(low_top_n_dists)
        tau_l = evaluate_proj_temporal_temporal_corr(high_rank=high_ranks, low_rank=low_ranks)

        if self.verbose:
            print("Spatial/Temporal ranking preserving (train) for {}-th epoch {}:\t{:.3f}".format(epoch, n_neighbors, tau_l.mean()))
        return float(tau_l.mean())


    def eval_spatial_temporal_nn_test(self, epoch, n_neighbors):
        # find n temporal neighbors
        epoch_num = (self.data_provider.e - self.data_provider.s) // self.data_provider.p + 1
        l = self.data_provider.test_num
        high_dists = np.zeros((l, epoch_num))
        low_dists = np.zeros((l, epoch_num))

        self.trainer.model.eval()

        curr_data = self.data_provider.test_representation(epoch)
        curr_embedding = self.trainer.model.encoder(
            torch.from_numpy(curr_data).to(dtype=torch.float32, device=self.trainer.DEVICE)).cpu().detach().numpy()
        for t in range(epoch_num):
            data = self.data_provider.test_representation(t * self.data_provider.p + self.data_provider.s)
            embedding = self.trainer.model.encoder(
                torch.from_numpy(data).to(dtype=torch.float32, device=self.trainer.DEVICE)).cpu().detach().numpy()

            high_dist = np.linalg.norm(curr_data - data, axis=1)
            low_dist = np.linalg.norm(curr_embedding - embedding, axis=1)
            high_dists[:, t] = high_dist
            low_dists[:,t] = low_dist
        
        # retrive top n temporal dists
        # high_top_n_arg = np.argsort(high_dists, axis=1)[:, :n_neighbors].squeeze()
        high_orders = np.argsort(high_dists, axis=1)
        high_top_n_arg = np.zeros((high_orders.shape[0], n_neighbors), dtype=int)
        for i in range(len(high_orders)):
            high_top_n_arg[i] = np.argwhere(high_orders[i]<n_neighbors).squeeze()

        high_top_n_dists_t = np.zeros(high_top_n_arg.shape)
        low_top_n_dists_t = np.zeros(high_top_n_arg.shape)
        for t in range(len(high_top_n_arg)):
            high_top_n_dists_t[t] = high_dists[t, high_top_n_arg[t]]
            low_top_n_dists_t[t] = low_dists[t, high_top_n_arg[t]]


        # find n spatial neighbors(consider the training and testing data together)
        metric = "euclidean"
        train_data = self.data_provider.train_representation(epoch)
        test_data = self.data_provider.test_representation(epoch)
        data = np.concatenate((train_data, test_data), axis=0)
        self.trainer.model.eval()
        embedding = self.trainer.model.encoder(
            torch.from_numpy(data).to(dtype=torch.float32, device=self.trainer.DEVICE)).cpu().detach().numpy()

        n_trees = 5 + int(round((data.shape[0]) ** 0.5 / 20.0))
        n_iters = max(5, int(round(np.log2(data.shape[0]))))
        # get nearest neighbors
        nnd = NNDescent(
            data,
            n_neighbors=n_neighbors,
            metric=metric,
            n_trees=n_trees,
            n_iters=n_iters,
            max_candidates=60,
            verbose=True
        )
        _, high_dists = nnd.neighbor_graph
        nnd = NNDescent(
            embedding,
            n_neighbors=n_neighbors,
            metric=metric,
            n_trees=n_trees,
            n_iters=n_iters,
            max_candidates=60,
            verbose=True
        )
        _, low_dists = nnd.neighbor_graph

        high_dists = high_dists[-l:]
        low_dists = low_dists[-l:]

        # retrive top n spatial dists
        # high_top_n_arg = np.argsort(high_dists, axis=1).squeeze()
        # high_top_n_dists_s = np.zeros(high_top_n_arg.shape)
        # low_top_n_dists_s = np.zeros(high_top_n_arg.shape)
        # for t in range(len(high_top_n_arg)):
        #     high_top_n_dists_s[t] = high_dists[t, high_top_n_arg[t]]
        #     low_top_n_dists_s[t] = low_dists[t, high_top_n_arg[t]]
        high_top_n_dists_s = np.copy(high_dists)
        low_top_n_dists_s = np.copy(low_dists)
        
        high_top_n_dists = np.concatenate((high_top_n_dists_t, high_top_n_dists_s), axis=1)
        low_top_n_dists = np.concatenate((low_top_n_dists_t, low_top_n_dists_s), axis=1)
        high_ranks = np.argsort(high_top_n_dists)
        low_ranks = np.argsort(low_top_n_dists)
        tau_l = evaluate_proj_temporal_temporal_corr(high_rank=high_ranks, low_rank=low_ranks)
    
        if self.verbose:
            print("Spatial/Temporal ranking preserving (test) for {}-th epoch {}:\t{:.3f}".format(epoch, n_neighbors, tau_l.mean()))
        return float(tau_l.mean())


    def eval_temporal_corr_train(self, n_grain=1):
        """evalute training temporal preserving property"""
        eval = dict()

        for n_epoch in range(self.data_provider.s+self.data_provider.p*n_grain, self.data_provider.e+1, n_grain*self.data_provider.p):
            prev_data = self.data_provider.train_representation(n_epoch - self.data_provider.p*n_grain)
            prev_embedding = self.trainer.model.encoder(
                torch.from_numpy(prev_data).to(dtype=torch.float32, device=self.trainer.DEVICE)).cpu().detach().numpy()

            curr_data = self.data_provider.train_representation(n_epoch)
            curr_embedding = self.trainer.model.encoder(
                torch.from_numpy(curr_data).to(dtype=torch.float32, device=self.trainer.DEVICE)).cpu().detach().numpy()

            dists = np.linalg.norm(prev_data - curr_data, axis=1)
            embedding_dists = np.linalg.norm(prev_embedding - curr_embedding, axis=1)
            corr = evaluate_temporal_epoch_corr(dists, embedding_dists)
            eval[n_epoch] = corr
            if self.verbose:
                print("{:d}-{:d} (train) corr value: {:.3f}".format(n_epoch - self.data_provider.p*n_grain, n_epoch, corr))

        return eval
        
    def eval_temporal_corr_test(self, n_grain=1):
        """evalute testing temporal preserving property"""
        eval = dict()

        for n_epoch in range(self.data_provider.s+self.data_provider.p*n_grain, self.data_provider.e+1, self.data_provider.p*n_grain):
            prev_data = self.data_provider.test_representation(n_epoch-self.data_provider.p*n_grain)
            prev_embedding = self.trainer.model.encoder(
                torch.from_numpy(prev_data).to(dtype=torch.float32, device=self.trainer.DEVICE)).cpu().detach().numpy()

            curr_data = self.data_provider.test_representation(n_epoch)
            curr_embedding = self.trainer.model.encoder(
                torch.from_numpy(curr_data).to(dtype=torch.float32, device=self.trainer.DEVICE)).cpu().detach().numpy()

            dists = np.linalg.norm(prev_data-curr_data, axis=1)
            embedding_dists = np.linalg.norm(prev_embedding-curr_embedding, axis=1)

            corr = evaluate_temporal_epoch_corr(dists, embedding_dists)
            eval[n_epoch] = corr
            if self.verbose:
                print("{:d}-{:d} (test) corr value: {:.3f}".format(n_epoch - self.data_provider.p*n_grain, n_epoch, corr))

        return eval
    

    #################################### helper functions #############################################

    def save_eval(self, n_neighbors, file_name="evaluation"):
        # save result
        save_dir = os.path.join(self.data_provider.model_path, file_name + ".json")
        if not os.path.exists(save_dir):
            evaluation = dict()
        else:
            f = open(save_dir, "r")
            evaluation = json.load(f)
            f.close()
        n_key = str(n_neighbors)
        evaluation[n_key] = dict()
        evaluation[n_key]["nn_train"] = dict()
        evaluation[n_key]["nn_test"] = dict()
        evaluation[n_key]["b_train"] = dict()
        evaluation[n_key]["b_test"] = dict()
        evaluation["ppr_train"] = dict()
        evaluation["ppr_test"] = dict()
        evaluation["ranking_train"] = dict()
        evaluation["ranking_test"] = dict()

        for epoch in range(self.data_provider.s, self.data_provider.e+1, self.data_provider.p):

            evaluation[n_key]["nn_train"][epoch] = self.eval_nn_train(epoch, n_neighbors)
            evaluation[n_key]["nn_test"][epoch] = self.eval_nn_test(epoch, n_neighbors)

            evaluation[n_key]["b_train"][epoch] = self.eval_b_train(epoch, n_neighbors)
            evaluation[n_key]["b_test"][epoch] = self.eval_b_test(epoch, n_neighbors)

            evaluation["ppr_train"][epoch] = self.eval_inv_train(epoch)
            evaluation["ppr_test"][epoch] = self.eval_inv_test(epoch)
            evaluation["ranking_train"][epoch] = dict()
            evaluation["ranking_test"][epoch] = dict()
            for k in [1,3,5,7]:
                evaluation["ranking_train"][epoch][k] = self.eval_temporal_ranking_corr_train(epoch, k)
                evaluation["ranking_test"][epoch][k] = self.eval_temporal_ranking_corr_test(epoch, k)

        t_train_val, t_train_std = self.eval_temporal_train(n_neighbors)
        evaluation[n_key]["temporal_train_mean"] = t_train_val
        evaluation[n_key]["temporal_train_std"] = t_train_std
        t_test_val, t_test_std = self.eval_temporal_test(n_neighbors)
        evaluation[n_key]["temporal_test_mean"] = t_test_val
        evaluation[n_key]["temporal_test_std"] = t_test_std
        # corr_train = self.eval_temporal_corr_train()
        # evaluation["temporal_corr_train"] = corr_train
        # corr_test = self.eval_temporal_corr_test()
        # evaluation["temporal_corr_test"] = corr_test

        with open(save_dir, "w") as f:
            json.dump(evaluation, f)
        if self.verbose:
            print("Successfully save evaluation with {:d} neighbors...".format(n_neighbors))




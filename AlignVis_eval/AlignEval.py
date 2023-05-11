from abc import ABC, abstractmethod
import os
import json

import numpy as np
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cosine

from singleVis.eval.evaluate import *
from singleVis.backend import *
from singleVis.utils import is_B, js_div
from singleVis.visualizer import visualizer
from AlignVis.utils import *
from sklearn.cluster import KMeans

class EvaluatorAbstractClass(ABC):
    def __init__(self, ref_projector, ref_provider, tar_provider, *args, **kwargs):
        self.tar_provider = tar_provider
        self.ref_projector = ref_projector
        self.ref_provider = ref_provider
    



class Evaluator(EvaluatorAbstractClass):
    def __init__(self, ref_projector, ref_provider, tar_provider, TAR_EPOCH, REF_EPOCH,verbose=1):
        self.tar_provider = tar_provider
        self.ref_projector = ref_projector
        self.ref_provider = ref_provider
        self.TAR_EPOCH = TAR_EPOCH
        self.REF_EPOCH = REF_EPOCH
        self.verbose =  verbose
################################# nearest neibour preserving #############################################################
    ################################# ref in ref #########################################################
    def eval_nn_train_ref_in_ref(self, n_neighbors):
        train_data = self.ref_provider.train_representation(self.REF_EPOCH)
        embedding = self.ref_projector.batch_project(self.REF_EPOCH, train_data)
        val = evaluate_proj_nn_perseverance_knn(train_data, embedding, n_neighbors=n_neighbors, metric="euclidean")
        if self.verbose:
            print("#train# nn preserving ref in ref: {:.2f}/{:d} in epoch {:d}".format(val, n_neighbors, self.REF_EPOCH))
        return val
    def eval_nn_test_ref_in_ref(self, n_neighbors):
        test_data = self.ref_provider.test_representation(self.REF_EPOCH)
        embedding = self.ref_projector.batch_project(self.REF_EPOCH, test_data)
        val = evaluate_proj_nn_perseverance_knn(test_data, embedding, n_neighbors=n_neighbors, metric="euclidean")
        if self.verbose:
            print("#test# nn preserving ref in ref: {:.2f}/{:d} in epoch {:d}".format(val, n_neighbors, self.REF_EPOCH))
        return val
    ################################# tar in ref #########################################################
    def eval_nn_train_tar_in_ref(self, n_neighbors):
        train_data = self.tar_provider.train_representation(self.REF_EPOCH)
        embedding = self.ref_projector.batch_project(self.REF_EPOCH, train_data)
        val = evaluate_proj_nn_perseverance_knn(train_data, embedding, n_neighbors=n_neighbors, metric="euclidean")
        if self.verbose:
            print("#train# nn preserving tar in ref: {:.2f}/{:d} in epoch {:d}".format(val, n_neighbors, self.REF_EPOCH))
        return val
    def eval_nn_train_tar_in_ref(self, n_neighbors):
        test_data = self.tar_provider.test_representation(self.REF_EPOCH)
        embedding = self.ref_projector.batch_project(self.REF_EPOCH, test_data)
        val = evaluate_proj_nn_perseverance_knn(test_data, embedding, n_neighbors=n_neighbors, metric="euclidean")
        if self.verbose:
            print("#test# nn preserving tar in ref: {:.2f}/{:d} in epoch {:d}".format(val, n_neighbors, self.REF_EPOCH))
        return val
    ################################# linear #########################################################
    def eval_nn_train_linear(self, n_neighbors, R):
        """
            R: numpy: 512 * 512:
                linear transformation matrix: map target to ref's space
        """
        init_data = self.tar_provider.train_representation(self.TAR_EPOCH)
        train_data = np.dot(init_data,R)
        embedding = self.projector.batch_project(self.REF_EPOCH, train_data)
        val = evaluate_proj_nn_perseverance_knn(init_data, embedding, n_neighbors=n_neighbors, metric="euclidean")
        if self.verbose:
            print("#train# nn preserving for linear: {:.2f}/{:d} in epoch {:d}".format(val, n_neighbors, self.REF_EPOCH))
        return val
    def eval_nn_test_linear(self, n_neighbors, R):
        """
            R: numpy: 512 * 512:
                linear transformation matrix: map target to ref's space
        """
        init_data = self.tar_provider.test_representation(self.TAR_EPOCH)
        test_data = np.dot(init_data,R)
        embedding = self.projector.batch_project(self.REF_EPOCH, test_data)
        val = evaluate_proj_nn_perseverance_knn(init_data, embedding, n_neighbors=n_neighbors, metric="euclidean")
        if self.verbose:
            print("#test# nn preserving for linear: {:.2f}/{:d} in epoch {:d}".format(val, n_neighbors, self.REF_EPOCH))
        return val
    ################################# autoencoder #########################################################
    def eval_nn_train_autoencoder(self, n_neighbors, autoencoder):
        """
            autoencoder: Autoencoder
                autoencoder trained on ref and tar
                autoencoder.encoder: map target to reference's space
                autoencoder.decoder: map reference to target's space
        """
        init_data = self.tar_provider.train_representation(self.TAR_EPOCH)
        train_data = autoencoder.encoder(torch.Tensor(init_data)).detach().numpy()
        embedding = self.ref_projector.batch_project(self.REF_EPOCH, train_data)
        val = evaluate_proj_nn_perseverance_knn(init_data, embedding, n_neighbors=n_neighbors, metric="euclidean")
        if self.verbose:
            print("#train# nn preserving for autoencoder: {:.2f}/{:d} in epoch {:d}".format(val, n_neighbors, self.REF_EPOCH))
        return val
    def eval_nn_test_autoencoder(self, n_neighbors, autoencoder):
        """
            autoencoder: Autoencoder
                autoencoder trained on ref and tar
                autoencoder.encoder: map target to reference's space
                autoencoder.decoder: map reference to target's space
        """
        init_data = self.tar_provider.test_representation(self.TAR_EPOCH)
        test_data = autoencoder.encoder(torch.Tensor(init_data)).detach().numpy()
        embedding = self.ref_projector.batch_project(self.REF_EPOCH, test_data)
        val = evaluate_proj_nn_perseverance_knn(init_data, embedding, n_neighbors=n_neighbors, metric="euclidean")
        if self.verbose:
            print("#test# nn preserving for autoencoder: {:.2f}/{:d} in epoch {:d}".format(val, n_neighbors, self.REF_EPOCH))
        return val
    
################################# boundary neibour preserving #############################################################
    def eval_boundary_nn_ref_in_ref(self, ref_features, n_neighbors=15):
        train_data = self.ref_provider.train_representation(self.REF_EPOCH)
        border_centers = ref_features
        low_center = self.ref_projector.batch_project(self.REF_EPOCH, border_centers)
        low_train = self.ref_projector.batch_project(self.REF_EPOCH, train_data)
    
        val = evaluate_proj_boundary_perseverance_knn(train_data,
                                                      low_train,
                                                      border_centers,
                                                      low_center,
                                                      n_neighbors=n_neighbors)
        print("#train# boundary preserving: {:.2f}/{:d} in epoch {:d}".format(val, n_neighbors, self.REF_EPOCH))

        return val
    
    def eval_boundary_nn_tar_in_ref(self, tar_features, n_neighbors=15):
        train_data = self.tar_provider.train_representation(self.TAR_EPOCH)
        border_centers = tar_features
        low_center = self.ref_projector.batch_project(self.REF_EPOCH, border_centers)
        low_train = self.ref_projector.batch_project(self.REF_EPOCH, train_data)
    
        val = evaluate_proj_boundary_perseverance_knn(train_data,
                                                      low_train,
                                                      border_centers,
                                                      low_center,
                                                      n_neighbors=n_neighbors)
        print("#train# boundary preserving: {:.2f}/{:d} in epoch {:d}".format(val, n_neighbors, self.REF_EPOCH))

        return val
    
    def eval_boundary_nn_linear(self, tar_features, R, n_neighbors=15):
        init_data = self.tar_provider.train_representation(self.TAR_EPOCH)
      
        train_data = np.dot(init_data, R)

        border_centers = np.dot(tar_features, R)

        low_center = self.ref_projector.batch_project(self.REF_EPOCH, border_centers)
        low_train = self.ref_projector.batch_project(self.REF_EPOCH, train_data)
    
        val = evaluate_proj_boundary_perseverance_knn(init_data,
                                                      low_train,
                                                      tar_features,
                                                      low_center,
                                                      n_neighbors=n_neighbors)
        print("#train# boundary preserving for linear: {:.2f}/{:d} in epoch {:d}".format(val, n_neighbors, self.TAR_EPOCH))

        return val
    
    def eval_boundary_nn_autoencoder(self, tar_features, autoencoder, n_neighbors=15):
        init_data = self.tar_provider.train_representation(self.TAR_EPOCH)
        encoded_tar = autoencoder.encoder(torch.Tensor(init_data))
        train_data = encoded_tar.detach().numpy()

        border_centers = autoencoder.encoder(torch.Tensor(tar_features)).detach().numpy()

        low_center = self.ref_projector.batch_project(self.REF_EPOCH, border_centers)
        low_train = self.ref_projector.batch_project(self.REF_EPOCH, train_data)
    
        val = evaluate_proj_boundary_perseverance_knn(init_data,
                                                      low_train,
                                                      tar_features,
                                                      low_center,
                                                      n_neighbors=n_neighbors)
        print("#train# boundary preserving: {:.2f}/{:d} in epoch {:d}".format(val, n_neighbors, self.TAR_EPOCH))

        return val

    def eval_boundary_nn_autoencoder_test(self, tar_features, autoencoder, n_neighbors=15):
        init_data = self.tar_provider.test_representation(self.TAR_EPOCH)
        encoded_tar = autoencoder.encoder(torch.Tensor(init_data))
        test_data = encoded_tar.detach().numpy()

        border_centers = autoencoder.encoder(torch.Tensor(tar_features)).detach().numpy()

        low_center = self.ref_projector.batch_project(self.REF_EPOCH, border_centers)
        low_train = self.ref_projector.batch_project(self.REF_EPOCH, test_data)
    
        val = evaluate_proj_boundary_perseverance_knn(init_data,
                                                      low_train,
                                                      tar_features,
                                                      low_center,
                                                      n_neighbors=n_neighbors)
        print("#test# boundary preserving: {:.2f}/{:d} in epoch {:d}".format(val, n_neighbors, self.TAR_EPOCH))

        return val
        
################################# prediction preserving #############################################################


    def eval_ppr_ref_in_ref(self):

        pred = self.ref_provider.get_pred(self.REF_EPOCH, self.ref_provider.train_representation(self.REF_EPOCH)).argmax(axis=1)
        
     
        train_data = self.ref_provider.train_representation(self.REF_EPOCH)


        embedding = self.ref_projector.batch_project(self.REF_EPOCH, train_data)
        inv_data = self.ref_projector.batch_inverse(self.REF_EPOCH, embedding)
       

        new_pred = self.ref_provider.get_pred(self.REF_EPOCH, inv_data).argmax(axis=1)

        val = evaluate_inv_accu(pred, new_pred)
       
        print("#train# ref in ref PPR: {:.2f} in epoch {:d}".format(val, self.REF_EPOCH))
        return val

    def eval_ppr_autoencoder(self , autoencoder):

        pred = self.tar_provider.get_pred(self.TAR_EPOCH, self.tar_provider.train_representation(self.TAR_EPOCH)).argmax(axis=1)
        
        init_data = self.tar_provider.train_representation(self.TAR_EPOCH)
        encoded_tar = autoencoder.encoder(torch.Tensor(init_data))
        train_data = encoded_tar.detach().numpy()


        embedding = self.ref_projector.batch_project(self.REF_EPOCH, train_data)
        inv_data = self.ref_projector.batch_inverse(self.REF_EPOCH, embedding)
        new_inv = autoencoder.decoder(torch.Tensor(inv_data))
        new_inv = new_inv.detach().numpy()

        new_pred = self.tar_provider.get_pred(self.REF_EPOCH, new_inv).argmax(axis=1)

        val = evaluate_inv_accu(pred, new_pred)
       
        print("#train# autoencoder PPR: {:.2f} in epoch {:d}".format(val, self.REF_EPOCH))
        return val
    

    ########################################## prediction preserving ##############################################################

    def eval_prediction_preserving(self, autoencoder):
        """
            We assess prediction preservation from two perspectives: first, for sample pairs belonging to the same cluster,
            we strive to maintain the same prediction results when represented in the lower-dimensional space. Second, for 
            samples in different subsets, we ensure that they remain distinct in the lower-dimensional space and are classified 
            into two specific, separate classes.

        """
        tar_pred = self.tar_provider.get_pred(self.TAR_EPOCH, self.tar_provider.train_representation(self.TAR_EPOCH)).argmax(axis=1)
        ref_pred = self.ref_provider.get_pred(self.REF_EPOCH, self.ref_provider.train_representation(self.REF_EPOCH)).argmax(axis=1)
        pred_same_list = np.where(tar_pred == ref_pred)[0]
        pred_diff_list = np.where(tar_pred != ref_pred)[0]


        ####################### get target prediction ###################################
        ref_train_data = self.ref_provider.train_representation(self.REF_EPOCH)
        ref_embedding = self.ref_projector.batch_project(self.REF_EPOCH, ref_train_data)
        ref_inv_data = self.ref_projector.batch_inverse(self.REF_EPOCH, ref_embedding)
        ref_new_pred = self.ref_provider.get_pred(self.REF_EPOCH, ref_inv_data).argmax(axis=1)

        
        ####################### get target prediction ###################################
        init_tar_data = self.tar_provider.train_representation(self.TAR_EPOCH)
        encoded_tar = autoencoder.encoder(torch.Tensor(init_tar_data))
        encoded_tar = encoded_tar.detach().numpy()
        embedding = self.ref_projector.batch_project(self.REF_EPOCH, encoded_tar)
        inv_tar_data = self.ref_projector.batch_inverse(self.REF_EPOCH, embedding)
        new_inv_tar_data = autoencoder.decoder(torch.Tensor(inv_tar_data))
        new_inv_tar_data = new_inv_tar_data.detach().numpy()
        new_tar_pred = self.tar_provider.get_pred(self.REF_EPOCH, new_inv_tar_data).argmax(axis=1)


        pred_preserving = np.where(ref_new_pred==ref_pred)[0]
        tar_pred_preserving = np.where(new_tar_pred==tar_pred)[0]
        val_same = np.intersect1d(pred_same_list ,np.intersect1d(pred_preserving, tar_pred_preserving))
        val_diff = np.intersect1d(pred_diff_list ,tar_pred_preserving)
        
        print("refere  predction preserving:{}/{} {:.2f}".format(len(pred_preserving),len(ref_pred),len(pred_preserving)/len(ref_pred) ))
        print("target  predction preserving:{}/{} {:.2f}".format(len(tar_pred_preserving),len(tar_pred),len(tar_pred_preserving)/len(tar_pred) ))
        print("ref&tar pred_same preserving:{}/{} {:.2f}".format(len(val_same), len(pred_same_list), len(val_same)/len(pred_same_list) ))
        print("ref&tar pred_diff preserving:{}/{} {:.2f}".format(len(val_diff), len(pred_diff_list), len(val_diff)/len(pred_diff_list) ))

        return val_same, val_diff
    
    def eval_move_direction_preserving(self,autoencoder,mes_val_for_diff,mes_val_for_same ):
        """
            Cluster all data and compute the centroid for each cluster. Evaluate the distance between each sample 
            pair and each cluster's centroid.We assess the similarity using the mean absolute error (MAE). If the 
            MAE > m, we consider the sample pair to be different. We then evaluate their differing direction (closer 
            to a distinct cluster), aiming to faithfully represent this direction in a lower-dimensional space.
        """
        ####### get top2 high top3 low
        tar_h, ref_h, long_, short_ = self.get_high_dimension_top(autoencoder, mes_val_for_diff,mes_val_for_same)
        tar_l, ref_l = self.get_low_dimension_top(autoencoder,n_clusters=10)
        
        ref_preserving_num = 0
        tar_distance_preserving_num = 0
        """
            caluculate target reference high dimensional and low dimensional top3 class. evaluate their distance preserving
        """
        for i in range(len(ref_l)):
            if np.any(np.isin(ref_l[i], ref_h[i])):
                ref_preserving_num = ref_preserving_num + 1
            if np.any(np.isin(tar_l[i], tar_h[i])):
                tar_distance_preserving_num = tar_distance_preserving_num + 1
        print("all reference distance preserving {}/{}".format(ref_preserving_num, len(ref_l)))
        print("all target distance preserving {}/{}".format(tar_distance_preserving_num, len(ref_l)))
              
        """
            The subset of small displacement actions in the low-dimensional space preserves the same relative positions.
        """
        short_distance_preserving_num = 0
        for i in range(len(ref_l[short_])):
            if np.any(np.isin(tar_l[i], tar_h[i])) and np.any(np.isin(ref_l[i], tar_l[i])):
                short_distance_preserving_num = short_distance_preserving_num + 1
        print("short distance move direction preserving {}/{}".format(short_distance_preserving_num,len(ref_l[short_])))
        """
            Calculate the extended movement distance for subsets in low-dimensional spaces, 
            as they each have distinct movement directions.
        """
        long_distance_preserving_num = 0
        for i in range(len(ref_l[long_])):
            if np.any(np.isin(tar_l[i], tar_h[i])) and np.any(np.isin(ref_l[i], ref_h[i])):
                long_distance_preserving_num = long_distance_preserving_num +1
        print("long distance move direction preserving {}/{}".format(long_distance_preserving_num, len(ref_l[long_])))


    def get_high_dimension_top(self, autoencoder, mes_val_for_diff,mes_val_for_same):
        """
            Obtain the top 2 nearest clusters for each sample in the high-dimensional space.
        """
        tar_pred_softmax = self.tar_provider.get_pred(self.TAR_EPOCH, self.tar_provider.train_representation(self.TAR_EPOCH))
        ref_pred_softmax = self.ref_provider.get_pred(self.REF_EPOCH, self.ref_provider.train_representation(self.REF_EPOCH))
        #### find long distance move set and short distance move by calculate softmax res mean absolute error and top 2 classes
        longDistanceMove = []
        shortDistanceMove = []
        tar_top_classes = np.argsort(tar_pred_softmax, axis=1)[:, ::-1][:, :2]
        ref_top_classes = np.argsort(ref_pred_softmax, axis=1)[:, ::-1][:, :2]
        for i in range(len(tar_pred_softmax)):
            #### get each sample's top 2 classes
         
            mes_val = EMAE(tar_pred_softmax[i], ref_pred_softmax[i])
            if mes_val > mes_val_for_diff and (not np.array_equal(tar_top_classes[i],ref_top_classes[i])) :
                longDistanceMove.append(i)
            elif mes_val <= mes_val_for_same and np.array_equal(tar_top_classes[i],ref_top_classes[i]) :
                shortDistanceMove.append(i)
        print("long distance move subet number is {}, short distance move set number is {}".format(len(longDistanceMove),len(shortDistanceMove)))


        
        
    
        return tar_top_classes, ref_top_classes,longDistanceMove,shortDistanceMove
    
    def get_low_dimension_top(self, autoencoder,n_clusters=10):
        """
            Obtain the top 3 nearest clusters for each sample in the low-dimensional space.
        """

        ref_pred = self.ref_provider.get_pred(self.REF_EPOCH, self.ref_provider.train_representation(self.REF_EPOCH)).argmax(axis=1)
        tar_pred = self.tar_provider.get_pred(self.TAR_EPOCH, self.tar_provider.train_representation(self.TAR_EPOCH)).argmax(axis=1)
        
        ##################### get reference embedding and target embedding #########
        ref_train_data = self.ref_provider.train_representation(self.REF_EPOCH)
        ref_embedding = self.ref_projector.batch_project(self.REF_EPOCH, ref_train_data)

        init_data = self.tar_provider.train_representation(self.TAR_EPOCH)
        encoded_tar = autoencoder.encoder(torch.Tensor(init_data))
        train_data = encoded_tar.detach().numpy()
        tar_embedding = self.ref_projector.batch_project(self.REF_EPOCH, train_data)

        # set the number of clusters
        n_clusters = n_clusters

        # fit KMeans to the sample and get the cluster centers
        ref_kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(ref_embedding)
        ref_centers = ref_kmeans.cluster_centers_

        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(tar_embedding)
        centers = kmeans.cluster_centers_

        # get the indices of the 10 closest samples to each center
        center_indices = []
        ref_centers_indices = []
        for i in range(n_clusters):
            dist_to_center = np.linalg.norm(tar_embedding - centers[i], axis=1)
            closest_indices = np.argsort(dist_to_center)[:10]
            center_indices.append(closest_indices)

            ref_dist_to_center = np.linalg.norm(ref_embedding - ref_centers[i], axis=1)
            ref_closest_indices = np.argsort(ref_dist_to_center)[:10]
            ref_centers_indices.append(ref_closest_indices)

        # print the center sample indices for each cluster
        tar_label_ = []
        ref_label_ = []
        for i in range(n_clusters):
            tar_label_.append(tar_pred[center_indices[i]][0])
            ref_label_.append(ref_pred[ref_centers_indices[i]][0])
            # print("Cluster {}: {},{}".format(i, center_indices[i], tar_pred[center_indices[i]]))
            # print("ref Cluster {}: {},{}".format(i, ref_centers_indices[i], ref_pred[ref_centers_indices[i]]))
        # build the label_dict dictionary
        tar_label_dict = {}
        ref_label_dict = {}
        for i, label in enumerate(tar_label_):
            tar_label_dict[i] = label
        for i, label in enumerate(ref_label_):
            ref_label_dict[i] = label

        # compute the distances between each sample and each cluster center
        distances = kmeans.transform(tar_embedding)
        tar_top_clusters = np.argsort(distances, axis=1)[:, :3]
        tar_top_classess = np.vectorize(tar_label_dict.get)(tar_top_clusters)
        #### ref
        ref_distances = ref_kmeans.transform(ref_embedding)
        ref_top_clusters = np.argsort(ref_distances, axis=1)[:, :3]
        ref_top_classess = np.vectorize(ref_label_dict.get)(ref_top_clusters)

        return tar_top_classess,ref_top_classess
    
    ############## boundary sample preserving ###########################
    def eval_boundary_align_sensitivity(self, autoencoder, tar_b_features, ref_b_features):
        ############## init ###################
        #init target
        tar_b_pred = self.tar_provider.get_pred(self.TAR_EPOCH, tar_b_features)
        tar_b_pred = tar_b_pred + 1e-8
        tar_sort_preds = np.sort(tar_b_pred, axis=1)
        tar_diff = (tar_sort_preds[:, -1] - tar_sort_preds[:, -2]) / (tar_sort_preds[:, -1] - tar_sort_preds[:, 0])
        tar_border = np.zeros(len(tar_diff), dtype=np.uint8) + 0.05
        tar_border[tar_diff < 0.15] = 1

        #init reference
        ref_b_pred = self.ref_provider.get_pred(self.REF_EPOCH, ref_b_features)
        ref_b_pred = ref_b_pred + 1e-8
        ref_sort_preds = np.sort(ref_b_pred, axis=1)
        ref_diff = (ref_sort_preds[:, -1] - ref_sort_preds[:, -2]) / (ref_sort_preds[:, -1] - ref_sort_preds[:, 0])
        ref_border = np.zeros(len(ref_diff), dtype=np.uint8) + 0.05
        ref_border[ref_diff < 0.15] = 1
        
        ##### get all boundary list
        all_boundary_list = []
        for i in range(len(ref_border)):
            if ref_border[i] == 1 and tar_border[i] == 1:
                all_boundary_list.append(i)
        
        #### get ref low dimensional border
        ref_b_embedding = self.ref_projector.batch_project(self.REF_EPOCH, ref_b_features)
        ref_b_inv = self.ref_projector.batch_inverse(self.REF_EPOCH,ref_b_embedding)
        ref_b_pred_l = self.ref_provider.get_pred(self.REF_EPOCH, ref_b_inv)
        ref_b_pred_l = ref_b_pred_l  + 1e-8
        ref_sort_preds_l  = np.sort(ref_b_pred_l , axis=1)
        ref_diff_l  = (ref_sort_preds_l[:, -1] - ref_sort_preds_l[:, -2]) / (ref_sort_preds_l[:, -1] - ref_sort_preds_l[:, 0])
        ref_border_l = np.zeros(len(ref_diff_l), dtype=np.uint8) + 0.05
        ref_border_l[ref_diff_l < 0.15] = 1
        
        #### get target low dimensional border
        tar_b_embedding = self.ref_projector.batch_project(self.REF_EPOCH, autoencoder.encoder(torch.Tensor(ref_b_features)).detach().numpy())
        tar_b_inv = self.ref_projector.batch_inverse(self.REF_EPOCH,tar_b_embedding)
        tar_b_inv = autoencoder.decoder(torch.Tensor(tar_b_inv)).detach().numpy()
        tar_b_pred_l = self.tar_provider.get_pred(self.REF_EPOCH, tar_b_inv)
        tar_b_pred_l = tar_b_pred_l  + 1e-8
        tar_sort_preds_l  = np.sort(tar_b_pred_l , axis=1)
        tar_diff_l  = (tar_sort_preds_l[:, -1] - tar_sort_preds_l[:, -2]) / (tar_sort_preds_l[:, -1] - tar_sort_preds_l[:, 0])
        tar_border_l = np.zeros(len(tar_diff_l), dtype=np.uint8) + 0.05
        tar_border_l[tar_diff_l < 0.15] = 1

        all_boundary_list_l = []
        sim = []
        sim_l = []
        target_preserving = []
        ref_preserving = []
        for i in range(len(ref_border)):
            if ref_border_l[i] == 1 and tar_border_l[i] == 1:
                all_boundary_list_l.append(i)
            if ref_border[i] == tar_border[i]:
                sim.append(i)
            if ref_border_l[i] == tar_border_l[i]:
                sim_l.append(i)
            if tar_border_l[i] == tar_border[i]:
                 target_preserving.append(i)
            if ref_border_l[i] == ref_border[i]:
                 ref_preserving.append(i)


        print("boundary sample preserving{}/{}".format(len(all_boundary_list_l),len(all_boundary_list)))
        print("boundary sample preserving{}/{}".format(len(sim_l),len(sim)))
        print("target keep boundary and non boundary{}/{}".format(len(target_preserving),len(tar_b_features)))
        print("target keep boundary and non boundary{}/{}".format(len(ref_preserving),len(ref_b_features)))
        

        print("boundary sample preserving{}/{}".format(len(all_boundary_list_l),len(all_boundary_list)))
        






    



  
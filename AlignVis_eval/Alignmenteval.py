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
        


  
from abc import ABC, abstractmethod
from pynndescent import NNDescent
from sklearn.utils import check_random_state
from umap.umap_ import fuzzy_simplicial_set
import numpy as np

"""
transform reference base on target
"""
class ReferenceTransformerAbstractClass(ABC):
    @abstractmethod
    def __init__(self, ref_data_provider, ref_projector, tar_data_provider,* args, **kawargs):
        pass

class ReferenceTransformer(ReferenceTransformerAbstractClass):
    def __init__(self, ref_data_provider, ref_projector, tar_data_provider):
        self.ref_data_provider = ref_data_provider
        self.ref_projector = ref_projector
        self.tar_data_provider = tar_data_provider
    

    def _construct_fuzzy_complex(self, n_neighbors, train_data):
        """
        construct a vietoris-rips complex
        """
        # number of trees in random projection forest
        n_trees = min(64, 5 + int(round((train_data.shape[0]) ** 0.5 / 20.0)))
        # max number of nearest neighbor iters to perform
        n_iters = max(5, int(round(np.log2(train_data.shape[0]))))
        # distance metric
        metric = "euclidean"
        # get nearest neighbors
        nnd = NNDescent(
            train_data,
            n_neighbors=n_neighbors,
            metric=metric,
            n_trees=n_trees,
            n_iters=n_iters,
            max_candidates=60,
            verbose=True
        )
        knn_indices, knn_dists = nnd.neighbor_graph
        random_state = check_random_state(None)
        complex, sigmas, rhos = fuzzy_simplicial_set(
            X=train_data,
            n_neighbors=n_neighbors,
            metric=metric,
            random_state=random_state,
            knn_indices=knn_indices,
            knn_dists=knn_dists,
        )
        return complex, sigmas, rhos, knn_indices

    def get_pred_diff(self, epoch):
        ref_pred_res = self.ref_data_provider.get_pred(epoch,  self.ref_data_provider.train_representation(epoch)).argmax(axis=1)
        tar_pred_res = self.tar_data_provider.get_pred(epoch,  self.tar_data_provider.train_representation(epoch)).argmax(axis=1)
        diff_list = []
        for i in range(len(ref_pred_res)):
            if ref_pred_res[i] != tar_pred_res[i]:
                diff_list.append(i)
        return diff_list,ref_pred_res,tar_pred_res
    
    def get_new_ref_by_tar_knn(self, epoch , n_neighbors):
        ref_train_data = self.ref_data_provider.train_representation(epoch)
        diff_list,ref_pred_res,tar_pred_res = self.get_pred_diff(epoch)

        tar_complex, tar_sigmas, tar_rhos, tar_knn_indices = self._construct_fuzzy_complex(n_neighbors, self.ref_data_provider.train_representation(epoch))
        # ref_complex, ref_sigmas, ref_rhos, ref_knn_indices = _construct_fuzzy_complex(n_neighbors, self.tar_data_provider.train_representation(epoch))
        for i in range(len(diff_list)):
            index = diff_list[i]
            nearest_index_in_tar = tar_knn_indices[index][1]
            if ref_pred_res[nearest_index_in_tar] == tar_pred_res[nearest_index_in_tar]:
                ref_train_data[index] = self.ref_data_provider.train_representation(epoch)[nearest_index_in_tar]
        
        return ref_train_data

    
    


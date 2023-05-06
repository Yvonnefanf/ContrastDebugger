from abc import ABC, abstractmethod
from AlignVis.DataInit import DataInit
import numpy as np
import os
import json
from singleVis.utils import *

from sklearn.cluster import KMeans

class AlignSimilarityScalerAbstractClass(ABC):
    @abstractmethod
    def __init__(self, REF_PATH, REF_CONFIG_PATH, TAR_PATH, TAR_CONFIG_PATH,  REF_EPOCH, TAR_EPOCH, DEVICE, * args, **kawargs):
        pass

class AlignSimilarityScaler(AlignSimilarityScalerAbstractClass):
    """
        Clustering representations from multiple models is a crucial challenge in cross-domain similarity analysis, 
        which aims to identify similar patterns across different domains. Given two sets of representations $X_1$ 
        and $X_2$ of the same set of $n$ data points obtained from two different models, the objective is to cluster 
        the representations separately and compare the resulting cluster assignments to evaluate the similarity 
        between the two sets of representations.
    """
    def __init__(self, REF_PATH, REF_CONFIG_PATH, TAR_PATH, TAR_CONFIG_PATH,  REF_EPOCH, TAR_EPOCH, DEVICE) -> None:

        ref_datainit = DataInit(REF_CONFIG_PATH,REF_PATH,REF_EPOCH,DEVICE)
        tar_datainit = DataInit(TAR_CONFIG_PATH,TAR_PATH,TAR_EPOCH,DEVICE)
        ref_model, ref_provider, ref_train_data, ref_prediction, ref_prediction_res, ref_scores = ref_datainit.getData()
        tar_model, tar_provider, tar_train_data, tar_prediction, tar_prediction_res, tar_scores = tar_datainit.getData()

        self.REF_PATH = REF_PATH
        self.REF_EPOCH = REF_EPOCH
        self.ref_model = ref_model
        self.ref_provider = ref_provider
        self.TAR_EPOCH = TAR_EPOCH
        self.tar_provider = tar_provider
        self.tar_model = tar_model
        self.tar_provider = tar_provider
        self.DEVICE = DEVICE

    def representation_normalization(self, ref_representaion, tar_representaion):
        # =============          Reference        ============ #
        # Compute the mean and standard deviation of the representations
        ref_rep_mean = ref_representaion.mean(axis=0)
        ref_rep_std = ref_representaion.std(axis=0)
        # Normalize the representations
        ref_rep_norm = (ref_representaion - ref_rep_mean) / ref_rep_std


        # =============           Target         ============ #
        # Compute the mean and standard deviation of the representations
        tar_rep_mean = tar_representaion.mean(axis=0)
        tar_rep_std = tar_representaion.std(axis=0)
        # Normalize the representations
        tar_rep_norm = (tar_representaion - tar_rep_mean) / tar_rep_std

        return ref_rep_norm, tar_rep_norm
    
    def get_cluster(self, ref_rep_norm, tar_rep_norm, n_clusters=20):
        # Initialize the k-means clustering algorithm
        kmeans = KMeans(n_clusters=n_clusters)
        # Cluster the representations
        ref_cluster_labels = kmeans.fit_predict(ref_rep_norm)
        tar_cluster_labels = kmeans.fit_predict(tar_rep_norm)

        return ref_cluster_labels, tar_cluster_labels
    
    def get_jaccard_similarities(self,ref_rep, tar_rep,n_clusters):
        jaccard_similarities = []

        ref_rep_norm, tar_rep_norm = self.representation_normalization(ref_rep,tar_rep)
        ref_cluster_labels, tar_cluster_labels = self.get_cluster(ref_rep_norm, tar_rep_norm,n_clusters)

        for i in range(ref_cluster_labels.shape[0]):
            ref_cluster = ref_cluster_labels[i]
            tar_cluster = tar_cluster_labels[i]
    
            if ref_cluster != -1 and tar_cluster != -1:
     
                ref_cluster_set = set(np.where(ref_cluster_labels == ref_cluster)[0])
                tar_cluster_set = set(np.where(tar_cluster_labels == tar_cluster)[0])

                # print("ref_cluster",ref_cluster,tar_cluster,"tar_cluster_labels", len(ref_cluster_set),len(tar_cluster_set))
                jaccard_sim = len(ref_cluster_set.intersection(tar_cluster_set)) / len(ref_cluster_set.union(tar_cluster_set))
                jaccard_similarities.append(jaccard_sim)

        return jaccard_similarities





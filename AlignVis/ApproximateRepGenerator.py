import torch
from abc import ABC, abstractmethod
import torch.optim as optim
from AlignVis.losses import KNNOverlapLoss, CKALoss, PredictionLoss, ConfidenceLoss

import sys
sys.path.append("..")
import numpy as np
from scipy.special import softmax
import torch.nn.functional as F
from singleVis.utils import *
import os
from scipy.spatial import procrustes



class ApproximateRefGeneratorAbstractClass(ABC):
    @abstractmethod
    def __init__(self, data_provider) -> None:
        pass


class ApproximateRefGenerator(ApproximateRefGeneratorAbstractClass):
    '''
        generate approximate representations in reference difference dataset
        and use the new generated representations to replace the reference represnetation
        constrain: 1. KNN(Xr') -> KNN (Xt) 2. CKA(Xr', Xt) -> 1 3. CKA(X)

    '''
    def __init__(self, ref_provider, tar_provider, REF_EPOCH, TAR_EPOCH,tar_model, ref_model, DEVICE) -> None:

        """Init parameters for approximate reference generator

        Parameters
        ----------
        ref_data_provider : data.DataProvider
            reference data provider
        tar_data_provider : data.DataProvider
            target data provider 
        REF_EPOCH : int
            reference epoch number
        TAR_EPOCH : int
            target epoch number
        """
        self.ref_provider = ref_provider
        self.tar_provider = tar_provider
        self.REF_EPOCH = REF_EPOCH
        self.TAR_EPOCH = TAR_EPOCH
        self.tar_model = tar_model
        self.ref_model = ref_model
        self.DEVICE = DEVICE
        self.split=-1

        ### reference train data and target train data
        self.ref_train_data = self.ref_provider.train_representation(REF_EPOCH).squeeze()
        self.tar_train_data = self.tar_provider.train_representation(TAR_EPOCH).squeeze()
        ### prediction results of reference train data and target train data
        self.ref_prediction = self.ref_provider.get_pred(REF_EPOCH, self.ref_train_data)
        self.tar_prediction = self.tar_provider.get_pred(TAR_EPOCH, self.tar_train_data)
        ### label results of reference train data and target train data
        self.ref_prediction_argmax = self.ref_prediction.argmax(axis=1)
        self.tar_prediction_argmax = self.tar_prediction.argmax(axis=1)
        ##### confidence score list of reference and target 
        self.ref_conf_score = np.amax(softmax(self.ref_prediction, axis=1), axis=1)
        self.tar_conf_score = np.amax(softmax(self.tar_prediction, axis=1), axis=1)
    
    def get_prediction(self, input_representation, tar_representation):


        tar_representation = tar_representation.detach().numpy()
        input_representation = input_representation.detach().numpy()

        tar = self.tar_provider.get_pred(self.TAR_EPOCH, tar_representation)
        prediction = self.ref_provider.get_pred(self.REF_EPOCH, input_representation)

        return prediction,tar
    
    def getHighDiscrepancyRepresentationSet(self, percentile=95):
        """
            percentile: array_like of float
                Percentile or sequence of percentiles to compute, which must be between 0 and 100 inclusive.

        """
        X = self.ref_train_data
        Y = self.tar_train_data
        # Perform Procrustes analysis to transform X to Y
        mtx1, mtx2, disparity = procrustes(X, Y)
        # Compute the transformation degree for each row of the matrix
        degree = np.sqrt(np.sum((mtx1 - X)**2, axis=1))

        # Find the subset of samples with the highest transformation degree
        threshold = np.percentile(degree, percentile)
        high_discrepancy_representation_set = np.where(degree > threshold)[0]

        print("Number of samples with highest transformation degree:", len(high_discrepancy_representation_set))

        return high_discrepancy_representation_set
    



    def generate_representation_by_cka(self,ref, tar,epoch=1000,K_ALPHA = 10,C_ALPHA=10,P_ALPHA=0.1,alpha_for_pred_ref=1):
        
        x = torch.Tensor(tar)
        z = torch.Tensor(ref)
        y = torch.from_numpy(ref)
        y.requires_grad = True
        # need_update_arr = np.arange(51,len(diff_combine_same))
        mask = torch.zeros_like(y, dtype=torch.bool)
        # mask[need_update_arr, :] = True
        weight_decay = 1e-4
        optimizer = optim.Adam([y], lr=1e-2,weight_decay=weight_decay)
  
        x = torch.Tensor(tar)

        for i in range(epoch):
            
            ##### knn loss
            knn_overlap_loss = KNNOverlapLoss(k=10)
            ##### knn overleap different with target
            knn_loss = knn_overlap_loss(input=y, target=x)
            ##### knn overleap different with reference
            knn_loss_with_ref = knn_overlap_loss(input=y, target=z)

            #### CKA loss
            cka_loss_f = CKALoss(gamma=None, alpha=1e-3)
            cka_loss = cka_loss_f(x,y,z)


            #### Prediction loss
            pred_loss_fn = PredictionLoss(self.tar_model, self.ref_model, self.tar_provider, self.ref_provider,self.TAR_EPOCH, self.REF_EPOCH, self.DEVICE, alpha_for_pred_ref)
            pred_loss = pred_loss_fn(y, ref)

            #### Confidence loss
            confidence_loss_fn = ConfidenceLoss()
            p_x,p_y = self.get_prediction(y,x)
            confidence_loss = confidence_loss_fn(torch.Tensor(p_x),torch.Tensor(p_y))

            optimizer.zero_grad()
            # loss = loss
            combined_loss = K_ALPHA * (knn_loss + knn_loss_with_ref) + C_ALPHA * cka_loss + P_ALPHA * (pred_loss + confidence_loss)
            # loss.backward()
            combined_loss.backward()
            optimizer.step()

            # Print the loss value every 100 iterations
            if i % 9 == 0:
                print(f"Iteration {i}: CKA loss = {cka_loss.item():.10f}")
                print(f"               Prediction loss = {pred_loss.item():.10f}")
                print(f"               KNN loss = {knn_loss.item():.10f}")
                print(f"              KNN loss with ref = {knn_loss_with_ref.item():.10f}")
                print(f"               Confidence different loss = {confidence_loss.item():.10f}")
        
        return y.detach().numpy()

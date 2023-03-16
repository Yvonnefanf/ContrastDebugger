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


    def generate_representation_by_cka(self,indicates,epoch=1000,K_ALPHA = 10,C_ALPHA=10,P_ALPHA=0.1,alpha_for_pred_ref=1):
        tar = self.tar_train_data[indicates]
        ref = self.ref_train_data[indicates]
        x = torch.Tensor(tar)
        z = torch.Tensor(ref)
        y = torch.from_numpy(ref)
        y.requires_grad = True
        # need_update_arr = np.arange(51,len(diff_combine_same))
        mask = torch.zeros_like(y, dtype=torch.bool)
        # mask[need_update_arr, :] = True
        optimizer = optim.Adam([y], lr=1e-2)
  
        x = torch.Tensor(tar)

        for i in range(epoch):
            
            ##### knn loss
            knn_overlap_loss = KNNOverlapLoss(k=10)
            knn_loss = knn_overlap_loss(input=y, target=x)

            #### CKA loss
            cka_loss_f = CKALoss(gamma=None, alpha=1e-3)
            cka_loss = cka_loss_f(x,y,z)


            #### Prediction loss
            pred_loss_fn = PredictionLoss(self.tar_model, self.ref_model, self.tar_provider, self.ref_provider,self.TAR_EPOCH, self.REF_EPOCH, self.DEVICE, alpha_for_pred_ref)
            pred_loss = pred_loss_fn(y, indicates)

            #### Confidence loss
            # confidence_loss_fn = ConfidenceLoss()
            # confidence_loss = confidence_loss_fn(x,y)

            optimizer.zero_grad()
            # loss = loss
            combined_loss = K_ALPHA * knn_loss + C_ALPHA * cka_loss + P_ALPHA * pred_loss
            # loss.backward()
            combined_loss.backward()
            optimizer.step()

            # Print the loss value every 100 iterations
            if i % 9 == 0:
                print(f"Iteration {i}: CKA loss = {cka_loss.item():.10f}")
                print(f"Iteration {i}: prediction loss = {pred_loss.item():.10f}")
                print(f"Iteration {i}: KNN loss = {knn_loss.item():.10f}")
        
        return y.detach().numpy()

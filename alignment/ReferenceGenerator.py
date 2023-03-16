import torch

from abc import ABC, abstractmethod
import torch
import torch.optim as optim
import torch.nn as nn
import os

import sys
sys.path.append("..")
import numpy as np
from scipy.special import softmax

from pynndescent import NNDescent

import math
from singleVis.utils import *
from alignment.CKA_utils import *
from alignment.utils import *
import torch.nn.functional as F

# Define the deep neural network model
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    


class ReferenceGeneratorAbstractClass(ABC):
    @abstractmethod
    def __init__(self, data_provider) -> None:
        pass

class ReferenceGenerator(ReferenceGeneratorAbstractClass):
    '''generate the reference based on CCA
    '''
    def __init__(self, ref_provider, tar_provider, REF_EPOCH, TAR_EPOCH,tar_model, ref_model, DEVICE) -> None:
        """Init parameters for spatial edge constructor

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


    
    def prediction_function(self, Epoch, model_path, model):
        #TODO

        model_location = os.path.join(model_path, "Model", "Epoch_{:d}".format(Epoch), "subject_model.pth")
        model.load_state_dict(torch.load(model_location, map_location=torch.device("cpu")))
        model.to(self.DEVICE)
        model.eval()

        model = torch.nn.Sequential(*(list(model.children())[self.split:]))
        model.to(self.DEVICE)
        model.eval()
      
        return model
    
    def get_pred(self, epoch, data, model_path, model):
        '''
        get the prediction score for data in epoch_id
        :param data: numpy.ndarray
        :param epoch_id:
        :return: pred, numpy.ndarray
        '''
        prediction_func = self.prediction_function(epoch,model_path, model)

        # data = torch.from_numpy(data)
        data = data.to(self.DEVICE)
        pred = batch_run(prediction_func, data)
        return pred.squeeze()
    
    def pred_loss_function(self,epoch, adjusted_input,indicates):
        target_output = self.tar_prediction[indicates]
        output = self.get_pred(epoch, adjusted_input,self.tar_provider.content_path, self.tar_model)
        ref_output = self.get_pred(epoch, adjusted_input,self.ref_provider.content_path, self.ref_model)
        loss_output = F.mse_loss(torch.tensor(output), torch.tensor(target_output))
        ref_loss_output = F.mse_loss(torch.tensor(ref_output), torch.tensor(target_output))
        loss_Rep = F.mse_loss(adjusted_input, torch.tensor(self.tar_provider.train_representation(epoch)[indicates]))
        loss = loss_output + loss_Rep + ref_loss_output
        return loss
    
    
    def subsetClassify(self, mes_val_for_diff, mes_val_for_same, conf_val_for_diff=0.3,conf_val_for_same=0.2 ):
        high_distance_indicates = []
        low_distance_indicates = []
        absolute_alignment_indicates = []
        predict_label_diff_indicates = []
        predict_confidence_diff_indicates = []
        ####### definate high_distance_indicates and low_distance_indicates base on EMAE 
        for i in range(len(self.ref_prediction)):
            mes_val = EMAE(self.ref_prediction[i], self.tar_prediction[i])
            if mes_val > mes_val_for_diff:
                high_distance_indicates.append(i)
            elif mes_val < mes_val_for_same:
                low_distance_indicates.append(i)
        for i in range(len(self.ref_prediction)):
            if self.tar_prediction_argmax[i] == self.ref_prediction_argmax[i]:
                if math.fabs(self.ref_conf_score[i] - self.tar_conf_score[i]) < conf_val_for_same and  (i in low_distance_indicates):
                    absolute_alignment_indicates.append(i)
                elif math.fabs(self.ref_conf_score[i] - self.tar_conf_score[i]) > conf_val_for_diff:
                    predict_confidence_diff_indicates.append(i)
            else:
                predict_label_diff_indicates.append(i)

        print('absolute alignment indicates number:',len(absolute_alignment_indicates),'label diff indicates number:',len(predict_label_diff_indicates),'confidence diff indicates number:',len(predict_confidence_diff_indicates),"high distance number:",len(high_distance_indicates))
        return absolute_alignment_indicates,predict_label_diff_indicates,predict_confidence_diff_indicates,high_distance_indicates
        
    
    def kernel_HSIC_cka_loss(self, X, Y, gamma=None):
        K_xx = kernel_HSIC(X, X, gamma)
        K_yy = kernel_HSIC(Y, Y, gamma)
        # K_xy = rbf_kernel(X, Y, 1e-2) 
        K_xy = kernel_HSIC(X, Y, gamma)   
        cka_loss = 1 - torch.mean(K_xy) / torch.sqrt(K_xx * K_yy)
        return cka_loss
    
    def kernel_HSIC_cka_loss_consider_init(self, X, Y, Z, gamma=None, alpha1 = 100000, alpha2=0.01):
        K_xx = kernel_HSIC(X, X, gamma)
        K_yy = kernel_HSIC(Y, Y, gamma)
        K_zz = kernel_HSIC(Z, Z, gamma)
        K_xy = kernel_HSIC(X, Y, gamma)
        K_xz = kernel_HSIC(X, Z, gamma)
        K_yz = kernel_HSIC(Y, Z, gamma)
        cka_loss1 = 1 - torch.mean(K_xy) / torch.sqrt(K_xx * K_yy)
        cka_loss2 = 1 - torch.mean(K_yz) / torch.sqrt(K_yy * K_zz)
        loss = alpha1 * cka_loss1 + alpha2 * cka_loss2
        return loss

    
    
    
    def generate_representation_by_cka(self,epoch,indicates):
        # absolute_alignment_indicates,predict_label_diff_indicates,predict_confidence_Diff_indicates,high_distance_indicates = self.subsetClassify(mes_val_for_diff, mes_val_for_same)
        # diff_combine_same = np.concatenate((absolute_alignment_indicates, predict_label_diff_indicates), axis=0)
        # indicates = predict_label_diff_indicates
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
        # params_to_optimize = torch.nn.Parameter(y[mask])
        # optimizer = optim.Adagrad([params_to_optimize], lr=0.01)
        x = torch.Tensor(tar)

        for i in range(epoch):
            # loss = self.kernel_HSIC_cka_loss(x,y)
            loss = self.kernel_HSIC_cka_loss_consider_init(z,y,x)

            loss2 = self.pred_loss_function(self.TAR_EPOCH, y, indicates)
     

            optimizer.zero_grad()
            # loss = loss
            combined_loss = 100 * loss + loss2
            # loss.backward()
            combined_loss.backward()
            optimizer.step()

            # Print the loss value every 100 iterations
            if i % 9 == 0:
                print(f"Iteration {i}: CKA loss = {loss.item():.10f}")
                print(f"Iteration {i}: prediction loss = {loss2.item():.10f}")
        
        return y.detach().numpy()
    

    
    def neibour_graph_build(self, train_data, n_neighbors):
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

        return knn_indices, knn_dists

    
        

    



        

        





    


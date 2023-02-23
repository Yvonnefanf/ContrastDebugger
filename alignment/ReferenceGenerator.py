import torch

from abc import ABC, abstractmethod
import torch
import torch.optim as optim
import torch.nn as nn

import sys
sys.path.append("..")
import numpy as np
from scipy.special import softmax
from CKA_utils.CKA import CKA, CudaCKA
import math

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
    

def EMAE(Y, y, a=1.5):
    """
    param：
        Y: 原始序列（假定波动较大）
        y: 拟合序列（假定波动较小）
        a: 指数的自变量，≥1，该值越大，则两序列间的残差（特别是残差的离群值）对EMAE返回值影响的强化作用越明显；
        当a=1时，EMAE化简为MAE。
    return：
        指数MAE值，该值的大小与两条序列间平均偏差程度成正比，该值越大，平均偏差程度越大；
        且两序列间的残差（特别是残差的离群值）对EMAE的影响比MAE大。
    """

    Y, y = np.array(Y), np.array(y)
    Y[Y < 0] = 0  # 使指数的底数≥1，则所有指数均为递增函数
    y[y < 0] = 0
    emae = sum(abs((Y+1)**a - (y+1)**a)) / len(Y)

    return emae


class ReferenceGeneratorAbstractClass(ABC):
    @abstractmethod
    def __init__(self, data_provider) -> None:
        pass

class ReferenceGenerator(ReferenceGeneratorAbstractClass):
    '''generate the reference based on CCA
    '''
    def __init__(self, ref_provider, tar_provider, REF_EPOCH, TAR_EPOCH) -> None:
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



    def subsetClassify(self, mes_val_for_diff, mes_val_for_same):
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
                if math.fabs(self.ref_conf_score[i] - self.tar_conf_score[i]) < 0.1 and  (i in low_distance_indicates):
                    absolute_alignment_indicates.append(i)
                elif math.fabs(self.ref_conf_score[i] - self.tar_conf_score[i]) > 0.3:
                    predict_confidence_diff_indicates.append(i)
            else:
                predict_label_diff_indicates.append(i)

        print('absolute alignment indicates number:',len(absolute_alignment_indicates),'label diff indicates number:',len(predict_label_diff_indicates),'confidence diff indicates number:',len(predict_confidence_diff_indicates))
        return absolute_alignment_indicates,predict_label_diff_indicates,predict_confidence_diff_indicates
        

    def CKAcaculator(self, X, Y):
        np_cka = CKA()
        value = np_cka.kernel_CKA(X,Y)
        return value
    
    # Define the CCA loss function
    def cca_loss(self, x, y):
        # Normalize the input data
        x_normalized = torch.nn.functional.normalize(x, dim=0)
        y_normalized = torch.nn.functional.normalize(y, dim=0)
    
        # Compute the covariance matrix of the normalized input data
        cov = torch.matmul(x_normalized.T, y_normalized)
    
        # Compute the singular value decomposition of the covariance matrix
        u, s, v = torch.svd(cov)
    
        # Compute the canonical correlation coefficients
        cca_coef = s[:min(x.shape[1], y.shape[1])]
    
        # Normalize the CCA coefficients
        cca_coef_norm = cca_coef / torch.max(cca_coef)
    
        # Compute the loss function
        loss = 1 - cca_coef_norm.sum()
    
        return loss
    
    # Define the CKA loss function
    def cka_loss(self, x, y):
        # Compute the Gram matrix of the input data
        x_gram = torch.matmul(x, x.t())
        y_gram = torch.matmul(y, y.t())
    
        # Compute the normalization factors for the Gram matrices
        x_norm = torch.norm(x_gram, p='fro')
        y_norm = torch.norm(y_gram, p='fro')
    
        # Compute the centered Gram matrix of the input data
        x_centered = x_gram - torch.mean(x_gram, dim=1, keepdim=True) - torch.mean(x_gram, dim=0, keepdim=True) + torch.mean(x_gram)
        y_centered = y_gram - torch.mean(y_gram, dim=1, keepdim=True) - torch.mean(y_gram, dim=0, keepdim=True) + torch.mean(y_gram)
    
        # Compute the normalization factors for the centered Gram matrices
        x_centered_norm = torch.norm(x_centered, p='fro')
        y_centered_norm = torch.norm(y_centered, p='fro')
    
        # Compute the centered kernel alignment between the input data
        cka = torch.trace(torch.matmul(x_centered, y_centered)) / (x_centered_norm * y_centered_norm)
    
        # Compute the loss function
        loss = 1 - cka
    
        return loss
 
    def rbf(self, X, sigma=None):
        # X = torch.tensor(X)
        GX = torch.matmul(X, X.T)
        KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
        if sigma is None:
            mdist = torch.median(KX[KX != 0])
            sigma = torch.sqrt(mdist)
        KX *= - 0.5 / (sigma * sigma)
        KX = torch.exp(KX)
        return KX
    
    def kernel_HSIC(self, X, Y, gamma):
        n1 = X.shape[0]
        n2 = Y.shape[0]
        H1 = torch.eye(n1) - torch.ones(n1, n1) / n1
        H2 = torch.eye(n2) - torch.ones(n2, n2) / n2
        K1 = torch.matmul(torch.matmul(H1, self.rbf(X, gamma)), H1)
        K2 = torch.matmul(torch.matmul(H2, self.rbf(Y, gamma)), H2)
        hsic = torch.trace(torch.matmul(K1, K2))
        return hsic
    
    def kernel_HSIC_cka_loss(self, X, Y, gamma=None):
        K_xx = self.kernel_HSIC(X, X, gamma)
        K_yy = self.kernel_HSIC(Y, Y, gamma)
        # K_xy = rbf_kernel(X, Y, 1e-2) 
        K_xy = self.kernel_HSIC(X, Y, gamma)   
        cka_loss = 1 - torch.mean(K_xy) / torch.sqrt(K_xx * K_yy)
        return cka_loss
    
    
    def generate_representation_by_cka(self,mes_val_for_diff, mes_val_for_same,epoch):
        absolute_alignment_indicates,predict_label_diff_indicates,predict_confidence_Diff_indicates = self.subsetClassify(mes_val_for_diff, mes_val_for_same)

        diff_combine_same = np.concatenate((absolute_alignment_indicates, predict_label_diff_indicates), axis=0)
        ref_diff = self.ref_train_data[predict_label_diff_indicates]
        ref_same = self.ref_train_data[absolute_alignment_indicates]
        tar = self.tar_train_data[diff_combine_same]
        ref = self.ref_train_data[diff_combine_same]
        x = torch.Tensor(tar)
        y1 = torch.randn(ref_diff.shape[0], ref_diff.shape[1], requires_grad = True)
        
       
        y = torch.cat([torch.Tensor(ref_same),y1], 0)
        

        x = torch.Tensor(self.tar_train_data[predict_label_diff_indicates])
        y = torch.randn(ref_diff.shape[0], ref_diff.shape[1], requires_grad = True)
        optimizer = optim.Adam([y], lr=1e-2)

        # print(y.shape(),tar.shape(),result.shape())

        for i in range(epoch):
            # y_features = model(y)
            loss = self.kernel_HSIC_cka_loss(x,y)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print the loss value every 100 iterations
            if i % 9 == 0:
                print(f"Iteration {i}: CKA loss = {loss.item():.4f}")
        # cka_value = self.rbf_kernel_cka(x, y_features, sigma=1.0)
        # print(f"Final RBF kernel CKA value: {cka_value.item()}")
        
        return y.detach().numpy()
    



        

        





    


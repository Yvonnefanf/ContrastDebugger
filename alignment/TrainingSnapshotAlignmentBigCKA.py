from abc import ABC, abstractmethod
import os
import sys
sys.path.append("..")
import torch
import numpy as np
from CKA_utils.CKA import CKA, CudaCKA, CCA_val

from alignment.CKA_utils import *
from torch.autograd import Variable


class TrainingSnapshotAlignmentAbstractClass(ABC):
    @abstractmethod
    def __init__(self, ref_data_provider, tar_data_provider, ref_EPOCH, tar_EPOCH, ALPHA, * args, **kawargs):
        pass

class TrainingSnapshotAlignment(TrainingSnapshotAlignmentAbstractClass):
    def __init__(self, ref_data_provider, tar_data_provider,ref_EPOCH, tar_EPOCH,ALPHA):
        self.ref_data_provider = ref_data_provider
        self.tar_data_provider = tar_data_provider
        self.ref_EPOCH = ref_EPOCH
        self.tar_EPOCH = tar_EPOCH
        self.ALPHA = ALPHA
    
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


    # Most of the time we iterate for a fixed number of training steps rather than iterating until the loss falls below a threshold.

    # 1.Calculate gradient g of the loss with respect to the matrix R. 
    # 2. Update R (Rnew = Rold - αg) . α is the learning rate which is a scalar.

    # alignment_embeddings
    def align_embeddings(self,X: np.ndarray, Y: np.ndarray,
                          train_steps:int,
                          learning_rate: float=0.0003,
                          seed: int=129) -> np.ndarray:
        '''
        Finding the optimal R with gradient descent algorithm
        Inputs:
            X: a matrix of dimension (m,n) where the colums are the contrast representation 
            Y: a matrix of dimension (m,n) where the colums are the reference representation
           learning_rate: positive float - describes how big steps will  gradient descent algorithm do.
        Outputs:
            R: a matrix of dimension (n,n) - the projection matrix that minimizes the F norm ||projector(X R) - projector ( Y )||^2
        '''
        # the number of columns in X is the number of dimensions for a word vector (e.g. 300)
        # R is a square matrix with length equal to the number of dimensions in th  word embedding
        # R = np.random.rand(X.shape[1], X.shape[1])
        R = Variable(torch.ones(X.shape[1],X.shape[1]),requires_grad=True)

        m = len(X)
        X = torch.Tensor(X)
        Y = torch.Tensor(Y)
        
        # train_steps = 100000
        for i in range(train_steps):
            loss1 = ((( X.matmul(R) - Y)**2).sum())/m
            loss2 = self.kernel_HSIC_cka_loss(X.matmul(R),Y)
            if i% 99 == 0:
                print(f"iteration {i}, loss1 {loss1}",'loss2', {loss2}) 

            loss = loss1 + self.ALPHA * loss2

            loss.backward()

            R.data = R.data - learning_rate * R.grad.data

            R.grad.data.zero_()
    
        return R

    
    

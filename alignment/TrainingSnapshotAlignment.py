from abc import ABC, abstractmethod
import os
import sys
sys.path.append("..")
import torch
import numpy as np
from CKA_utils.CKA import CKA, CudaCKA, CCA_val


class TrainingSnapshotAlignmentAbstractClass(ABC):
    @abstractmethod
    def __init__(self, ref_data_provider, tar_data_provider, ref_EPOCH, tar_EPOCH, * args, **kawargs):
        pass

class TrainingSnapshotAlignment(TrainingSnapshotAlignmentAbstractClass):
    def __init__(self, ref_data_provider, tar_data_provider,ref_EPOCH, tar_EPOCH):
        self.ref_data_provider = ref_data_provider
        self.tar_data_provider = tar_data_provider
        self.ref_EPOCH = ref_EPOCH
        self.tar_EPOCH = tar_EPOCH
    

    def compute_loss(self, X, Y, R):
        '''
        Inputs: 
           X: a matrix of dimension (m,n) where the columns are the English embeddings.
           Y: a matrix of dimension (m,n) where the columns correspong to the French embeddings.
           R: a matrix of dimension (n,n) - transformation matrix from English to French vector space embeddings.
        Outputs:
           L: a matrix of dimension (m,n) - the value of the loss function for given X, Y and R.
        '''
        ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
        # m is the number of rows in X
        m = len(X)

        # diff is XR - Y
        diff = np.dot(X, R) - Y

        # diff_squared is the element-wise square of the difference
        diff_squared = diff**2

        # sum_diff_squared is the sum of the squared elements
        sum_diff_squared = diff_squared.sum()

        # loss is the sum_diff_squared divided by the number of examples (m)
        loss = sum_diff_squared/m
        ### END CODE HERE ###
        return loss
    def compute_gradient(self, X, Y, R):
        '''
            the gradient of the loss with respect to the matrix encodes how much a tiny change 
        in some coordinate of that matrix affect the change of loss function.
            Gradient descent uses that information to iteratively change matrix R until we reach 
        a point where the loss is minimized.
        Inputs: 
            X: a matrix of dimension (m,n) where the colums are the contrast representation 
            Y: a matrix of dimension (m,n) where the colums are the reference representation
            R: a matrix of dimension (n,n) - transformation matrix from Y2d to X2d
        Outputs:
        g: a matrix of dimension (n,n) - gradient of the loss function L for given X, Y and R.
        '''
        # m is the number of rows in X
        m = len(X)

        rows, columns = X.shape

        gradient = (np.dot(X.T, np.dot(X, R) - Y) * 2)/rows
        assert gradient.shape == (columns, columns)
        ### END CODE HERE ###
        return gradient


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
        R = np.random.rand(X.shape[1], X.shape[1])
        # R = Variable(torch.ones(X.shape[1],X.shape[1]),requires_grad=True)
        
        # train_steps = 100000
        for i in range(train_steps):
            if i%1000 == 0:
                loss = self.compute_loss(X,Y,R)
                print(f"iteration {i}, loss {loss}") 



            ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
            # use the function that you defined to compute the gradient
            gradient = self.compute_gradient(X, Y, R)
       
        
            # update R by subtracting the learning rate times gradient
            R -= learning_rate * gradient
            ### END CODE HERE ###
    
        return R

    
    

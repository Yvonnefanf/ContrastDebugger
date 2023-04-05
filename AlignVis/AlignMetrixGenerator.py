from abc import ABC, abstractmethod
import os
import sys
sys.path.append("..")
import torch
import numpy as np
from CKA_utils.CKA import CKA, CudaCKA, CCA_val
from AlignVis.losses import KNNOverlapLoss, CKALoss, PredictionLoss, ConfidenceLoss

from alignment.CKA_utils import *
from torch.autograd import Variable
from sklearn.neighbors import NearestNeighbors
from typing import Tuple

class TrainingSnapshotAlignmentAbstractClass(ABC):
    @abstractmethod
    def __init__(self, ref_data_provider, tar_data_provider, ref_EPOCH, tar_EPOCH, projector, visualizer, * args, **kawargs):
        pass

class TrainingSnapshotAlignment(TrainingSnapshotAlignmentAbstractClass):
    def __init__(self, ref_data_provider, tar_data_provider,ref_EPOCH, tar_EPOCH,projector,visualizer):
        self.ref_data_provider = ref_data_provider
        self.tar_data_provider = tar_data_provider
        self.ref_EPOCH = ref_EPOCH
        self.tar_EPOCH = tar_EPOCH
        self.projector = projector
        self.visualizer = visualizer
    
    def kernel_HSIC_cka_loss(self, X, Y, gamma=None):
        K_xx = kernel_HSIC(X, X, gamma)
        K_yy = kernel_HSIC(Y, Y, gamma)
        # K_xy = rbf_kernel(X, Y, 1e-2) 
        K_xy = kernel_HSIC(X, Y, gamma)   
        cka_loss = 1 - torch.mean(K_xy) / torch.sqrt(K_xx * K_yy)
        return cka_loss
    
    def kernel_HSIC_cka_loss_consider_init(self, X, Y, Z, alpha2=1e-3, gamma=None):
        K_xx = kernel_HSIC(X, X, gamma)
        K_yy = kernel_HSIC(Y, Y, gamma)
        K_zz = kernel_HSIC(Z, Z, gamma)
        K_xy = kernel_HSIC(X, Y, gamma)
        K_xz = kernel_HSIC(X, Z, gamma)
        K_yz = kernel_HSIC(Y, Z, gamma)
        cka_loss1 = 1 - torch.mean(K_xy) / torch.sqrt(K_xx * K_yy)
        cka_loss2 = 1 - torch.mean(K_yz) / torch.sqrt(K_yy * K_zz)
        loss = cka_loss1 + alpha2 * cka_loss2
        return loss

    
    def get_decision_view_grid(self, visualizer, epoch):

        x_min, y_min, x_max, y_max = visualizer.get_epoch_plot_measures(epoch)
        # create grid
        xs = np.linspace(x_min, x_max, 200)
        ys = np.linspace(y_min, y_max, 200)
        grid = np.array(np.meshgrid(xs, ys))
        grid = np.swapaxes(grid.reshape(grid.shape[0], -1), 0, 1)
        
        return grid

    
    def get_align_indicates(self, projector, epoch, representation, x_diff=0.02,y_diff=0.02):

        """
            Get the grid indicates that is near to the embeddings
            Parameters
            ========================
            projector: Projetcor
                use to get embedding
            epoch: int
                the epoch number
            representation: numpy
                high dimensional representaions need to be aligned
            x_diff: float | default 0.02
                Compute absolute differences between x coordinates
            y_diff: float | default 0.02
                Compute absolute differences between y coordinates
            ========================
        """

        embedding = projector.batch_project(epoch, representation)

        grid = self.get_decision_view_grid(self.visualizer, epoch)

        # Compute absolute differences between x and y coordinates
        diff_x = np.abs(embedding[:, 0][:, np.newaxis] - grid[:, 0])
        diff_y = np.abs(embedding[:, 1][:, np.newaxis] - grid[:, 1])
        # Find indices where both conditions are satisfied
        indices = np.where((diff_x < x_diff) & (diff_y < y_diff))
        print('len of align:' ,len(indices[0]))

        return indices
    
    def get_grid_align_indicates(self, embedding, vis, epoch,x_min_b,y_min_b):
        x_min, y_min, x_max, y_max = vis.get_epoch_plot_measures(epoch)
        # create grid
        xs = np.linspace(x_min, x_max, 200)
        ys = np.linspace(y_min, y_max, 200)
        grid = np.array(np.meshgrid(xs, ys))
        grid = np.swapaxes(grid.reshape(grid.shape[0], -1), 0, 1)

        # Compute absolute differences between x and y coordinates
        diff_x = np.abs(embedding[:, 0][:, np.newaxis] - grid[:, 0])
        diff_y = np.abs(embedding[:, 1][:, np.newaxis] - grid[:, 1])
        # Find indices where both conditions are satisfied
        indices = np.where((diff_x < x_min_b) & (diff_y < y_min_b))
        print('len of align:' ,len(indices[0]))
        return indices,grid
    
    # K-nearest neighbors overlap loss function
    def knn_overlap_loss(self, X, Y, k):

        # Ensure X and Y are tensors
        assert isinstance(X, torch.Tensor), "X must be a torch.Tensor."
        assert isinstance(Y, torch.Tensor), "Y must be a torch.Tensor."

        # Replace NaN values with zeros
        X = torch.where(torch.isnan(X), torch.zeros_like(X), X)
        Y = torch.where(torch.isnan(Y), torch.zeros_like(Y), Y)

        # Check for infinity values
        assert not torch.isinf(X).any(), "X contains infinity values."
        assert not torch.isinf(Y).any(), "Y contains infinity values."

        X_np = X.detach().numpy()
        Y_np = Y.detach().numpy()
    
        nbrs_X = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(X_np)
        nbrs_Y = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(Y_np)
        _, indices_X = nbrs_X.kneighbors(X_np)
        _, indices_Y = nbrs_Y.kneighbors(Y_np)
    
        # Ignore the first column, as it contains the point itself (distance 0)
        indices_X = indices_X[:, 1:]
        indices_Y = indices_Y[:, 1:]
    
        common_neighbors = 0
        for i in range(X_np.shape[0]):
            common_neighbors += len(np.intersect1d(indices_X[i], indices_Y[i]))
    
        loss = 1 - (common_neighbors / (X_np.shape[0] * k))
        return torch.tensor(loss, requires_grad=True)
    
    def get_mini_batch(self, X: torch.Tensor, Y: torch.Tensor, batch_idx: int, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:

        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(X))
    
        return X[start_idx:end_idx], Y[start_idx:end_idx]

    def align_embeddings_batch(self, X: np.ndarray, Y: np.ndarray,
                     train_steps: int = 5000,
                     batch_size: int = 100,
                     CKA_LAMBDA: int = 1,
                     CKA_LAMBAD_FOR_INIT = 1e-3,
                     N_LAMBDA: int = 1,
                     K_neibour: int = 10,
                     learning_rate: float = 0.0003,
                     seed: int = 129) -> np.ndarray:

        R = Variable(torch.ones(X.shape[1], X.shape[1]), requires_grad=True)

        X = torch.Tensor(X)
        Y = torch.Tensor(Y)

        n_samples = len(X)
        n_batches = (n_samples + batch_size - 1) // batch_size

        for i in range(train_steps):
            batch_idx = i % n_batches

            X_batch, Y_batch = self.get_mini_batch(X, Y, batch_idx, batch_size)

            loss1 = (((X_batch.matmul(R) - Y_batch) ** 2).sum()) / len(X_batch)
        
            loss2 = self.kernel_HSIC_cka_loss_consider_init(X_batch.matmul(R), Y_batch, X_batch, CKA_LAMBAD_FOR_INIT)

            loss3 = self.knn_overlap_loss(X_batch.matmul(R), Y_batch, K_neibour)

            ##### knn loss
            # knn_overlap_loss = KNNOverlapLoss(k=K_neibour)
            # knn_loss = knn_overlap_loss(input=X_batch.matmul(R), target=Y_batch)


            if i % 199 == 0:
                print(f"batch_idx {batch_idx},iteration {i}, loss1 {loss1}", "loss2", {loss2}, "neibour_loss", {loss3})

            loss = loss1 + CKA_LAMBDA * loss2 + N_LAMBDA * loss3

            loss.backward()
            
            # # Add this line after loss.backward()
            # if R.grad is not None:
            #     torch.nn.utils.clip_grad_norm_(R, max_norm=1.0)


            R.data = R.data - learning_rate * R.grad.data

            R.grad.data.zero_()

        return R
    
    # alignment_embeddings
    def align_embeddings(self,X: np.ndarray, Y: np.ndarray,
                          train_steps:int = 5000,
                          CKA_LAMBDA:int=1,
                          CKA_LAMBAD_FOR_INIT=1e-3,
                          N_LAMBDA:int=1,
                          K_neibour:int=10,
                          learning_rate: float=0.0003,
                          seed: int=129) -> np.ndarray:
        '''
        Finding the optimal R with gradient descent algorithm
        Inputs:
            X: a matrix of dimension (m,n) where the colums are the need_transform representation 
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
            

        for i in range(train_steps):
            loss1 = ((( X.matmul(R) - Y)**2).sum())/m
            loss2 = self.kernel_HSIC_cka_loss_consider_init(X.matmul(R),Y,X,CKA_LAMBAD_FOR_INIT)
            loss3 = self.knn_overlap_loss(X.matmul(R),Y,K_neibour)
            if i% 19 == 0:
                print(f"iteration {i}, loss1 {loss1}",'loss2', {loss2}, "neibour_loss",{loss3}) 

            loss = loss1 + CKA_LAMBDA * loss2 + N_LAMBDA *loss3

            loss.backward()

            R.data = R.data - learning_rate * R.grad.data

            R.grad.data.zero_()
    
        return R
    

    
    

import torch
import sys
import os

import argparse


from singleVis.data import NormalDataProvider
from singleVis.projector import Projector
from singleVis.SingleVisualizationModel import VisModel

from torch import nn


import scipy
import numpy as np
from sklearn.cross_decomposition import CCA

########################################################################################################################
#                                                     LOAD PARAMETERS                                                  #
########################################################################################################################
parser = argparse.ArgumentParser(description='Process hyperparameters...')
parser.add_argument('--content_path', type=str)
parser.add_argument('--reference_path', type=str)

args = parser.parse_args()
CONTENT_PATH = args.content_path
REFERENCE_PATH = args.reference_path


sys.path.append(CONTENT_PATH)
sys.path.append(REFERENCE_PATH)

from config import config

GPU_ID = config["GPU"]
EPOCH_START = config["EPOCH_START"]
EPOCH_END = config["EPOCH_END"]
EPOCH_PERIOD = config["EPOCH_PERIOD"]


ref_provider = NormalDataProvider(REFERENCE_PATH, net, EPOCH_START, EPOCH_END, EPOCH_PERIOD, split=-1, device=DEVICE, classes=CLASSES,verbose=1)

data_provider = NormalDataProvider(content_path, net, EPOCH_START, EPOCH_END, EPOCH_PERIOD, split=-1, device=DEVICE, classes=CLASSES,verbose=1)

model = VisModel(ENCODER_DIMS, DECODER_DIMS)
projector = TimeVisProjector(vis_model=model, content_path=REFERENCE_PATH, vis_model_name=VIS_MODEL_NAME, device="cpu")




def compute_gradient(X, Y, R):
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

    diff = np.dot(X,R) - Y
    # diff_squared is the element-wise square of the difference
    diff_squared = diff**2

    # sum_diff_squared is the sum of the squared elements
    sum_diff_squared = diff_squared.sum()

    loss = sum_diff_squared/m

    gradient = (np.dot(X.T, np.dot(X, R) - Y) * 2)/rows
    assert gradient.shape == (columns, columns)
    ### END CODE HERE ###
    return gradient, loss

# alignment_embeddings
def align_embeddings(X: np.ndarray, Y: np.ndarray,
                      train_steps: int=4000,
                      learning_rate: float=0.0005,
                      seed: int=129) -> np.ndarray:
    '''
    Finding the optimal R with gradient descent algorithm
    Inputs:
        X: a matrix of dimension (m,n) where the colums are the contrast representation 
        Y: a matrix of dimension (m,n) where the colums are the reference representation
        train_steps: positive int - describes how many steps will gradient descent algorithm do.
        learning_rate: positive float - describes how big steps will  gradient descent algorithm do.
    Outputs:
        R: a matrix of dimension (n,n) - the projection matrix that minimizes the F norm ||projector(X R) - projector ( Y )||^2
    '''
    # the number of columns in X is the number of dimensions for a word vector (e.g. 300)
    # R is a square matrix with length equal to the number of dimensions in th  word embedding
    R = np.random.rand(X.shape[1], X.shape[1])

    for i in range(train_steps):

        ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
        # use the function that you defined to compute the gradient
        gradient, loss = compute_gradient(X, Y, R)
       
        print(f"iteration {i} loss is {loss:.4f}")
         # update R by subtracting the learning rate times gradient
        R -= learning_rate * gradient
        ### END CODE HERE ###
    
    return R

for i in range(EPOCH_START, EPOCH_END+1, EPOCH_PERIOD):
    contrast_data = data_provider.train_representation(i).squeeze()
    ref_data = ref_provider.train_representation(i).squeeze()

    R = align_embeddings(contrast_data,ref_data)
    vis.savefig(i, path=os.path.join(save_dir, "{}_{}_{}.png".format(DATASET, i, VIS_METHOD)))
    with open 




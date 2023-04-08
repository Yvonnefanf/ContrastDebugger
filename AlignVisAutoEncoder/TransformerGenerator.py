from abc import ABC, abstractmethod
from AlignVis.DataInit import DataInit
import numpy as np
import os
import json
from singleVis.utils import *

import torch.optim as optim
import numpy as np
from pyemd import emd
import torch.nn as nn
from scipy.spatial.distance import cdist
from sklearn.neighbors import kneighbors_graph
from AlignVisAutoEncoder.autoencoder import SimpleAutoencoder
from AlignVisAutoEncoder.data import DataLoaderInit

class TransformerGeneratorAbstractClass(ABC):
    @abstractmethod
    def __init__(self, REF_PATH, REF_CONFIG_PATH, TAR_PATH, TAR_CONFIG_PATH,  REF_EPOCH, TAR_EPOCH, DEVICE, * args, **kawargs):
        pass

class TransformerGenerator(TransformerGeneratorAbstractClass):
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
    
    def earth_movers_distance(self, X, Y, k=5):
        X, Y = X.detach().numpy(), Y.detach().numpy()
    
        # Compute KNN graphs
        X_knn_graph = kneighbors_graph(X, k, mode='distance')
        Y_knn_graph = kneighbors_graph(Y, k, mode='distance')
    
        # Convert to dense NumPy arrays
        X_knn_matrix = X_knn_graph.toarray()
        Y_knn_matrix = Y_knn_graph.toarray()

        # Calculate the EMD between the KNN distance matrices
        distance_matrix = cdist(X_knn_matrix, Y_knn_matrix)
        first_histogram = np.ones(X_knn_matrix.shape[0]) / X_knn_matrix.shape[0]
        second_histogram = np.ones(Y_knn_matrix.shape[0]) / Y_knn_matrix.shape[0]

        return emd(first_histogram, second_histogram, distance_matrix)

    def frobenius_norm_loss(self, predicted, target):
        return torch.norm(predicted - target, p='fro') / predicted.numel()
    
    def transformer_generation(self,num_epochs=200,learning_rate=1e-5):
        ### data loader
        data_loader = DataLoaderInit(self.ref_train_data, self.tar_train_data)
        dataloader = data_loader.get_data_loader()
        autoencoder = SimpleAutoencoder(512,512)

        # Define the loss function and the optimizer
        optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate,weight_decay=1e-5)
        # Training loop
        for epoch in range(num_epochs):
            for data_X, data_Y in dataloader: # Assuming you have a DataLoader instance with paired data (X, Y)
                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass (encoding Y and decoding to X's space)
                transformed_Y = autoencoder.encoder(data_Y)
                recon_X = autoencoder.decoder(transformed_Y)

                # loss = recon_loss + alpha * topological_loss
                topological_loss_encoder = self.earth_movers_distance(data_Y, transformed_Y)
                topological_loss_decoder = self.earth_movers_distance(data_Y, recon_X)
                loss_f_decoder = self.frobenius_norm_loss(recon_X, data_Y) + topological_loss_decoder
                loss_f_encoder = self.frobenius_norm_loss(transformed_Y, data_X) + topological_loss_encoder

                loss = loss_f_decoder + loss_f_encoder 

                # Backward pass
                loss.backward()

                # Update the weights
                optimizer.step()




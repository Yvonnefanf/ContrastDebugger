from abc import ABC, abstractmethod
from AlignVis.DataInit import DataInit
import numpy as np
import os
import json
from singleVis.utils import *

from sklearn.cluster import KMeans

from AlignVisAutoEncoder.autoencoder import SimpleAutoencoder
from AlignVisAutoEncoder.data_loader import DataLoaderInit

import torch.optim as optim
import numpy as np
from pyemd import emd
import torch.nn as nn
from scipy.spatial.distance import cdist
from sklearn.neighbors import kneighbors_graph
import torch.nn.functional as F
from AlignVis.AlignSimilarityScaler import AlignSimilarityScaler

class AutoEncoderGeneratorAbstractClass(ABC):
    @abstractmethod
    def __init__(self, REF_PATH, REF_CONFIG_PATH, TAR_PATH, TAR_CONFIG_PATH,  REF_EPOCH, TAR_EPOCH, DEVICE, * args, **kawargs):
        pass

class AutoEncoderGenerator(AutoEncoderGeneratorAbstractClass):

    def __init__(self, REF_PATH, REF_CONFIG_PATH, TAR_PATH, TAR_CONFIG_PATH,  REF_EPOCH, TAR_EPOCH, projector, DEVICE) -> None:

        ref_datainit = DataInit(REF_CONFIG_PATH,REF_PATH,REF_EPOCH,DEVICE)
        tar_datainit = DataInit(TAR_CONFIG_PATH,TAR_PATH,TAR_EPOCH,DEVICE)
        ref_model, ref_provider, ref_train_data, ref_prediction, ref_prediction_res, ref_scores = ref_datainit.getData()
        tar_model, tar_provider, tar_train_data, tar_prediction, tar_prediction_res, tar_scores = tar_datainit.getData()

        self.REF_PATH = REF_PATH
        self.TAR_PATH = TAR_PATH
        self.REF_EPOCH = REF_EPOCH
        self.ref_model = ref_model
        self.ref_provider = ref_provider
        self.TAR_EPOCH = TAR_EPOCH
        self.tar_provider = tar_provider
        self.tar_model = tar_model
        self.tar_provider = tar_provider
        self.projector = projector
        self.DEVICE = DEVICE
        
    ##### for topology preserving
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
    
    ####### try to map reference and target to the other space
    def frobenius_norm_loss(self, predicted, target):
        return torch.norm(predicted - target, p='fro') / predicted.numel()
    
    ##### keep them prediction result
    def prediction_loss(self,trans_X, Y):
    
        target_output = self.tar_provider.get_pred(self.TAR_EPOCH, Y.detach().numpy())
        # tar_output = self.get_pred(self.TAR_EPOCH, adjusted_input, self.tar_provider.content_path, self.tar_model)
        ref_output = self.tar_provider.get_pred(self.TAR_EPOCH, trans_X.detach().numpy())
        loss_ref_output = F.mse_loss(torch.tensor(ref_output), torch.tensor(target_output))
        loss_Rep = F.mse_loss(trans_X, Y)
        
        # loss = loss_tar_output + loss_Rep + self.alpha_for_pred_ref * loss_ref_output
        loss =  loss_Rep + 1 * loss_ref_output
        return loss
    
    # Define a contrastive loss function
    def contrastive_loss(self, x1, x2, y, margin=1.0):
        y = torch.from_numpy(y)
        # Compute the Euclidean distance between x1 and x2
        distance = F.pairwise_distance(x1, x2)

        # Compute the contrastive loss
        loss_contrastive = torch.mean(y * torch.pow(distance, 2) + (1 - y) * torch.pow(torch.clamp(margin - distance, min=0.0), 2))

        return loss_contrastive

    
    def label_flip_loss(self, X, Y, encoded_Y):
    
        pred = self.ref_provider.get_pred(self.REF_EPOCH, X.detach().numpy()).argmax(axis=1)
        new_pred_origin = self.tar_provider.get_pred(self.REF_EPOCH, Y.detach().numpy())
        new_pred = new_pred_origin.argmax(axis=1)
        flip_indices = [i for i, (x, y) in enumerate(zip(pred, new_pred)) if x != y]

        embedding_ref = self.projector.batch_project(self.REF_EPOCH, X.detach().numpy())
        embedding_trans = self.projector.batch_project(self.REF_EPOCH, encoded_Y.detach().numpy())
        inv_ref_data = self.projector.batch_inverse(self.REF_EPOCH, embedding_ref)
        inv_trans_data = self.projector.batch_inverse(self.REF_EPOCH, embedding_trans)

        low_pred = self.ref_provider.get_pred(self.REF_EPOCH, inv_ref_data).argmax(axis=1)
        low_new_pred_origin = self.tar_provider.get_pred(self.REF_EPOCH, inv_trans_data)
        low_new_pred = low_new_pred_origin.argmax(axis=1)

        low_flip_indices = [i for i, (x, y) in enumerate(zip(low_pred, low_new_pred)) if x != y]
        loss_intersection = set(flip_indices).intersection(low_flip_indices)

        loss_ppr = F.mse_loss(torch.tensor(new_pred_origin), torch.tensor(low_new_pred_origin))

        inv_trans_data_tensor = torch.tensor(inv_trans_data)
        recon_loss = F.mse_loss(inv_trans_data_tensor, Y)

        # Compute the edge loss
        edge_mask = (torch.tensor(pred) == torch.tensor(new_pred)).float()  # create a mask indicating which indices share the same label
        edge_mask = edge_mask.unsqueeze(-1)
        edge_loss = ((inv_trans_data_tensor - Y)**2 * edge_mask).sum() / edge_mask.sum()  # mean squared error of values at edge indices

        if len(flip_indices) == 0:
            flip_loss = 0.001
        else:
            flip_loss = len(loss_intersection) / len(flip_indices)
      
        loss = 0.1 * flip_loss + 0.2 * loss_ppr + 0.35 * recon_loss + 0.35 * edge_loss
        # loss =  abs(diff_pred - diff_low_pred)
        return loss

    

    def load_encoder_data(self, batch_size=500, input_dim=512, output_dim=512):

        autoencoder = SimpleAutoencoder(input_dim,output_dim)

        ######### train sample #############
        data_loader = DataLoaderInit(self.ref_provider.train_representation(self.REF_EPOCH), self.tar_provider.train_representation(self.TAR_EPOCH),batch_size)
        dataloader = data_loader.get_data_loader()

        return dataloader,autoencoder
    
    def encoder_trainer(self, saved_path, num_epochs = 20,learning_rate = 1e-5):
        dataloader,autoencoder = self.load_encoder_data()
        AlignSimilarity_scaler = AlignSimilarityScaler(self.REF_PATH, self.REF_PATH, self.TAR_PATH, self.TAR_PATH, 200,200, self.DEVICE)

        optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate,weight_decay=1e-5)

        alpha = 1 # weight for topological loss, adjust this according to your requirements

        # Training loop
        for epoch in range(num_epochs):
            for data_X, data_Y in dataloader: # Assuming you have a DataLoader instance with paired data (X, Y)
                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass (encoding Y and decoding to X's space)
                transformed_Y = autoencoder.encoder(data_Y)
                recon_X = autoencoder.decoder(transformed_Y)

                topological_loss_encoder = self.earth_movers_distance(data_Y, transformed_Y)
                topological_loss_decoder = self.earth_movers_distance(data_Y, recon_X)
        
                loss_f_decoder = self.frobenius_norm_loss(recon_X, data_Y) + 10* topological_loss_decoder
                loss_f_encoder = self.frobenius_norm_loss(transformed_Y, data_X) + topological_loss_encoder
        
                ###### get current similairrty
                sim_list = AlignSimilarity_scaler.get_jaccard_similarities(data_X, data_Y,10)

                # Create a binary label tensor indicating whether each pair is similar or dissimilar
                sim_array = np.array(sim_list)
                # print(sim_array)
                labels = (sim_array > 0.99).astype(float)
                # print(labels)

                loss_contrastive = self.contrastive_loss(data_X, transformed_Y, labels)

                # pred_loss = prediction_loss(recon_X, data_Y)

                flip_loss = self.label_flip_loss(data_X, data_Y, recon_X)

                # loss = loss_f_decoder + loss_f_encoder + 0.01 * pred_loss + 0.1 * flip_loss
        
                loss = loss_f_decoder + loss_f_encoder + 0.01 * flip_loss + loss_contrastive
                # Backward pass
                loss.backward()

                # Update the weights
                optimizer.step()


            # Print the loss for each epoch
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Loss decoder: {loss_f_decoder.item():.4f},Loss encoder: {loss_f_encoder.item():.4f},flip_loss: {flip_loss},loss_contrastive{loss_contrastive}')

        torch.save({'epoch': self.TAR_EPOCH,
                            'model_state_dict': autoencoder.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss}, saved_path)
        return autoencoder
    
    def encoder_active_learning():

        return




    

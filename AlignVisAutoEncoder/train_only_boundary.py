####### dropout resnet18 vs without dropout
#### 
import torch
import sys
sys.path.append("..")
import numpy as np

TAR_PATH = "/home/yifan/Exp/Dropout/0.8/experiment1"
REF_PATH = "/home/yifan/dataset/clean/pairflip/cifar10/0"


ENCODER_DIMS=[512,256,256,256,256,2]
DECODER_DIMS= [2,256,256,256,256,512]
VIS_MODEL_NAME = 'vis'

DEVICE='cuda:1'
########## initulize reference data and target data
from AlignVis.DataInit import DataInit
REF_EPOCH = 200
TAR_EPOCH = 190
tar_datainit = DataInit(TAR_PATH,TAR_PATH,TAR_EPOCH,DEVICE)
ref_datainit = DataInit(REF_PATH,REF_PATH,REF_EPOCH,DEVICE)

ref_model, ref_provider, ref_train_data, ref_prediction, ref_prediction_res, ref_scores = ref_datainit.getData()
tar_model, tar_provider, tar_train_data, tar_prediction, tar_prediction_res, tar_scores = tar_datainit.getData()


from AlignVis.ReferenceGenerator import ReferenceGenerator
gen = ReferenceGenerator(ref_provider=ref_provider, tar_provider=tar_provider,REF_EPOCH=REF_EPOCH,TAR_EPOCH=TAR_EPOCH,ref_model=ref_model,tar_model=tar_model,DEVICE=DEVICE)

absolute_alignment_indicates,predict_label_diff_indicates,predict_confidence_Diff_indicates,high_distance_indicates = gen.subsetClassify(18,0.8,0.3,0.05)


from AlignVis_Visualizer.visualizer import visualizer
from singleVis.SingleVisualizationModel import VisModel
from singleVis.projector import TimeVisProjector
model = VisModel(ENCODER_DIMS, DECODER_DIMS)

I = np.eye(512)
projector = TimeVisProjector(vis_model=model, content_path=REF_PATH, vis_model_name=VIS_MODEL_NAME, device="cpu")
vis = visualizer(ref_provider, I,I, np.dot(ref_provider.train_representation(TAR_EPOCH),I), projector, 200,[0,1],'tab10')


from singleVis.spatial_edge_constructor import kcSpatialEdgeConstructor
spatial_cons = kcSpatialEdgeConstructor(data_provider=tar_provider, init_num=300, s_n_epochs=0, b_n_epochs=0, n_neighbors=15, MAX_HAUSDORFF=0.4, ALPHA=0, BETA=0.1)


####### generate boundary ponits for tar and ref respectively
from AlignVis.AlignmentBoundaryGenerator import AlignmentBoundaryGenerator
BoundaryGen = AlignmentBoundaryGenerator(REF_PATH,REF_PATH,TAR_PATH,TAR_PATH,REF_EPOCH,TAR_EPOCH,DEVICE)

import torch
###### get border sample features
import torch.nn as nn

import os
ref_border_path = os.path.join(TAR_PATH,"Model", "Epoch_{:d}".format(REF_EPOCH),
                                          "aligned_ref_border.npy")
tar_border_path = os.path.join(TAR_PATH,"Model", "Epoch_{:d}".format(TAR_EPOCH),
                                          "aligned_tar_border.npy")
if os.path.exists(ref_border_path) and os.path.exists(tar_border_path):
    print("load positive boundary samples")
    ref_features = np.load(ref_border_path).squeeze()
    tar_features = np.load(tar_border_path).squeeze()

ref_neg_border_path = os.path.join(TAR_PATH,"Model", "Epoch_{:d}".format(TAR_EPOCH),
                                          "aligned_ref_neg_border.npy")
tar_neg_border_path = os.path.join(TAR_PATH,"Model", "Epoch_{:d}".format(TAR_EPOCH),
                                          "aligned_tar_neg_border.npy")
if os.path.exists(ref_neg_border_path) and os.path.exists(tar_neg_border_path):
     print("load negative boundary samples")
     ref_neg_features = np.load(ref_neg_border_path).squeeze()
     tar_neg_features = np.load(tar_neg_border_path).squeeze()




######### initialize autoencoder and dataloader #########################
from AlignVisAutoEncoder.autoencoder import SimpleAutoencoder
from AlignVisAutoEncoder.data_loader import DataLoaderInit
input_dim = 512
output_dim = 512

autoencoder = SimpleAutoencoder(input_dim,output_dim)
######### train sample + generated boundary sample's => input  #############
indicates = np.random.choice(np.arange(50000), size=20000, replace=False)
input_X = np.concatenate((ref_provider.train_representation(REF_EPOCH)[indicates], ref_features,ref_neg_features),axis=0)
input_Y = np.concatenate((tar_provider.train_representation(TAR_EPOCH)[indicates], tar_features,tar_neg_features),axis=0)
data_loader_b = DataLoaderInit(input_X, input_Y, batch_size=100)
data_loader = DataLoaderInit(ref_provider.train_representation(REF_EPOCH), tar_provider.train_representation(TAR_EPOCH), batch_size=5000)
dataloader = data_loader.get_data_loader()
dataloader_b = data_loader_b.get_data_loader()



import torch.optim as optim
import numpy as np
from pyemd import emd
import torch.nn as nn
from scipy.spatial.distance import cdist
from sklearn.neighbors import kneighbors_graph
import torch.nn.functional as F
from AlignVis.losses import KNNOverlapLoss, CKALoss, PredictionLoss, ConfidenceLoss

pre_trained_model_loc = os.path.join(TAR_PATH, "Model", "Epoch_{}".format(TAR_EPOCH),"test_use_complex{}.pth".format(4))


checkpoint = torch.load(pre_trained_model_loc)
autoencoder.load_state_dict(checkpoint['model_state_dict'])

def earth_movers_distance(X, Y, k=5):
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

def frobenius_norm_loss(predicted, target):
    return torch.norm(predicted - target, p='fro') / predicted.numel()

def prediction_loss(trans_X, Y):
    
    target_output = tar_provider.get_pred(TAR_EPOCH, Y.detach().numpy())
    # tar_output = self.get_pred(self.TAR_EPOCH, adjusted_input, self.tar_provider.content_path, self.tar_model)
    ref_output = tar_provider.get_pred(TAR_EPOCH, trans_X.detach().numpy())

    loss_ref_output = F.mse_loss(torch.tensor(ref_output), torch.tensor(target_output))
    loss_Rep = F.mse_loss(trans_X, Y)
        
    # loss = loss_tar_output + loss_Rep + self.alpha_for_pred_ref * loss_ref_output
    loss =  loss_Rep + 1 * loss_ref_output
    return loss


import torch
import torch.nn as nn
import numpy as np

from scipy.sparse import csr_matrix

def complex_loss_func(complex1: csr_matrix, sigmas1, rhos1, knn_indices1, 
                      complex2: csr_matrix, sigmas2, rhos2, knn_indices2):
    # Convert sparse CSR matrices to dense format
    complex1_dense = complex1.toarray().flatten()
    complex2_dense = complex2.toarray().flatten()
    
    # Compute Jaccard distance between the two complexes
    union = np.union1d(complex1_dense, complex2_dense)
    intersection = np.intersect1d(complex1_dense, complex2_dense)
    if len(union) != 0:
        complex_diff = 1 - len(intersection) / len(union)
    else:
        complex_diff = 0
    
    # Compute difference in sigmas
    sigma_diff = np.linalg.norm(sigmas1 - sigmas2)
    
    # Compute difference in rhos
    rho_diff = np.linalg.norm(rhos1 - rhos2)
    
    # Compute difference in nearest neighbors
    knn_diff = np.linalg.norm(knn_indices1 - knn_indices2)
    
    # Compute final loss as a weighted sum of the above components
    loss = complex_diff + sigma_diff + rho_diff + knn_diff
    
    return loss




# Define hyperparameters
num_epochs = 6
batch_size = 50
learning_rate = 1e-4

# Define the loss function and the optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate,weight_decay=1e-5)
if pre_trained_model_loc != "":
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


alpha = 1 # weight for topological loss, adjust this according to your requirements


# Training loop
for epoch in range(num_epochs):
    # Initialize a list to store the predictions of unlabelled data
    unlabelled_preds = []
    for data_X, data_Y in dataloader_b: # Assuming you have a DataLoader instance with paired data (X, Y)
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass (encoding Y and decoding to X's space)
        transformed_Y = autoencoder.encoder(data_Y)
        recon_X = autoencoder.decoder(transformed_Y)

        topological_loss_encoder = earth_movers_distance(data_Y, transformed_Y)
        topological_loss_decoder = earth_movers_distance(data_X, recon_X)
        
        loss_f_decoder = frobenius_norm_loss(recon_X, data_X) + 10 * topological_loss_decoder
        loss_f_encoder = frobenius_norm_loss(transformed_Y, data_X) + 10 * topological_loss_encoder

        complex, sigmas_t, rhos_t, knn_idxs_t = spatial_cons._construct_fuzzy_complex(data_Y.detach().numpy())
        complex_2, sigmas_t_2, rhos_t_2, knn_idxs_t2 = spatial_cons._construct_fuzzy_complex(transformed_Y.detach().numpy())
        
  
        loss_complex = complex_loss_func(complex, sigmas_t, rhos_t, knn_idxs_t, complex_2, sigmas_t_2, rhos_t_2, knn_idxs_t2)

        pred_loss = prediction_loss(recon_X, data_Y)

        #### CKA loss
        cka_loss_f = CKALoss(gamma=None, alpha=1e-8)
        cka_loss = cka_loss_f(data_Y,transformed_Y,recon_X)

        loss = loss_f_decoder + loss_f_encoder + 0.02 * pred_loss + 0.1 * cka_loss + 0.001 * loss_complex

        # Backward pass
        loss.backward()

        # Update the weights
        optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Loss decoder: {loss_f_decoder.item():.4f},Loss encoder: {loss_f_encoder.item():.4f},pred_loss,{pred_loss.item():.4f},CKA,{cka_loss.item():.4f},loss_complex:{loss_complex}')

    # Print the loss for each epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Loss decoder: {loss_f_decoder.item():.4f},Loss encoder: {loss_f_encoder.item():.4f},pred_loss,{pred_loss.item():.4f},CKA,{cka_loss.item():.4f},loss_complex:{loss_complex}')
    torch.save({
    'epoch': TAR_EPOCH,
    'model_state_dict': autoencoder.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss
    }, os.path.join(TAR_PATH, "Model", "Epoch_{}".format(TAR_EPOCH),"test_use_complex_boundary{}.pth".format(epoch)))



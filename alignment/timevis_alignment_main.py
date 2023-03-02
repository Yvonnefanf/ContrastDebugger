import torch
import sys
import os
import time
import numpy as np
import argparse
import json
import sys
sys.path.append("..")

from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from umap.umap_ import find_ab_params

from singleVis.custom_weighted_random_sampler import CustomWeightedRandomSampler
from singleVis.SingleVisualizationModel import VisModel, SingleVisualizationModel
from singleVis.losses import SmoothnessLoss, HybridLoss, UmapLoss, ReconstructionLoss, SingleVisLoss
# from singleVis.edge_dataset import DataHandler
from singleVis.edge_dataset import HybridDataHandler
# from singleVis.trainer import SingleVisTrainer
from singleVis.trainer import HybridVisTrainer
from singleVis.data import NormalDataProvider
from singleVis.spatial_edge_constructor import kcSpatialEdgeConstructor
##### use alignment_tempory
from alignment.spatial_edge_constructor import  kcHybridSpatialEdgeConstructor
from alignment.temporal_edge_constructor import  GlobalTemporalEdgeConstructor, LocalTemporalEdgeConstructor
from alignment.ReferenceGenerator import ReferenceGenerator

########################################################################################################################
#                                                    VISUALIZATION SETTING                                             #
########################################################################################################################
VIS_METHOD= "TimeVis"
########################################################################################################################
#                                                     LOAD PARAMETERS                                                  #
########################################################################################################################
parser = argparse.ArgumentParser(description='Process hyperparameters...')
parser.add_argument('--content_path', type=str)
parser.add_argument('--reference_path', type=str)
args = parser.parse_args()

CONTENT_PATH = args.content_path
REF_PATH = args.reference_path
sys.path.append(CONTENT_PATH)
from config import config

# record output information
now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time())) 
sys.stdout = open(os.path.join(CONTENT_PATH, now+".txt"), "w")

SETTING = config["SETTING"]
CLASSES = config["CLASSES"]
DATASET = config["DATASET"]
PREPROCESS = config["VISUALIZATION"]["PREPROCESS"]
GPU_ID = config["GPU"]
EPOCH_START = config["EPOCH_START"]
EPOCH_END = config["EPOCH_END"]
EPOCH_PERIOD = config["EPOCH_PERIOD"]

# Training parameter (subject model)
TRAINING_PARAMETER = config["TRAINING"]
NET = TRAINING_PARAMETER["NET"]
LEN = TRAINING_PARAMETER["train_num"]
# Training parameter (visualization model)
VISUALIZATION_PARAMETER = config["VISUALIZATION"]
LAMBDA = VISUALIZATION_PARAMETER["LAMBDA"]
B_N_EPOCHS = VISUALIZATION_PARAMETER["BOUNDARY"]["B_N_EPOCHS"]
L_BOUND = VISUALIZATION_PARAMETER["BOUNDARY"]["L_BOUND"]
INIT_NUM = VISUALIZATION_PARAMETER["INIT_NUM"]
ALPHA = VISUALIZATION_PARAMETER["ALPHA"]
BETA = VISUALIZATION_PARAMETER["BETA"]
MAX_HAUSDORFF = VISUALIZATION_PARAMETER["MAX_HAUSDORFF"]
HIDDEN_LAYER = VISUALIZATION_PARAMETER["HIDDEN_LAYER"]
ENCODER_DIMS = VISUALIZATION_PARAMETER["ENCODER_DIMS"]
DECODER_DIMS = VISUALIZATION_PARAMETER["DECODER_DIMS"]
S_N_EPOCHS = VISUALIZATION_PARAMETER["S_N_EPOCHS"]
T_N_EPOCHS = VISUALIZATION_PARAMETER["T_N_EPOCHS"]
N_NEIGHBORS = VISUALIZATION_PARAMETER["N_NEIGHBORS"]
PATIENT = VISUALIZATION_PARAMETER["PATIENT"]
MAX_EPOCH = VISUALIZATION_PARAMETER["MAX_EPOCH"]
S_LAMBDA = VISUALIZATION_PARAMETER["S_LAMBDA"]
S_LAMBDA = 100

VIS_MODEL_NAME = VISUALIZATION_PARAMETER["VIS_MODEL_NAME"]

EVALUATION_NAME = VISUALIZATION_PARAMETER["EVALUATION_NAME"]

SEGMENTS = [(EPOCH_START, EPOCH_END)]

# define hyperparameters
DEVICE = torch.device("cuda:{}".format(GPU_ID) if torch.cuda.is_available() else "cpu")

import Model.model as subject_model
net = eval("subject_model.{}()".format(NET))

########################################################################################################################
#                                                    TRAINING SETTING                                                  #
########################################################################################################################


data_provider = NormalDataProvider(CONTENT_PATH, net, EPOCH_START, EPOCH_END, EPOCH_PERIOD, split=-1, device=DEVICE, classes=CLASSES,verbose=1)
if PREPROCESS:
    data_provider._meta_data()
    if B_N_EPOCHS >0:
        data_provider._estimate_boundary(LEN//10, l_bound=L_BOUND)

# model = SingleVisualizationModel(input_dims=512, output_dims=2, units=256, hidden_layer=HIDDEN_LAYER)
model = VisModel(ENCODER_DIMS, DECODER_DIMS)
# model = SingleVisualizationModel(input_dims=512, output_dims=2, units=256, hidden_layer=HIDDEN_LAYER)


negative_sample_rate = 5
min_dist = .1
_a, _b = find_ab_params(1.0, min_dist)
umap_loss_fn = UmapLoss(negative_sample_rate, DEVICE, _a, _b, repulsion_strength=1.0)
recon_loss_fn = ReconstructionLoss(beta=1.0)
# criterion = SingleVisLoss(umap_loss_fn, recon_loss_fn, lambd=LAMBDA)
smooth_loss_fn = SmoothnessLoss(margin=0.5)
criterion = HybridLoss(umap_loss_fn, recon_loss_fn, smooth_loss_fn, lambd1=LAMBDA, lambd2=S_LAMBDA)


optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=.1)

t0 = time.time()
###########
ref_model = VisModel(ENCODER_DIMS, DECODER_DIMS)
##### load reference model
ref_save_model = torch.load(os.path.join(REF_PATH, "Model", "vis.pth"), map_location=torch.device("cpu"))
ref_model.load_state_dict(ref_save_model["state_dict"])
ref_provider = NormalDataProvider(REF_PATH, net, EPOCH_START, EPOCH_END, EPOCH_PERIOD, split=-1, device=DEVICE, classes=CLASSES,verbose=1)
#### get absolute alignment indicates
ReferenceGenerator = ReferenceGenerator(ref_provider=ref_provider, tar_provider=data_provider,REF_EPOCH=200,TAR_EPOCH=200)
absolute_alignment_indicates,predict_label_diff_indicates,predict_confidence_Diff_indicates = ReferenceGenerator.subsetClassify(35, 1)

prev_selected = absolute_alignment_indicates
# with open(os.path.join(REF_PATH, "selected_idxs", "selected_{}.json".format(200)), "r") as f:
#     prev_selected = json.load(f)
prev_data = torch.from_numpy(ref_provider.train_representation(200)[prev_selected]).to(dtype=torch.float32)
prev_embedding = ref_model.encoder(prev_data).detach().numpy()
print("Resume from with {} points...".format(len(prev_embedding)))

start_point = len(SEGMENTS)-1
# c0=None
# d0=None

with open(os.path.join(REF_PATH, "selected_idxs", "baseline.json".format(200)), "r") as f:
    c0, d0 = json.load(f)
spatial_cons = kcHybridSpatialEdgeConstructor(data_provider=data_provider, init_num=INIT_NUM, s_n_epochs=S_N_EPOCHS, b_n_epochs=B_N_EPOCHS, n_neighbors=N_NEIGHBORS, MAX_HAUSDORFF=MAX_HAUSDORFF, ALPHA=ALPHA, BETA=BETA, init_idxs=prev_selected, init_embeddings=prev_embedding, c0=c0, d0=d0)
s_edge_to, s_edge_from, s_probs, feature_vectors, embedded, coefficient, time_step_nums, time_step_idxs_list, knn_indices, sigmas, rhos, attention, (c0,d0) = spatial_cons.construct()


temporal_cons = GlobalTemporalEdgeConstructor(X=feature_vectors, time_step_nums=time_step_nums, sigmas=sigmas, rhos=rhos, n_neighbors=N_NEIGHBORS, n_epochs=T_N_EPOCHS)


t_edge_to, t_edge_from, t_probs = temporal_cons.construct()
t1 = time.time()

edge_to = np.concatenate((s_edge_to, t_edge_to),axis=0)
edge_from = np.concatenate((s_edge_from, t_edge_from), axis=0)
probs = np.concatenate((s_probs, t_probs), axis=0)
probs = probs / (probs.max()+1e-3)
eliminate_zeros = probs>1e-3
edge_to = edge_to[eliminate_zeros]
edge_from = edge_from[eliminate_zeros]
probs = probs[eliminate_zeros]

# dataset = DataHandler(edge_to, edge_from, feature_vectors, attention)
dataset = HybridDataHandler(edge_to, edge_from, feature_vectors, attention, embedded, coefficient)
n_samples = int(np.sum(S_N_EPOCHS * probs) // 1)
# chose sampler based on the number of dataset
if len(edge_to) > 2^24:
    sampler = CustomWeightedRandomSampler(probs, n_samples, replacement=True)
else:
    sampler = WeightedRandomSampler(probs, n_samples, replacement=True)
edge_loader = DataLoader(dataset, batch_size=1000, sampler=sampler)

########################################################################################################################
#                                                       TRAIN                                                          #
########################################################################################################################

# trainer = SingleVisTrainer(model, criterion, optimizer, lr_scheduler, edge_loader=edge_loader, DEVICE=DEVICE)

trainer = HybridVisTrainer(model, criterion, optimizer, lr_scheduler,edge_loader=edge_loader, DEVICE=DEVICE)

t2=time.time()
trainer.train(PATIENT, MAX_EPOCH)
t3 = time.time()

save_dir = data_provider.model_path

trainer.save(save_dir=save_dir, file_name="{}".format(VIS_MODEL_NAME))


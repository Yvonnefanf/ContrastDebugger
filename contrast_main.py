import torch
import sys
import os
import time
import numpy as np
import argparse

from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from umap.umap_ import find_ab_params

from singleVis.custom_weighted_random_sampler import CustomWeightedRandomSampler
from singleVis.SingleVisualizationModel import VisModel
from singleVis.losses import UmapLoss, ReconstructionLoss, SingleVisLoss
from singleVis.edge_dataset import DataHandler
from singleVis.trainer import SingleVisTrainer
from singleVis.data import NormalDataProvider
from singleVis.spatial_edge_constructor import kcSpatialAlignmentEdgeConstructor
# from singleVis.temporal_edge_constructor import GlobalTemporalEdgeConstructor
from singleVis.alignment_edge_constructor import LocalAlignmentEdgeConstructor
from singleVis.projector import TimeVisProjector
from singleVis.eval.evaluator import Evaluator

########################################################################################################################
#                                                    VISUALIZATION SETTING                                             #
########################################################################################################################
VIS_METHOD= "TimeVis"
########################################################################################################################
#                                                     LOAD PARAMETERS                                                  #
########################################################################################################################
parser = argparse.ArgumentParser(description='Process hyperparameters...')
parser.add_argument('--content_path', type=str)
# reference data path
parser.add_argument('--reference_path', type=str)

args = parser.parse_args()

CONTENT_PATH = args.content_path
REF_PATH = args.reference_path
sys.path.append(REF_PATH)
sys.path.append(CONTENT_PATH)


from config import config



# record output information
now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time())) 
sys.stdout = open(os.path.join(CONTENT_PATH, now+".txt"), "w")

SETTING = config["SETTING"]
CLASSES = config["CLASSES"]
DATASET = config["DATASET"]
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
PREPROCESS = VISUALIZATION_PARAMETER["PREPROCESS"]
LAMBDA = VISUALIZATION_PARAMETER["LAMBDA"]
B_N_EPOCHS = VISUALIZATION_PARAMETER["BOUNDARY"]["B_N_EPOCHS"]
L_BOUND = VISUALIZATION_PARAMETER["BOUNDARY"]["L_BOUND"]
INIT_NUM = VISUALIZATION_PARAMETER["INIT_NUM"]
ALPHA = VISUALIZATION_PARAMETER["ALPHA"]
BETA = VISUALIZATION_PARAMETER["BETA"]
MAX_HAUSDORFF = VISUALIZATION_PARAMETER["MAX_HAUSDORFF"]
# HIDDEN_LAYER = VISUALIZATION_PARAMETER["HIDDEN_LAYER"]
ENCODER_DIMS = VISUALIZATION_PARAMETER["ENCODER_DIMS"]
DECODER_DIMS = VISUALIZATION_PARAMETER["DECODER_DIMS"]
S_N_EPOCHS = VISUALIZATION_PARAMETER["S_N_EPOCHS"]
T_N_EPOCHS = VISUALIZATION_PARAMETER["T_N_EPOCHS"]
N_NEIGHBORS = VISUALIZATION_PARAMETER["N_NEIGHBORS"]
PATIENT = VISUALIZATION_PARAMETER["PATIENT"]
MAX_EPOCH = VISUALIZATION_PARAMETER["MAX_EPOCH"]

VIS_MODEL_NAME = 'contrast'
EVALUATION_NAME = VISUALIZATION_PARAMETER["EVALUATION_NAME"]

SEGMENTS = [(EPOCH_START, EPOCH_END)]



# contrast vis -> reference


# define hyperparameters
DEVICE = torch.device("cuda:{}".format(GPU_ID) if torch.cuda.is_available() else "cpu")

import Model.model as subject_model
net = eval("subject_model.{}()".format(NET))

########################################################################################################################
#                                                    TRAINING SETTING                                                  #
########################################################################################################################

##### load reference data ##############

ref_provider = NormalDataProvider(REF_PATH, net, EPOCH_START, EPOCH_END, EPOCH_PERIOD, split=-1, device=DEVICE, classes=CLASSES,verbose=1)

data_provider = NormalDataProvider(CONTENT_PATH, net, EPOCH_START, EPOCH_END, EPOCH_PERIOD, split=-1, device=DEVICE, classes=CLASSES,verbose=1)
if PREPROCESS:
    data_provider._meta_data()
    if B_N_EPOCHS >0:
        data_provider._estimate_boundary(LEN//10, l_bound=L_BOUND)

# model = SingleVisualizationModel(input_dims=512, output_dims=2, units=256, hidden_layer=HIDDEN_LAYER)
model = VisModel(ENCODER_DIMS, DECODER_DIMS)

negative_sample_rate = 5
min_dist = .1
_a, _b = find_ab_params(1.0, min_dist)
umap_loss_fn = UmapLoss(negative_sample_rate, DEVICE, _a, _b, repulsion_strength=1.0)
recon_loss_fn = ReconstructionLoss(beta=1.0)
criterion = SingleVisLoss(umap_loss_fn, recon_loss_fn, lambd=LAMBDA)
# projector = Projector(vis_model=model, content_path=CONTENT_PATH, segments=SEGMENTS, device=DEVICE)
ref_projector = TimeVisProjector(vis_model=model, content_path=REF_PATH, vis_model_name=VIS_MODEL_NAME, device=DEVICE)
projector = TimeVisProjector(vis_model=model, content_path=CONTENT_PATH, vis_model_name=VIS_MODEL_NAME, device=DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=.1)

t0 = time.time()

spatial_cons = kcSpatialAlignmentEdgeConstructor(data_provider=data_provider, init_num=INIT_NUM, s_n_epochs=S_N_EPOCHS, b_n_epochs=B_N_EPOCHS, n_neighbors=N_NEIGHBORS, MAX_HAUSDORFF=MAX_HAUSDORFF, ALPHA=ALPHA, BETA=BETA, ref_provider=ref_provider)
s_edge_to, s_edge_from, s_probs, feature_vectors, time_step_nums, time_step_idxs_list, knn_indices, sigmas, rhos, attention, ref_edge_to, ref_edge_from, ref_weight, ref_feature_vectors, ref_knn_indices, ref_sigmas, ref_rhos, ref_attention = spatial_cons.construct()
alignment_cons = LocalAlignmentEdgeConstructor(X=feature_vectors, time_step_nums=time_step_nums, sigmas=sigmas, rhos=rhos, n_neighbors=N_NEIGHBORS, n_epochs=T_N_EPOCHS, persistent=2,time_step_idxs_list=time_step_idxs_list, knn_indices=knn_indices, ref_edge_to=ref_edge_to, ref_edge_from=ref_edge_from, ref_weight = ref_weight, ref_X=ref_feature_vectors, ref_knn_indices=ref_knn_indices, ref_sigmas=ref_sigmas, ref_rhos=ref_rhos, ref_provider=ref_provider, ref_projector=ref_projector)
t_edge_to, t_edge_from, t_probs = alignment_cons.construct()
# alignment_cons = GlobalAlignmentContrastEdgeConstructor(X = feature_vectors)
t1 = time.time()

edge_to = np.concatenate((s_edge_to, t_edge_to),axis=0)
edge_from = np.concatenate((s_edge_from, t_edge_from), axis=0)
probs = np.concatenate((s_probs, t_probs), axis=0)
probs = probs / (probs.max()+1e-3)
eliminate_zeros = probs>1e-3
edge_to = edge_to[eliminate_zeros]
edge_from = edge_from[eliminate_zeros]
probs = probs[eliminate_zeros]

dataset = DataHandler(edge_to, edge_from, feature_vectors, attention)
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

trainer = SingleVisTrainer(model, criterion, optimizer, lr_scheduler, edge_loader=edge_loader, DEVICE=DEVICE)

t2=time.time()
trainer.train(PATIENT, MAX_EPOCH)
t3 = time.time()

save_dir = data_provider.model_path
trainer.record_time(save_dir, "time_{}_{}.json".format(VIS_METHOD, VIS_MODEL_NAME), "complex_construction", t1-t0)
trainer.record_time(save_dir, "time_{}_{}.json".format(VIS_METHOD, VIS_MODEL_NAME), "training", t3-t2)
trainer.save(save_dir=save_dir, file_name="{}".format(VIS_MODEL_NAME))


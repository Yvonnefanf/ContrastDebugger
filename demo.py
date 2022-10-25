"""This serve as an template of Visualization in pytorch."""
import torch
import sys
import os
import json
import time
import numpy as np
import argparse

from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from umap.umap_ import find_ab_params

from singleVis.custom_weighted_random_sampler import CustomWeightedRandomSampler

# Define different visualization components in the following file and import them
from singleVis.SingleVisualizationModel import VisModel
from singleVis.losses import Loss # and other Losses 
from singleVis.edge_dataset import DataHandlerAbstractClass
from singleVis.trainer import TrainerAbstractClass
from singleVis.data import DataProviderAbstractClass
from singleVis.spatial_edge_constructor import SpatialEdgeConstructor, SpatialEdgeConstructorAbstractClass
from singleVis.projector import ProjectorAbstractClass
from singleVis.eval.evaluator import EvaluatorAbstractClass

########################################################################################################################
#                                                     VIS PARAMETERS                                                   #
########################################################################################################################
VIS_METHOD = "your visualization name"

########################################################################################################################
#                                                     LOAD PARAMETERS                                                  #
########################################################################################################################
parser = argparse.ArgumentParser(description='Process hyperparameters...')
parser.add_argument('--content_path', type=str)
args = parser.parse_args()

CONTENT_PATH = args.content_path
sys.path.append(CONTENT_PATH)

# Define your dataset hyperparameters in config
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
S_N_EPOCHS = VISUALIZATION_PARAMETER["S_N_EPOCHS"]
T_N_EPOCHS = VISUALIZATION_PARAMETER["T_N_EPOCHS"]
N_NEIGHBORS = VISUALIZATION_PARAMETER["N_NEIGHBORS"]
PATIENT = VISUALIZATION_PARAMETER["PATIENT"]
MAX_EPOCH = VISUALIZATION_PARAMETER["MAX_EPOCH"]

VIS_MODEL_NAME = VISUALIZATION_PARAMETER["VIS_MODEL_NAME"]
EVALUATION_NAME = VISUALIZATION_PARAMETER["EVALUATION_NAME"]

DEVICE = torch.device("cuda:{}".format(GPU_ID) if torch.cuda.is_available() else "cpu")

import Model.model as subject_model
net = eval("subject_model.{}()".format(NET))

########################################################################################################################
#                                                    TRAINING SETTING                                                  #
########################################################################################################################
# Define data_provider
data_provider = DataProviderAbstractClass(CONTENT_PATH, net, EPOCH_START, EPOCH_END, EPOCH_PERIOD, split=-1)
if PREPROCESS:
    data_provider._meta_data()
    if B_N_EPOCHS >0:
        data_provider._estimate_boundary(LEN//10, l_bound=L_BOUND)

# Define your own visualization models
model = VisModel(encoder_dims=[100,20,2], decoder_dims=[2,50,100])  # placeholder
# Define your own Losses
criterion = Loss()
# Define your own Projector
projector = ProjectorAbstractClass()

# Define your own training parameters
optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=.1)
# Define your own Edge dataset
t0 = time.time()
spatial_cons = SpatialEdgeConstructorAbstractClass(data_provider)
edge_to, edge_from, probs, feature_vectors = spatial_cons.construct()
t1 = time.time()

# remove edges with low weight (optional) 
probs = probs / (probs.max()+1e-3)
eliminate_zeros = probs>1e-3
edge_to = edge_to[eliminate_zeros]
edge_from = edge_from[eliminate_zeros]
probs = probs[eliminate_zeros]

# save result
save_dir = os.path.join(data_provider.model_path, "{}_time_{}.json".format(VIS_METHOD, VIS_MODEL_NAME))
if not os.path.exists(save_dir):
    evaluation = dict()
else:
    f = open(save_dir, "r")
    evaluation = json.load(f)
    f.close()

evaluation["complex_construction"] = round(t1-t0, 3)
with open(save_dir, 'w') as f:
    json.dump(evaluation, f)
print("constructing {} edge dataset in {:.1f} seconds.".format(VIS_METHOD, t1-t0))

dataset = DataHandlerAbstractClass(edge_to, edge_from, feature_vectors)

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

trainer = TrainerAbstractClass(model, criterion, optimizer, lr_scheduler, edge_loader, DEVICE)

t2=time.time()
trainer.train(PATIENT, MAX_EPOCH)
t3 = time.time()

# save result
save_file = os.path.join(data_provider.model_path, "{}_time_{}.json".format(VIS_METHOD, VIS_MODEL_NAME))
if not os.path.exists(save_file):
    evaluation = dict()
else:
    f = open(save_file, "r")
    evaluation = json.load(f)
    f.close()

evaluation["training"] = round(t3-t2, 3)
with open(save_file, 'w') as f:
    json.dump(evaluation, f)

save_dir = "path/to/model"
trainer.save(save_dir=save_dir, file_name="{}".format(VIS_MODEL_NAME))

########################################################################################################################
#                                                      VISUALIZATION                                                   #
########################################################################################################################
from singleVis.visualizer import VisualizerAbstractClass
# Define your Visualizer
vis = VisualizerAbstractClass(data_provider, projector)
save_dir = "path/to/generated/imgs"
os.makedirs(save_dir)

for epoch in range(EPOCH_START, EPOCH_END+1, EPOCH_PERIOD):
    vis.savefig(epoch, path=os.path.join(save_dir, "{}_{}_{}.png".format(DATASET, epoch, VIS_METHOD)))

    
########################################################################################################################
#                                                       EVALUATION                                                     #
########################################################################################################################
eval_epochs = range(EPOCH_START, EPOCH_END, EPOCH_PERIOD)
# Define your evaluator
evaluator = EvaluatorAbstractClass(data_provider, projector)
for eval_epoch in eval_epochs:
    evaluator.save_epoch_eval(eval_epoch, 15, temporal_k=5, file_name="{}".format(EVALUATION_NAME))

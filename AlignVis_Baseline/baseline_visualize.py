\
import torch
import sys
import os
sys.path.append("..")

import argparse


from singleVis.data import NormalDataProvider
from singleVis.projector import Projector
from singleVis.SingleVisualizationModel import VisModel
# from singleVis.SingleVisualizationModel import SingleVisualizationModel
########################################################################################################################
#                                                     LOAD PARAMETERS                                                  #
########################################################################################################################
parser = argparse.ArgumentParser(description='Process hyperparameters...')
parser.add_argument('--content_path', type=str)
args = parser.parse_args()

CONTENT_PATH = args.content_path
sys.path.append(CONTENT_PATH)
from config import config

VIS_METHOD= "TimeVis"

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


VIS_MODEL_NAME = VISUALIZATION_PARAMETER["VIS_MODEL_NAME"]
EVALUATION_NAME = VISUALIZATION_PARAMETER["EVALUATION_NAME"]

S_N_EPOCHS = VISUALIZATION_PARAMETER["S_N_EPOCHS"]
T_N_EPOCHS = VISUALIZATION_PARAMETER["T_N_EPOCHS"]
N_NEIGHBORS = VISUALIZATION_PARAMETER["N_NEIGHBORS"]
PATIENT = VISUALIZATION_PARAMETER["PATIENT"]
MAX_EPOCH = VISUALIZATION_PARAMETER["MAX_EPOCH"]

SEGMENTS = [(EPOCH_START, EPOCH_END)]



ENCODER_DIMS = VISUALIZATION_PARAMETER["ENCODER_DIMS"]
DECODER_DIMS = VISUALIZATION_PARAMETER["DECODER_DIMS"]


model = VisModel(ENCODER_DIMS, DECODER_DIMS)
# model = SingleVisualizationModel(input_dims=512, output_dims=2, units=256, hidden_layer=HIDDEN_LAYER)
# define hyperparameters
# DEVICE = torch.device("cuda:{}".format(GPU_ID) if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")
import Model.model as subject_model
# net = resnet18()
net = eval("subject_model.{}()".format(NET))

data_provider = NormalDataProvider(CONTENT_PATH, net, EPOCH_START, EPOCH_END, EPOCH_PERIOD, split=-1, device=DEVICE, classes=CLASSES,verbose=1)
if PREPROCESS:
    data_provider.initialize(LEN//10, l_bound=L_BOUND)
projector = Projector(vis_model=model, content_path=CONTENT_PATH, segments=SEGMENTS, device=DEVICE,vis_name="baseline")

from singleVis.visualizer import visualizer
vis = visualizer(data_provider, projector, 200, 'tab10')
save_dir = os.path.join(CONTENT_PATH , "baseline")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for i in range(EPOCH_START, EPOCH_END+1, EPOCH_PERIOD):
    vis.savefig(i, path=os.path.join(save_dir, "{}_{}_{}.png".format(DATASET, i, VIS_METHOD)))
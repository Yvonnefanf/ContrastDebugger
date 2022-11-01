from sklearn import datasets
import torch
import sys
import os

import argparse

from umap.umap_ import find_ab_params

from singleVis.SingleVisualizationModel import SingleVisualizationModel
from singleVis.data import NormalDataProvider
from singleVis.eval.evaluator import Evaluator
from singleVis.projector import Projector


########################################################################################################################
#                                                     LOAD PARAMETERS                                                  #
########################################################################################################################


parser = argparse.ArgumentParser(description='Process hyperparameters...')
parser.add_argument('--content_path', type=str)
parser.add_argument('-d','--dataset', choices=['online','cifar10', 'mnist', 'fmnist', 'cifar10_full', 'mnist_full', 'fmnist_full'])
parser.add_argument('-p',"--preprocess", choices=[0,1], default=0)
parser.add_argument('-g',"--gpu_id", type=int, choices=[0,1,2,3], default=0)
args = parser.parse_args()

CONTENT_PATH = args.content_path
sys.path.append(CONTENT_PATH)
from config import config


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

# define hyperparameters
DEVICE = torch.device("cuda:{}".format(GPU_ID) if torch.cuda.is_available() else "cpu")



content_path = CONTENT_PATH
sys.path.append(content_path)

import Model.model as subject_model
# net = resnet18()
net = eval("subject_model.{}()".format(NET))
classes = ("airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


########################################################################################################################
#                                                    TRAINING SETTING                                                  #
########################################################################################################################

data_provider = NormalDataProvider(CONTENT_PATH, net, EPOCH_START, EPOCH_END, EPOCH_PERIOD, split=-1, device=DEVICE, classes=CLASSES,verbose=1)
if PREPROCESS:
    data_provider.initialize(LEN//10, l_bound=L_BOUND)

model = SingleVisualizationModel(input_dims=512, output_dims=2, units=256, hidden_layer=HIDDEN_LAYER)
projector = Projector(vis_model=model, content_path=CONTENT_PATH, segments=SEGMENTS, device=DEVICE)



########################################################################################################################
#                                                       EVALUATION                                                     #
########################################################################################################################
EVAL_EPOCH_DICT = {
    "mnist":[1,2,5,10,13,16,20],
    "fmnist":[1,2,6,11,25,30,36,50],
    "cifar10":list(range(1, 201))
}

eval_epochs = EVAL_EPOCH_DICT[DATASET]
evaluator = Evaluator(data_provider, projector)

for eval_epoch in eval_epochs:
    evaluator.save_epoch_eval(eval_epoch, 10, temporal_k=3, file_name="test_evaluation_timevis")
    evaluator.save_epoch_eval(eval_epoch, 15, temporal_k=5, file_name="test_evaluation_timevis")
    evaluator.save_epoch_eval(eval_epoch, 20, temporal_k=7,file_name="test_evaluation_timevis")
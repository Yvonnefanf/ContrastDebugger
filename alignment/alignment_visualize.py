import torch
import sys
import os

import argparse


from singleVis.data import NormalDataProvider
from singleVis.projector import Projector
from singleVis.SingleVisualizationModel import VisModel
########################################################################################################################
#                                                     LOAD PARAMETERS                                                  #
########################################################################################################################
parser = argparse.ArgumentParser(description='Process hyperparameters...')
parser.add_argument('--target_path', type=str)
parser.add_argument('--reference_path', type=str,default='/home/yifan/dataset/clean/pairflip/cifar10/0')
args = parser.parse_args()

TARGET_PATH = args.target_path
REFERENCE_PARH = args.reference_path

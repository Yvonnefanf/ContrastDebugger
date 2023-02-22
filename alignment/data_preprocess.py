"""The DataProvider class serve as a helper module for retriving subject model data"""
from abc import ABC, abstractmethod

import os
import gc
import time

from singleVis.utils import *
from singleVis.eval.evaluate import evaluate_inv_accu
import copy

"""
DataContainder module
1. prepare data
2. estimate boundary
3. provide data
"""
class DataPreProcessAbstractClass(ABC):
    
    def __init__(self, content_path, model, epoch_start, epoch_end, epoch_period):
        self.mode = "abstract"
        self.content_path = content_path
        self.model = model
        self.s = epoch_start
        self.e = epoch_end
        self.p = epoch_period
        
    @property
    @abstractmethod
    def train_num(self):
        pass

    @property
    @abstractmethod
    def test_num(self):
        pass

    @abstractmethod
    def _meta_data(self):
        pass



class DataPreProcess(DataPreProcessAbstractClass):
    def __init__(self, content_path, model, epoch_start, epoch_end, epoch_period, split, device, classes):
        self.content_path = content_path
        self.model = model
        self.s = epoch_start
        self.e = epoch_end
        self.p = epoch_period
        self.split = split
        self.DEVICE = device
        self.classes = classes

        self.model_path = os.path.join(self.content_path, "Model")
 

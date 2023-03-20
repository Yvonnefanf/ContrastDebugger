from abc import ABC, abstractmethod
from AlignVis.DataInit import DataInit
import numpy as np
import os
import json
from singleVis.utils import *

class AlignmentBoundaryGeneratorAbstractClass(ABC):
    @abstractmethod
    def __init__(self, REF_PATH, REF_CONFIG_PATH, TAR_PATH, TAR_CONFIG_PATH,  REF_EPOCH, TAR_EPOCH, DEVICE, * args, **kawargs):
        pass

class AlignmentBoundaryGenerator(AlignmentBoundaryGeneratorAbstractClass):
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
    
    def get_ref_boundary(self):
        points, selectedindx = self.get_boundary_point(self.REF_PATH, self.ref_model, self.REF_EPOCH, self.DEVICE)

        return points,selectedindx

        
    def get_boundary_point(self, DEVICE,l_bound=0.6,num_adv_eg=5000):
        ## Loading training data
        training_data_path = os.path.join(self.ref_provider.content_path, "Training_data")
        training_data = torch.load(os.path.join(training_data_path, "training_dataset_data.pth"),
                                   map_location=DEVICE)
        index_file = os.path.join(self.ref_provider.content_path,"Model", "Epoch_{:d}".format(self.REF_EPOCH), "index.json")
        index = load_labelled_data_index(index_file)
        training_data = training_data[index]
        ####

        ## Loading training data
        tar_training_data_path = os.path.join(self.tar_provider.content_path, "Training_data")
        tar_training_data = torch.load(os.path.join(tar_training_data_path, "training_dataset_data.pth"),
                                   map_location=DEVICE)
        tar_index_file = os.path.join(self.tar_provider.content_path,"Model", "Epoch_{:d}".format(self.TAR_EPOCH), "index.json")
        tar_index = load_labelled_data_index(tar_index_file)
        tar_training_data = tar_training_data[tar_index]

        border_points, tar_border,_, _ = get_aligned_border_points(model=self.ref_model, input_x=training_data, tar_model = self.tar_model, tar_input_x = tar_training_data, device=DEVICE, l_bound=l_bound, num_adv_eg=num_adv_eg, lambd=0.05, verbose=0)

        return border_points, tar_border


import torch
import sys
sys.path.append("..")
from singleVis.data import NormalDataProvider
from scipy.special import softmax
import numpy as np
import os
import json

class DataInit():
    def __init__(self, config_path, content_path, cur_epoch):
        self.content_path = content_path
        self.config_path = config_path
        self.cur_epoch = cur_epoch

    def get_conf(self, predction):
        scores = np.amax(softmax(predction, axis=1), axis=1)
        return scores

    def getData(self):
        sys.path.append(self.content_path)
        sys.path.append(self.config_path)

        with open(os.path.join(self.config_path,'config.json'), 'r') as f:
            config = json.load(f)
        ####### parameter ######
        TRAINING_PARAMETER = config["TRAINING"]
        NET = TRAINING_PARAMETER["NET"]
        CLASSES = config["CLASSES"]
        GPU_ID = config["GPU"]
        EPOCH_START = config["EPOCH_START"]
        EPOCH_END = config["EPOCH_END"]
        EPOCH_PERIOD = config["EPOCH_PERIOD"]
        DEVICE = torch.device("cuda:{}".format(GPU_ID) if torch.cuda.is_available() else "cpu")
        print('NET',NET)
        import Model.model as subject_model
        net = eval("subject_model.{}()".format(NET))

        # sys.path.remove()
        sys.path.remove(self.content_path)

        data_provider = NormalDataProvider(self.content_path, net,EPOCH_START, EPOCH_END, EPOCH_PERIOD, split=-1, device=DEVICE, classes=CLASSES,verbose=1)
        train_data = data_provider.train_representation(self.cur_epoch).squeeze()
        prediction = data_provider.get_pred(self.cur_epoch, train_data)
        prediction_label = prediction.argmax(axis=1)
        confidence = self.get_conf(prediction)

        return net, data_provider, train_data, prediction, prediction_label, confidence
        






 

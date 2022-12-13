import sys
import os
import numpy as np
import torch
import json
import collections
CONTENT_PATH = "/home/yifan/dataset/clean/pairflip/cifar10/0"

sys.path.append(CONTENT_PATH)
from config import config
TRAINING_PARAMETER = config["TRAINING"]
NET = TRAINING_PARAMETER["NET"]
LEN = TRAINING_PARAMETER["train_num"]
EPOCH_START = config["EPOCH_START"]
EPOCH_END = config["EPOCH_END"]
EPOCH_PERIOD = config["EPOCH_PERIOD"]
import Model.model as subject_model
net = eval("subject_model.{}()".format(NET))
model_path = os.path.join(CONTENT_PATH, "Model")

index = list(range(LEN))
for n_epoch in range(EPOCH_START,EPOCH_END + EPOCH_PERIOD, EPOCH_PERIOD):
    m_path = os.path.join(model_path, "epoch={:03d}.ckpt".format(n_epoch-1))
    save_param =  torch.load(m_path)
    state_dict = save_param["state_dict"]

    order_dict =  collections.OrderedDict()
    for key in state_dict.keys():
        new_key = key.replace("model.","")
        order_dict[new_key] = state_dict[key]

    save_dir = os.path.join(model_path, "Epoch_{}".format((n_epoch) // EPOCH_PERIOD)) 
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    torch.save(order_dict, os.path.join(save_dir, "subject_model.pth"))
    with open(os.path.join(save_dir, "index.json"),"w") as f:
        json.dump(index, f)
    os.remove(m_path)


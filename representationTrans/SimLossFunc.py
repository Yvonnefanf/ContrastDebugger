import torch
import torch.nn as nn
import torch.nn.functional as func

class SimLossFunc(nn.Module):
    def __init__(self, sim, ref_2d, X, projector ):
        super(SimLossFunc, self).__init__()
        self.sim = sim
        self.ref_2d = ref_2d
        self.X = X
        self.projector = projector
        return
    def forward(self,x,y):
        print(self,x,y)
        print(self.sim,self.X)
        loss= 1
        return loss
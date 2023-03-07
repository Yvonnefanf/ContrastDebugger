from abc import ABC, abstractmethod
import os
import sys
sys.path.append("..")
import torch
import numpy as np
from CKA_utils.CKA import CKA, CudaCKA, CCA_val


class SequenceAlignmentAbstractClass(ABC):
    @abstractmethod
    def __init__(self, ref_data_provider, tar_data_provider, ref_EPOCH, tar_EPOCH, * args, **kawargs):
        pass

class SequenceAlignment(SequenceAlignmentAbstractClass):
    def __init__(self, ref_data_provider, tar_data_provider,ref_MAX_EPOCH, tar_MAx_EPOCH):
        self.ref_data_provider = ref_data_provider
        self.tar_data_provider = tar_data_provider
        self.ref_MAX_EPOCH = ref_MAX_EPOCH
        self.tar_MAx_EPOCH = tar_MAx_EPOCH
    


    def get_alignment_list_useGPU(self):
        device = torch.device('cuda:1')
        cuda_cka = CudaCKA(device)
        X = torch.Tensor(self.ref_data_provider.train_representation(200))
        Y = torch.Tensor(self.tar_data_provider.train_representation(200))
        print('Linear CKA, between X and Y: {}'.format(cuda_cka.linear_CKA(X, Y)))
    
    def get_alignment_list(self):
        np_cka = CKA()
        X = self.ref_data_provider.train_representation(200)
        Y = self.tar_data_provider.train_representation(200)
        print('Linear CKA, between X and Y: {}'.format(np_cka.linear_CKA(X, Y)))
    
    def get_alignment_list_byCCA(self,ref_EPOCH, tar_EPOCH):
        np_cca = CCA_val()
        X = self.ref_data_provider.train_representation(ref_EPOCH)
        Y = self.tar_data_provider.train_representation(tar_EPOCH)
        
        print('Linear CCA, between ref epoch{} and target epoch{} : {}'.format(ref_EPOCH, tar_EPOCH, np_cca.cal_CCA(X, Y)))


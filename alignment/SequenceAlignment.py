from abc import ABC, abstractmethod
import os
import sys
sys.path.append("..")
import torch
import numpy as np
from CKA_utils.CKA import CKA, CudaCKA, CCA_val
from alignment.ReferenceGenerator import ReferenceGenerator


class SequenceAlignmentAbstractClass(ABC):
    @abstractmethod
    def __init__(self, ref_data_provider, tar_data_provider, ref_EPOCH_START, ref_EPOCH_END, tar_EPOCH_START, tar_EPOCH_END, * args, **kawargs):
        self.ref_data_provider = ref_data_provider
        self.tar_data_provider = tar_data_provider
        pass

class SequenceAlignment(SequenceAlignmentAbstractClass):
    def __init__(self, ref_data_provider, tar_data_provider, ref_EPOCH_START, ref_EPOCH_END, tar_EPOCH_START, tar_EPOCH_END):
        """
        Parameter
        --------------
        ref_data_provider :  data.DataProvider 
            reference data provider
        tar_data_provider : data.DataProvider 
            target data prvider
        ref_EPOCH_START: int 
            reference training process alignment start epoch
        ref_EPOCH_END: int 
            reference training process alignment end epoch
        tar_EPOCH_START: int 
            target training process alignment start epoch
        tar_EPOCH_END: int 
            target training process alignment end epoch
        --------------
        """

        self.ref_data_provider = ref_data_provider
        self.tar_data_provider = tar_data_provider
        self.ref_EPOCH_START = ref_EPOCH_START
        self.ref_EPOCH_END = ref_EPOCH_END
        self.tar_EPOCH_START = tar_EPOCH_START
        self.tar_EPOCH_END = tar_EPOCH_END

       
    def getAlignment(self, model,device,mse_val_for_diff=15.0,mse_val_for_same=1.5,conf_val_for_diff=0.3,conf_val_for_same=0.05,min_adsolute_sample_num=300,align_min_CKA_val=0.91):
        """
        get aligbment list

        Parameter
        --------------
        model: model
        device: device
        mse_val_for_diff: float
            mse value , if the mse value between samplek in ref and tar > mse_val_for_diff, 
            the sample is considered diff_list
        mse_val_for_same: float
            mse value , if the mse value between samplek in ref and tar < mse_val_for_same, 
            the sample is considered same_list
        conf_val_for_diff: float
            confidence value, if the confidence value 
            |ref_con_samplek - tar_con_samplek| > conf_val_for_diff , 
            the sample is considered confidence_diff_list
        conf_val_for_same: float
            confidence value, if the confidence value 
            |ref_con_samplek - tar_con_samplek| < conf_val_for_same , 
            the sample is considered confidence_same_list
        min_adsolute_sample_num: int
            the minimun number of alignmnet subset, 
            if the we only can find less number alinmnet samples, 
            we skip this epoch(CKA = 0)
        align_min_CKA_val: float
            the minimun value of CKA, if the CKA < align_min_CKA_val,
            we think it can not be alignmnet
        --------------
        """
        np_cka = CKA()
        alignmentList = []
        absolute_align_sample_list = []
        for i in range(self.tar_EPOCH_START, self.tar_EPOCH_END,1):
            ad_align_list = []
            CKA_val_list = []
            for j in range(self.ref_EPOCH_START, self.ref_EPOCH_END, 1):
                tar_cur_epoch = self.tar_EPOCH_END - i + self.tar_EPOCH_START
                ref_cur_epoch = self.ref_EPOCH_END - j + self.ref_EPOCH_START
                tar_representation = self.tar_data_provider.train_representation(tar_cur_epoch)
                ref_representation = self.ref_data_provider.train_representation(ref_cur_epoch)
                referenceGenerator = ReferenceGenerator(ref_provider=self.ref_data_provider, tar_provider=self.tar_data_provider,REF_EPOCH=ref_cur_epoch,TAR_EPOCH=tar_cur_epoch,model=model,DEVICE=device)
                absolute_alignment_indicates,predict_label_diff_indicates,predict_confidence_Diff_indicates,high_distance_indicates = referenceGenerator.subsetClassify(mse_val_for_diff,mse_val_for_same,conf_val_for_diff,conf_val_for_same)
                ad_align_list.insert(0, absolute_alignment_indicates)
                cka_val = 0
                align = { 'ref': -1 , 'tar': tar_cur_epoch, 'cka':cka_val, 'absolute_alignment_set_num':-1}
                if len(absolute_alignment_indicates) > min_adsolute_sample_num:
                    cka_val = np_cka.kernel_CKA(ref_representation[absolute_alignment_indicates], tar_representation[absolute_alignment_indicates])
                    if cka_val  > align_min_CKA_val:
                        self.ref_EPOCH_END = ref_cur_epoch
                        align = { 'ref': self.ref_EPOCH_END , 'tar': tar_cur_epoch, 'cka': cka_val, 'absolute_alignment_set_num':len(absolute_alignment_indicates)}
                        print('CKA between reference epoch: ',ref_cur_epoch,' and target epoch: ', tar_cur_epoch, 'is :', cka_val)
                        break

                
            # CKA_val_list.insert(0, cka_val)
            # absolute_align_sample_list.insert(0, ad_align_list)
            # high_CKA_indicates = []
            # for k in range(len(CKA_val_list)):
            #     if CKA_val_list[k] > align_min_CKA_val:
            #         high_CKA_indicates.append(k)

            # if len(high_CKA_indicates):
            #     self.ref_EPOCH_END = high_CKA_indicates[len(high_CKA_indicates)-1] + self.ref_EPOCH_START + 1
            #     align = { 'ref': self.ref_EPOCH_END , 'tar': tar_cur_epoch}
            # else:
            #     print('ref epoch', ref_cur_epoch, 'tar epoch', tar_cur_epoch, 'can not be align')
                # align = {'ref': -1, 'tar': tar_cur_epoch}
            
            alignmentList.append(align)

        
        return alignmentList, absolute_align_sample_list

            




    
    


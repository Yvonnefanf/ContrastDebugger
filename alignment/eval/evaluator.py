from abc import ABC, abstractmethod
import os
import json
import numpy as np
from representationTrans.eval.evaluate import *
from scipy.special import softmax


class EvaluatorAbstractClass(ABC):
    def __init__(self, ref_provider, tar_provider, data, projector, *args, **kwargs):
        self.ref_provider = ref_provider
        self.tar_provider = tar_provider

        self.projector = projector

    @abstractmethod
    def evaluate_predcition(labels, ori_pred, new_pred):
        pass

class Evaluator(EvaluatorAbstractClass):
    def __init__(self, ref_provider, tar_provider, projector, verbose=1,*args, **kwargs):
        self.ref_provider = ref_provider
        self.tar_provider = tar_provider
        self.verbose =verbose
        self.projector = projector

    def evaluate_predcition(self, epoch, data):
        """
        get inverse confidence for a single point
        :param epoch: int
        :param data: numpy.ndarray
        :return l: boolean, whether reconstruction data have the same prediction
        :return conf_diff: float, (0, 1), confidence difference
        """
        embedding = self.projector.batch_project(epoch, data)
        recon = self.projector.batch_inverse(epoch, embedding)

        ori_pred = self.tar_provider.get_pred(epoch, data)
        new_pred = self.tar_provider.get_pred(epoch, recon)
        ori_pred = softmax(ori_pred, axis=1)
        new_pred = softmax(new_pred, axis=1)

        old_label = ori_pred.argmax(-1)
        new_label = new_pred.argmax(-1)
        l = old_label == new_label

        old_conf = [ori_pred[i, old_label[i]] for i in range(len(old_label))]
        new_conf = [new_pred[i, old_label[i]] for i in range(len(old_label))]
        old_conf = np.array(old_conf)
        new_conf = np.array(new_conf)

        conf_diff = old_conf - new_conf
        return l, conf_diff
    
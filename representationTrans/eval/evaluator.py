from abc import ABC, abstractmethod
import os
import json
import numpy as np
from representationTrans.eval.evaluate import *


class EvaluatorAbstractClass(ABC):
    def __init__(self, data_provider, projector, *args, **kwargs):
        self.data_provider = data_provider
        self.projector = projector

    @abstractmethod
    def evaluate_confidence(labels, ori_pred, new_pred):
        pass


class Evaluator(EvaluatorAbstractClass):
    def __init__(self, data_provider, projector, verbose=1,*args, **kwargs):
        self.data_provider = data_provider
        self.projector = projector
        self.verbose =verbose
    
    def evaluate_confidence(self, labels, ori_pred, new_pred):
        """
        the confidence difference between original data and reconstruction data
        :param labels: ndarray, shape(N,), the original prediction for each point
        :param ori_pred: ndarray, shape(N,10), the prediction of original data
        :param new_pred: ndarray, shape(N,10), the prediction of reconstruction data
        :return diff: float, the mean of confidence difference for each point
        """
        old_conf = [ori_pred[i, labels[i]] for i in range(len(labels))]
        new_conf = [new_pred[i, labels[i]] for i in range(len(labels))]
        old_conf = np.array(old_conf)
        new_conf = np.array(new_conf)
    
        diff = np.abs(old_conf - new_conf)
        # return diff.mean(), diff.max(), diff.min()
        return diff.mean()
    
    def eval_nn_test(self, epoch, train_data, embedding, n_neighbors):
        # train_data = self.data_provider.train_representation(epoch)
        # test_data = self.data_provider.test_representation(epoch)
        # fitting_data = np.concatenate((train_data, test_data), axis=0)
        # embedding = self.projector.batch_project(epoch, train_data)
        val = evaluate_proj_nn_perseverance_knn(train_data, embedding, n_neighbors=n_neighbors, metric="euclidean")
       
        print("#test# nn preserving : {:.2f}/{:d} in epoch {:d}".format(val, n_neighbors, epoch))
        return val


    def batch_inv_preserve(self, epoch, data):
        """
        get inverse confidence for a single point
        :param epoch: int
        :param data: numpy.ndarray
        :return l: boolean, whether reconstruction data have the same prediction
        :return conf_diff: float, (0, 1), confidence difference
        """
        embedding = self.projector.batch_project(epoch, data)
        recon = self.projector.batch_inverse(epoch, embedding)
    
        ori_pred = self.data_provider.get_pred(epoch, data)
        new_pred = self.data_provider.get_pred(epoch, recon)
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


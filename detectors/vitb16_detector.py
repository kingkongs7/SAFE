'''
# author: Yulin Yao
# email: yuliny@bupt.edu.cn
# date: 2025-0304
# description: Class for the Vitb16Detector
'''

import os
import datetime
import logging
import numpy as np
from sklearn import metrics
from typing import Union
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

from metrics.base_metrics_class import calculate_metrics_for_train

from .base_detector import AbstractDetector
from transformers import ViTModel
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC

logger = logging.getLogger(__name__)

@DETECTOR.register_module(module_name='vitb16')
class Vitb16Detector(AbstractDetector):
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.head = nn.Linear(768, 2)
        self.loss_func = self.build_loss(config)

    def build_backbone(self, config):
        return ViTModel.from_pretrained("/data/yyl/model/deepfake/DeepfakeBench-main/training/pretrained/google-vit-base-patch32-224-in21k")
    
    def build_loss(self, config):
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()
        return loss_func
    
    def features(self, data_dict: dict) -> torch.tensor:
        feat = self.backbone(data_dict['image'])['pooler_output']
        return feat
    
    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.head(features)
    
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        loss = self.loss_func(pred, label)
        loss_dict = {'overall': loss}
        return loss_dict
    
    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict
    
    def forward(self, data_dict: dict, inference=False) -> dict:
        # get the features by backbone
        features = self.features(data_dict)
        # get the prediction by classifier
        pred = self.classifier(features)
        # get the probability of the pred
        prob = torch.softmax(pred, dim=1)[:, 1]
        # build the prediction dict for each output
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features}
        return pred_dict
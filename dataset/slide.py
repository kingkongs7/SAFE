import sys
sys.path.append('.')

import os
import cv2
import lmdb
import torch
import random
import math

import numpy as np
import albumentations as A
import torchvision.transforms as T

from copy import deepcopy
from tqdm import tqdm
from multiprocessing import Pool
from scipy.spatial.distance import pdist
from scipy.fft import fft2, fftshift
from torch.utils.data import Sampler
from training.dataset.abstract_dataset import DeepfakeAbstractBaseDataset

class SlideSampler(Sampler):
    def __init__(self, label_list, stats_list, total_epochs=10, T0=1, T1=5, initial_ratio=0.3, final_ratio=1.0,
                 schedule="linear",
                 k=10, tau=0.5):
        super().__init__(None)
        self.schedule = schedule.lower()
        self.labels = np.array(label_list)
        self.stats  = np.array(stats_list, dtype=float)
        self.total_epochs = total_epochs
        self.initial_ratio = initial_ratio
        self.final_ratio = final_ratio
        self.T0 = T0
        self.T1 = T1
        self.k = k          
        self.tau = tau    

        # init index
        self.real_indices = np.where(self.labels == 0)[0]
        self.fake_indices = np.where(self.labels == 1)[0]

        self.current_epoch = 0
        self.losses = None

    def _ratio(self, x: float):
        r0, r1 = self.initial_ratio, self.final_ratio
        if self.schedule == "linear":
            return r0 + (r1 - r0) * x
        elif self.schedule == "convex":
            return r0 + (r1 - r0) * (x ** 2)
        elif self.schedule == "concave":
            return r0 + (r1 - r0) * (x ** 0.5)
        elif self.schedule == "cos":
            return r1 - (r1 - r0) * 0.5 * (1 + np.cos(np.pi * x))
        elif self.schedule == "sigmoid":
            s = 1 / (1 + np.exp(-self.k * (x - 0.5)))
            return r0 + (r1 - r0) * s
        elif self.schedule == "step":
            return r0 if x < self.tau else r1
        else:
            raise ValueError(f"Unknown schedule {self.schedule}")
        
    def _compute_gamma(self):
        if self.current_epoch <= self.T0:
            return 0.8
        elif self.current_epoch <= self.T1:
            return 0.8 - 0.5*(self.current_epoch - self.T0)/(self.T1 - self.T0)
        else:
            return 0.2

    def update_state(self, epoch, losses=None):
        self.current_epoch = epoch
        if losses == None:
            self.losses = None
        else:
            self.losses = np.array(losses)

    def __iter__(self):
        if self.losses  is None:
            raise RuntimeError("Losses not updated! Call update_state() first.")
        
        normalized_stats = (self.stats-np.min(self.stats)) / (np.max(self.stats) - np.min(self.stats) + 1e-6)
        normalized_dynamic = self.losses / (np.max(self.losses) + 1e-6)
        
        selection_scores = (1 - normalized_stats) + (1 - normalized_dynamic)
        
        if self.current_epoch <= self.T0:
            fake_scores = selection_scores[self.fake_indices]
            n_select = int(len(self.fake_indices) * self.initial_ratio)
            selected_fake = self.fake_indices[np.argsort(-fake_scores)[:n_select]] 
            
            indices = np.concatenate([self.real_indices,  selected_fake])
            
        elif self.current_epoch <= self.T1:
            progress = (self.current_epoch - self.T0) / (self.T1 - self.T0)
            current_ratio = self._ratio(progress)
            selected_count = int(len(self.labels) * current_ratio)
            indices = np.argsort(-selection_scores)[:selected_count]
            
        else:
            weights = np.exp(selection_scores) 
            weights /= weights.sum() 
            indices = np.random.choice(len(self.labels),  len(self.labels),  p=weights, replace=False)
        
        np.random.shuffle(indices) 
        return iter(indices.tolist()) 

    def __len__(self):
        return len(list(self.__iter__()))
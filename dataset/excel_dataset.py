import sys
sys.path.append('.')

import os
import cv2
import lmdb
import torch
import random
import math

import numpy as np
import pandas as pd
import albumentations as A
import torchvision.transforms as T

from PIL import Image
from copy import deepcopy
from tqdm import tqdm
from multiprocessing import Pool
from scipy.spatial.distance import pdist
from scipy.fft import fft2, fftshift
from torch.utils.data import Sampler
from training.dataset.abstract_dataset import DeepfakeAbstractBaseDataset

class ExcelDataset(DeepfakeAbstractBaseDataset):
    def __init__(self, config=None, mode='train'):
        super().__init__(config, mode)
        '''
        self.image_list = []
        self.label_list = []
        self.config = config
        self.mode = mode
        self.compression = config['compression']
        self.frame_num = config['frame_num'][mode]
        '''
        df = pd.read_csv('./scores_with_text.csv')

        self.stats_list = df['fft_var'].tolist()
        df["label"] = (df["class"] != "origin").astype(int)
        self.label_list = df['label'].tolist()
        self.data_dict = {
            'image': df['image_path'].tolist(),
            'label': df['label'].tolist(),
            'text_label': df['caption_long'].tolist()
        }

    def __getitem__(self, index, no_norm=False):
        """
        Returns the data point at the given index.

        Args:
            index (int): The index of the data point.

        Returns:
            A tuple containing the image tensor, the label tensor, the landmark tensor,
            and the mask tensor.
        """
        # Get the image paths and label
        image_paths = self.data_dict['image'][index]
        label = self.data_dict['label'][index]

        if not isinstance(image_paths, list):
            image_paths = [image_paths]  # for the image-level IO, only one frame is used

        image_tensors = []
        landmark_tensors = []
        mask_tensors = []
        augmentation_seed = None

        for image_path in image_paths:
            # Initialize a new seed for data augmentation at the start of each video
            if self.video_level and image_path == image_paths[0]:
                augmentation_seed = random.randint(0, 2**32 - 1)

            # Get the mask and landmark paths
            mask_path = image_path.replace('frames', 'masks')  # Use .png for mask
            landmark_path = image_path.replace('frames', 'landmarks').replace('.png', '.npy')  # Use .npy for landmark

            # Load the image
            # image = self.load_rgb(image_path)

            try:
                image = self.load_rgb(image_path)
            except Exception as e:
                # Skip this image and return the first one
                print(f"Error loading image at index {index}: {e}")
                return self.__getitem__(0)
            image = np.array(image)  # Convert to numpy array for data augmentation

            # Load mask and landmark (if needed)
            if self.config['with_mask']:
                mask = self.load_mask(mask_path)
            else:
                mask = None
            if self.config['with_landmark']:
                landmarks = self.load_landmark(landmark_path)
            else:
                landmarks = None

            # Do Data Augmentation
            if self.mode == 'train' and self.config['use_data_augmentation']:
                image_trans, landmarks_trans, mask_trans = self.data_aug(image, landmarks, mask, augmentation_seed)
            else:
                image_trans, landmarks_trans, mask_trans = deepcopy(image), deepcopy(landmarks), deepcopy(mask)
            

            # To tensor and normalize
            if not no_norm:
                image_trans = self.normalize(self.to_tensor(image_trans))
                if self.config['with_landmark']:
                    landmarks_trans = torch.from_numpy(landmarks)
                if self.config['with_mask']:
                    mask_trans = torch.from_numpy(mask_trans)

            image_tensors.append(image_trans)
            landmark_tensors.append(landmarks_trans)
            mask_tensors.append(mask_trans)

        if self.video_level:
            # Stack image tensors along a new dimension (time)
            image_tensors = torch.stack(image_tensors, dim=0)
            # Stack landmark and mask tensors along a new dimension (time)
            if not any(landmark is None or (isinstance(landmark, list) and None in landmark) for landmark in landmark_tensors):
                landmark_tensors = torch.stack(landmark_tensors, dim=0)
            if not any(m is None or (isinstance(m, list) and None in m) for m in mask_tensors):
                mask_tensors = torch.stack(mask_tensors, dim=0)
        else:
            # Get the first image tensor
            image_tensors = image_tensors[0]
            # Get the first landmark and mask tensors
            if not any(landmark is None or (isinstance(landmark, list) and None in landmark) for landmark in landmark_tensors):
                landmark_tensors = landmark_tensors[0]
            if not any(m is None or (isinstance(m, list) and None in m) for m in mask_tensors):
                mask_tensors = mask_tensors[0]

        return image_tensors, label, landmark_tensors, mask_tensors, self.data_dict['text_label'][index]


    @staticmethod
    def collate_fn(batch):
        """
        Collate a batch of data points.

        Args:
            batch (list): A list of tuples containing the image tensor, the label tensor,
                          the landmark tensor, and the mask tensor.

        Returns:
            A tuple containing the image tensor, the label tensor, the landmark tensor,
            and the mask tensor.
        """
        # Separate the image, label, landmark, and mask tensors
        images, labels, landmarks, masks, text_label = zip(*batch)
        
        # Stack the image, label, landmark, and mask tensors
        images = torch.stack(images, dim=0)
        labels = torch.LongTensor(labels)
        
        # Special case for landmarks and masks if they are None
        if not any(landmark is None or (isinstance(landmark, list) and None in landmark) for landmark in landmarks):
            landmarks = torch.stack(landmarks, dim=0)
        else:
            landmarks = None

        if not any(m is None or (isinstance(m, list) and None in m) for m in masks):
            masks = torch.stack(masks, dim=0)
        else:
            masks = None

        # Create a dictionary of the tensors
        data_dict = {}
        data_dict['image'] = images
        data_dict['label'] = labels
        data_dict['text_label'] = text_label
        data_dict['landmark'] = landmarks
        data_dict['mask'] = masks
        return data_dict
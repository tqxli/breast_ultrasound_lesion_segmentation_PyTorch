import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import os
from os import listdir
from torchvision import transforms
from numpy import clip
from skimage import io
from skimage.color import rgb2gray
from skimage.util import img_as_float, img_as_ubyte
from skimage.transform import resize
from PIL import Image
import random

class BUSIDataProcessor(Dataset):
    def __init__(self, imgs_dir, masks_dir, resize_img=True):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir

        # Specify desired transformations here:
        self.transformations = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=10),
            transforms.RandomRotation(degrees=10)
        ])

        self.resize_img = resize_img
        self.imgs_ids = sorted(os.listdir(imgs_dir))
        self.mask_ids = sorted(os.listdir(masks_dir))

    @classmethod
    def preprocess(cls, img, resize_img=True, expand_channel=False, adjust_label=False, normalize=False):
        w, h = img.shape[0], img.shape[1]

        if expand_channel:
            if len(img.shape) == 2:
                img = np.dstack([img]*3)
        # For mask, 
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
            
        img = img.transpose((2, 0, 1))
        
        if resize_img:
            new_size = 256
            assert new_size <= w or new_size <= h, 'Resize cannot be greater than image size'
            if expand_channel:
                img = resize(img, (3, new_size, new_size))
            else: 
                img = resize(img, (1, new_size, new_size))
            
        # Standarize pixel values
        if normalize:
            img_min = img.min(axis=(1, 2), keepdims=True)
            img_max = img.max(axis=(1, 2), keepdims=True)
                
            img = (img - img_min)/(img_max-img_min)
            img = (img - img.mean()) / img.std()
            img = clip(img, -1.0, 1.0)
            img = (img + 1.0) / 2.0
            
        # For mask to have values between 0 and 1
        if adjust_label:
            coords = np.where(img != 0)
            img[coords] = 1

        return img

    def __getitem__(self, i):
        img_idx = self.imgs_ids[i]
        mask_idx = self.mask_ids[i]
        img_file = self.imgs_dir + img_idx
        mask_file = self.masks_dir + mask_idx
        
        mask = Image.open(mask_file)
        img = Image.open(img_file)
        
        # Ensure same transformations to image and mask
        seed = np.random.randint(2147483647) 
        random.seed(seed) 
        torch.manual_seed(seed) 
        if self.transformations is not None:
            img = self.transformations(img)
                              
        random.seed(seed) 
        torch.manual_seed(seed) 
        if self.transformations is not None:
            mask = self.transformations(mask)

        img = np.asarray(img).astype('float32')
        mask = np.asarray(mask).astype('float32')

        img = self.preprocess(img, self.resize_img, expand_channel=False, adjust_label=False, normalize=True)
        mask = self.preprocess(mask, self.resize_img, expand_channel=False, adjust_label=True, normalize=False)
        
        return (torch.from_numpy(img), torch.from_numpy(mask))

    def __len__(self):
        return len(self.imgs_ids)
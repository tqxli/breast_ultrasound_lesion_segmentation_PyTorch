import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os
from torchvision import transforms
from numpy import clip
import pandas as pd
from skimage import io
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
from skimage.transform import resize
from PIL import Image
import random
import io
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2

class BUSIDataProcessor(Dataset):
    def __init__(self, imgs_dir, masks_dir, labels_dir=None):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.images_filesnames = sorted(os.listdir(imgs_dir))
        self.labels_dir = labels_dir
        self.labels, self.negative_samples_idx = None, []

        if labels_dir is not None: 
            self.labels = pd.read_csv(labels_dir)['labels'].to_numpy()
            self.negative_samples_idx = (self.labels == 2)
        
        # Specify data augmentations here
        self.transformations = A.Compose([
                A.Resize(256, 256),
                A.OneOf([
                    A.RandomRotate90(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
                    A.RandomBrightnessContrast(p=0.2),
                    A.GridDistortion(p=0.2),
                    A.ElasticTransform(p=0.2)
                ]), 
        ])

    def get_normal_samples_idx(self):
        return self.normal_samples_idx
        
    def __getitem__(self, i):
        img_filename = self.images_filesnames[i]
        
        img = cv2.imread(os.path.join(self.imgs_dir, img_filename)).astype(np.float32) # [H, W, 3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.masks_dir, img_filename), cv2.IMREAD_GRAYSCALE).astype(np.float32) # [H, W]
        mask = np.expand_dims(mask, -1) # [H, W, 1] 

        random.seed(123456)
        if self.transformations is not None: 
            transformed = self.transformations(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']

        img = normalize(img) 
        mask = mask / 255.0 

        img = np.transpose(img, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        
        if self.labels_dir is not None:
            label = np.asarray(self.labels[i])
            return (torch.from_numpy(img), torch.from_numpy(mask), torch.from_numpy(label)) 

        return (torch.from_numpy(img), torch.from_numpy(mask)) 

    def __len__(self):
        return len(self.images_filesnames)

    def normalize(pixels):
        mean, std = pixels.mean(), pixels.std()
        pixels = (pixels - mean) / std
        pixels = np.clip(pixels, -1.0, 1.0)
        pixels = (pixels + 1.0) / 2.0   

        return pixels

class TestDataset(Dataset):
    def __init__(self, imgs_dir):
        self.imgs_dir = imgs_dir
        self.imgs_id = os.listdir(imgs_dir)
        self.processor = BUSIDataProcessor(imgs_dir=None, masks_dir=None)

    def __len__(self):
        return len(self.imgs_id)

    def __getitem__(self, i):
        img = io.imread(self.imgs_dir + self.imgs_id[i])
        processed_img = self.processor.preprocess(img, resize_img=True, expand_channel=False, adjust_label=False, normalize=True)
        
        return {processed_img}
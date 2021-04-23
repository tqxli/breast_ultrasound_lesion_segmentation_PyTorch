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
    def __init__(self, imgs_dir, masks_dir, resize_img=False):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir

        # Specify data augmentations
        self.transformations = A.Compose([
                A.Resize(256, 256),
                A.RandomRotate90(p=0.5)
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
                A.RandomBrightnessContrast(p=0.2),
                A.GridDistortion(p=0.5),
                A.ElasticTransform(),
                ToTensorV2(),
        ])

        # self.transformations = transforms.Compose([
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomVerticalFlip(),
        #     transforms.RandomAffine(degrees=10),
        #     transforms.RandomRotation(degrees=10)
        # ])

        self.resize_img = resize_img
        self.imgs_ids = sorted(os.listdir(imgs_dir))
        self.mask_ids = sorted(os.listdir(masks_dir))

    @classmethod
    def preprocess(cls, img, resize_img=False, expand_channel=False, adjust_label=False, normalize=False):
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
        
        # mask = Image.open(mask_file)
        # img = Image.open(img_file)
        
        # Ensure same transformations to image and mask
        # seed = np.random.randint(2147483647) 
        # random.seed(seed) 
        # torch.manual_seed(seed) 
        # if self.transformations is not None:
        #     img = self.transformations(img)
                              
        # random.seed(seed) 
        # torch.manual_seed(seed) 
        # if self.transformations is not None:
        #     mask = self.transformations(mask)

        img = cv2.imread(img_file).astype(np.float32)
        mask = cv2.imread(mask_file).astype(np.float32)

        #img = np.expand_dims(img, axis=2).transpose((2, 0, 1)) 
        #mask = np.expand_dims(mask, dim=2).transpose((2, 0, 1))

        # img = np.asarray(img).astype('float32')
        # mask = np.asarray(mask).astype('float32')
        img = self.preprocess(img, self.resize_img, expand_channel=True, adjust_label=False, normalize=True)
        mask = self.preprocess(mask, self.resize_img, expand_channel=True, adjust_label=True, normalize=False)
        
        # return (torch.from_numpy(img), torch.from_numpy(mask))        
        random.seed(11)
        if self.transformations is not None: 
            transformed = self.transformations(image=img, mask=mask)
            transformed_image = transformed['image']
            transformed_mask = transformed['mask']

        return (transformed_image, transformed_mask)

    def __len__(self):
        return len(self.imgs_ids)

class BUSIDataProcessor_with_labels(Dataset):
    def __init__(self, imgs_dir, masks_dir, labels_dir, resize_img=True):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.labels_dir = labels_dir
        self.labels = pd.read_csv(labels_dir)['labels'].to_numpy()

        self.normal_samples_idx = (self.labels == 2)

        # Specify desired data transformations here:
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

    def get_normal_samples_idx(self):
        return self.normal_samples_idx

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
        label = np.asarray(self.labels[i])
        
        return (torch.from_numpy(img), torch.from_numpy(mask), torch.from_numpy(label))

    def __len__(self):
        return len(self.imgs_ids)

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
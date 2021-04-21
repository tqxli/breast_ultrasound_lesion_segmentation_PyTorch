from utils import gif
from skimage import io, color
import os
import cv2
import nrrd
import numpy as np
import argparse
from parse_config import ConfigParser
import torch

def segment_3d_input(model, path_to_image):
    if path_to_image:
        # Read from directory
        image_list = sorted(os.listdir(path_to_image))
        
        segmented_volume = []

        for image in image_list:
            img_path = os.path.join(path_to_image, image)
            img = io.imread(img_path) 
            mask = model.get_prediction(img_path) 
            
            segmented_img = apply_mask_to_image(img, mask)      
            segmented_volume.append(segmented_img)

        return segmented_volume
    else:
        print("Image Path Required!")

def apply_mask_to_image(img, mask):
    """
    Apply mask to input image with only contours.
    """
    img_size = img.shape[0]
    mask = cv2.resize(mask, dsize=(img_size, img_size))

    # Find contour of the mask
    imgray = mask
    ret,thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on image
    segmented_img = cv2.drawContours(img, contours, -1, (0,255,0), 3)

    return segmented_img

def saveImgWithSegmentations(segmented_volume, volume_name, save_dir):    
    """
    Save the 3d ultrasound with segmentations as nrrd and gif (for better visualization)
    """
    # Convert from RGB to grayscale
    gray3d = []
    for i in range(len(segmented_volume)):
        gray3d.append(color.rgb2gray(segmented_volume[i]))
    gray3d = np.asarray(gray3d)
    gray3d = np.transpose(gray3d, (2, 1, 0))

    filename = volume_name + '_segmentation'

    # Save as nrrd
    nrrd.write(filename=save_dir+filename+'.nrrd', data=gray3d)
    print('Successfully save results as nrrd.')

    # Save as gif
    img_seq = gray3d.transpose((2, 1, 0)) * 255.0
    img_seq = img_seq.astype(np.uint8)
    if gif(filename, img_seq):
        print('Successfully save results as gif.')


if __name__ == "__main__"':
    # Load checkpoints for inference
    path_to_checkpoint = '/content/drive/MyDrive/exp_results/models/Attn_ResUNet_plus_classification/0421_214038/checkpoint-epoch100.pth' 
    checkpoint = torch.load(path_to_checkpoint)

    # Select test images
    path_to_image = '/content/drive/MyDrive/data/sample_test_volumes/'
    volume_list = os.listdir(path_to_image)
    volume_name = volume_list[0]

    # Initialize model
    args = argparse.ArgumentParser(description='Inference configuration')

    args.add_argument('--config', type=str, default='options/default.json', 
                      help='config path to correct model architecture')

    config = ConfigParser.from_args(args)
    
    # build model architecture, load checkpoints
    model = config.init_obj('arch', models)
    model.load_state_dict(checkpoint)

    # prepare for GPU environment
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    
    # Inference on test images
    prediction_3d = segment_3d_input(model, path_to_image+volume_name)

    # Save results as nrrd & gif
    saveImgWithSegmentations(prediction_3d, volume_name, save_dir='/content/drive/MyDrive/exp_results/test/')
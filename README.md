(Experimental) Implementation for several variants of U-Net in **PyTorch** for breast ultrasound lesion segmentation. 

### Platform
Primary expectation is to enable model training & visualization in **Google Colab**, while scripts can be employed for other GPU available environments.

### Training Dataset
**Breast Ultrasound Images Dataset (Dataset BUSI)** [Data Access](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset)
<u>Number of patients</u>: 600 females.
<u>Number of images</u>: 780 PNG images with an average size of 500*500.
<u>Classes</u>: Normal, Benign, Malignant.
Both ground truth masks and original images are provided.

### Models
- UNet
- Attention UNet
- UNet++
- RCNN-UNet

### General Workflow
#### Option 1: 
Check ``` train_with_Colab.ipynb``` and follow the instructions.
#### Option 2: 
1. To download & preprocess the BUSI dataset, run ```data_proprocessing.py```.
2. Edit the configuration settings in ```configs/<your_config_filename>.yaml```, or use the default one. 
3. Run ```python train.py --config configs/<your_config_filename>.yaml``` to start training.
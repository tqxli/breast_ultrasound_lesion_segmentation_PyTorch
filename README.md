(Experimental) Implementation for several variants of U-Net in **PyTorch** for breast ultrasound lesion segmentation. UNDER CONSTRUCTION!

### Platform
Primary expectation is to enable model training & visualization in **Google Colab**, while scripts can be employed for other GPU available environments.

### Training Dataset
**Breast Ultrasound Images Dataset (Dataset BUSI)** [Data Access](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset)
* Number of patients: 600 females.
* Number of images: 780 PNG images with an average size of 500*500.
* Classes: Normal, Benign, Malignant.

Both ground truth masks and original images are provided.

### Models
* UNet
* Attention UNet
* UNet++
* RCNN-UNet

### General Workflow
#### Option 1: 
Check ``` train_with_Colab.ipynb``` and follow the instructions.
#### Option 2: 
1. To download & preprocess the BUSI dataset, run ```BUSI/BUSI_prepare_for_trainging.py```.
2. Edit the configuration settings in ```options/<your_config_filename>.json```.
3. Run ```python train.py --config options/<your_config_filename>.json --device 'index to GPU device' --resume 'path/to/latest/checkpoint' or None``` to start training.
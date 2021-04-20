## Breast Ultrasound Segmentation

Easy implementation for different models in **PyTorch** for breast ultrasound lesion segmentation. 

### Platform
Main expectation is to enable better model training & visualization in **Google Colab**, while the scripts can be employed for other GPU available environments.

### Training Dataset
**Breast Ultrasound Images Dataset (Dataset BUSI)** 
[Dataset Public Access](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset)
* Number of patients: 600 females.
* Number of images: 780 PNG images with an average size of 500*500.
* Classes: Normal, Benign, Malignant.

Both ground truth masks and original images are provided. 

Normal images are not used during training.

### Models (work in progress)
* UNet with ResNet backbone (ResUNet)
* ResUNet with attention
* Attention UNet
* Multi-scale Attention Net (MA-Net)

### General Workflow
#### Option 1: 
Check ``` train_with_Colab.ipynb``` and follow the instructions inside.
#### Option 2: 
1. To download the BUSI dataset, run 
    ```python BUSI/BUSI_prepare_for_training.py```.
    You can also prepare it differently.
2. Change configuration settings in ```options/<your_config_filename>.json``` or use the default json file. 
3. To enable Tensorboard visualization, make sure ```tensorboard``` option is turned on in the config file. ```"tensorboard: true"```
    At the project root, open Tensorboard server:
    ```
    mkdir exp_results
    mkdir exp_results/log
    tensorboard --logdir exp_results/log/
    ```
    The server will then open at ```http://localhost:6006```

3. To start training, run
    ```
    python train.py --config options/<your_config_filename>.json 
    --device 'index to GPU device' 
    --resume 'path/to/latest/checkpoint' or None
    ``` 

### References
**Template**: 
PyTorch Template Project https://github.com/victoresque/pytorch-template#pytorch-template-project
**Library**: 
Segmentation Models https://smp.readthedocs.io/en/latest/

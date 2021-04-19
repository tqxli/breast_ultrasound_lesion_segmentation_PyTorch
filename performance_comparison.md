#### Experiments results

Model | Strategy | Batch Size | Epoch | Loss fcn | Train Loss | Val Loss |  Train IoU | Train Dice | Val IoU | Val Dice 
:-----------: | :----: | :---: | :-----:| :-----:| :----: | :----: | :----: | :----: | :----: | :----: | :----: 
Attn UNet | t.f.s| 16 | 150 | BCE + DiceBCE | 0.57 | 0.61 | 0.44 | 0.64 | 0.41 | 0.62 
Resnet18 UNet | ImageNet| 16 | 100 | BCE + BCEDice | 0.57 | 0.61 | 0.44 | 0.64 | 0.41 | 0.62

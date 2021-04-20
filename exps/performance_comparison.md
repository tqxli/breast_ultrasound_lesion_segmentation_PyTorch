#### Experiments results

| Model              | Strategy | Batch Size | Epoch | Loss fcn           | Train IoU | Train Dice | Val IoU | Val Dice |
| :----------------: | :------: | :--------: | :----:| :-----------------:| :-------: | :--------: | :-----: | :------: | 
| Attn UNet          | t.f.s    | 16         | 100   | BCE(50) + Dice(50) | 0.55      | 0.66       | 0.52    | 0.64     |
| Resnet18 UNet      | ImageNet | 16         | 50    | BCE                | 0.73      | 0.82       | 0.70    | 0.78     |
| Resnet18 UNet      | ImageNet | 16         | 50    | DiceBCE            | 0.71      | 0.80       | 0.70    | 0.79     |
| Resnet34 UNet      | ImageNet | 16         | 50    | DiceBCE            | 0.73      | 0.81       | 0.70    | 0.79     |
| Attn Resnet18 UNet | ImageNet | 16         | 50    | DiceBCE            | 0.72      | 0.81       | 0.71    | 0.80     |
| MA Net             | ImageNet | 16         | 50    | DiceBCE            | 0.61      | 0.71       | 0.56    | 0.66     |

#### Observations
Adding attention to the decoder path gives (marginal) improvement.
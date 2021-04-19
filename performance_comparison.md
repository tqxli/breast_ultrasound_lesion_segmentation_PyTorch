#### Experiments results

| Model | Strategy | Batch Size | Epoch | Loss fcn | Train Loss | Val Loss |  Train IoU | Train Dice | Val IoU | Val Dice |
| :-----------: | :----: | :---: | :-----:| :-----:| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| Attn UNet | t.f.s| 16 | 50 | BCE | 0.13 | 0.15 | 0.44 | 0.64 | 0.41 | 0.62 |
| Attn UNet | finetuned from t.f.s| 16 | 100 | DiceBCE | 0.57 | 0.61 | 0.44 | 0.64 | 0.41 | 0.62 |


"""
Two ways for checking model architectures:
1. Use torchsummary.
2. Define your model classes as a BaseModel (instead of a nn.Module),
    which specifies a __str__() function.
"""

from __future__ import print_function
import torch
import torchsummary

from models import ResNet_UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_test = ResNet_UNet()
model_test.to(device)

torchsummary.summary(model_test, (3, 256, 256))
print(model_test)
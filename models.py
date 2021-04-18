from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

class conv_block(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channel_in, channel_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(channel_out),
            nn.ReLU(inplace=True), # inplace=True: modify the input directly, without allocating any additional memory.
            nn.Conv2d(channel_out, channel_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(channel_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv(x)

        return x

class up_conv_block(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(up_conv_block, self).__init__()
        self.up_conv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(channel_in,channel_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(channel_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up_conv(x)

        return x

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi

        return out

#class UNet(nn.Module):

#class ResNet_UNet(nn.Module):

class Attention_UNet(BaseModel):
    def __init__(self, input_channel=3, output_channel=1):
        super(Attention_UNet, self).__init__()
        
        filters = [64, 128, 256, 512, 1024]
        self.conv1 = conv_block(input_channel, filters[0])
        self.conv2 = conv_block(filters[0], filters[1])
        self.conv3 = conv_block(filters[1], filters[2])
        self.conv4 = conv_block(filters[2], filters[3])
        self.conv5 = conv_block(filters[3], filters[4])

        self.up_sampling5 = up_conv_block(filters[4], filters[3])
        self.up_sampling4 = up_conv_block(filters[3], filters[2])
        self.up_sampling3 = up_conv_block(filters[2], filters[1])
        self.up_sampling2 = up_conv_block(filters[1], filters[0])

        self.up_conv5 = conv_block(filters[4], filters[3])
        self.up_conv4 = conv_block(filters[3], filters[2])
        self.up_conv3 = conv_block(filters[2], filters[1])
        self.up_conv2 = conv_block(filters[1], filters[0])

        self.att5 = Attention_block(filters[3], filters[3], filters[2])
        self.att4 = Attention_block(filters[2], filters[2], filters[1])
        self.att3 = Attention_block(filters[1], filters[1], filters[0])
        self.att2 = Attention_block(filters[0], filters[0], int(filters[0]/2))
        
        self.final = nn.Conv2d(filters[0], output_channel, kernel_size=1, stride=1, padding=0)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder path
        x1 = self.conv1(x)
        
        x2 = self.maxpool(x1)
        x2 = self.conv2(x2)

        x3 = self.maxpool(x2)
        x3 = self.conv3(x3)

        x4 = self.maxpool(x3)
        x4 = self.conv4(x4)

        x5 = self.maxpool(x4)
        x5 = self.conv5(x5)
        
        # Decoder path
        d5 = self.up_sampling5(x5)
        a4 = self.att5(d5, x4)
        d5 = torch.cat((a4, d5), dim=1)
        d5 = self.up_conv5(d5)
        
        d4 = self.up_sampling4(d5)
        a3 = self.att4(d4, x3)
        d4 = torch.cat((a3, d4), dim=1)
        d4 = self.up_conv4(d4)

        d3 = self.up_sampling3(d4)
        a2 = self.att3(d3, x2)
        d3 = torch.cat((a2, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up_sampling2(d3)
        a1 = self.att2(d2, x1)
        d2 = torch.cat((a1, d2), dim=1)
        d2 = self.up_conv2(d2)

        out = self.final(d2)

        return out
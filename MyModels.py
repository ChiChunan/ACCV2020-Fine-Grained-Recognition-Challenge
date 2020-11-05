import torch
import torchvision
from torch import nn as nn
from torchvision import models


class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, is_downsample):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        if is_downsample:
            self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=32, bias=False)
        else:
            self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3 = nn.Conv2d(out_channel, out_channel, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential()
        if is_downsample:
            self.downsample = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=(2, 2), bias=False),
                                   nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)

        return out

class  MyModels(nn.Module):
    def __init__(self, model_ft, num_classes):
        super(MyModels, self).__init__()
        self.trunk = nn.Sequential(*list(model_ft.children())[:-2])
        self.block1 = BasicBlock(2048, 8192, is_downsample=True)
#        self.block2 = BasicBlock(8192, 8192, is_downsample=False)
#        self.block3 = BasicBlock(8192, 8192, is_downsample=False)
        self.mean = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(8192, num_classes)

    def forward(self, x):
        x = self.trunk(x)
        x = self.block1(x)
#        x = self.block2(x)
#        x = self.block3(x)
        x = self.mean(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)

        return x

def initialize_model(model_name, num_classes, features_extract, use_pretrained=True):
    if model_name == 'ResNext':
        model_ft = models.resnet.resnext101_32x8d(pretrained=use_pretrained)
        if features_extract:
            for param in model_ft.parameters():
                param.requires_grad = False

        model_ft = MyModels(model_ft, num_classes)
        input_size =  224
        print('model is ResNext101-8x32d')
    else:
        print('model not implemented')
        return None, None

    return model_ft, input_size
from typing import Tuple
import torch
import torch.nn as nn  #专门为神经网络设计的模块化接口
import kornia.augmentation as K  #kornia是一个几何计算机视觉库


class RandomTransform(nn.Module):  #nn.Module是一个抽象概念，既可以表示神经网络的某个层，也可以表示包含很多层的神经网络
    def __init__(self, size: Tuple[int, int], randcrop: bool, augment: bool):
        super().__init__()
        if randcrop:
            self.crop = K.RandomCrop(size)
        else:
            self.crop = K.CenterCrop(size)
        self.flip = nn.Sequential(K.RandomVerticalFlip(p=0.5),
                                  K.RandomHorizontalFlip(p=0.5))
        self.augment = augment

    def forward(self, img, disparity, conf=None):
        if conf is None:
            input = torch.cat([img, disparity], dim=0)
        else:
            input = torch.cat([img, disparity, conf], dim=0)
        input = self.crop(input)
        if self.augment:
            input = self.flip(input)
        img = input[:, :3]
        disparity = input[:, [3]]
        if conf is None:
            return img, disparity
        else:
            conf = input[:, [4]]
            return img, disparity, conf


class HyperspectralDepthTransform(nn.Module):
    def __init__(self, size: Tuple[int, int], randcrop: bool, augment: bool):
        super().__init__()
        if randcrop:
            self.crop = K.RandomCrop(size)
        else:
            self.crop = K.CenterCrop(size)
        self.flip = nn.Sequential(K.RandomVerticalFlip(p=0.5),
                                  K.RandomHorizontalFlip(p=0.5))
        self.augment = augment

    def forward(self, img, depthmap, hyperspectral):
        hyperspectral_dim = hyperspectral.shape[0]
        input = torch.cat([img, depthmap, hyperspectral], dim=0)
        input = self.crop(input)
        if self.augment:
            input = self.flip(input)

        input = input.squeeze(dim=0)
        img = input[:3, ...]
        depthmap = input[3, ...].unsqueeze(0)
        hyperspectral = input[-hyperspectral_dim:, ...]
        return img, depthmap, hyperspectral




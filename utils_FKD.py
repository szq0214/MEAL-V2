import os
import torch
import torch.distributed
import torch.nn as nn
import torchvision
from torchvision.ops import roi_align
from torchvision.transforms import functional as t_F
from torch.nn import functional as F
from torchvision.datasets.folder import ImageFolder
from torch.nn.modules import loss
from torchvision.transforms import InterpolationMode
import random
import numpy as np


class RandomResizedCrop_FKD(torchvision.transforms.RandomResizedCrop):
    def __init__(self, **kwargs):
        super(RandomResizedCrop_FKD, self).__init__(**kwargs)

    def __call__(self, img, coords, status):
        i = coords[0].item() * img.size[1]
        j = coords[1].item() * img.size[0]
        h = coords[2].item() * img.size[1]
        w = coords[3].item() * img.size[0]

        if self.interpolation == 'bilinear':
            inter = InterpolationMode.BILINEAR
        elif self.interpolation == 'bicubic':
            inter = InterpolationMode.BICUBIC
        return t_F.resized_crop(img, i, j, h, w, self.size, inter)


class RandomHorizontalFlip_FKD(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img, coords, status):
    
        if status == True:
            return t_F.hflip(img)
        else:
            return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class Compose_FKD(torchvision.transforms.Compose):
    def __init__(self, **kwargs):
        super(Compose_FKD, self).__init__(**kwargs)

    def __call__(self, img, coords, status):
        for t in self.transforms:
            if type(t).__name__ == 'RandomResizedCrop_FKD':
                img = t(img, coords, status)
            elif type(t).__name__ == 'RandomCrop_FKD':
                img, coords = t(img)
            elif type(t).__name__ == 'RandomHorizontalFlip_FKD':
                img = t(img, coords, status)
            else:
                img = t(img)
        return img


class ImageFolder_FKD(torchvision.datasets.ImageFolder):
    def __init__(self, **kwargs):
        self.num_crops = kwargs['num_crops']
        self.softlabel_path = kwargs['softlabel_path']
        kwargs.pop('num_crops')
        kwargs.pop('softlabel_path')
        super(ImageFolder_FKD, self).__init__(**kwargs)

    def __getitem__(self, index):

            path, target = self.samples[index]

            label_path = os.path.join(self.softlabel_path, '/'.join(path.split('/')[-4:]).split('.')[0] + '.tar')

            label = torch.load(label_path, map_location=torch.device('cpu'))

            coords, flip_status, output = label

            rand_index = torch.randperm(len(output))#.cuda()
            output_new = []

            sample = self.loader(path)
            sample_all = [] 
            target_all = []

            for i in range(self.num_crops):
                if self.transform is not None:
                    output_new.append(output[rand_index[i]])
                    sample_new = self.transform(sample, coords[rand_index[i]], flip_status[rand_index[i]])
                    sample_all.append(sample_new)
                    target_all.append(target)
                else:
                    coords = None
                    flip_status = None
                if self.target_transform is not None:
                    target = self.target_transform(target)

            return sample_all, target_all, output_new


def Recover_soft_label(label, label_type, n_classes):
    if label_type == 'hard':
        return torch.zeros(label.size(0), n_classes).scatter_(1, label.view(-1, 1), 1)
    elif label_type == 'smoothing':
        index = label[:,0].to(dtype=int)
        value = label[:,1]
        minor_value = (torch.ones_like(value) - value)/(n_classes-1)
        minor_value = minor_value.reshape(-1,1).repeat_interleave(n_classes, dim=1)
        soft_label = (minor_value * torch.ones(index.size(0), n_classes)).scatter_(1, index.view(-1, 1), value.view(-1, 1))
        return soft_label
    elif label_type == 'marginal_smoothing_k5':
        index = label[:,0,:].to(dtype=int)
        value = label[:,1,:]
        minor_value = (torch.ones(label.size(0),1) - torch.sum(value, dim=1, keepdim=True))/(n_classes-5)
        minor_value = minor_value.reshape(-1,1).repeat_interleave(n_classes, dim=1)
        soft_label = (minor_value * torch.ones(index.size(0), n_classes)).scatter_(1, index, value)
        return soft_label
    elif label_type == 'marginal_renorm':
        index = label[:,0,:].to(dtype=int)
        value = label[:,1,:]
        soft_label = torch.zeros(index.size(0), n_classes).scatter_(1, index, value)
        soft_label = F.normalize(soft_label, p=1.0, dim=1, eps=1e-12)
        return soft_label
    elif label_type == 'marginal_smoothing_k10':
        index = label[:,0,:].to(dtype=int)
        value = label[:,1,:]
        minor_value = (torch.ones(label.size(0),1) - torch.sum(value, dim=1, keepdim=True))/(n_classes-10)
        minor_value = minor_value.reshape(-1,1).repeat_interleave(n_classes, dim=1)
        soft_label = (minor_value * torch.ones(index.size(0), n_classes)).scatter_(1, index, value)
        return soft_label
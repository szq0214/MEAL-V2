"""Dataset class for loading imagenet data."""

import os

from torch.utils import data as data_utils
from torchvision import datasets as torch_datasets
from torchvision import transforms

from utils_FKD import RandomResizedCrop_FKD,RandomHorizontalFlip_FKD,ImageFolder_FKD,Compose_FKD
from torchvision.transforms import InterpolationMode

def get_train_loader(imagenet_path, batch_size, num_workers, image_size):
    train_dataset = ImageNet(imagenet_path, image_size, is_train=True)
    return data_utils.DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size, pin_memory=True,
        num_workers=num_workers)

def get_train_loader_FKD(imagenet_path, batch_size, num_workers, image_size, num_crops, softlabel_path):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = ImageFolder_FKD(
        num_crops=num_crops,
        softlabel_path=softlabel_path,
        root=os.path.join(imagenet_path, 'train'),
        transform=Compose_FKD(transforms=[
            RandomResizedCrop_FKD(size=224,
                                  scale=(0.08, 1),
                                  interpolation='bilinear'), 
            RandomHorizontalFlip_FKD(),
            transforms.ToTensor(),
            normalize,
        ]))
    return data_utils.DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size, pin_memory=True,
        num_workers=num_workers)

def get_val_loader(imagenet_path, batch_size, num_workers, image_size):
    val_dataset = ImageNet(imagenet_path, image_size, is_train=False)
    return data_utils.DataLoader(
        val_dataset, shuffle=False, batch_size=batch_size, pin_memory=True,
        num_workers=num_workers)


class ImageNet(torch_datasets.ImageFolder):
    """Dataset class for ImageNet dataset.

    Arguments:
        root_dir (str): Path to the dataset root directory, which must contain
            train/ and val/ directories.
        is_train (bool): Whether to read training or validation images.
    """
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(self, root_dir, im_size, is_train):
        if is_train:
            root_dir = os.path.join(root_dir, 'train')
            transform = transforms.Compose([
                transforms.RandomResizedCrop(im_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(ImageNet.MEAN, ImageNet.STD),
            ])
        else:
            root_dir = os.path.join(root_dir, 'val')
            transform = transforms.Compose([
                transforms.Resize(int(256/224*im_size)),
                transforms.CenterCrop(im_size),
                transforms.ToTensor(),
                transforms.Normalize(ImageNet.MEAN, ImageNet.STD),
            ])
        super().__init__(root_dir, transform=transform)



import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision import datasets
import os
PBN_DATAPATH = './data'


def load_cifar100(trans):
    mean = [x / 255 for x in [129.3, 124.1, 112.4]]
    std = [x / 255 for x in [68.2, 65.4, 70.4]]

    train_transform = transforms.Compose(trans + [
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)])
        
    train_data = torchvision.datasets.CIFAR100(root=PBN_DATAPATH, 
                                            train=True,
                                            download=True, 
                                            transform=train_transform)

    test_data = torchvision.datasets.CIFAR100(root=PBN_DATAPATH, 
                                            train=False,
                                            download=True, 
                                            transform=test_transform)
    return train_data, test_data



def load_tiny_imagenet_200(trans):
    mean = [x / 255 for x in [127.5, 127.5, 127.5]]
    std = [x / 255 for x in [127.5, 127.5, 127.5]]

    train_transform = transforms.Compose(trans+[
            transforms.RandomCrop(64, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)])
    
    train_root = os.path.join(PBN_DATAPATH, 'tiny-imagenet-200/train')  # this is path to training images folder
    validation_root = os.path.join(PBN_DATAPATH, 'tiny-imagenet-200/val/images')  # this is path to validation images folder
    train_data = datasets.ImageFolder(train_root, transform=train_transform)
    test_data = datasets.ImageFolder(validation_root, transform=test_transform)
    
    return train_data, test_data


def load_data(args, batch_size, data_name):
    trans = [transforms.RandomHorizontalFlip()]

    if data_name.lower() == 'cifar100':
        train_data, test_data = load_cifar100(trans)
    elif data_name.lower() == 'tinyimagenet200':
        train_data, test_data = load_tiny_imagenet_200(trans)
    else:
        raise f'{data_name} is not supported'
    
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    dataloaders = {
        "train": trainloader,
        "val": testloader
    }

    return dataloaders
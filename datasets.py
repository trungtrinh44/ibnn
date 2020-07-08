import torch
import torchvision
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import DataLoader, Subset


def get_data_loader(dataset, batch_size=64, validation=False, validation_fraction=0.1, random_state=42, root_dir='data/'):
    if dataset == 'mnist':
        train_data = torchvision.datasets.MNIST(root_dir, train=True, download=True,
                                                transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(
                                                        (0.1307,), (0.3081,))
                                                ]))
        test_data = torchvision.datasets.MNIST(root_dir, train=False, download=True,
                                               transform=torchvision.transforms.Compose([
                                                   torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Normalize(
                                                       (0.1307,), (0.3081,))
                                               ]))
        test_loader = DataLoader(
            test_data, batch_size=batch_size, pin_memory=True, shuffle=False)
        if validation:
            train_idx, valid_idx = train_test_split(np.arange(len(train_data.targets)),
                                                    test_size=validation_fraction,
                                                    shuffle=True, random_state=random_state,
                                                    stratify=train_data.targets)
            train_loader = DataLoader(Subset(
                train_data, train_idx), batch_size=batch_size, pin_memory=True, shuffle=True)
            valid_loader = DataLoader(Subset(
                train_data, valid_idx), batch_size=batch_size, pin_memory=True, shuffle=False)
            return train_loader, valid_loader, test_loader
        else:
            train_loader = DataLoader(
                train_data, batch_size=batch_size, pin_memory=True, shuffle=True)
            return train_loader, test_loader
    if dataset == 'fmnist':
        train_data = torchvision.datasets.FashionMNIST(root_dir, train=True, download=True,
                                                       transform=torchvision.transforms.Compose([
                                                           torchvision.transforms.ToTensor(),
                                                           torchvision.transforms.Normalize(
                                                               (0.2860,), (0.3205,))
                                                       ]))
        test_data = torchvision.datasets.FashionMNIST(root_dir, train=False, download=True,
                                                      transform=torchvision.transforms.Compose([
                                                          torchvision.transforms.ToTensor(),
                                                          torchvision.transforms.Normalize(
                                                              (0.2860,), (0.3205,))
                                                      ]))
        test_loader = DataLoader(
            test_data, batch_size=batch_size, pin_memory=True, shuffle=False)
        if validation:
            train_idx, valid_idx = train_test_split(np.arange(len(train_data.targets)),
                                                    test_size=validation_fraction,
                                                    shuffle=True, random_state=random_state,
                                                    stratify=train_data.targets)
            train_loader = DataLoader(Subset(
                train_data, train_idx), batch_size=batch_size, pin_memory=True, shuffle=True)
            valid_loader = DataLoader(Subset(
                train_data, valid_idx), batch_size=batch_size, pin_memory=True, shuffle=False)
            return train_loader, valid_loader, test_loader
        else:
            train_loader = DataLoader(
                train_data, batch_size=batch_size, pin_memory=True, shuffle=True)
            return train_loader, test_loader
    if dataset == 'cifar10':
        train_data = torchvision.datasets.CIFAR10(
            root_dir, train=True, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Pad(4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomCrop(32),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
        test_data = torchvision.datasets.CIFAR10(root_dir, train=False, download=True,
                                                 transform=torchvision.transforms.Compose([
                                                     torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
        test_loader = DataLoader(
            test_data, batch_size=batch_size, pin_memory=True, shuffle=False)
        if validation:
            valid_data = torchvision.datasets.CIFAR10(root_dir, train=True, download=True,
                                                      transform=torchvision.transforms.Compose([
                                                          torchvision.transforms.ToTensor(),
                                                          torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
            train_idx, valid_idx = train_test_split(np.arange(len(train_data.targets)),
                                                    test_size=validation_fraction,
                                                    shuffle=True, random_state=random_state,
                                                    stratify=train_data.targets)
            train_loader = DataLoader(Subset(
                train_data, train_idx), batch_size=batch_size, pin_memory=True, shuffle=True)
            valid_loader = DataLoader(Subset(
                valid_data, valid_idx), batch_size=batch_size, pin_memory=True, shuffle=False)
            return train_loader, valid_loader, test_loader
        else:
            train_loader = DataLoader(
                train_data, batch_size=batch_size, pin_memory=True, shuffle=True)
            return train_loader, test_loader
    raise NotImplementedError('Dataset is not supported')

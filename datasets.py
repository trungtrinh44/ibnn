import sys

import numpy as np
import torch
import torchvision
import webdataset as wds
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset


def infinite_wrapper(loader):
    while True:
        for x in loader:
            yield x


def identity(x):
    return x


def worker_urls(urls):
    result = wds.worker_urls(urls)
    print("worker_urls returning", len(result),
          "of", len(urls), "urls", file=sys.stderr)
    return result


def imagenet_loader(train_batch_size=128, test_batch_size=200, shuffle_buffer=1000, num_train_workers=4, num_test_workers=1,
                    trainshards='./data/imagenet/shards/imagenet-train-{000000..001281}.tar', valshards='./data/imagenet/shards/imagenet-val-{000000..000049}.tar'):
    trainsize = 1281167
    normalize = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            normalize,
        ]
    )
    val_transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize(256), torchvision.transforms.CenterCrop(
            224), torchvision.transforms.ToTensor(), normalize]
    )
    num_batches = trainsize // train_batch_size
    train_dataset = (
        wds.Dataset(trainshards, length=num_batches,
                    shard_selection=worker_urls)
        .shuffle(shuffle_buffer)
        .decode("pil")
        .to_tuple("jpg;png;jpeg cls")
        .map_tuple(train_transform, identity)
        .batched(train_batch_size)
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=None, shuffle=False, num_workers=num_train_workers,
    )
    val_dataset = (
        wds.Dataset(valshards, length=50000//test_batch_size, shard_selection=worker_urls)
        .decode("pil")
        .to_tuple("jpg;png;jpeg cls")
        .map_tuple(val_transform, identity)
        .batched(test_batch_size)
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=None, shuffle=False, num_workers=num_test_workers,
    )
    return train_loader, val_loader


def get_data_loader(dataset, batch_size=64, validation=False, validation_fraction=0.1, random_state=42, root_dir='data/', test_only=False, train_only=False, augment=True, degree=0):
    if dataset == 'mnist' and degree != 0:
        test_data = torchvision.datasets.MNIST(root_dir, train=False, download=True,
                                               transform=torchvision.transforms.Compose([
                                                   torchvision.transforms.RandomRotation(
                                                       (degree, degree)),
                                                   torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Normalize(
                                                       (0.1307,), (0.3081,))
                                               ]))
        test_loader = DataLoader(
            test_data, batch_size=batch_size, pin_memory=True, shuffle=False)
        return test_loader
    if dataset == 'fmnist' and degree != 0:
        test_data = torchvision.datasets.FashionMNIST(root_dir, train=False, download=True,
                                                      transform=torchvision.transforms.Compose([
                                                          torchvision.transforms.RandomRotation(
                                                              (degree, degree)),
                                                          torchvision.transforms.ToTensor(),
                                                          torchvision.transforms.Normalize(
                                                              (0.2860,), (0.3205,))
                                                      ]))
        test_loader = DataLoader(
            test_data, batch_size=batch_size, pin_memory=True, shuffle=False)
        return test_loader
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
        if test_only:
            return test_loader
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
        if test_only:
            return test_loader
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
    if dataset == 'fmnist_mnist_test':
        test_data = torchvision.datasets.FashionMNIST(root_dir, train=False, download=True,
                                                      transform=torchvision.transforms.Compose([
                                                          torchvision.transforms.ToTensor(),
                                                          torchvision.transforms.Normalize(
                                                              (0.1307,), (0.3081,))
                                                      ]))
        test_loader = DataLoader(
            test_data, batch_size=batch_size, pin_memory=True, shuffle=False)
        return test_loader
    if dataset == 'kmnist_mnist_test':
        test_data = torchvision.datasets.KMNIST(root_dir, train=False, download=True,
                                                transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(
                                                        (0.1307,), (0.3081,))
                                                ]))
        test_loader = DataLoader(
            test_data, batch_size=batch_size, pin_memory=True, shuffle=False)
        return test_loader
    if dataset == 'mnist_fmnist_test':
        test_data = torchvision.datasets.MNIST(root_dir, train=False, download=True,
                                               transform=torchvision.transforms.Compose([
                                                   torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Normalize(
                                                       (0.2860,), (0.3205,))
                                               ]))
        test_loader = DataLoader(
            test_data, batch_size=batch_size, pin_memory=True, shuffle=False)
        return test_loader
    if dataset == 'kmnist_fmnist_test':
        test_data = torchvision.datasets.KMNIST(root_dir, train=False, download=True,
                                                transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(
                                                        (0.2860,), (0.3205,))
                                                ]))
        test_loader = DataLoader(
            test_data, batch_size=batch_size, pin_memory=True, shuffle=False)
        return test_loader
    if dataset == 'cifar10':
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        augment_transform = [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip()
        ] if augment else []
        train_data = torchvision.datasets.CIFAR10(
            root_dir, train=True, download=True,
            transform=torchvision.transforms.Compose([
                *augment_transform,
                transform
            ]))
        if train_only:
            train_loader = DataLoader(
                train_data, batch_size=batch_size, pin_memory=True, shuffle=True, drop_last=True)
            return train_loader
        test_data = torchvision.datasets.CIFAR10(root_dir, train=False, download=True,
                                                 transform=transform)
        test_loader = DataLoader(
            test_data, batch_size=batch_size, pin_memory=True, shuffle=False)
        if test_only:
            return test_loader
        if validation:
            valid_data = torchvision.datasets.CIFAR10(root_dir, train=True, download=True,
                                                      transform=transform)
            train_idx, valid_idx = train_test_split(np.arange(len(train_data.targets)),
                                                    test_size=validation_fraction,
                                                    shuffle=True, random_state=random_state,
                                                    stratify=train_data.targets)
            train_loader = DataLoader(Subset(
                train_data, train_idx), batch_size=batch_size, pin_memory=True, shuffle=True, drop_last=True)
            valid_loader = DataLoader(Subset(
                valid_data, valid_idx), batch_size=batch_size, pin_memory=True, shuffle=False, drop_last=True)
            return train_loader, valid_loader, test_loader
        else:
            train_loader = DataLoader(
                train_data, batch_size=batch_size, pin_memory=True, shuffle=True, drop_last=True)
            return train_loader, test_loader
    if dataset == 'cifar100':
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        augment_transform = [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip()
        ] if augment else []
        train_data = torchvision.datasets.CIFAR100(
            root_dir, train=True, download=True,
            transform=torchvision.transforms.Compose([
                *augment_transform,
                transform
            ]))
        if train_only:
            train_loader = DataLoader(
                train_data, batch_size=batch_size, pin_memory=True, shuffle=True, drop_last=True)
            return train_loader
        test_data = torchvision.datasets.CIFAR100(root_dir, train=False, download=True,
                                                  transform=transform)
        test_loader = DataLoader(
            test_data, batch_size=batch_size, pin_memory=True, shuffle=False)
        if test_only:
            return test_loader
        if validation:
            valid_data = torchvision.datasets.CIFAR100(root_dir, train=True, download=True,
                                                       transform=transform)
            train_idx, valid_idx = train_test_split(np.arange(len(train_data.targets)),
                                                    test_size=validation_fraction,
                                                    shuffle=True, random_state=random_state,
                                                    stratify=train_data.targets)
            train_loader = DataLoader(Subset(
                train_data, train_idx), batch_size=batch_size, pin_memory=True, shuffle=True, drop_last=True)
            valid_loader = DataLoader(Subset(
                valid_data, valid_idx), batch_size=batch_size, pin_memory=True, shuffle=False, drop_last=True)
            return train_loader, valid_loader, test_loader
        else:
            train_loader = DataLoader(
                train_data, batch_size=batch_size, pin_memory=True, shuffle=True, drop_last=True)
            return train_loader, test_loader
    if dataset == 'vgg_cifar10':
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        augment_transform = [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip()
        ] if augment else []
        train_data = torchvision.datasets.CIFAR10(
            root_dir, train=True, download=True,
            transform=torchvision.transforms.Compose([
                *augment_transform,
                transform
            ]))
        if train_only:
            train_loader = DataLoader(
                train_data, batch_size=batch_size, pin_memory=True, shuffle=True, drop_last=True)
            return train_loader
        test_data = torchvision.datasets.CIFAR10(root_dir, train=False, download=True,
                                                 transform=transform)
        test_loader = DataLoader(
            test_data, batch_size=batch_size, pin_memory=True, shuffle=False)
        if test_only:
            return test_loader
        if validation:
            valid_data = torchvision.datasets.CIFAR10(root_dir, train=True, download=True,
                                                      transform=transform)
            train_idx, valid_idx = train_test_split(np.arange(len(train_data.targets)),
                                                    test_size=validation_fraction,
                                                    shuffle=True, random_state=random_state,
                                                    stratify=train_data.targets)
            train_loader = DataLoader(Subset(
                train_data, train_idx), batch_size=batch_size, pin_memory=True, shuffle=True, drop_last=True)
            valid_loader = DataLoader(Subset(
                valid_data, valid_idx), batch_size=batch_size, pin_memory=True, shuffle=False, drop_last=True)
            return train_loader, valid_loader, test_loader
        else:
            train_loader = DataLoader(
                train_data, batch_size=batch_size, pin_memory=True, shuffle=True, drop_last=True)
            return train_loader, test_loader
    if dataset == 'vgg_cifar100':
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        augment_transform = [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip()
        ] if augment else []
        train_data = torchvision.datasets.CIFAR100(
            root_dir, train=True, download=True,
            transform=torchvision.transforms.Compose([
                *augment_transform,
                transform
            ]))
        if train_only:
            train_loader = DataLoader(
                train_data, batch_size=batch_size, pin_memory=True, shuffle=True, drop_last=True)
            return train_loader
        test_data = torchvision.datasets.CIFAR100(root_dir, train=False, download=True,
                                                  transform=transform)
        test_loader = DataLoader(
            test_data, batch_size=batch_size, pin_memory=True, shuffle=False)
        if test_only:
            return test_loader
        if validation:
            valid_data = torchvision.datasets.CIFAR100(root_dir, train=True, download=True,
                                                       transform=transform)
            train_idx, valid_idx = train_test_split(np.arange(len(train_data.targets)),
                                                    test_size=validation_fraction,
                                                    shuffle=True, random_state=random_state,
                                                    stratify=train_data.targets)
            train_loader = DataLoader(Subset(
                train_data, train_idx), batch_size=batch_size, pin_memory=True, shuffle=True, drop_last=True)
            valid_loader = DataLoader(Subset(
                valid_data, valid_idx), batch_size=batch_size, pin_memory=True, shuffle=False, drop_last=True)
            return train_loader, valid_loader, test_loader
        else:
            train_loader = DataLoader(
                train_data, batch_size=batch_size, pin_memory=True, shuffle=True, drop_last=True)
            return train_loader, test_loader
    if dataset == 'svhn_cifar10_test':
        test_data = torchvision.datasets.SVHN(root_dir, split='test', download=True,
                                              transform=torchvision.transforms.Compose([
                                                  torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
        test_loader = DataLoader(
            test_data, batch_size=batch_size, pin_memory=True, shuffle=False)
        return test_loader
    if dataset == 'greyscalecifar10':
        train_data = torchvision.datasets.CIFAR10(
            root_dir, train=True, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Grayscale(num_output_channels=1),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.4809,), (0.2174,))]))
        test_data = torchvision.datasets.CIFAR10(root_dir, train=False, download=True,
                                                 transform=torchvision.transforms.Compose([
                                                     torchvision.transforms.Grayscale(
                                                         num_output_channels=1),
                                                     torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.Normalize(
                                                         (0.4809,), (0.2174,))]))
        test_loader = DataLoader(
            test_data, batch_size=batch_size, pin_memory=True, shuffle=False)
        if test_only:
            return test_loader
        if validation:
            valid_data = torchvision.datasets.CIFAR10(root_dir, train=True, download=True,
                                                      transform=torchvision.transforms.Compose([
                                                          torchvision.transforms.Grayscale(
                                                              num_output_channels=1),
                                                          torchvision.transforms.ToTensor(),
                                                          torchvision.transforms.Normalize(
                                                              (0.4809,), (0.2174,))]))
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
    if dataset == 'semeion_mnist_test':
        test_data = torchvision.datasets.SEMEION(root_dir, download=True,
                                                 transform=torchvision.transforms.Compose([
                                                     torchvision.transforms.Pad(
                                                         6),
                                                     torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.Normalize(
                                                         (0.1307,), (0.3081,))
                                                 ]))
        test_loader = DataLoader(test_data, batch_size=batch_size,
                                 pin_memory=True, shuffle=False)
        return test_loader
    if dataset == 'semeion_fmnist_test':
        test_data = torchvision.datasets.SEMEION(root_dir, download=True,
                                                 transform=torchvision.transforms.Compose([
                                                     torchvision.transforms.Pad(
                                                         6),
                                                     torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.Normalize(
                                                         (0.2860,), (0.3205,))
                                                 ]))
        test_loader = DataLoader(test_data, batch_size=batch_size,
                                 pin_memory=True, shuffle=False)
        return test_loader
    raise NotImplementedError('Dataset is not supported')

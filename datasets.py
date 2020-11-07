import torch
import torchvision
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import DataLoader, Subset, distributed


def infinite_wrapper(loader):
    while True:
        for x in loader:
            yield x

def get_distributed_data_loader(dataset, num_replicas, rank, train_batch_size=64, test_batch_size=64, seed=42, root_dir='data/'):
    if dataset == 'cifar10':
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        augment_transform = [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip()
        ]
        train_transform = torchvision.transforms.Compose([
            *augment_transform,
            transform
        ])
        train_data = torchvision.datasets.CIFAR10(root_dir, train=True, download=True, transform=train_transform)
        train_sampler = distributed.DistributedSampler(train_data, seed=seed, shuffle=True, drop_last=True)
        train_loader = DataLoader(train_data, batch_size=train_batch_size, pin_memory=True, shuffle=False, num_workers=0, sampler=train_sampler, drop_last=True)

        test_data = torchvision.datasets.CIFAR10(root_dir, train=False, download=True, transform=transform)
        test_sampler = distributed.DistributedSampler(test_data, shuffle=False, drop_last=False)
        test_loader = DataLoader(test_data, batch_size=test_batch_size, pin_memory=True, shuffle=False, num_workers=0, sampler=test_sampler, drop_last=False)
        
        return train_loader, test_loader, train_sampler, test_sampler
    if dataset == 'cifar100':
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        augment_transform = [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip()
        ]
        train_transform = torchvision.transforms.Compose([
            *augment_transform,
            transform
        ])
        train_data = torchvision.datasets.CIFAR100(root_dir, train=True, download=True, transform=train_transform)
        train_sampler = distributed.DistributedSampler(train_data, seed=seed, shuffle=True, drop_last=True)
        train_loader = DataLoader(train_data, batch_size=train_batch_size, pin_memory=True, shuffle=False, num_workers=0, sampler=train_sampler, drop_last=True)

        test_data = torchvision.datasets.CIFAR100(root_dir, train=False, download=True, transform=transform)
        test_sampler = distributed.DistributedSampler(test_data, shuffle=False, drop_last=False)
        test_loader = DataLoader(test_data, batch_size=test_batch_size, pin_memory=True, shuffle=False, num_workers=0, sampler=test_sampler)
        
        return train_loader, test_loader, train_sampler, test_sampler
    if dataset == 'vgg_cifar10':
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        augment_transform = [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip()
        ]
        train_transform = torchvision.transforms.Compose([
            *augment_transform,
            transform
        ])
        train_data = torchvision.datasets.CIFAR10(root_dir, train=True, download=True, transform=train_transform)
        train_sampler = distributed.DistributedSampler(train_data, seed=seed, shuffle=True, drop_last=True)
        train_loader = DataLoader(train_data, batch_size=train_batch_size, pin_memory=True, shuffle=False, num_workers=0, sampler=train_sampler, drop_last=True)

        test_data = torchvision.datasets.CIFAR10(root_dir, train=False, download=True, transform=transform)
        test_sampler = distributed.DistributedSampler(test_data, shuffle=False, drop_last=False)
        test_loader = DataLoader(test_data, batch_size=test_batch_size, pin_memory=True, shuffle=False, num_workers=0, sampler=test_sampler)
        
        return train_loader, test_loader, train_sampler, test_sampler
    if dataset == 'vgg_cifar100':
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        augment_transform = [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip()
        ]
        train_transform = torchvision.transforms.Compose([
            *augment_transform,
            transform
        ])
        train_data = torchvision.datasets.CIFAR100(root_dir, train=True, download=True, transform=train_transform)
        train_sampler = distributed.DistributedSampler(train_data, seed=seed, shuffle=True, drop_last=True)
        train_loader = DataLoader(train_data, batch_size=train_batch_size, pin_memory=True, shuffle=False, num_workers=0, sampler=train_sampler, drop_last=True)

        test_data = torchvision.datasets.CIFAR100(root_dir, train=False, download=True, transform=transform)
        test_sampler = distributed.DistributedSampler(test_data, shuffle=False, drop_last=False)
        test_loader = DataLoader(test_data, batch_size=test_batch_size, pin_memory=True, shuffle=False, num_workers=0, sampler=test_sampler)
        
        return train_loader, test_loader, train_sampler, test_sampler

def get_data_loader(dataset, batch_size=64, validation=False, validation_fraction=0.1, random_state=42, root_dir='data/', test_only=False, train_only=False, augment=True, degree=0):
    if dataset == 'mnist' and degree != 0:
        test_data = torchvision.datasets.MNIST(root_dir, train=False, download=True,
                                               transform=torchvision.transforms.Compose([
                                                   torchvision.transforms.RandomRotation((degree, degree)),
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
                                                          torchvision.transforms.RandomRotation((degree, degree)),
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
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
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
            train_loader = DataLoader(train_data, batch_size=batch_size, pin_memory=True, shuffle=True, drop_last=True)
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
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
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
            train_loader = DataLoader(train_data, batch_size=batch_size, pin_memory=True, shuffle=True, drop_last=True)
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
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
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
            train_loader = DataLoader(train_data, batch_size=batch_size, pin_memory=True, shuffle=True, drop_last=True)
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
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
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
            train_loader = DataLoader(train_data, batch_size=batch_size, pin_memory=True, shuffle=True, drop_last=True)
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

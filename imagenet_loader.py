import sys
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import webdataset as wds

def make_train_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )


def make_val_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    return transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize,]
    )

def worker_urls(urls):
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        rank = 0
        world_size = 1
    result = wds.worker_urls(urls[rank::world_size])
    print("worker_urls returning", len(result), "of", len(urls), "urls", file=sys.stderr)
    return result

def identity(x):
    return x

def make_train_loader_wds(trainshards, trainsize, batch_size, workers, shuffle_buffer):
    print("=> using WebDataset loader")
    train_transform = make_train_transform()
    num_batches = trainsize // batch_size
    dataset = (
        wds.Dataset(trainshards, length=num_batches, shard_selection=worker_urls)
        .shuffle(shuffle_buffer)
        .decode("pil")
        .to_tuple("jpg;png;jpeg cls")
        .map_tuple(train_transform, identity)
        .batched(batch_size//torch.distributed.get_world_size())
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=None, shuffle=False, num_workers=workers,
    )
    return loader

def make_val_loader_wds(valshards, valsize, batch_size, workers):
    print("=> using WebDataset loader")
    train_transform = make_train_transform()
    num_batches = valsize // batch_size
    dataset = (
        wds.Dataset(valshards, length=num_batches, shard_selection=worker_urls)
        .decode("pil")
        .to_tuple("jpg;png;jpeg cls")
        .map_tuple(train_transform, identity)
        .batched(batch_size//torch.distributed.get_world_size())
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=None, shuffle=False, num_workers=workers,
    )
    return loader



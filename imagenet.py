import argparse
import ast
import logging
import logging.handlers
import os
from bisect import bisect
from itertools import chain
from time import sleep
import numpy as np
import torch
import time
import torch.distributed as dist
import torch.distributions as D
import torch.multiprocessing as mp
import torch.nn as nn
import torchvision
import webdataset as wds
import json
import sys
from torch.nn.parallel import DistributedDataParallel as DDP

from imagenet_loader import ImageFolderLMDB
from models import count_parameters
from models.resnet50 import resnet50
from models.utils import bn_update

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
END_MSG = "PROCESS_END"

def identity(x):
    return x

def worker_urls(urls):
    result = wds.worker_urls(urls)
    print("worker_urls returning", len(result),
          "of", len(urls), "urls", file=sys.stderr)
    return result

class StoreDictKeyPair(argparse.Action):
    def __init__(self, option_strings, dest, keys, nargs=None, const=None, default=None, type=None, choices=None, required=False, help=None, metavar=None):
        super().__init__(option_strings, dest, nargs, const,
                         default, type, choices, required, help, metavar)
        self.keys = keys

    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = ast.literal_eval(values)
        my_dict = {
            k: v for k, v in my_dict.items() if k in self.keys
        }
        setattr(namespace, self.dest, my_dict)


def worker_configurer(queue):
    h = logging.handlers.QueueHandler(queue)  # Just the one handler needed
    root = logging.getLogger()
    root.addHandler(h)
    # send all messages, for demo; no other level or filter logic applied.
    root.setLevel(logging.INFO)


def listener_configurer(path):
    root = logging.getLogger()
    file_handler = logging.FileHandler(path)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    root.addHandler(console_handler)
    root.setLevel(logging.INFO)


def listener_process(queue, path, n_processes):
    listener_configurer(path)
    end_count = 0
    while end_count < n_processes:
        while not queue.empty():
            record = queue.get()
            if record.message == END_MSG:
                end_count += 1
            logger = logging.getLogger(record.name)
            # No level or filter logic applied - just do it!
            logger.handle(record)
        sleep(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--kl_weight', help="kl_weight",
                        action=StoreDictKeyPair, keys=('kl_min', 'kl_max', 'last_iter'))
    parser.add_argument('--batch_size', action=StoreDictKeyPair,
                        keys=('train', 'test'), help='batch size per gpu', default={'train':256,'test':500})
    parser.add_argument('--root', type=str, help='Directory')
    parser.add_argument('--prior', dest="prior",
                        action=StoreDictKeyPair, keys=('mean', 'std'))
    parser.add_argument('--n_components', default=2,
                        type=int, help='Number of components')
    parser.add_argument('--det_params', help="det_params",
                        action=StoreDictKeyPair, keys=('lr', 'weight_decay'), default={'lr':0.1,'weight_decay':0.0001})
    parser.add_argument('--sto_params', help="sto_params",
                        action=StoreDictKeyPair, keys=('lr', 'weight_decay'))
    parser.add_argument('--sgd_params', help="sgd_params",
                        action=StoreDictKeyPair, keys=('momentum', 'dampening', 'nesterov'), default={'momentum': 0.9, 'dampening': 0.0, 'nesterov': True})
    parser.add_argument('--num_epochs', default=135,
                        type=int, help='Number of epochs')
    parser.add_argument('--num_sample', help='num_sample',
                        action=StoreDictKeyPair, keys=('train', 'test'))
    parser.add_argument('--logging_freq', default=1,
                        type=int, help='Logging frequency')
    parser.add_argument('--test_freq', default=15,
                        type=int, help='Testing frequency')
    parser.add_argument('--schedule', help='lr schedule',
                        action=StoreDictKeyPair, keys=('det', 'sto'),
                        default={'det': [(0.1, 30), (0.01, 60), (0.001, 80)], 'sto': []})
    parser.add_argument('--warmup', help='warm up epochs', type=int, default=5)
    parser.add_argument('--posterior', help="posterior",
                        action=StoreDictKeyPair, keys=('mean_init', 'std_init'))
    parser.add_argument('--use_pretrained', action='store_true')
    parser.add_argument('--nodes', default=1, type=int, metavar='N',
                        help='number of nodes (default: 1)')
    parser.add_argument('--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--traindir', default='data/imagenet/train.lmdb', type=str)
    parser.add_argument('--valdir', default='data/imagenet/val.lmdb', type=str)
    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes
    args.total_batch_size = args.batch_size['train']
    args.batch_size['train'] //= args.world_size
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8888'
    if args.nr == 0:
        os.makedirs(args.root, exist_ok=True)
        print(args)
        with open(os.path.join(args.root, 'config.json'), 'w') as out:
            json.dump(vars(args), out, indent=2)
    ctx = mp.get_context('spawn')
    queue = ctx.Queue(-1)
    listener = mp.Process(
        target=listener_process, args=(queue, os.path.join(args.root, f'train.log'), args.gpus))
    listener.start()
    ctx = mp.spawn(train, nprocs=args.gpus, args=(args, queue), join=True)
#     ctx.join()


def get_kl_weight(epoch, args):
    kl_max = args.kl_weight['kl_max']
    kl_min = args.kl_weight['kl_min']
    last_iter = args.kl_weight['last_iter']
    value = (kl_max-kl_min)/last_iter
    return min(kl_max, kl_min + epoch*value)


def schedule(step, steps_per_epoch, warm_up, multipliers):
    lr_epoch = step / steps_per_epoch
    factor = 1.0
    if warm_up >= 1:
        factor = min(1.0, lr_epoch / warm_up)
    for mult, start_epoch in multipliers[::-1]:
        if lr_epoch >= start_epoch:
            return mult
    return factor



def vb_loss(model, x, y, n_sample):
    y = y.unsqueeze(1).expand(-1, n_sample)
    logits, kl = model(x, n_sample, return_kl=True)
    logp = D.Categorical(logits=logits).log_prob(y).mean()
    return -logp, kl


def parallel_nll(model, x, y, n_sample):
    n_components = model.module.n_components
    indices = torch.empty(x.size(0)*n_sample,
                          dtype=torch.long, device=x.device)
    prob = torch.cat([model(x, n_sample, indices=torch.full((x.size(0)*n_sample,), idx,
                                                            out=indices, device=x.device, dtype=torch.long)) for idx in range(n_components)], dim=1)
    logp = D.Categorical(logits=prob).log_prob(
        y.unsqueeze(1).expand(-1, n_components*n_sample))
    logp = torch.logsumexp(logp, 1) - torch.log(torch.tensor(n_components *
                                                             n_sample, dtype=torch.float32, device=x.device))
    return -logp.mean(), prob


def get_model(args, gpu, dataloader):
    step_per_epoch = 1281167 // args.total_batch_size
    model = resnet50(deterministic_pretrained=args.use_pretrained, n_components=args.n_components,
                     prior_mean=args.prior['mean'], prior_std=args.prior['std'], posterior_mean_init=args.posterior['mean_init'], posterior_std_init=args.posterior['std_init'])
    model.cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    detp = []
    stop = []
    for name, param in model.named_parameters():
        if 'posterior' in name or 'prior' in name:
            stop.append(param)
        else:
            detp.append(param)
    optimizer = torch.optim.SGD(
        [{
            'params': detp,
            **args.det_params
        }, {
            'params': stop,
            **args.sto_params
        }], **args.sgd_params)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [lambda step: schedule(step, step_per_epoch, args.warmup, args.schedule['det']), lambda step: schedule(step, step_per_epoch, args.warmup, args.schedule['sto'])])
    model = DDP(model, device_ids=[gpu])
    return model, optimizer, scheduler


def get_dataloader(args, rank):
    trainshards='./data/imagenet/shards/imagenet-train-{000000..001281}.tar'
    valshards='./data/imagenet/shards/imagenet-val-{000000..000049}.tar'
    trainsize = 1281167
    train_batch_size = args.batch_size['train']
    test_batch_size = args.batch_size['test']
    shuffle_buffer = 1000
    num_test_workers = num_train_workers = args.workers
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            normalize,
        ]
    )
    val_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(), 
            normalize
        ]
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


def get_logger(args, logger, rank):
    fh = logging.FileHandler(os.path.join(args.root, f'process{rank}.log'))
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def test_nll(model, loader, num_sample):
    model.eval()
    with torch.no_grad():
        nll = 0
        acc = 0
        for bx, by in loader:
            bx = bx.cuda(non_blocking=True)
            by = by.cuda(non_blocking=True)
            with torch.cuda.amp.autocast():
                bnll, pred = parallel_nll(model, bx, by, num_sample)
            nll += bnll.item() * bx.size(0)
            acc += (pred.exp().mean(1).argmax(-1) == by).sum().item()
        acc /= len(loader.dataset)
        nll /= len(loader.dataset)
    return nll, acc


def train(gpu, args, queue):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    worker_configurer(queue)
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(
        backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    logger = logging.getLogger(f'worker-{rank}')
    logger.info('Get data loader')
    train_loader, test_loader = get_dataloader(args, rank)
    if rank == 0:
        logger.info(
            f"Train size: {len(train_loader.dataset)}, test size: {len(test_loader.dataset)}")
    n_batch = len(train_loader)
    logger.info(
        f"Train size: {len(train_loader)}, test size: {len(test_loader)}")
    torch.manual_seed(args.seed + rank*123)
    model, optimizer, scheduler = get_model(args, 0, train_loader)
    if rank == 0:
        count_parameters(model, logger)
        logger.info(str(model))
    checkpoint_dir = os.path.join(args.root, 'checkpoint_{}.pt')
    model.train()
    scaler = torch.cuda.amp.GradScaler()
    for i in range(args.num_epochs):
        t0 = time.time()
        lls = []
        for bx, by in train_loader:
            bx = bx.cuda(non_blocking=True)
            by = by.cuda(non_blocking=True)
            with torch.cuda.amp.autocast():
                loglike, kl = vb_loss(model, bx, by, args.num_sample['train'])
                klw = get_kl_weight(i, args)
                loss = loglike + klw*kl/(n_batch*args.total_batch_size)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            # loss.backward()
            # optimizer.step()
            scheduler.step()
            scaler.update()
            
            optimizer.zero_grad()

            print("VB Epoch %d: loglike: %.4f, kl: %.4f, kl weight: %.4f, lr1: %.4f, lr2: %.4f" % (i, loglike.item(), kl.item(), klw, optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr']))
            lls.append(loglike.item())
        t1 = time.time()
        if (i+1) % args.logging_freq == 0:
            logger.info("VB Epoch %d: loglike: %.4f, kl: %.4f, kl weight: %.4f, lr1: %.4f, lr2: %.4f, time: %.1f",
                        i, np.mean(lls).item(), kl.item(), klw, optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'], t1-t0)
        if (i+1) % args.test_freq == 0:
            if rank == 0:
                torch.save(model.module.state_dict(), checkpoint_dir.format(str(i)))
                logger.info('Save checkpoint')
            nll, acc = test_nll(model, test_loader,
                                args.num_sample['test'])
            logger.info(
                "VB Epoch %d: test NLL %.4f, acc %.4f", i, nll, acc)
            model.train()
    if rank == 0:
        torch.save(model.module.state_dict(), checkpoint_dir.format('final'))
        logger.info('Save checkpoint')
    tnll = 0
    acc = 0
    nll_miss = 0
    model.eval()
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.cuda(non_blocking=True)
            by = by.cuda(non_blocking=True)
            indices = torch.empty(
                bx.size(0)*args.num_sample['test'], dtype=torch.long, device=bx.device)
            prob = torch.cat([model(bx, args.num_sample['test'], indices=torch.full((bx.size(0)*args.num_sample['test'],),
                                                                                    idx, out=indices, device=bx.device, dtype=torch.long)) for idx in range(model.module.n_components)], dim=1)
            y_target = by.unsqueeze(
                1).expand(-1, args.num_sample['test']*model.module.n_components)
            bnll = D.Categorical(logits=prob).log_prob(y_target)
            bnll = torch.logsumexp(bnll, dim=1) - torch.log(torch.tensor(
                args.num_sample['test']*model.module.n_components, dtype=torch.float32, device=bnll.device))
            tnll -= bnll.sum().item()
            vote = prob.exp().mean(dim=1)
            pred = vote.argmax(dim=1)

            y_miss = pred != by
            if y_miss.sum().item() > 0:
                nll_miss -= bnll[y_miss].sum().item()
            acc += (pred == by).sum().item()
    nll_miss /= len(test_loader.dataset) - acc
    tnll /= len(test_loader.dataset)
    acc /= len(test_loader.dataset)
    logger.info("Test data: acc %.4f, nll %.4f, nll miss %.4f" %
                (acc, tnll, nll_miss))
    logger.info(END_MSG)


if __name__ == '__main__':
    main()

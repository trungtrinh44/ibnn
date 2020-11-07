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
import torch.distributed as dist
import torch.distributions as D
import torch.multiprocessing as mp
import torch.nn as nn
import torchvision
import json
from torch.nn.parallel import DistributedDataParallel as DDP

from datasets import get_distributed_data_loader
from models import (BayesianVGG16, BayesianWideResNet28x10, DetVGG16,
                    DetWideResNet28x10, StoVGG16, StoWideResNet28x10,
                    count_parameters)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
END_MSG = "PROCESS_END"

class StoreDictKeyPair(argparse.Action): 
    def __init__(self, option_strings, dest, keys, nargs=None, const=None, default=None, type=None, choices=None, required=False, help=None, metavar=None): 
        super().__init__(option_strings, dest, nargs, const, default, type, choices, required, help, metavar) 
        self.keys=keys 
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
    formatter = logging.Formatter('%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s')
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
            logger.handle(record)  # No level or filter logic applied - just do it!
        sleep(1)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--model', default='StoWideResNet28x10', type=str, help='model name')
    parser.add_argument('--kl_weight', dest="kl_weight", action=StoreDictKeyPair, keys=('kl_min', 'kl_max', 'last_iter'))
    parser.add_argument('--batch_size', dest='batch_size', action=StoreDictKeyPair, keys=('train', 'test'), help='batch size per gpu')
    parser.add_argument('--root', type=str, help='Directory')
    parser.add_argument('--prior', dest="prior", action=StoreDictKeyPair, keys=('mean', 'std'))
    parser.add_argument('--n_components', default=2, type=int, help='Number of components')
    parser.add_argument('--det_params', dest="det_params", action=StoreDictKeyPair, keys=('lr', 'weight_decay'))
    parser.add_argument('--sto_params', dest="sto_params", action=StoreDictKeyPair, keys=('lr', 'weight_decay'))
    parser.add_argument('--sgd_params', dest="sgd_params", action=StoreDictKeyPair, keys=('momentum', 'dampening', 'nesterov'))
    parser.add_argument('--num_epochs', default=300, type=int, help='Number of epochs')
    parser.add_argument('--num_sample', dest='num_sample', action=StoreDictKeyPair, keys=('train', 'test'))
    parser.add_argument('--logging_freq', default=1, type=int, help='Logging frequency')
    parser.add_argument('--test_freq', default=15, type=int, help='Testing frequency')
    parser.add_argument('--dataset', default='cifar100', type=str, help='Name of dataset')
    parser.add_argument('--lr_ratio', dest='lr_ratio', action=StoreDictKeyPair, keys=('det', 'sto'))
    parser.add_argument('--posterior', dest="posterior", action=StoreDictKeyPair, keys=('mean_init', 'std_init'))
    parser.add_argument('--milestones', nargs=2, type=float)
    parser.add_argument('--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('--nr', default=0, type=int,
                        help='ranking within the nodes')
    args = parser.parse_args()
    if args.dataset == 'cifar100' or args.dataset == 'vgg_cifar100':
        args.num_classes = 100
    elif args.dataset == 'cifar10' or args.dataset == 'vgg_cifar10':
        args.num_classes = 10
    args.world_size = args.gpus * args.nodes
    args.total_batch_size = args.batch_size['train']
    args.batch_size['train'] //=  args.world_size
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8888'
    if args.nr == 0:
        os.makedirs(args.root, exist_ok=True)
        print(args)
        with open(os.path.join(args.root, 'config.json'), 'w') as out:
            json.dump(vars(args), out, indent=2)
    ctx = mp.get_context('spawn')
    queue = ctx.Queue(-1)
    ctx = mp.spawn(train, nprocs=args.gpus, args=(args, queue), join=False)
    listener = mp.Process(
        target=listener_process, args=(queue, os.path.join(args.root, f'train.log'), args.gpus))
    listener.start()
    ctx.join()

def get_kl_weight(epoch, args):
    kl_max = args.kl_weight['kl_max']
    kl_min = args.kl_weight['kl_min']
    last_iter = args.kl_weight['last_iter']
    value = (kl_max-kl_min)/last_iter
    return min(kl_max, kl_min + epoch*value)

def schedule(num_epochs, epoch, milestones, lr_ratio):
    t = epoch / num_epochs
    m1, m2 = milestones
    if t <= m1:
        factor = 1.0
    elif t <= m2:
        factor = 1.0 - (1.0 - lr_ratio) * (t - m1) / (m2 - m1)
    else:
        factor = lr_ratio
    return factor

def vb_loss(model, x, y, n_sample):
    y = y.unsqueeze(1).expand(-1, n_sample)
    logits, kl = model(x, n_sample, return_kl=True)
    logp = D.Categorical(logits=logits).log_prob(y).mean()
    return -logp, kl

def parallel_nll(model, x, y, n_sample):
    n_components = model.module.n_components
    indices = torch.empty(x.size(0)*n_sample, dtype=torch.long, device=x.device)
    prob = torch.cat([model(x, n_sample, indices=torch.full((x.size(0)*n_sample,), idx, out=indices, device=x.device, dtype=torch.long)) for idx in range(n_components)], dim=1)
    logp = D.Categorical(logits=prob).log_prob(y.unsqueeze(1).expand(-1, n_components*n_sample))
    logp = torch.logsumexp(logp, 1) - torch.log(torch.tensor(n_components*n_sample, dtype=torch.float32, device=x.device))
    return -logp.mean(), prob

def get_model(args, gpu):
    if args.model == 'StoWideResNet28x10':
        model = StoWideResNet28x10(args.num_classes, args.n_components, args.prior['mean'], args.prior['std'])
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
            },{
                'params': stop,
                **args.sto_params
            }], **args.sgd_params)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [lambda e: schedule(args.num_epochs, e, args.milestones, args.lr_ratio['det']), lambda e: schedule(args.num_epochs, e, args.milestones, args.lr_ratio['sto'])])
    elif args.model == 'StoVGG16':
        model = StoVGG16(args.num_classes, args.n_components, args.prior['mean'], args.prior['std'], args.posterior['mean_init'], args.posterior['std_init'])
        model.cuda(gpu)
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
            },{
                'params': stop,
                **args.sto_params
            }], **args.sgd_params)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [lambda e: schedule(args.num_epochs, e, args.milestones, args.lr_ratio['det']), lambda e: schedule(args.num_epochs, e, args.milestones, args.lr_ratio['sto'])])
    
    model = DDP(model, device_ids=[gpu])
    return model, optimizer, scheduler


def get_dataloader(args, rank):
    return get_distributed_data_loader(dataset=args.dataset, num_replicas=args.world_size, rank=rank, train_batch_size=args.batch_size['train'], 
                                       test_batch_size=args.batch_size['test'], seed=args.seed)


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
            bnll, pred = parallel_nll(model, bx, by, num_sample)
            nll += bnll.item() * bx.size(0)
            acc += (pred.exp().mean(1).argmax(-1) == by).sum().item()
        acc /= len(loader.dataset)
        nll /= len(loader.dataset)
    return nll, acc


def train(gpu, args, queue):
    worker_configurer(queue)
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    logger = logging.getLogger(f'worker-{rank}')
    train_loader, test_loader, train_sampler, test_sampler = get_dataloader(args, rank)
    if rank == 0:
        logger.info(f"Train size: {len(train_loader.dataset)}, test size: {len(test_loader.dataset)}")
    n_batch = len(train_loader)
    logger.info(f"Train size: {len(train_loader)}, test size: {len(test_loader)}")
    torch.manual_seed(args.seed + rank*123)
    torch.cuda.set_device(gpu)
    model, optimizer, scheduler = get_model(args, gpu)
    if rank == 0:
        count_parameters(model, logger)
        logger.info(str(model))
    checkpoint_dir = os.path.join(args.root, 'checkpoint.pt')
    if args.model.startswith('Sto'):
        model.train()
        for i in range(args.num_epochs):
            train_sampler.set_epoch(i)
            for bx, by in train_loader:
                bx = bx.cuda(non_blocking=True)
                by = by.cuda(non_blocking=True)
                loglike, kl = vb_loss(model, bx, by, args.num_sample['train'])
                klw = get_kl_weight(i, args)
                loss = loglike + klw*kl/(n_batch*args.total_batch_size)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            if (i+1) % args.logging_freq == 0:
                logger.info("VB Epoch %d: loglike: %.4f, kl: %.4f, kl weight: %.4f, lr1: %.4f, lr2: %.4f",
                            i, loglike.item(), kl.item(), klw, optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'])
            if (i+1) % args.test_freq == 0:
                if rank == 0:
                    torch.save(model.module.state_dict(), checkpoint_dir)
                    logger.info('Save checkpoint')
                with torch.no_grad():
                    nll, acc = test_nll(model, test_loader, args.num_sample['test'])
#                 if rank == 0:
                logger.info("VB Epoch %d: test NLL %.4f, acc %.4f", i, nll, acc)
                model.train()
        if rank == 0:
            torch.save(model.module.state_dict(), checkpoint_dir)
            logger.info('Save checkpoint')
        tnll = 0
        acc = 0
        nll_miss = 0
        model.eval()
        with torch.no_grad():
            for bx, by in test_loader:
                bx = bx.cuda(non_blocking=True)
                by = by.cuda(non_blocking=True)
                indices = torch.empty(bx.size(0)*args.num_sample['test'], dtype=torch.long, device=bx.device)
                prob = torch.cat([model(bx, args.num_sample['test'], indices=torch.full((bx.size(0)*args.num_sample['test'],), idx, out=indices, device=bx.device, dtype=torch.long)) for idx in range(model.module.n_components)], dim=1)
                y_target = by.unsqueeze(1).expand(-1, args.num_sample['test']*model.module.n_components)
                bnll = D.Categorical(logits=prob).log_prob(y_target)
                bnll = torch.logsumexp(bnll, dim=1) - torch.log(torch.tensor(args.num_sample['test']*model.module.n_components, dtype=torch.float32, device=bnll.device))
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
        #if rank == 0:
        logger.info("Test data: acc %.4f, nll %.4f, nll miss %.4f" % (acc, tnll, nll_miss))
    logger.info(END_MSG)

if __name__ == '__main__':
    main()

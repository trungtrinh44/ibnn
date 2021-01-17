import argparse
import ast
import os
from time import sleep
import numpy as np
import torch
import time
import torch.distributed as dist
import torch.distributions as D
import torch.multiprocessing as mp
import torch.nn as nn
import torchvision
import json
from torch.nn.parallel import DistributedDataParallel as DDP
import random

from imagenet_loader import get_dali_train_loader, get_dali_val_loader
from models import count_parameters
from models.resnet50 import resnet50

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
END_MSG = "PROCESS_END"


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--kl_weight', help="kl_weight",
                        action=StoreDictKeyPair, keys=('kl_min', 'kl_max', 'last_iter'))
    parser.add_argument('--batch_size', action=StoreDictKeyPair,
                        keys=('train', 'test'), help='batch size per gpu', default={'train':256,'test':500})
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
    parser.add_argument('--start_epoch', default=0,
                        type=int, help='Start from epoch')
    parser.add_argument('--schedule', help='lr schedule',
                        action=StoreDictKeyPair, keys=('det', 'sto'),
                        default={'det': [(0.1, 30), (0.01, 60), (0.001, 80)], 'sto': []})
    parser.add_argument('--warmup', help='warm up epochs', type=int, default=5)
    parser.add_argument('--posterior', help="posterior",
                        action=StoreDictKeyPair, keys=('mean_init', 'std_init'))
    parser.add_argument('--use_pretrained', action='store_true')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--traindir', default='data/imagenet/train.lmdb', type=str)
    parser.add_argument('--valdir', default='data/imagenet/val.lmdb', type=str)
    args = parser.parse_args()
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.rank = int(os.environ["RANK"])
    args.root = os.environ["ROOT_DIR"]

    args.gpu = args.local_rank % torch.cuda.device_count()
    print(args.gpu)
    torch.cuda.set_device(args.gpu)
    dist.init_process_group(backend='nccl', init_method='env://')
    
    args.world_size = torch.distributed.get_world_size()
    args.total_batch_size = args.batch_size['train']
    args.batch_size['train'] //= args.world_size
    print("Using seed = {}".format(args.seed))
    torch.manual_seed(args.seed + args.local_rank)
    torch.cuda.manual_seed(args.seed + args.local_rank)
    np.random.seed(seed=args.seed + args.local_rank)
    random.seed(args.seed + args.local_rank)
    train(args)

def get_kl_weight(iteration, args):
    kl_max = args.kl_weight['kl_max']
    kl_min = args.kl_weight['kl_min']
    last_iter = args.kl_weight['last_iter']*args.step_per_epoch
    value = (kl_max-kl_min)/last_iter
    return min(kl_max, kl_min + iteration*value)


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
                                                             n_sample, dtype=logp.dtype, device=x.device))
    return -logp.mean(), prob


def get_model(args, dataloader):
    args.step_per_epoch = step_per_epoch = 1281167 // args.total_batch_size
    model = resnet50(deterministic_pretrained=args.use_pretrained, n_components=args.n_components,
                     prior_mean=args.prior['mean'], prior_std=args.prior['std'], posterior_mean_init=args.posterior['mean_init'], posterior_std_init=args.posterior['std_init'])
    model.cuda()
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    detp = []
    stop = []
    bnp = []
    for name, param in model.named_parameters():
        if 'posterior' in name or 'prior' in name:
            stop.append(param)
        else:
            detp.append(param)
    for n in model.modules():
        if isinstance(n, torch.nn.modules.batchnorm._BatchNorm):
            bnp.extend([n.weight, n.bias])
    detp = list(set(detp) - set(bnp))
    optimizer = torch.optim.SGD(
        [{
            'params': detp,
            'initial_lr': args.det_params['lr'],
            **args.det_params
        }, {
            'params': stop,
            'initial_lr': args.sto_params['lr'],
            **args.sto_params
        }, {
            'params': bnp,
            'lr': args.det_params['lr'],
            'initial_lr': args.det_params['lr'],
            'weight_decay': 0.0
        }], **args.sgd_params)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [
        lambda step: lr_cosine_policy(epoch=step, total_epochs=args.num_epochs*step_per_epoch, 
                                      warmup_length=args.warmup*step_per_epoch), 
        lambda step: schedule(step, step_per_epoch, args.warmup, args.schedule['sto']), 
        lambda step: lr_cosine_policy(epoch=step, total_epochs=args.num_epochs*step_per_epoch, 
                                      warmup_length=args.warmup*step_per_epoch)
    ])
    args.step_per_epoch = step_per_epoch
    model = DDP(model, device_ids=[args.gpu])
    return model, optimizer, scheduler


def get_dataloader(args):
    get_train_loader = get_dali_train_loader(dali_cpu=True)
    get_val_loader = get_dali_val_loader()
    train_loader, train_loader_len = get_train_loader(
        path=[f"data/imagenet/tf_records/train/train-{i:05d}-of-01024" for i in range(1024)],
        index_path=[f"data/imagenet/tf_records/train/train-{i:05d}-of-01024.txt" for i in range(1024)],
        batch_size=args.batch_size["train"],
        num_classes=1000,
        one_hot=False,
        start_epoch=args.start_epoch,
        workers=args.workers,
    )

    val_loader, val_loader_len = get_val_loader(
        path=[f"data/imagenet/tf_records/validation/validation-{i:05d}-of-00128" for i in range(128)],
        index_path=[f"data/imagenet/tf_records/validation/validation-{i:05d}-of-00128.txt" for i in range(128)],
        batch_size=args.batch_size["test"],
        num_classes=1000,
        one_hot=False,
        workers=args.workers
    )

    return train_loader, val_loader, train_loader_len, val_loader_len

def lr_cosine_policy(warmup_length, epoch, total_epochs):
    if epoch < warmup_length:
        factor = epoch / warmup_length
    else:
        e = epoch - warmup_length
        es = total_epochs - warmup_length
        factor = 0.5 * (1 + np.cos(np.pi * e / es))
    return factor

def test_nll(model, loader, num_sample):
    model.eval()
    with torch.no_grad():
        nll = 0
        acc = 0
        count = 0
        for bx, by in loader:
            count += bx.size(0)
            with torch.cuda.amp.autocast():
                bnll, pred = parallel_nll(model, bx, by, num_sample)
            nll += bnll.item() * bx.size(0)
            acc += (pred.exp().mean(1).argmax(-1) == by).sum().item()
        acc /= count
        nll /= count
    return nll, acc


def train(args):
    print('Get data loader')
    train_loader, test_loader, train_loader_len, test_loader_len = get_dataloader(args)
    n_batch = train_loader_len
    print(f"Train size: {train_loader_len}, test size: {test_loader_len}")
    model, optimizer, scheduler = get_model(args, train_loader)
    if args.rank == 0:
        count_parameters(model)
        print(str(model))
    checkpoint_dir = os.path.join(args.root, 'checkpoint_{}.pt')
    model.train()
    scaler = torch.cuda.amp.GradScaler()
    iteration = 0
    for i in range(args.num_epochs):
        t0 = time.time()
        lls = []
        for bx, by in train_loader:
            optimizer.zero_grad()
            klw = get_kl_weight(iteration, args)
            iteration += 1
            with torch.cuda.amp.autocast():
                loglike, kl = vb_loss(model, bx, by, args.num_sample['train'])
                loss = loglike + klw*kl/(n_batch*args.total_batch_size)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            print("VB Epoch %d: loglike: %.4f, kl: %.4f, kl weight: %.4f, lr1: %.4f, lr2: %.4f, time: %.2f" % (i, loglike.item(), kl.item(), klw, optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'], time.time()-t0))
            lls.append(loglike.item())
        t1 = time.time()
        if (i+1) % args.logging_freq == 0:
            print("VB Epoch %d: loglike: %.4f, kl: %.4f, kl weight: %.4f, lr1: %.4f, lr2: %.4f, time: %.1f" % (i, np.mean(lls).item(), kl.item(), klw, optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'], t1-t0))
        if (i+1) % args.test_freq == 0:
            if args.rank == 0:
                torch.save(model.module.state_dict(), checkpoint_dir.format(str(i)))
                print('Save checkpoint')
            nll, acc = test_nll(model, test_loader,
                                args.num_sample['test'])
            print("VB Epoch %d: test NLL %.4f, acc %.4f" % (i, nll, acc))
            model.train()
    if args.rank == 0:
        torch.save(model.module.state_dict(), checkpoint_dir.format('final'))
        print('Save checkpoint')
    # tnll = 0
    # acc = 0
    # nll_miss = 0
    # model.eval()
    # with torch.no_grad():
    #     for bx, by in test_loader:
    #         bx = bx.cuda(non_blocking=True)
    #         by = by.cuda(non_blocking=True)
    #         indices = torch.empty(
    #             bx.size(0)*args.num_sample['test'], dtype=torch.long, device=bx.device)
    #         prob = torch.cat([model(bx, args.num_sample['test'], indices=torch.full((bx.size(0)*args.num_sample['test'],),
    #                                                                                 idx, out=indices, device=bx.device, dtype=torch.long)) for idx in range(model.module.n_components)], dim=1)
    #         y_target = by.unsqueeze(
    #             1).expand(-1, args.num_sample['test']*model.module.n_components)
    #         bnll = D.Categorical(logits=prob).log_prob(y_target)
    #         bnll = torch.logsumexp(bnll, dim=1) - torch.log(torch.tensor(
    #             args.num_sample['test']*model.module.n_components, dtype=torch.float32, device=bnll.device))
    #         tnll -= bnll.sum().item()
    #         vote = prob.exp().mean(dim=1)
    #         pred = vote.argmax(dim=1)

    #         y_miss = pred != by
    #         if y_miss.sum().item() > 0:
    #             nll_miss -= bnll[y_miss].sum().item()
    #         acc += (pred == by).sum().item()
    # nll_miss /= len(test_loader.dataset) - acc
    # tnll /= len(test_loader.dataset)
    # acc /= len(test_loader.dataset)
    # print("Test data: acc %.4f, nll %.4f, nll miss %.4f" % (acc, tnll, nll_miss))
    print(END_MSG)


if __name__ == '__main__':
    main()
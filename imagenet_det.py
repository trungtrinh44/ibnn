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
#from torch.nn.parallel import DistributedDataParallel as DDP
from apex.parallel import DistributedDataParallel as DDP
from apex.parallel import convert_syncbn_model
import random
from apex import amp
from imagenet_loader import get_dali_train_loader, get_dali_val_loader
from models import count_parameters

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
    parser.add_argument('--start_checkpoint', default="",
                        type=str, help='Start checkpoint')
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
    parser.add_argument('--not_normalize', action='store_true')
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


def get_model(args, dataloader):
    args.step_per_epoch = step_per_epoch = 1281167 // args.total_batch_size
    model = torchvision.models.resnet50(False)
    if args.start_checkpoint != "":
        model.load_state_dict(torch.load(args.start_checkpoint, 'cpu'))
    model.cuda()
    #model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = convert_syncbn_model(model)
    optimizer = torch.optim.SGD(
        [{
            'params': model.parameters(),
            'initial_lr': args.det_params['lr'],
            **args.det_params
        }], **args.sgd_params)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [
        lambda step: schedule(step, step_per_epoch, args.warmup, args.schedule['det']),
    ], last_epoch=step_per_epoch*args.start_epoch-1)
    model, optimizer = amp.initialize(
        model,
        optimizer,
        opt_level="O1",
        loss_scale=128 #"dynamic" if args.dynamic_loss_scale else args.static_loss_scale,
    )
    model = DDP(model)
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
        normalize=not args.not_normalize
    )

    val_loader, val_loader_len = get_val_loader(
        path=[f"data/imagenet/tf_records/validation/validation-{i:05d}-of-00128" for i in range(128)],
        index_path=[f"data/imagenet/tf_records/validation/validation-{i:05d}-of-00128.txt" for i in range(128)],
        batch_size=args.batch_size["test"],
        num_classes=1000,
        one_hot=False,
        workers=args.workers,
        normalize=not args.not_normalize
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

def test_nll(model, loader):
    model.eval()
    with torch.no_grad():
        nll = 0
        acc = 0
        for bx, by in loader:
#            with torch.cuda.amp.autocast():
            pred = model(bx)
            bnll = torch.nn.functional.cross_entropy(pred, by, reduction='sum')
            nll += bnll.item()
            acc += (pred.argmax(-1) == by).sum().item()
            torch.cuda.synchronize()
        acc /= 50000
        nll /= 50000
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
#    scaler = torch.cuda.amp.GradScaler()
    for i in range(args.start_epoch, args.num_epochs):
        t0 = time.time()
        lls = []
        for bx, by in train_loader:
#            with torch.cuda.amp.autocast():
            pred = model(bx)
            loss = torch.nn.functional.cross_entropy(pred, by)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                 scaled_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
#            scaler.scale(loss).backward()
#            scaler.step(optimizer)
#            scaler.update()
            scheduler.step()
            torch.cuda.synchronize()
            print("Epoch %d: loglike: %.4f, lr1: %.4f, time: %.2f" % (i, loss.item(), optimizer.param_groups[0]['lr'], time.time()-t0))
            lls.append(loss.item())
        t1 = time.time()
        if (i+1) % args.logging_freq == 0:
            print("Epoch %d: loglike: %.4f, lr1: %.4f, time: %.2f" % (i, np.mean(lls).item(), optimizer.param_groups[0]['lr'], t1-t0))
        if (i+1) % args.test_freq == 0:
            if args.rank == 0:
                torch.save(model.module.state_dict(), checkpoint_dir.format(str(i)))
                print('Save checkpoint')
            nll, acc = test_nll(model, test_loader)
            print("Epoch %d: test NLL %.4f, acc %.4f" % (i, nll, acc))
            model.train()
    if args.rank == 0:
        torch.save(model.module.state_dict(), checkpoint_dir.format('final'))
        print('Save checkpoint')
    print(END_MSG)


if __name__ == '__main__':
    main()

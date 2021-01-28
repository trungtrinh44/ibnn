import argparse
import ast
import os
from time import sleep
import numpy as np
import torch
import time
import torch.distributions as D
import torch.nn as nn
import torchvision
import json
import random
from tqdm import tqdm
from imagenet_loader import get_dali_val_loader
from models import count_parameters, ECELoss
from models.resnet50 import resnet50

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--n_components', default=2,
                        type=int, help='Number of components')
    parser.add_argument('--num_sample', type=int, default=1)
    parser.add_argument('--checkpoint', default="",
                        type=str, help='Start checkpoint')
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--valdir', default='data/imagenet/val.lmdb', type=str)
    parser.add_argument('--not_normalize', action='store_true')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(seed=args.seed)
    random.seed(args.seed)
    train(args)

def parallel_nll(model, x, y, n_sample):
    n_components = model.module.n_components
    prob = model(x, n_sample * n_components)
    logp = D.Categorical(logits=prob).log_prob(y.unsqueeze(1).expand(-1, n_components*n_sample))
    logp = torch.logsumexp(logp, 1) - torch.log(torch.tensor(n_components * n_sample, dtype=logp.dtype, device=x.device))
    return -logp.sum(), prob


def get_model(args):
    model = resnet50(deterministic_pretrained=False, n_components=args.n_components,
                     prior_mean=1.0, prior_std=0.1, posterior_mean_init=(1.0, 0.75), posterior_std_init=(0.05, 0.02))
    model.cuda()
    return model


def get_dataloader(args):
    get_val_loader = get_dali_val_loader()
    val_loader, val_loader_len = get_val_loader(
        path=[f"data/imagenet/tf_records/validation/validation-{i:05d}-of-00128" for i in range(128)],
        index_path=[f"data/imagenet/tf_records/validation/validation-{i:05d}-of-00128.txt" for i in range(128)],
        batch_size=args.batch_size,
        num_classes=1000,
        one_hot=False,
        workers=args.workers,
        normalize=not args.not_normalize
    )

    return val_loader, val_loader_len

def test_nll(model, loader, num_sample):
    model.eval()
    all_preds = []
    with torch.no_grad():
        nll = torch.zeros(())
        acc = torch.zeros(())
        for bx, by in loader:
            bnll, pred = parallel_nll(model, bx, by, num_sample)
            nll += bnll
            acc += (pred.exp().mean(1).argmax(-1) == by).sum()
            all_preds.append(pred)
            torch.cuda.synchronize()
    return nll, acc, torch.cat(all_preds, dim=0)


def train(args):
    print('Get data loader')
    test_loader, test_loader_len = get_dataloader(args)
    model = get_model(args)
    model.eval()
    amp_cp = torch.load(args.checkpoint, 'cpu')
    model.load_state_dict(amp_cp['model'])
    del amp_cp
    tnll = 0
    acc = [0, 0, 0]
    nll_miss = 0
    y_prob = []
    y_true = []
    y_prob_all = []
    model.eval()
    with torch.no_grad():
        for bx, by in tqdm(test_loader):
            bx = bx.cuda(non_blocking=True)
            by = by.cuda(non_blocking=True)
            prob = model(bx, args.num_sample*model.n_components)
            y_target = by.unsqueeze(1).expand(-1, args.num_sample*model.n_components)
            bnll = D.Categorical(logits=prob).log_prob(y_target)
            bnll = torch.logsumexp(bnll, dim=1) - torch.log(torch.tensor(args.num_sample*model.n_components, dtype=torch.float32, device=bnll.device))
            tnll -= bnll.sum().item()
            vote = prob.exp().mean(dim=1)
            top3 = torch.topk(vote, k=3, dim=1, largest=True, sorted=True)[1]
            y_prob_all.append(prob.exp().cpu().numpy())
            y_prob.append(vote.cpu().numpy())
            y_true.append(by.cpu().numpy())
            y_miss = top3[:, 0] != by
            if y_miss.sum().item() > 0:
                nll_miss -= bnll[y_miss].sum().item()
            for k in range(3):
                acc[k] += (top3[:, k] == by).sum().item()
    nll_miss /= 50000 - acc[0]
    tnll /= 50000
    for k in range(3):
        acc[k] /= 50000
    acc = np.cumsum(acc)
    y_prob = np.concatenate(y_prob, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    y_prob_all = np.concatenate(y_prob_all, axis=0)
    np.save(os.path.join(args.out_dir, 'predictions.npy'), y_prob)
    ece = ECELoss(15)
    ece_val = ece(torch.from_numpy(y_prob), torch.from_numpy(y_true)).item()
    result = {
        'nll': float(tnll),
        'nll_miss': float(nll_miss),
        'ece': float(ece_val),
        **{
            f"top-{k}": float(a) for k, a in enumerate(acc, 1)
        }
    }
    with open(os.path.join(args.out_dir, 'result.json'), 'w') as out:
        json.dump(result, out)


if __name__ == '__main__':
    main()

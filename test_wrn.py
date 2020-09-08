import argparse
import json
import os

import numpy as np
import torch
import torch.distributions as D
from scipy.stats import entropy
from sklearn.metrics import confusion_matrix

from datasets import get_data_loader
from models import DetWideResNet, StoWideResNet, DropWideResNet, get_wrn_model_from_config, ECELoss
from utils import (plot_auc, plot_calibration_curve, plot_filters, plot_mean_std,
                   plot_prior_var, plot_samples, RunningMeanStd)


def test_stochastic(model, dataloader, device, num_test_sample, n_noise, n_classes):
    tnll = 0
    acc = [0, 0, 0]
    nll_miss = 0
    y_prob = []
    y_true = []
    y_prob_all = []
    model.eval()
    with torch.no_grad():
        for bx, by in dataloader:
            bx = bx.to(device)
            by = by.to(device)
            prob = model.forward(bx, num_test_sample*model.n_components, indices=torch.arange(0, model.n_components, device=bx.device).repeat((bx.size(0)*num_test_sample,)))
            y_target = by.unsqueeze(1).repeat(1, num_test_sample*model.n_components)
            bnll = D.Categorical(logits=prob).log_prob(y_target)
            bnll = torch.logsumexp(bnll, dim=1) - torch.log(torch.tensor(num_test_sample*model.n_components, dtype=torch.float32, device=bnll.device))
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
    nll_miss /= len(dataloader.dataset) - acc[0]
    tnll /= len(dataloader.dataset)
    for k in range(3):
        acc[k] /= len(dataloader.dataset)
    acc = np.cumsum(acc)
    y_prob = np.concatenate(y_prob, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    y_prob_all = np.concatenate(y_prob_all, axis=0)
    return y_prob_all, y_prob, y_true, acc, tnll, nll_miss


def test_model_deterministic(model, dataloader, device):
    tnll = 0
    acc = [0, 0, 0]
    nll_miss = 0
    y_prob = []
    y_true = []
    model.eval()
    with torch.no_grad():
        for bx, by in dataloader:
            bx = bx.to(device)
            by = by.to(device)
            prob = model(bx)
            y_prob.append(prob.exp().cpu().numpy())
            y_true.append(by.cpu().numpy())
            top3 = torch.topk(prob, k=3, dim=1, largest=True, sorted=True)[1]
            tnll += torch.nn.functional.nll_loss(prob, by).item() * len(by)
            y_miss = top3[:, 0] != by
            if y_miss.sum().item() > 0:
                prob_miss = prob[y_miss]
                by_miss = by[y_miss]
                nll_miss += torch.nn.functional.nll_loss(
                    prob_miss, by_miss).item() * len(by_miss)
            for k in range(3):
                acc[k] += (top3[:, k] == by).sum().item()
    nll_miss /= len(dataloader.dataset) - acc[0]
    tnll /= len(dataloader.dataset)
    for k in range(3):
        acc[k] /= len(dataloader.dataset)
    acc = np.cumsum(acc)
    y_prob = np.concatenate(y_prob, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    return y_prob, y_true, acc, tnll, nll_miss


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=str)
    parser.add_argument('--num_samples', '-n', type=int, default=10)
    parser.add_argument('--device', '-d', type=str, default='cuda')
    parser.add_argument('--width', '-w', type=int, default=32)
    parser.add_argument('--height', type=int, default=32)
    parser.add_argument('--in_channels', '-i', type=int, default=3)
    parser.add_argument('--classes', '-c', type=int, default=10)
    parser.add_argument('--experiment', '-e', type=str, default='wrn_cifar10')
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    parser.add_argument('--n_rows', '-r', type=int, default=2)
    parser.add_argument('--n_noise', type=int, default=5)
    parser.add_argument('--ece_bins', type=int, default=15)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    num_sample = args.num_samples
    checkpoint = os.path.join(args.root, 'checkpoint.pt')
    with open(os.path.join(args.root, 'config.json')) as inp:
        config = json.load(inp)
    args.root = os.path.join(args.root, args.experiment)
    os.makedirs(args.root, exist_ok=True)
    text_path = os.path.join(args.root, 'result.json')
    test_loader = get_data_loader(args.experiment, args.batch_size, test_only=True)
    model = get_wrn_model_from_config(config, args.width, args.height, args.in_channels, args.classes)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    if config['model_type'] == 'deterministic':
        y_prob, y_true, acc, tnll, nll_miss = test_model_deterministic(model, test_loader, device)
    else:
        y_prob_all, y_prob, y_true, acc, tnll, nll_miss = test_stochastic(model, test_loader, device, args.num_samples, args.n_noise, args.classes)
    pred_entropy = entropy(y_prob, axis=1)
    np.save(os.path.join(args.root, 'pred_entropy.npy'), pred_entropy)
    ece = ECELoss(args.ece_bins)
    ece_val = ece(torch.from_numpy(y_prob), torch.from_numpy(y_true)).item()
    result = {
        'nll': float(tnll),
        'nll_miss': float(nll_miss),
        'ece': float(ece_val),
        'predictive_entropy': {
            'mean': float(pred_entropy.mean()),
            'std': float(pred_entropy.std())
        },
        **{
            f"top-{k}": float(a) for k, a in enumerate(acc, 1)
        }
    }
    with open(text_path, 'w') as out:
        json.dump(result, out)

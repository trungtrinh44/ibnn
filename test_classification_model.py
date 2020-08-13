import argparse
import json
import os

import numpy as np
import torch
import torch.distributions as D
from scipy.stats import entropy
from sklearn.metrics import confusion_matrix

from datasets import get_data_loader
from models import DeterministicLeNet, StochasticLeNet, DropoutLeNet, get_model_from_config
from utils import (plot_auc, plot_calibration_curve, plot_filters, plot_mean_std,
                   plot_prior_var, plot_samples, RunningMeanStd)


def test_stochastic(model, dataloader, device, num_test_sample, n_noise, n_classes):
    tnll = 0
    acc = [0, 0, 0]
    nll_miss = 0
    y_prob = []
    y_true = []
    y_prob_all = []
    conv1_outs = []
    conv2_outs = []
    model.eval()
    with torch.no_grad():
        first = True
        for bx, by in dataloader:
            bx = bx.to(device)
            by = by.to(device)
            prob, c1o, c2o, f1o, f2o = model(bx, num_test_sample, return_conv=True)
            conv1_outs.append(c1o[:, :n_noise].cpu().numpy())
            conv2_outs.append(c2o[:, :n_noise].cpu().numpy())
            bnll = (torch.logsumexp(D.Categorical(logits=prob).log_prob(by.unsqueeze(1).repeat(1, num_test_sample)), dim=1)
                    - torch.log(torch.tensor(num_test_sample, dtype=torch.float32, device=prob.device)))
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
            if first:
                conv1_ms = [RunningMeanStd(dim=0, shape=c1o.shape[2:]) for _ in range(n_classes)]
                conv2_ms = [RunningMeanStd(dim=0, shape=c2o.shape[2:]) for _ in range(n_classes)]
                fc1_ms = [RunningMeanStd(dim=0, shape=f1o.shape[2:]) for _ in range(n_classes)]
                fc2_ms = [RunningMeanStd(dim=0, shape=f2o.shape[2:]) for _ in range(n_classes)]
                for ic in range(n_classes):
                    conv1_ms[ic].to(device)
                    conv2_ms[ic].to(device)
                    fc1_ms[ic].to(device)
                    fc2_ms[ic].to(device)
                first = False
            for ic in range(n_classes):
                conv1_ms[ic].update(torch.flatten(c1o[by==ic], start_dim=0, end_dim=1))
                conv2_ms[ic].update(torch.flatten(c2o[by==ic], start_dim=0, end_dim=1))
                fc1_ms[ic].update(torch.flatten(f1o[by==ic], start_dim=0, end_dim=1))
                fc2_ms[ic].update(torch.flatten(f2o[by==ic], start_dim=0, end_dim=1))
    nll_miss /= len(dataloader.dataset) - acc[0]
    tnll /= len(dataloader.dataset)
    for k in range(3):
        acc[k] /= len(dataloader.dataset)
    acc = np.cumsum(acc)
    y_prob = np.concatenate(y_prob, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    y_prob_all = np.concatenate(y_prob_all, axis=0)
    conv1_outs = np.concatenate(conv1_outs, axis=0)
    conv2_outs = np.concatenate(conv2_outs, axis=0)
    conv1_ms = [(ms.mean().cpu().numpy(), ms.var().cpu().numpy()) for ms in conv1_ms]
    conv2_ms = [(ms.mean().cpu().numpy(), ms.var().cpu().numpy()) for ms in conv2_ms]
    fc1_ms = [(ms.mean().cpu().numpy(), ms.var().cpu().numpy()) for ms in fc1_ms]
    fc2_ms = [(ms.mean().cpu().numpy(), ms.var().cpu().numpy()) for ms in fc2_ms]
    return y_prob_all, y_prob, y_true, conv1_outs, conv2_outs, acc, tnll, nll_miss, conv1_ms, conv2_ms, fc1_ms, fc2_ms


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
    parser.add_argument('--num_samples', '-n', type=int, default=200)
    parser.add_argument('--device', '-d', type=str, default='cuda')
    parser.add_argument('--width', '-w', type=int, default=28)
    parser.add_argument('--height', type=int, default=28)
    parser.add_argument('--in_channels', '-i', type=int, default=1)
    parser.add_argument('--classes', '-c', type=int, default=10)
    parser.add_argument('--experiment', '-e', type=str, default='mnist')
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    parser.add_argument('--n_rows', '-r', type=int, default=2)
    parser.add_argument('--n_noise', type=int, default=5)
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
    model = get_model_from_config(config, args.width, args.height, args.in_channels, args.classes)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    if config['model_type'] == 'deterministic':
        y_prob, y_true, acc, tnll, nll_miss = test_model_deterministic(model, test_loader, device)
    else:
        y_prob_all, y_prob, y_true, conv1_outs, conv2_outs, acc, tnll, nll_miss, conv1_ms, conv2_ms, fc1_ms, fc2_ms = test_stochastic(model, test_loader, device,
                                                                                                                                      args.num_samples, args.n_noise, args.classes)
        test_image = torch.tensor(test_loader.dataset.data).numpy()
        plot_samples(y_true, y_prob_all, test_image,
                     args.classes, os.path.join(args.root, 'samples.png'))
        for ic in range(args.classes):
            plot_mean_std(conv1_ms[ic], conv2_ms[ic], fc1_ms[ic], fc2_ms[ic], os.path.join(args.root, f'mean_var_{ic}.png'))
        plot_filters(y_true, y_prob_all, test_image, args.classes, conv1_outs,
                     conv2_outs, args.root, n_noise=args.n_noise, n_samples=1)
        
    plot_auc(y_true, y_prob, args.classes, args.n_rows, args.classes // args.n_rows, os.path.join(args.root, 'auc.png'))
    plot_calibration_curve(y_true, y_prob, args.classes, args.n_rows, args.classes//args.n_rows, os.path.join(args.root, 'calibration.png'))
    pred_entropy = entropy(y_prob, axis=1)
    np.save(os.path.join(args.root, 'pred_entropy.npy'), pred_entropy)
    result = {
        'mean_predictive_entropy': float(pred_entropy.mean()),
        'nll': float(tnll),
        'nll_miss': float(nll_miss),
        **{
            f"top-{k}": float(a) for k, a in enumerate(acc, 1)
        }
    }
    with open(text_path, 'w') as out:
        json.dump(result, out)

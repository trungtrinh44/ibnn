import argparse
import json
import os

import numpy as np
import torch

from datasets import get_data_loader
from models import DeterministicLeNet, StochasticLeNet
from utils import plot_auc, plot_calibration_curve


def test_stochastic(model, dataloader, device, num_test_sample, path, mll):
    tnll = 0
    acc = [0, 0, 0]
    nll_miss = 0
    y_prob = []
    y_true = []
    if mll:
        func = model.marginal_loglikelihood_loss
    else:
        func = model.negative_loglikelihood
    model.eval()
    with torch.no_grad():
        for bx, by in dataloader:
            bx = bx.to(device)
            by = by.to(device)
            prob = model(bx, num_test_sample, mll)
            tnll += func(bx, by,
                         num_test_sample).item() * len(by)
            vote = prob.exp().mean(dim=1)
            top3 = torch.topk(vote, k=3, dim=1, largest=True, sorted=True)[1]
            y_prob.append(vote.cpu().numpy())
            y_true.append(by.cpu().numpy())
            y_miss = top3[:, 0] != by
            if y_miss.sum().item() > 0:
                by_miss = by[y_miss]
                bx_miss = bx[y_miss]
                nll_miss += func(
                    bx_miss, by_miss, num_test_sample).item() * len(by_miss)
            for k in range(3):
                acc[k] += (top3[:, k] == by).sum().item()
    nll_miss /= len(dataloader.dataset) - acc[0]
    tnll /= len(dataloader.dataset)
    for k in range(3):
        acc[k] /= len(dataloader.dataset)
    acc = np.cumsum(acc)
    with open(path, 'w') as out:
        accs = ", ".join(f"top-{k}: {a:.4f}" for k, a in enumerate(acc, 1))
        out.write(
            f"Test data: {accs}, nll {tnll:.4f}, nll miss {nll_miss:.4f}\n")
    y_prob = np.concatenate(y_prob, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    return y_prob, y_true


def test_model_deterministic(model, dataloader, device, path):
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
    with open(path, 'w') as out:
        accs = ", ".join(f"top-{k}: {a:.4f}" for k, a in enumerate(acc, 1))
        out.write(
            f"Test data: {accs}, nll {tnll:.4f}, nll miss {nll_miss:.4f}\n")
    y_prob = np.concatenate(y_prob, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    return y_prob, y_true


if __name__ == "__main__":
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
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    num_sample = args.num_samples
    checkpoint = os.path.join(args.root, 'checkpoint.pt')
    text_path = os.path.join(args.root, 'result.txt')
    with open(os.path.join(args.root, 'config.json')) as inp:
        config = json.load(inp)
    _, test_loader = get_data_loader(args.experiment, args.batch_size, False)
    if config['model_type'] == 'deterministic':
        model = DeterministicLeNet(args.width, args.height, args.in_channels,
                                   config['conv_hiddens'], config['fc_hidden'], args.classes, config['init_method'], config['activation'])
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model.to(device)
        y_prob, y_true = test_model_deterministic(
            model, test_loader, device, text_path)
    else:
        model = StochasticLeNet(args.width, args.height, args.in_channels, config['conv_hiddens'],
                                config['fc_hidden'], args.classes, config['init_method'], config['activation'],
                                config['init_mean'], config['init_log_std'], config['noise_type'], config['noise_size'], single_prior_mean=config.get('single_prior_mean', False), single_prior_std=config.get('single_prior_std', False))
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model.to(device)
        y_prob, y_true = test_stochastic(model, test_loader, device, args.num_samples,
                                         text_path, config['vb_iteration'] == 0)
    plot_auc(y_true, y_prob, args.classes, args.n_rows, args.classes//args.n_rows, os.path.join(args.root, 'auc.pdf'))
    plot_calibration_curve(y_true, y_prob, args.classes, args.n_rows, args.classes//args.n_rows, os.path.join(args.root, 'calibration.pdf'))

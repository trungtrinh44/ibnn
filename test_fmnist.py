import argparse
import json
import os

import numpy as np
import torch

from datasets import get_data_loader
from models import DeterministicLeNet, StochasticLeNet
from utils import (plot_auc, plot_calibration_curve, plot_filters,
                   plot_prior_var, plot_samples)


def test_stochastic(model, dataloader, device, num_test_sample, path, n_noise):
    tnll = 0
    acc = [0, 0, 0]
    nll_miss = 0
    y_prob = []
    y_true = []
    y_prob_all = []
    conv1_outs = []
    conv2_outs = []
    func = model.negative_loglikelihood
    model.eval()
    with torch.no_grad():
        for bx, by in dataloader:
            bx = bx.to(device)
            by = by.to(device)
            prob, c1o, c2o, _ = model(bx, num_test_sample, return_conv=True)
            conv1_outs.append(c1o[:, :n_noise].cpu().numpy())
            conv2_outs.append(c2o[:, :n_noise].cpu().numpy())
            tnll += func(bx, by,
                         num_test_sample).item() * len(by)
            vote = prob.exp().mean(dim=1)
            top3 = torch.topk(vote, k=3, dim=1, largest=True, sorted=True)[1]
            y_prob_all.append(prob.exp().cpu().numpy())
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
    y_prob_all = np.concatenate(y_prob_all, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    conv1_outs = np.concatenate(conv1_outs, axis=0)
    conv2_outs = np.concatenate(conv2_outs, axis=0)
    return y_prob_all, y_prob, y_true, conv1_outs, conv2_outs


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
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    parser.add_argument('--n_rows', '-r', type=int, default=2)
    parser.add_argument('--n_noise', type=int, default=5)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    num_sample = args.num_samples
    checkpoint = os.path.join(args.root, 'checkpoint.pt')
    with open(os.path.join(args.root, 'config.json')) as inp:
        config = json.load(inp)
    test_loader = get_data_loader('fmnist_mnist_test', args.batch_size, False)
    if config['model_type'] == 'deterministic':
        model = DeterministicLeNet(28, 28, 1,
                                   config['conv_hiddens'], config['fc_hidden'], 10, config['init_method'], config['activation'])
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model.to(device)
        args.root = os.path.join(args.root, 'fmnist')
        text_path = os.path.join(args.root, 'result.txt')
        os.makedirs(args.root, exist_ok=True)
        y_prob, y_true = test_model_deterministic(
            model, test_loader, device, text_path)
    else:
        model = StochasticLeNet(28, 28, 1, config['conv_hiddens'],
                                config['fc_hidden'], 10, config['init_method'], config['activation'],
                                config['init_dist_mean'], config['init_dist_std'], config['init_prior_mean'], config['init_prior_std'],
                                config['noise_type'], config['noise_size'], use_abs=config.get('use_abs', True))
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model.to(device)
        args.root = os.path.join(args.root, 'fmnist')
        text_path = os.path.join(args.root, 'result.txt')
        os.makedirs(args.root, exist_ok=True)
        y_prob_all, y_prob, y_true, conv1_outs, conv2_outs = test_stochastic(model, test_loader, device,
                                                                             args.num_samples, text_path, n_noise=args.n_noise)
        test_image = torch.tensor(test_loader.dataset.data).numpy()
        plot_samples(y_true, y_prob_all, test_image,
                     10, os.path.join(args.root, 'samples.png'))
        plot_filters(y_true, y_prob_all, test_image, 10, conv1_outs, conv2_outs, args.root, n_noise=args.n_noise, n_samples=1)
    plot_auc(y_true, y_prob, 10, args.n_rows, 10 //
             args.n_rows, os.path.join(args.root, 'auc.pdf'))
    plot_calibration_curve(y_true, y_prob, 10, args.n_rows,
                           10//args.n_rows, os.path.join(args.root, 'calibration.pdf'))

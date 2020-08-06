import argparse
import json
import os

import numpy as np
import torch
import torch.distributions as D
from scipy.stats import entropy
from sklearn.metrics import confusion_matrix

from datasets import get_data_loader
from models import DeterministicLeNet, StochasticLeNet, DropoutLeNet
from utils import (plot_auc, plot_calibration_curve, plot_filters,
                   plot_prior_var, plot_samples)


def test_stochastic(model, dataloader, device, num_test_sample, n_noise):
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
    y_prob = np.concatenate(y_prob, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    y_prob_all = np.concatenate(y_prob_all, axis=0)
    conv1_outs = np.concatenate(conv1_outs, axis=0)
    conv2_outs = np.concatenate(conv2_outs, axis=0)
    return y_prob_all, y_prob, y_true, conv1_outs, conv2_outs, confusion_matrix(y_true, y_prob.argmax(axis=1)), acc, tnll, nll_miss

def test_dropout(model, dataloader, device, num_test_sample):
    tnll = 0
    acc = [0, 0, 0]
    nll_miss = 0
    y_prob = []
    y_true = []
    y_prob_all = []
    func = model.negative_loglikelihood
    model.eval()
    with torch.no_grad():
        for bx, by in dataloader:
            bx = bx.to(device)
            by = by.to(device)
            bnll, prob = func(bx, by, num_test_sample, return_prob=True)
            tnll += bnll.item() * len(by)
            vote = prob.exp().mean(dim=1)
            top3 = torch.topk(vote, k=3, dim=1, largest=True, sorted=True)[1]
            y_prob_all.append(prob.exp().cpu().numpy())
            y_prob.append(vote.cpu().numpy())
            y_true.append(by.cpu().numpy())
            y_miss = top3[:, 0] != by
            if y_miss.sum().item() > 0:
                miss_prob = prob[y_miss]
                by_miss = by[y_miss].unsqueeze(1).repeat(1, num_test_sample)
                logp_miss = D.Categorical(logits=miss_prob).log_prob(by_miss)
                nll_miss -= (torch.logsumexp(logp_miss, dim=1) - torch.log(torch.tensor(num_test_sample, dtype=torch.float32, device=logp_miss.device))).sum().item()
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
    text_path = os.path.join(args.root, 'result.json')
    with open(os.path.join(args.root, 'config.json')) as inp:
        config = json.load(inp)
    test_loader = get_data_loader(args.experiment, args.batch_size, test_only=True)
    if config['model_type'] == 'deterministic':
        model = DeterministicLeNet(28, 28, 1,
                                   config['conv_hiddens'], config['fc_hidden'], 10, config['init_method'], config['activation'])
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model.to(device)
        y_prob, y_true, acc, tnll, nll_miss = test_model_deterministic(
            model, test_loader, device)
    elif config['model_type'] == 'dropout':
        model = DropoutLeNet(28, 28, 1,
                             config['conv_hiddens'], config['fc_hidden'], 10, config['init_method'], config['activation'], config['dropout'])
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model.to(device)
        y_prob_all, y_prob, y_true, acc, tnll, nll_miss = test_dropout(model, test_loader, device, args.num_samples)
        test_image = torch.tensor(test_loader.dataset.data).numpy()
        plot_samples(y_true, y_prob_all, test_image,
                     10, os.path.join(args.root, 'samples.png'))
    else:
        model = StochasticLeNet(28, 28, 1, config['conv_hiddens'],
                                config['fc_hidden'], 10, config['init_method'], config['activation'],
                                config['init_dist_mean'], config['init_dist_std'], config['init_prior_mean'], config['init_prior_std'],
                                config['noise_type'], config['noise_size'], use_abs=config.get('use_abs', True))
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model.to(device)
        y_prob_all, y_prob, y_true, conv1_outs, conv2_outs, confusion_matrix, acc, tnll, nll_miss = test_stochastic(model, test_loader, device,
                                                                                                                    args.num_samples, args.n_noise)
        test_image = torch.tensor(test_loader.dataset.data).numpy()
        plot_samples(y_true, y_prob_all, test_image,
                     10, os.path.join(args.root, 'samples.png'))
        plot_filters(y_true, y_prob_all, test_image, 10, conv1_outs,
                     conv2_outs, args.root, n_noise=args.n_noise, n_samples=1)
    plot_auc(y_true, y_prob, 10, args.n_rows, 10 //
             args.n_rows, os.path.join(args.root, 'auc.pdf'))
    plot_calibration_curve(y_true, y_prob, 10, args.n_rows,
                           10//args.n_rows, os.path.join(args.root, 'calibration.pdf'))
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

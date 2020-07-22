import argparse
import json
import os

import numpy as np
import torch

from datasets import get_data_loader
from models import DeterministicLeNet, StochasticLeNet
from utils import plot_auc, plot_calibration_curve


def test_nll(model, dataloader, device, num_test_sample, path):
    tnll = 0
    acc = [0, 0, 0]
    nll_miss = 0
    model.eval()
    with torch.no_grad():
        for bx, by in dataloader:
            bx = bx.to(device)
            by = by.to(device)
            prob = model(bx, num_test_sample, False)
            tnll += model.negative_loglikelihood(bx, by,
                                                 num_test_sample).item() * len(by)
            vote = prob.mean(1)
            # onehot = torch.zeros((vote.size(0), vote.size(1), 10),
            #                      device=vote.device)
            # onehot.scatter_(2, vote.unsqueeze(2), 1)
            # vote = onehot.sum(dim=1)
            # vote /= vote.sum(dim=1, keepdims=True)
            top3 = torch.topk(vote, k=3, dim=1, largest=True, sorted=True)[1]

            y_miss = top3[:, 0] != by
            if y_miss.sum().item() > 0:
                by_miss = by[y_miss]
                bx_miss = bx[y_miss]
                nll_miss += model.negative_loglikelihood(
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


def test_mll(model, dataloader, device, num_test_sample, path):
    tnll = 0
    acc = [0, 0, 0]
    nll_miss = 0
    model.eval()
    with torch.no_grad():
        for bx, by in dataloader:
            bx = bx.to(device)
            by = by.to(device)
            prob = model(bx, num_test_sample, True)
            tnll += model.marginal_loglikelihood_loss(bx, by,
                                                      num_test_sample).item() * len(by)
            vote = prob.mean(1)
            # onehot = torch.zeros((vote.size(0), vote.size(1), 10),
            #                      device=vote.device)
            # onehot.scatter_(2, vote.unsqueeze(2), 1)
            # vote = onehot.sum(dim=1)
            # vote /= vote.sum(dim=1, keepdims=True)
            top3 = torch.topk(vote, k=3, dim=1, largest=True, sorted=True)[1]

            y_miss = top3[:, 0] != by
            if y_miss.sum().item() > 0:
                by_miss = by[y_miss]
                bx_miss = bx[y_miss]
                nll_miss += model.marginal_loglikelihood_loss(
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


def test_model_deterministic(model, dataloader, device, path):
    tnll = 0
    acc = [0, 0, 0]
    nll_miss = 0
    model.eval()
    with torch.no_grad():
        for bx, by in dataloader:
            bx = bx.to(device)
            by = by.to(device)
            prob = model(bx)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=str)
    parser.add_argument('--num_samples', '-n', type=int, default=200)
    parser.add_argument('--device', '-d', type=str, default='cuda')
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    num_sample = args.num_samples
    checkpoint = os.path.join(args.root, 'checkpoint.pt')
    text_path = os.path.join(args.root, 'semeion_result.txt')
    with open(os.path.join(args.root, 'config.json')) as inp:
        config = json.load(inp)
    test_loader = get_data_loader('semeion', args.batch_size, False)
    if config['model_type'] == 'deterministic':
        model = DeterministicLeNet(28, 28, 1,
                                   config['conv_hiddens'], config['fc_hidden'], 10, config['init_method'], config['activation'])
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model.to(device)
        test_model_deterministic(model, test_loader, device, text_path)
    else:
        model = StochasticLeNet(28, 28, 1, config['conv_hiddens'],
                                config['fc_hidden'], 10, config['init_method'], config['activation'],
                                config['init_mean'], config['init_log_std'], config['noise_type'], config['noise_size'],
                                single_prior_mean=config.get('single_prior_mean', False), single_prior_std=config.get('single_prior_std', False),
                                use_abs=config.get('use_abs', True))
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model.to(device)
        test_func = test_mll if config['vb_iteration'] == 0 else test_nll
        test_func(model, test_loader, device, args.num_samples, text_path)

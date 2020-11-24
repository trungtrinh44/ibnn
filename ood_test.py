import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.distributions as D
from scipy.stats import entropy
from sklearn.metrics import confusion_matrix

from datasets import get_data_loader
from models import ECELoss, get_model_from_config


def add_imnoise(img, mode, prob):
    imgn = img
    if mode == 'salt_pepper':
        prob_sp = torch.rand_like(imgn)
        imgn = torch.clone(imgn)
        imgn[prob_sp < prob] = 0.0
        imgn[prob_sp > 1 - prob] = 1.0
    elif mode == 'gaussian':
        noise = torch.randn_like(imgn)
        imgn = (1-prob)*imgn + noise*prob
    return imgn

def test_dropout(model, dataloader, device, num_test_sample, transform):
    total_uncertainty = []
    aleatoric_uncertainty = []
    model.train()
    with torch.no_grad():
        for bx, by in dataloader:
            bx = transform(bx).to(device)
            by = by.to(device)
            prob = torch.cat([
                model.forward(bx).unsqueeze(1) for _ in range(num_test_sample)
            ], dim=1)
            log_prob_mean = torch.logsumexp(prob, dim=1) - torch.log(torch.tensor(prob.size(1), dtype=torch.float32, device=device))
            total_uncertainty.append((-log_prob_mean*log_prob_mean.exp()).sum(dim=1))
            aleatoric_uncertainty.append(
                (-prob*prob.exp()).sum(dim=2).mean(dim=1)
            )
    return torch.cat(total_uncertainty, dim=0), torch.cat(aleatoric_uncertainty, dim=0)

def test_stochastic(model, dataloader, device, num_test_sample, transform):
    total_uncertainty = []
    aleatoric_uncertainty = []
    model.eval()
    with torch.no_grad():
        for bx, by in dataloader:
            bx = transform(bx).to(device)
            by = by.to(device)
            indices = torch.empty(bx.size(0)*num_test_sample,
                                  dtype=torch.long, device=bx.device)
            prob = torch.cat([
                model.forward(bx, num_test_sample, indices=torch.full((bx.size(0)*num_test_sample,), idx, out=indices, device=bx.device, dtype=torch.long)) for idx in range(model.n_components)
            ], dim=1)
            log_prob_mean = torch.logsumexp(prob, dim=1) - torch.log(torch.tensor(prob.size(1), dtype=torch.float32, device=device))
            total_uncertainty.append((-log_prob_mean*log_prob_mean.exp()).sum(dim=1))
            aleatoric_uncertainty.append(
                (-prob*prob.exp()).sum(dim=2).mean(dim=1)
            )
    return torch.cat(total_uncertainty, dim=0), torch.cat(aleatoric_uncertainty, dim=0)

def test_model_deterministic(model, dataloader, device, transform):
    total_uncertainty = []
    model.eval()
    with torch.no_grad():
        for bx, by in dataloader:
            bx = transform(bx).to(device)
            by = by.to(device)
            prob = model(bx)
            total_uncertainty.append((-prob*prob.exp()).sum(1))
    return torch.cat(total_uncertainty, dim=0)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=str)
    parser.add_argument('--num_samples', '-n', type=int, default=10)
    parser.add_argument('--device', '-d', type=str, default='cuda')
    parser.add_argument('--in_channels', '-i', type=int, default=3)
    parser.add_argument('--classes', '-c', type=int, default=10)
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    parser.add_argument('--ece_bins', type=int, default=15)
    parser.add_argument('--dropout', action='store_true')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    num_sample = args.num_samples
    checkpoint = os.path.join(args.root, 'checkpoint.pt')
    with open(os.path.join(args.root, 'config.json')) as inp:
        config = json.load(inp)
    args.root = os.path.join(args.root, config['dataset'])
    os.makedirs(args.root, exist_ok=True)
    text_path = os.path.join(args.root, 'result.json')
    test_loader = get_data_loader(
        config['dataset'], args.batch_size, test_only=True)
    model = get_model_from_config(config)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    if model.__class__.__name__.startswith('Det'):
        if args.dropout:
            probs = np.arange(0.0, 0.55, 0.05, np.float32)
            tm, tv, am, av, em, ev = [], [], [], [], [], []
            for prob in probs:
                total_uncertainty, aleatoric_uncertainty = test_dropout(model, test_loader, device, args.num_samples, lambda x: add_imnoise(x, 'salt_pepper', prob))
                epistemic_uncertainty = total_uncertainty - aleatoric_uncertainty
                tm.append(total_uncertainty.mean().item())
                tv.append(total_uncertainty.std().item())
                am.append(aleatoric_uncertainty.mean().item())
                av.append(aleatoric_uncertainty.std().item())
                em.append(epistemic_uncertainty.mean().item())
                ev.append(epistemic_uncertainty.std().item())
            pd.DataFrame({
                'probs': probs, 'aleatoric_mean': am, 'aleatoric_std': av, 'epistemic_mean': em, 'epistemic_std': ev, 'total_mean': tm, 'total_std': tv
            }).to_csv(os.path.join(args.root, 'dropout_salt_pepper_noise.csv'), index=False)
            probs = np.arange(0.0, 1.05, 0.1, np.float32)
            tm, tv, am, av, em, ev = [], [], [], [], [], []
            for prob in probs:
                total_uncertainty, aleatoric_uncertainty = test_dropout(model, test_loader, device, args.num_samples, lambda x: add_imnoise(x, 'gaussian', prob))
                epistemic_uncertainty = total_uncertainty - aleatoric_uncertainty
                tm.append(total_uncertainty.mean().item())
                tv.append(total_uncertainty.std().item())
                am.append(aleatoric_uncertainty.mean().item())
                av.append(aleatoric_uncertainty.std().item())
                em.append(epistemic_uncertainty.mean().item())
                ev.append(epistemic_uncertainty.std().item())
            pd.DataFrame({
                'probs': probs, 'aleatoric_mean': am, 'aleatoric_std': av, 'epistemic_mean': em, 'epistemic_std': ev, 'total_mean': tm, 'total_std': tv
            }).to_csv(os.path.join(args.root, 'dropout_gaussian_noise.csv'), index=False)
        else:
            probs = np.arange(0.0, 0.55, 0.05, np.float32)
            mean = []
            var = []
            for prob in probs:
                aleatoric_uncertainty = test_model_deterministic(model, test_loader, device, lambda x: add_imnoise(x, 'salt_pepper', prob))
                mean.append(aleatoric_uncertainty.mean().item())
                var.append(aleatoric_uncertainty.std().item())
            pd.DataFrame({
                'probs': probs, 'aleatoric_mean': mean, 'aleatoric_std': var
            }).to_csv(os.path.join(args.root, 'salt_pepper_noise.csv'), index=False)
            
            probs = np.arange(0.0, 1.05, 0.1, np.float32)
            mean = []
            var = []
            for prob in probs:
                aleatoric_uncertainty = test_model_deterministic(model, test_loader, device, lambda x: add_imnoise(x, 'gaussian', prob))
                mean.append(aleatoric_uncertainty.mean().item())
                var.append(aleatoric_uncertainty.std().item())
            pd.DataFrame({
                'probs': probs, 'aleatoric_mean': mean, 'aleatoric_std': var
            }).to_csv(os.path.join(args.root, 'gaussian_noise.csv'), index=False)
    elif model.__class__.__name__.startswith('Sto'):
        probs = np.arange(0.0, 0.55, 0.05, np.float32)
        tm, tv, am, av, em, ev = [], [], [], [], [], []
        for prob in probs:
            total_uncertainty, aleatoric_uncertainty = test_stochastic(model, test_loader, device, args.num_samples, lambda x: add_imnoise(x, 'salt_pepper', prob))
            epistemic_uncertainty = total_uncertainty - aleatoric_uncertainty
            tm.append(total_uncertainty.mean().item())
            tv.append(total_uncertainty.std().item())
            am.append(aleatoric_uncertainty.mean().item())
            av.append(aleatoric_uncertainty.std().item())
            em.append(epistemic_uncertainty.mean().item())
            ev.append(epistemic_uncertainty.std().item())
        pd.DataFrame({
            'probs': probs, 'aleatoric_mean': am, 'aleatoric_std': av, 'epistemic_mean': em, 'epistemic_std': ev, 'total_mean': tm, 'total_std': tv
        }).to_csv(os.path.join(args.root, 'salt_pepper_noise.csv'), index=False)
        probs = np.arange(0.0, 1.05, 0.1, np.float32)
        tm, tv, am, av, em, ev = [], [], [], [], [], []
        for prob in probs:
            total_uncertainty, aleatoric_uncertainty = test_stochastic(model, test_loader, device, args.num_samples, lambda x: add_imnoise(x, 'gaussian', prob))
            epistemic_uncertainty = total_uncertainty - aleatoric_uncertainty
            tm.append(total_uncertainty.mean().item())
            tv.append(total_uncertainty.std().item())
            am.append(aleatoric_uncertainty.mean().item())
            av.append(aleatoric_uncertainty.std().item())
            em.append(epistemic_uncertainty.mean().item())
            ev.append(epistemic_uncertainty.std().item())
        pd.DataFrame({
            'probs': probs, 'aleatoric_mean': am, 'aleatoric_std': av, 'epistemic_mean': em, 'epistemic_std': ev, 'total_mean': tm, 'total_std': tv
        }).to_csv(os.path.join(args.root, 'gaussian_noise.csv'), index=False)

import argparse
import json
import os

import numpy as np
import torch
import torch.distributions as D
from scipy.stats import entropy
from sklearn.metrics import confusion_matrix

from datasets import get_data_loader
from models import ECELoss, get_model_from_config, StoLayer
import pandas as pd

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

def main(args):
    device = args.device if torch.cuda.is_available() else 'cpu'
    checkpoint = os.path.join(args.root, 'checkpoint.pt')
    with open(os.path.join(args.root, 'config.json')) as inp:
        config = json.load(inp)
    args.root = os.path.join(args.root, 'det_checkpoints')
    os.makedirs(args.root, exist_ok=True)
    model = get_model_from_config(config)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    det_config = config.copy()
    det_config['model_name'] = f"Det{model.__class__.__name__[3:]}"
    det_model = get_model_from_config(det_config)
    test_loader = get_data_loader(config['dataset'], args.batch_size, test_only=True)
    ece = ECELoss(args.ece_bins)
    results = []
    predictions = []
    for index in ['ones', 'mean'] + list(range(config['n_components'])):
        det_model = StoLayer.convert_deterministic(model, index, det_model)
        torch.save(det_model.state_dict(), os.path.join(args.root, f'checkpoint_{index}.pt'))
        y_prob, y_true, acc, tnll, nll_miss = test_model_deterministic(det_model, test_loader, device)
        pred_entropy = entropy(y_prob, axis=1)
        ece_val = ece(torch.from_numpy(y_prob), torch.from_numpy(y_true)).item()
        predictions.append(y_prob)
        result = {
            'checkpoint': index,
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
        results.append(result)
    results = pd.DataFrame(results)
    results.to_csv(os.path.join(args.root, 'results.csv'), index=False)
    np.save(os.path.join(args.root, 'preds.npy'), np.array(predictions))

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=str)
    parser.add_argument('--device', '-d', type=str, default='cuda')
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    parser.add_argument('--ece_bins', type=int, default=15)
    args = parser.parse_args()
    main(args)

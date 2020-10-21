import argparse
import json
import os

import numpy as np
import torch

from datasets import get_data_loader
from utils import plot_error
from models import DeterministicLeNet, StochasticLeNet, DropoutLeNet, get_model_from_config
from scipy.stats import entropy

def test_noise(model, dataset, device, num_test_sample, epsilons, path):
    test_loader = get_data_loader(args.dataset, args.batch_size, False, test_only=True)
    entropies = [[] for _ in range(len(epsilons)+1)]
    pred_prob = [[] for _ in range(len(epsilons)+1)]
    for bx, by in test_loader:
        bx = bx.to(device)
        by = by.to(device)
        bx.requires_grad = True
        prob = model(bx, num_test_sample)
        loss = torch.distributions.Categorical(logits=prob).log_prob(by.unsqueeze(1)).mean(1).sum()
        model.zero_grad()
        loss.backward()
        x_grad = bx.grad.data.sign()
        pred_mean = prob.exp().mean(1)
        entropies[0].append(entropy(pred_mean.detach().cpu().numpy(), axis=1))
        pred_prob[0].append(torch.distributions.Categorical(probs=pred_mean.detach()).log_prob(by).exp().cpu().numpy())
        with torch.no_grad():
            for pp, ent, eps in zip(pred_prob[1:], entropies[1:], epsilons):
                prob = model(bx + eps*x_grad, num_test_sample)
                pred_mean = prob.exp().mean(1)
                ent.append(entropy(pred_mean.cpu().numpy(), axis=1))
                pp.append(torch.distributions.Categorical(probs=pred_mean).log_prob(by).exp().cpu().numpy())
    entropies = [np.concatenate(ent, axis=0) for ent in entropies]
    entropy_mean = [ent.mean() for ent in entropies]
    entropy_std = [ent.std() for ent in entropies]
    pred_prob = [np.concatenate(pp, axis=0) for pp in pred_prob]
    pred_mean = [pp.mean() for pp in pred_prob]
    pred_std =  [pp.std()  for pp in pred_prob]
    plot_error(x=np.array([0] + epsilons), xlabel=r'$\epsilon$',
               mean1=np.array(entropy_mean), std1=np.array(entropy_std),
               mean2=np.array(pred_mean), std2=np.array(pred_std),
               legend1=r'$\mathcal{H}(y|x)$', legend2=r'$p(y=t|x)$', ylabel1='nats', ylabel2='probs', save_path=path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=str)
    parser.add_argument('--num_samples', '-n', type=int, default=5)
    parser.add_argument('--device', '-d', type=str, default='cuda')
    parser.add_argument('--width', '-w', type=int, default=28)
    parser.add_argument('--height', type=int, default=28)
    parser.add_argument('--in_channels', '-i', type=int, default=1)
    parser.add_argument('--classes', '-c', type=int, default=10)
    parser.add_argument('--dataset', '-e', type=str, default='mnist')
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    parser.add_argument('--epsilons', type=float, nargs='+', default=[.2, .4, .6, .8, 1.0])
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    num_sample = args.num_samples
    checkpoint = os.path.join(args.root, 'checkpoint.pt')
    text_path = os.path.join(args.root, 'result.txt')
    with open(os.path.join(args.root, 'config.json')) as inp:
        config = json.load(inp)
    model = get_model_from_config(config, args.width, args.height, args.in_channels, args.classes)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    os.makedirs(os.path.join(args.root, 'fgsm'), exist_ok=True)
    test_noise(model, args.dataset, device, args.num_samples, args.epsilons, os.path.join(args.root, 'fgsm', args.dataset))

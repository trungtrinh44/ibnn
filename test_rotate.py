import argparse
import json
import os

import numpy as np
import torch

from datasets import get_data_loader
from utils import plot_error
from models import DeterministicLeNet, StochasticLeNet, DropoutLeNet, get_model_from_config
from scipy.stats import entropy

def draw_rotate(model, dataset, device, num_test_sample, degrees, path):
    model.eval()
    entropy_mean = []
    entropy_std = []
    for degree in degrees:
        test_loader = get_data_loader(args.dataset, args.batch_size, False, test_only=True, degree=degree)
        entropies = []
        with torch.no_grad():
            for bx, by in test_loader:
                bx = bx.to(device)
                prob = model(bx, num_test_sample)
                pred_mean = prob.exp().mean(1).cpu().numpy()
                entropies.append(entropy(pred_mean, axis=1))
        entropies = np.concatenate(entropies, axis=0)
        entropy_mean.append(entropies.mean())
        entropy_std.append(entropies.std())
    plot_error(np.array(degrees), np.array(entropy_mean), np.array(entropy_std), 'Degree', 'Predictive entropy', path)


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
    parser.add_argument('--degrees', type=int, nargs='+', default=list(range(0, 180, 20)))
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
    os.makedirs(os.path.join(args.root, 'rotate_test'), exist_ok=True)
    draw_rotate(model, args.dataset, device, args.num_samples, args.degrees, os.path.join(args.root, 'rotate_test', args.dataset))

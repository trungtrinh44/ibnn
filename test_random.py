import argparse
import json
import os

import numpy as np
import torch

from datasets import get_data_loader
from utils import plot_error2
from models import DeterministicLeNet, StochasticLeNet, DropoutLeNet, get_model_from_config
from scipy.stats import entropy

def test_random_scale(model, batch_size, device, num_test_sample, scales, path):
    model.eval()
    entropy_mean = []
    entropy_std = []
    random_tensor = torch.load('data/random_data.pt', map_location=device)
    for scale in scales:
        random_data = random_tensor*scale
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(random_data), batch_size, shuffle=False)
        entropies = []
        with torch.no_grad():
            for bx in test_loader:
                bx = bx[0].to(device)
                prob = model(bx, num_test_sample)
                pred_mean = prob.exp().mean(1)
                entropies.append(entropy(pred_mean.cpu().numpy(), axis=1))
        entropies = np.concatenate(entropies, axis=0)
        entropy_mean.append(entropies.mean())
        entropy_std.append(entropies.std())
    plot_error2(x=scales, xlabel='Scale',
               mean=np.array(entropy_mean), std=np.array(entropy_std),
               legend=r'$\mathcal{H}(y|x)$', ylabel='nats', save_path=path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=str)
    parser.add_argument('--num_samples', '-n', type=int, default=5)
    parser.add_argument('--device', '-d', type=str, default='cuda')
    parser.add_argument('--width', '-w', type=int, default=28)
    parser.add_argument('--height', type=int, default=28)
    parser.add_argument('--in_channels', '-i', type=int, default=1)
    parser.add_argument('--classes', '-c', type=int, default=10)
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    parser.add_argument('--scales', type=float, nargs='+', default=[1.0, 10.0, 100.0, 1000.0, 10000.0])
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
    test_random_scale(model, args.batch_size, device, args.num_samples, args.scales, os.path.join(args.root, 'random_scale'))

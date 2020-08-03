import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.decomposition import IncrementalPCA

from datasets import get_data_loader
from models import DeterministicLeNet, StochasticLeNet


def draw_pca(model, dataloader, device, num_test_sample, path):
    conv1_outs = []
    conv2_outs = []
    fc1_outs = []
    fc2_outs = []
    labels = []
    model.eval()
    pca1 = IncrementalPCA(n_components=2)
    pca2 = IncrementalPCA(n_components=2)
    pca3 = IncrementalPCA(n_components=2)
    pca4 = IncrementalPCA(n_components=2)
    with torch.no_grad():
        for bx, by in dataloader:
            bx = bx.to(device)
            bs = by.shape[0]
            fc2, c1o, c2o, fc1 = model(bx, num_test_sample, return_conv=True)
            fc2, c1o, c2o, fc1 = fc2.cpu().numpy(), c1o.cpu().numpy(), c2o.cpu().numpy(), fc1.cpu().numpy()
            conv1_outs.append(c1o)
            conv2_outs.append(c2o)
            fc1_outs.append(fc1)
            fc2_outs.append(fc2)
            labels.append(by.cpu().numpy())
            pca1.partial_fit(c1o.reshape(bs*num_test_sample, -1))
            pca2.partial_fit(c2o.reshape(bs*num_test_sample, -1))
            pca3.partial_fit(fc1.reshape(bs*num_test_sample, -1))
            pca4.partial_fit(fc2.reshape(bs*num_test_sample, -1))
    conv1_outs = np.concatenate(conv1_outs, axis=0)
    conv2_outs = np.concatenate(conv2_outs, axis=0)
    fc1_outs = np.concatenate(fc1_outs, axis=0)
    fc2_outs = np.concatenate(fc2_outs, axis=0)
    labels = np.concatenate(labels, axis=0)
    class_names = [f"Class {x}" for x in labels]
    class_order = [f"Class {x}" for x in range(labels.max()+1)]
    _, axes = plt.subplots(ncols=num_test_sample, figsize=(num_test_sample*6, 6), sharex='col', sharey='row')
    for i_noise in range(num_test_sample):
        ax = axes[i_noise]
        value = pca1.transform(conv1_outs[:, i_noise].reshape(len(conv1_outs), -1))
        sns.scatterplot(x=value[:,0], y=value[:,1], hue=class_names, hue_order=class_order, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'conv1.png'))
    plt.close()

    _, axes = plt.subplots(ncols=num_test_sample, figsize=(num_test_sample*6, 6), sharex='col', sharey='row')
    for i_noise in range(num_test_sample):
        ax = axes[i_noise]
        value = pca2.transform(conv2_outs[:, i_noise].reshape(len(conv2_outs), -1))
        sns.scatterplot(x=value[:,0], y=value[:,1], hue=class_names, hue_order=class_order, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'conv2.png'))
    plt.close()

    _, axes = plt.subplots(ncols=num_test_sample, figsize=(num_test_sample*6, 6), sharex='col', sharey='row')
    for i_noise in range(num_test_sample):
        ax = axes[i_noise]
        value = pca3.transform(fc1_outs[:, i_noise])
        sns.scatterplot(x=value[:,0], y=value[:,1], hue=class_names, hue_order=class_order, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'fc1.png'))
    plt.close()

    _, axes = plt.subplots(ncols=num_test_sample, figsize=(num_test_sample*6, 6), sharex='col', sharey='row')
    for i_noise in range(num_test_sample):
        ax = axes[i_noise]
        value = pca4.transform(fc2_outs[:, i_noise])
        sns.scatterplot(x=value[:,0], y=value[:,1], hue=class_names, hue_order=class_order, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'fc2.png'))
    plt.close()

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
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    num_sample = args.num_samples
    checkpoint = os.path.join(args.root, 'checkpoint.pt')
    text_path = os.path.join(args.root, 'result.txt')
    with open(os.path.join(args.root, 'config.json')) as inp:
        config = json.load(inp)
    test_loader = get_data_loader(args.dataset, args.batch_size, False, test_only=True)
    model = StochasticLeNet(args.width, args.height, args.in_channels, config['conv_hiddens'],
                            config['fc_hidden'], args.classes, config['init_method'], config['activation'],
                            config['init_dist_mean'], config['init_dist_std'], config['init_prior_mean'], config['init_prior_std'],
                            config['noise_type'], config['noise_size'], use_abs=config.get('use_abs', True))
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    os.makedirs(os.path.join(args.root, 'pca', args.dataset), exist_ok=True)
    draw_pca(model, test_loader, device, args.num_samples, os.path.join(args.root, 'pca', args.dataset))

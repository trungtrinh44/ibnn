import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import entropy
from sklearn.metrics import auc, roc_auc_score, roc_curve


def generate_weight_plot(weight: np.ndarray, nrows, ncols, **kwarg):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(
        8 * ncols, 6 * nrows), squeeze=False, **kwarg)
    for i, (ax, col) in enumerate(zip(itertools.chain.from_iterable(axes), weight.T)):
        ax.hist(col)
        ax.set_title(f"Component {i}")


def plot_calibration_curve(y_true, y_prob, n_classes, nrows, ncols, save_path=None):
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, sharex='col', sharey='row', figsize=(8*ncols, 6*nrows))
    fp, mv = [], []
    bins = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.1]
    for ax, cls_id in zip(itertools.chain.from_iterable(axes), range(n_classes)):
        prob = y_prob[:, cls_id]
        total_count = []
        pos_count = []
        pos = (y_true == cls_id).astype(np.int32)
        mean_predicted_value = []
        for xmin, xmax in zip(bins[:-1], bins[1:]):
            bin_prob = prob[prob >= xmin]
            bin_pos = pos[prob >= xmin]
            bin_pos = bin_pos[bin_prob < xmax]
            bin_prob = bin_prob[bin_prob < xmax]
            if bin_pos.shape[0] > 0:
                pos_count.append(bin_pos.sum())
                total_count.append(bin_pos.shape[0])
                mean_predicted_value.append(bin_prob.mean())
        mean_predicted_value = np.array(mean_predicted_value)
        pos_count = np.array(pos_count)
        total_count = np.array(total_count)
        fraction_of_positives = pos_count/total_count
        ax.plot(mean_predicted_value, fraction_of_positives, "s-", alpha=.4)
        for x, y, p, t in zip(mean_predicted_value, fraction_of_positives, pos_count, total_count):
            ax.annotate(r"$\frac{" + str(p) + "}" +
                        "{" + str(t) + "}$", (x, y), fontsize=15)
        ax.plot([0, 1], [0, 1], '--', color='black', alpha=.4)
        ax.set_title(f'Class {cls_id}')
        ax.set_xticks(np.arange(0.0, 1.01, 0.1))
        fp.append(fraction_of_positives)
        mv.append(mean_predicted_value)
        if cls_id % ncols == 0:
            ax.set_ylabel('Fraction of positives')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    return fp, mv


def plot_auc(y_true, y_prob, n_classes, nrows, ncols, save_path=None):
    lw = 2.0
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(
        8*ncols, 6*nrows), sharex='col', sharey='row')
    for i, ax in zip(range(n_classes), itertools.chain.from_iterable(axes)):
        fpr, tpr, _ = roc_curve((y_true == i).astype(np.int32), y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color='darkorange', lw=lw,
                label=f'ROC curve (area = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([0.0, 1.05])
        ax.legend(loc="lower right")
        if i % ncols == 0:
            ax.set_ylabel('True Positive Rate')
        if i / ncols >= nrows - 1:
            ax.set_xlabel('False Positive Rate')
    ovr_score = roc_auc_score(y_true, y_prob, multi_class='ovr')
    ovo_score = roc_auc_score(y_true, y_prob, multi_class='ovo')
    fig.suptitle(f"AUC score: OVR {ovr_score:.4f}, OVO {ovo_score:.4f}")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)


def sort_clear2blur(y_prob):
    return entropy(y_prob.mean(axis=1), axis=1).argsort()


def plot_image_filters(image, conv1, conv2, n_noise=5, save_path=None):
    n_filter = 64
    fig = plt.figure(figsize=((n_filter+2)*1, n_noise*5))
    gs = fig.add_gridspec(n_noise*2, 64+2)
    ax = fig.add_subplot(gs[0, :2])
    ax.set_title('image')
    ax.imshow(image, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    for i_noise in range(n_noise):
        for i_filter in range(32):
            ax = fig.add_subplot(gs[i_noise, 2*i_filter+2:2*i_filter+4])
            if i_noise % n_noise == 0:
                ax.set_title(f'filter {i_filter+1}')
            ax.imshow(conv1[i_noise, i_filter], cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
    for i_noise in range(n_noise):
        for i_filter in range(64):
            ax = fig.add_subplot(gs[i_noise+n_noise, i_filter+2])
            if i_noise % n_noise == 0:
                ax.set_title(f'filter {i_filter+1}')
            ax.imshow(conv2[i_noise, i_filter], cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.close()

def plot_error(x, mean, std, xlabel, legend, save_path):
    _, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, mean)
    ax.fill_between(x, mean-std, mean+std, alpha=.3, label=legend)
    ax.set_xlabel(xlabel)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_filters(y_true, y_prob, test_image, n_classes, conv1, conv2, root, n_noise=5, n_samples=2):
    cleartoblur = sort_clear2blur(y_prob)
    sort_y_test = y_true[cleartoblur]
    for i_class in range(n_classes):
        class_idx = cleartoblur[sort_y_test == i_class]
        class_idx = np.concatenate([class_idx[:n_samples], class_idx[-n_samples:]], axis=0)
        for idx, image_idx in enumerate(class_idx):
            image = test_image[image_idx].astype(np.float32)/255.0
            plot_image_filters(image, conv1[image_idx], conv2[image_idx], n_noise, os.path.join(root, f'{i_class}_{idx}.png'))


def plot_samples(y_true, y_prob, test_image, n_classes, save_path=None, n_samples=2):
    cleartoblur = sort_clear2blur(y_prob)
    sort_y_test = y_true[cleartoblur]
    fig, axes = plt.subplots(nrows=n_samples*4, ncols=n_classes, figsize=(4*n_classes, n_samples*4*4),
                             sharey='row', gridspec_kw={'height_ratios': [2, 1]*n_samples*2})
    for i in range(n_classes):
        im_ax = [axes[2*j][i] for j in range(4)]
        prob_ax = [axes[2*j+1][i] for j in range(4)]
        class_idx = cleartoblur[sort_y_test == i]
        class_idx = np.concatenate(
            [class_idx[:n_samples], class_idx[-n_samples:]], axis=0)
        for idx, ax1, ax2 in zip(class_idx, im_ax, prob_ax):
            image = test_image[idx].astype(np.float32)/255.0
            ax1.imshow(image, cmap='gray')
            ax1.set_xticks([])
            ax1.set_yticks([])
            parts = ax2.violinplot(y_prob[idx], positions=np.arange(n_classes))
            for pc in parts['bodies']:
                pc.set_edgecolor('black')
            for j in range(10):
                ax2.plot(np.random.normal(j, 0.04, len(
                    y_prob[idx, :, j])), y_prob[idx, :, j], 'b.', alpha=0.1)
            ax2.set_xticks(np.arange(n_classes))
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)


def plot_prior_var(data, title, save_path):
    plt.figure(figsize=(8, 6))
    plt.imshow(data)
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)

def plot_mean_std(conv1, conv2, fc1, fc2, save_path):
    fig = plt.figure(figsize=(128+1.5, 20))
    gs = fig.add_gridspec(20, 128+2, width_ratios=[1] + [2]*128 + [0.5])
    ax = fig.add_subplot(gs[:8, 0])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    ax.text(0.5, 0.5, 'Conv 1', fontsize='large', rotation='vertical', horizontalalignment='center', verticalalignment='center')
    vmin, vmax = conv1[0].min(), conv1[0].max()
    for i in range(32):
        ax = fig.add_subplot(gs[:4, 4*i+1:4*i+5])
        im = ax.imshow(conv1[0][i], vmin=vmin, vmax=vmax)
        if i == 0:
            ax.set_ylabel('Mean')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'Filter {i}')
    ax = fig.add_subplot(gs[:4, -1])
    fig.colorbar(im, cax=ax)
    vmin, vmax = conv1[1].min(), conv1[1].max()
    for i in range(32):
        ax = fig.add_subplot(gs[4:8, 4*i+1:4*i+5])
        im = ax.imshow(conv1[1][i], vmin=vmin, vmax=vmax)
        if i == 0:
            ax.set_ylabel('Variance')
        ax.set_xticks([])
        ax.set_yticks([])
    ax = fig.add_subplot(gs[4:8, -1])
    fig.colorbar(im, cax=ax)

    ax = fig.add_subplot(gs[10:14, 0])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    ax.text(0.5, 0.5, 'Conv 2', fontsize='large', rotation='vertical', horizontalalignment='center', verticalalignment='center')

    vmin, vmax = conv2[0].min(), conv2[0].max()
    for i in range(64):
        ax = fig.add_subplot(gs[10:12, 2*i+1:2*i+3])
        im = ax.imshow(conv2[0][i], vmin=vmin, vmax=vmax)
        if i == 0:
            ax.set_ylabel('Mean')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'Filter {i}')
    ax = fig.add_subplot(gs[10:12, -1])
    fig.colorbar(im, cax=ax)

    vmin, vmax = conv2[1].min(), conv2[1].max()
    for i in range(64):
        ax = fig.add_subplot(gs[12:14, 2*i+1:2*i+3])
        im = ax.imshow(conv2[1][i], vmin=vmin, vmax=vmax)
        if i == 0:
            ax.set_ylabel('Variance')
        ax.set_xticks([])
        ax.set_yticks([])
    ax = fig.add_subplot(gs[12:14, -1])
    fig.colorbar(im, cax=ax)

    ax = fig.add_subplot(gs[16:20, 0])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    ax.text(0.5, 0.5, 'FC 1', fontsize='large', rotation='vertical', horizontalalignment='center', verticalalignment='center')

    ax = fig.add_subplot(gs[16:20, 1:9])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Mean')
    vmin, vmax = fc1[0].min(), fc1[0].max()
    im = ax.imshow(fc1[0].reshape((16, 32)), vmin=vmin, vmax=vmax)
    fig.colorbar(im, cax=fig.add_subplot(gs[16:20, 9]))

    ax = fig.add_subplot(gs[16:20, 12:20])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Variance')
    vmin, vmax = fc1[1].min(), fc1[1].max()
    im = ax.imshow(fc1[1].reshape((16, 32)), vmin=vmin, vmax=vmax)
    fig.colorbar(im, cax=fig.add_subplot(gs[16:20, 20]))

    ax = fig.add_subplot(gs[16:20, 22])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    ax.text(0.5, 0.5, 'FC 2', fontsize='large', rotation='vertical', horizontalalignment='center', verticalalignment='center')

    ax = fig.add_subplot(gs[16:20, 23:33])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Mean')
    vmin, vmax = fc2[0].min(), fc2[0].max()
    im = ax.imshow(fc2[0].reshape((1, 10)), vmin=vmin, vmax=vmax)
    fig.colorbar(im, cax=fig.add_subplot(gs[16:20, 34]))

    ax = fig.add_subplot(gs[16:20, 38:48])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Variance')
    vmin, vmax = fc2[1].min(), fc2[1].max()
    im = ax.imshow(fc2[1].reshape((1, 10)), vmin=vmin, vmax=vmax)
    fig.colorbar(im, cax=fig.add_subplot(gs[16:24, 49]))

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

class RunningMeanStd(nn.Module):
    def __init__(self, dim, shape):
        super(RunningMeanStd, self).__init__()
        self.sum = nn.Parameter(torch.zeros(shape), requires_grad=False)
        self.square_sum = nn.Parameter(torch.zeros(shape), requires_grad=False)
        self.count = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.dim = dim
    
    def update(self, samples):
        self.count += samples.size(self.dim)
        self.sum += samples.sum(self.dim)
        self.square_sum += (samples**2).sum(self.dim)
    
    def mean(self):
        return self.sum/self.count

    def var(self):
        return self.square_sum/(self.count-1) - (self.sum/self.count)*(self.sum/(self.count-1))


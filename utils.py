import itertools

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve


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
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
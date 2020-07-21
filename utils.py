import itertools

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy
from sklearn.metrics import auc, roc_curve, roc_auc_score


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


def plot_samples(y_true, y_prob, test_image, n_classes, save_path=None):
    cleartoblur = entropy(y_prob.mean(axis=1), axis=1).argsort()
    sort_y_test = y_true[cleartoblur]
    fig, axes = plt.subplots(nrows=2*n_classes, ncols=10, figsize=(4*10, 8*n_classes),
                             sharey='row', gridspec_kw={'height_ratios': [2, 1]*n_classes})
    for i in range(n_classes):
        im_ax = axes[2*i]
        prob_ax = axes[2*i + 1]
        class_idx = cleartoblur[sort_y_test == i]
        class_idx = np.concatenate([class_idx[:5], class_idx[-5:]], axis=0)
        for idx, ax1, ax2 in zip(class_idx, im_ax, prob_ax):
            image = test_image[idx].astype(np.float32)/255.0
            ax1.imshow(image)
            ax1.set_xticks([])
            ax1.set_yticks([])
            parts = ax2.violinplot(y_prob[idx], positions=np.arange(10))
            for pc in parts['bodies']:
                pc.set_edgecolor('black')
            for j in range(10):
                ax2.plot(np.random.normal(j, 0.04, len(
                    y_prob[idx, :, j])), y_prob[idx, :, j], 'b.', alpha=0.1)
            ax2.set_xticks(np.arange(10))
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score, accuracy_score
from IPython.display import display as _display, Image as _Image
import io

colors = [pl.cm.Paired(i) for i in range(12)]


def _render(fig):
    """Render figure inline without fig.show() — no warnings ever."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    _display(_Image(buf.read()))
    pl.close(fig)


def distribution(data, transformed=False):
    fig = pl.figure(figsize=(11, 5))
    for i, feature in enumerate(['capital-gain', 'capital-loss']):
        ax = fig.add_subplot(1, 2, i + 1)
        ax.hist(data[feature], bins=25, color=colors[0])
        ax.set_title(f"'{feature}' Feature Distribution", fontsize=14)
        ax.set_xlabel("Value")
        ax.set_ylabel("Number of Records")
        ax.set_ylim((0, 2000))
        ax.set_yticks([0, 500, 1000, 1500, 2000])
        ax.set_yticklabels([0, 500, 1000, 1500, ">2000"])
    if transformed:
        fig.suptitle("Log-transformed Distributions of Continuous Census Data Features", fontsize=16, y=1.03)
    else:
        fig.suptitle("Skewed Distributions of Continuous Census Data Features", fontsize=16, y=1.03)
    fig.tight_layout()
    _render(fig)


def evaluate(results, accuracy, f1):
    fig, ax = pl.subplots(2, 3, figsize=(11, 8))
    bar_width = 0.3
    for k, learner in enumerate(results.keys()):
        for j, metric in enumerate(['train_time', 'acc_train', 'f_train', 'pred_time', 'acc_test', 'f_test']):
            for i in np.arange(3):
                ax[j // 3, j % 3].bar(i + k * bar_width, results[learner][i][metric], width=bar_width, color=colors[k])
                ax[j // 3, j % 3].set_xticks([0.45, 1.45, 2.45])
                ax[j // 3, j % 3].set_xticklabels(["1%", "10%", "100%"])
                ax[j // 3, j % 3].set_xlabel("Training Set Size")
                ax[j // 3, j % 3].set_xlim((-0.1, 3.0))
    ax[0, 0].set_ylabel("Time (in seconds)")
    ax[0, 1].set_ylabel("Accuracy Score")
    ax[0, 2].set_ylabel("F-score")
    ax[1, 0].set_ylabel("Time (in seconds)")
    ax[1, 1].set_ylabel("Accuracy Score")
    ax[1, 2].set_ylabel("F-score")
    ax[0, 0].set_title("Model Training")
    ax[0, 1].set_title("Accuracy Score on Training Subset")
    ax[0, 2].set_title("F-score on Training Subset")
    ax[1, 0].set_title("Model Predicting")
    ax[1, 1].set_title("Accuracy Score on Testing Set")
    ax[1, 2].set_title("F-score on Testing Set")
    ax[0, 1].axhline(y=accuracy, xmin=-0.1, xmax=3.0, linewidth=1, color='k', linestyle='dashed')
    ax[1, 1].axhline(y=accuracy, xmin=-0.1, xmax=3.0, linewidth=1, color='k', linestyle='dashed')
    ax[0, 2].axhline(y=f1, xmin=-0.1, xmax=3.0, linewidth=1, color='k', linestyle='dashed')
    ax[1, 2].axhline(y=f1, xmin=-0.1, xmax=3.0, linewidth=1, color='k', linestyle='dashed')
    ax[0, 1].set_ylim((0, 1))
    ax[0, 2].set_ylim((0, 1))
    ax[1, 1].set_ylim((0, 1))
    ax[1, 2].set_ylim((0, 1))
    patches = [mpatches.Patch(color=colors[i], label=learner) for i, learner in enumerate(results.keys())]
    pl.legend(handles=patches, bbox_to_anchor=(-.80, 2.53), loc='upper center', borderaxespad=0., ncol=3, fontsize='x-large')
    pl.suptitle("Performance Metrics for Three Supervised Learning Models", fontsize=16, x=0.63, y=1.05)
    pl.subplots_adjust(left=0.125, right=1.2, bottom=0.1, top=0.9, wspace=0.2, hspace=0.3)
    _render(fig)


def feature_plot(importances, X_train, y_train):
    indices = np.argsort(importances)[::-1]
    columns = X_train.columns.values[indices[:5]]
    values = importances[indices][:5]
    fig = pl.figure(figsize=(9, 5))
    pl.title("Normalized Weights for First Five Most Predictive Features", fontsize=16)
    pl.bar(np.arange(5) - 0.1, values, width=0.2, align="center", color=colors[0], label="Feature Weight")
    pl.bar(np.arange(5) + 0.1, np.cumsum(values), width=0.2, align="center", color=colors[1], label="Cumulative Feature Weight")
    pl.xticks(np.arange(5), columns, rotation=20)
    pl.xlim((-0.5, 4.5))
    pl.ylabel("Weight", fontsize=12)
    pl.xlabel("Feature", fontsize=12)
    pl.legend()
    pl.tight_layout()
    _render(fig)

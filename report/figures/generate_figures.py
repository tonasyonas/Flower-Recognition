# report/figures/generate_figures.py
"""
Generates all report figures from results/logs/*.csv.

Outputs:
  - parameter-efficiency.pdf  (Fig 1)
  - fewshot-curves.pdf        (Fig 2)
  - training-curves.pdf       (Fig 3)
  - prompt-length.pdf         (Fig 5)
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'results', 'logs'))
OUT_DIR = SCRIPT_DIR

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})


# === Test-set results from the evaluation runs. ===
# (method_name, trainable_params, test_acc, mpc_acc)
RESULTS = [
    ('VPT-Deep',       170_598,    0.9164, 0.9283),
    ('VPT-Shallow',    86_118,     0.8413, 0.8570),
    ('Frozen ResNet18',10_545_766, 0.8852, 0.9082),
    ('Label Smooth',   11_228_838, 0.8727, 0.8940),
    ('LS + Strong Aug',11_228_838, 0.8640, 0.8824),
    ('ResNet50',       23_717_030, 0.8593, 0.8807),
    ('Strong Aug',     11_228_838, 0.8442, 0.8611),
    ('Baseline',       11_228_838, 0.8442, 0.8611),
    ('Deformable',     11_415_534, 0.8133, 0.8346),
    ('MixUp',          11_228_838, 0.8052, 0.8350),
    ('Hybrid',         11_353_290, 0.7873, 0.8134),
    ('Dilated',        11_228_838, 0.7595, 0.7853),
    ('Depthwise',      1_623_270,  0.5703, 0.5971),
]


def fig_parameter_efficiency():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, params, _, mpc in RESULTS:
        marker = 'o'
        color = 'tab:red' if name.startswith('VPT') else ('tab:blue' if name == 'Baseline' else 'tab:gray')
        ax.scatter(params, mpc * 100, s=60, c=color, alpha=0.85, marker=marker)
        xytext = (5, 5) if name not in {'Baseline', 'Strong Aug'} else (5, -10)
        ax.annotate(name, (params, mpc * 100), xytext=xytext, textcoords='offset points', fontsize=8)
    ax.set_xscale('log')
    ax.set_xlabel('Trainable parameters (log scale)')
    ax.set_ylabel('Mean per-class accuracy (%)')
    ax.set_title('Parameter efficiency on Oxford Flowers 102')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'parameter-efficiency.pdf'))
    plt.close(fig)


FEWSHOT = {
    'VPT-Deep':        {1: 0.4637, 2: 0.6882, 5: 0.8490, 10: 0.9363},
    'Frozen ResNet18': {1: 0.3510, 2: 0.6578, 5: 0.8147, 10: 0.9167},
    'VPT-Shallow':     {1: 0.3784, 2: 0.5716, 5: 0.7745, 10: 0.8608},
    'Baseline':        {1: 0.3520, 2: 0.5931, 5: 0.7961, 10: 0.8833},
    'Deformable':      {1: 0.2696, 2: 0.5422, 5: 0.7745, 10: 0.8461},
    'Depthwise':       {1: 0.0696, 2: 0.1275, 5: 0.3941, 10: 0.6078},
}


def fig_fewshot_curves():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    shots = [1, 2, 5, 10]
    for name, results in FEWSHOT.items():
        ys = [results[k] * 100 for k in shots]
        ax.plot(shots, ys, 'o-', label=name, linewidth=2, markersize=7)
    ax.set_xticks(shots)
    ax.set_xlabel('Images per class ($k$)')
    ax.set_ylabel('Best validation accuracy (%)')
    ax.set_title('Few-shot learning curves')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fewshot-curves.pdf'))
    plt.close(fig)


def fig_training_curves():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    methods = [
        ('baseline', 'Baseline'),
        ('frozen', 'Frozen'),
        ('resnet50', 'ResNet50'),
        ('deformable', 'Deformable'),
        ('vpt_shallow', 'VPT-Shallow'),
        ('vpt_deep', 'VPT-Deep'),
    ]
    for key, label in methods:
        path = os.path.join(LOGS_DIR, f'{key}_training_log.csv')
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        ax.plot(df['epoch'], df['val_acc'] * 100, label=label, linewidth=1.6)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation accuracy (%)')
    ax.set_title('Training curves: validation accuracy per epoch')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'training-curves.pdf'))
    plt.close(fig)


PROMPT_LENGTH = {
    1: 0.9108,
    5: 0.9294,
    10: 0.9363,
    50: 0.9451,
}


def fig_prompt_length():
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = sorted(PROMPT_LENGTH.keys())
    ys = [PROMPT_LENGTH[x] * 100 for x in xs]
    ax.plot(xs, ys, 'o-', linewidth=2, markersize=8, color='tab:red')
    ax.set_xscale('log')
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel('Number of prompts per layer')
    ax.set_ylabel('Best validation accuracy (%)')
    ax.set_title('VPT-Deep: prompt length ablation')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'prompt-length.pdf'))
    plt.close(fig)


if __name__ == '__main__':
    fig_parameter_efficiency()
    fig_fewshot_curves()
    fig_training_curves()
    fig_prompt_length()
    print('Figures written to', OUT_DIR)

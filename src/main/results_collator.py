import os
import shutil
import seaborn as sns
import matplotlib.pyplot as plt
import json
import pandas as pd
import time
import numpy as np
import matplotlib.ticker as mticker

def load_auc_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data["Per-class AUC"]

def class_auc_collated(json_paths, model_names, class_names, output_dir, MODE):

    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
                os.remove(os.path.join(output_dir, file))
                print(f"Removed: {file}")
    time.sleep(3)
    model_aucs = {}
    for path, model_name in zip(json_paths, model_names):
        model_aucs[model_name] = load_auc_data(path)

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(class_names))
    width = 0.25

    color_map = {
        'CLIP': '#2ca02c',
        'RETFound': '#1f77b4',
        'UrFound': '#ff7f0e'
    }

    for i, model_name in enumerate(model_names):
        color = color_map[model_name]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, model_aucs[model_name], width,
                     label=model_name, color=color, edgecolor='black')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('AUC-ROC Score')
    ax.set_xlabel('Diabetic Retinopathy Severity')
    ax.set_title(f"{MODE} Tuned Models - AUC Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylim([0.5, 1.0])
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}")


COLOR_MAP = {
    'CLIP':     '#2ca02c',
    'RETFound': '#1f77b4',
    'UrFound':  '#ff7f0e',
}

PUBLICATION_RC = {
    'font.family':      'serif',
    'font.serif':       ['Times New Roman', 'DejaVu Serif'],
    'axes.labelsize':   13,
    'axes.titlesize':   14,
    'xtick.labelsize':  11,
    'ytick.labelsize':  11,
    'legend.fontsize':  11,
    'figure.dpi':       150,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.linewidth':   0.8,
    'xtick.direction':  'out',
    'ytick.direction':  'out',
}


def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _apply_rc():
    plt.rcParams.update(PUBLICATION_RC)


def _plot_single_metric(json_paths, model_names, output_dir, MODE,
                        metric_key, metric_display, filename):
    _apply_rc()

    values = [_load_json(p)[metric_key] for p in json_paths]
    n_models = len(model_names)

    fig, ax = plt.subplots(figsize=(7, 2.8))

    y_positions = np.arange(n_models)[::-1]

    v_min  = min(values)
    v_max  = max(values)
    margin = max((v_max - v_min) * 0.5, 0.05)
    x_lo   = max(0.0, v_min - margin)
    x_hi   = min(1.0, v_max + margin)

    for y, model, value in zip(y_positions, model_names, values):
        color = COLOR_MAP.get(model, 'C0')

        ax.hlines(y, x_lo, value, color=color, linewidth=2.0, alpha=0.7)
        ax.scatter(value, y, color=color, s=120, zorder=5,
                   edgecolors='white', linewidths=1.2)
        ax.text(value + (x_hi - x_lo) * 0.018, y, f'{value:.4f}',
                va='center', ha='left', fontsize=10,
                color=color, fontfamily='serif', fontweight='bold')

    ax.set_yticks(y_positions)
    ax.set_yticklabels(model_names, fontsize=11)
    ax.set_xlim(x_lo, x_hi + (x_hi - x_lo) * 0.18)
    ax.set_xlabel(metric_display, labelpad=8)
    ax.set_title(f'{MODE} — {metric_display}', pad=10)
    ax.grid(axis='x', alpha=0.3, linewidth=0.6)
    ax.axvline(x_lo, color='black', linewidth=0.6)

    plt.tight_layout()
    out = os.path.join(output_dir, filename)
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


def _label_bars(ax, bars, fmt='{:.3f}', fontsize=8.5, pad=0.004):
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2., h + pad,
            fmt.format(h),
            ha='center', va='bottom', fontsize=fontsize,
            fontfamily='serif'
        )


# -------------------------
# DR plotting functions
# -------------------------

def plot_auc_bars(json_paths, model_names, output_dir, MODE):
    _apply_rc()

    metrics  = ['macro_auc', 'weighted_auc']
    labels   = ['Macro AUC', 'Weighted AUC']
    x        = np.arange(len(labels))
    width    = 0.22
    n_models = len(model_names)

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (path, model) in enumerate(zip(json_paths, model_names)):
        data   = _load_json(path)
        vals   = [data[m] for m in metrics]
        offset = (i - (n_models - 1) / 2) * width
        color  = COLOR_MAP.get(model, f'C{i}')
        bars   = ax.bar(x + offset, vals, width,
                        label=model, color=color,
                        edgecolor='black', linewidth=0.6, zorder=3)
        _label_bars(ax, bars)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('AUC Score')
    ax.set_title(f'{MODE} — Macro & Weighted AUC Comparison', pad=12)
    ax.set_ylim(0.5, 1.05)
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))
    ax.grid(axis='y', which='major', alpha=0.3, linewidth=0.6, zorder=0)
    ax.grid(axis='y', which='minor', alpha=0.15, linewidth=0.4, zorder=0)
    ax.legend(frameon=True, framealpha=0.9, edgecolor='#cccccc')

    plt.tight_layout()
    out = os.path.join(output_dir, 'auc_bar_comparison.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


def plot_precision(json_paths, model_names, output_dir, MODE):
    _plot_single_metric(json_paths, model_names, output_dir, MODE,
                        'precision', 'Precision', 'precision.png')


def plot_recall(json_paths, model_names, output_dir, MODE):
    _plot_single_metric(json_paths, model_names, output_dir, MODE,
                        'recall', 'Recall', 'recall.png')


def plot_f1(json_paths, model_names, output_dir, MODE):
    _plot_single_metric(json_paths, model_names, output_dir, MODE,
                        'f1_score', 'F1 Score', 'f1_score.png')


def plot_qwk(json_paths, model_names, output_dir, MODE):
    _plot_single_metric(json_paths, model_names, output_dir, MODE,
                        'quadratic_weighted_kappa',
                        'Quadratic Weighted Kappa', 'qwk.png')


def plot_all_metrics(json_paths, model_names, output_dir, MODE):
    os.makedirs(output_dir, exist_ok=True)

    plot_auc_bars(json_paths, model_names, output_dir, MODE)
    plot_precision(json_paths, model_names, output_dir, MODE)
    plot_recall(json_paths, model_names, output_dir, MODE)
    plot_f1(json_paths, model_names, output_dir, MODE)
    plot_qwk(json_paths, model_names, output_dir, MODE)

    print(f'\nAll plots saved to: {output_dir}')


# -------------------------
# Glaucoma plotting functions
# keys available: accuracy, balanced_accuracy, macro_f1,
#                 sensitivity, specificity, macro_auc,
#                 weighted_auc, per_class_auc
# -------------------------

def plot_glaucoma_auc_bars(json_paths, model_names, output_dir, MODE):
    """macro auc and weighted auc side-by-side bar chart"""
    _apply_rc()

    metrics = ['macro_auc', 'weighted_auc']
    labels  = ['Macro AUC', 'Weighted AUC']
    x       = np.arange(len(labels))
    width   = 0.22
    n_models = len(model_names)

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (path, model) in enumerate(zip(json_paths, model_names)):
        data   = _load_json(path)
        vals   = [data[m] for m in metrics]
        offset = (i - (n_models - 1) / 2) * width
        color  = COLOR_MAP.get(model, f'C{i}')
        bars   = ax.bar(x + offset, vals, width,
                        label=model, color=color,
                        edgecolor='black', linewidth=0.6, zorder=3)
        _label_bars(ax, bars)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('AUC Score')
    ax.set_title(f'{MODE} Glaucoma — Macro & Weighted AUC', pad=12)
    ax.set_ylim(0.5, 1.05)
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))
    ax.grid(axis='y', which='major', alpha=0.3, linewidth=0.6, zorder=0)
    ax.grid(axis='y', which='minor', alpha=0.15, linewidth=0.4, zorder=0)
    ax.legend(frameon=True, framealpha=0.9, edgecolor='#cccccc')

    plt.tight_layout()
    out = os.path.join(output_dir, 'glaucoma_auc_bar_comparison.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')



def plot_glaucoma_balanced_accuracy(json_paths, model_names, output_dir, MODE):
    _plot_single_metric(json_paths, model_names, output_dir, MODE,
                        'balanced_accuracy', 'Balanced Accuracy',
                        'glaucoma_balanced_accuracy.png')


def plot_glaucoma_macro_f1(json_paths, model_names, output_dir, MODE):
    _plot_single_metric(json_paths, model_names, output_dir, MODE,
                        'macro_f1', 'Macro F1',
                        'glaucoma_macro_f1.png')


def plot_glaucoma_sensitivity(json_paths, model_names, output_dir, MODE):
    _plot_single_metric(json_paths, model_names, output_dir, MODE,
                        'sensitivity', 'Sensitivity (Glaucoma Recall)',
                        'glaucoma_sensitivity.png')


def plot_glaucoma_specificity(json_paths, model_names, output_dir, MODE):
    _plot_single_metric(json_paths, model_names, output_dir, MODE,
                        'specificity', 'Specificity (Healthy Recall)',
                        'glaucoma_specificity.png')


def plot_all_metrics_glaucoma(json_paths, model_names, output_dir, MODE):
    os.makedirs(output_dir, exist_ok=True)

    plot_glaucoma_auc_bars(json_paths, model_names, output_dir, MODE)
    plot_glaucoma_balanced_accuracy(json_paths, model_names, output_dir, MODE)
    plot_glaucoma_macro_f1(json_paths, model_names, output_dir, MODE)
    plot_glaucoma_sensitivity(json_paths, model_names, output_dir, MODE)
    plot_glaucoma_specificity(json_paths, model_names, output_dir, MODE)

    print(f'\nAll glaucoma plots saved to: {output_dir}')


if __name__ == "__main__":

    DR_NONLORA_TEST_RESULTS_DIR = "./testing/non-lora/results"
    DR_LORA_TEST_RESULTS_DIR = "./testing/lora-based/results"
    GLAUCOMA_RESNET50_TEST_RESULTS_DIR = "./testing/results/resnet50-glaucoma"
    GLAUCOMA_RESNET50_TEST_PLOTS_DIR = "../plots/baseline-plots/resnet50-glaucoma-testing-plots"
    #lora_jsons = [
    #]

    #non_lora_jsons = [
    #]
        

    if os.path.exists(GLAUCOMA_RESNET50_TEST_PLOTS_DIR):
        shutil.rmtree(GLAUCOMA_RESNET50_TEST_PLOTS_DIR)
        print(f"Removed directory: {GLAUCOMA_RESNET50_TEST_PLOTS_DIR}")

    os.makedirs(GLAUCOMA_RESNET50_TEST_PLOTS_DIR, exist_ok=True)
    print(f"Created directory: {GLAUCOMA_RESNET50_TEST_PLOTS_DIR}")

    resnet_50_glaucoma_jsons = [f"{GLAUCOMA_RESNET50_TEST_RESULTS_DIR}/resnet50_glaucoma_test_results.json"]

    dr_classes = [
        "No DR",
        "Mild",
        "Moderate",
        "Severe",
        "Proliferative DR"
    ]

    glaucoma_classes = ["Healthy", "Glaucoma"]

    model_names = ["ResNet50"]

    # DR plots
    # class_auc_collated(non_lora_jsons, model_names, dr_classes, "../plots/nonlora-final-plots", "NON-LORA")
    # class_auc_collated(lora_jsons, model_names, dr_classes, "../plots/lora-final-plots", "LORA")

    # Glaucoma plots
    plot_all_metrics_glaucoma(resnet_50_glaucoma_jsons, model_names, GLAUCOMA_RESNET50_TEST_PLOTS_DIR, "ResNet50-Glaucoma")

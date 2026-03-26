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
    'ResNet50': '#9467bd',   # purple — distinct from the three foundation models
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
# DR plotting functions (foundation models)
# JSON keys: precision, recall, f1_score, quadratic_weighted_kappa,
#            macro_auc, weighted_auc, Per-class AUC
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
# JSON keys: accuracy, balanced_accuracy, macro_f1,
#            sensitivity, specificity, macro_auc,
#            weighted_auc, per_class_auc
# -------------------------

def plot_glaucoma_auc_bars(json_paths, model_names, output_dir, MODE):
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


# -------------------------
# ResNet50 DR plotting functions
# JSON keys: accuracy, balanced_accuracy, macro_f1, qwk,
#            macro_auc, weighted_auc, per_class_auc
# Note: these keys differ from the foundation model DR JSONs which use
# precision, recall, f1_score, quadratic_weighted_kappa — so the
# existing plot_all_metrics cannot be reused here.
# -------------------------

def plot_resnet50_dr_auc_bars(json_paths, model_names, output_dir, MODE):
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
    ax.set_title(f'{MODE} DR Grading — Macro & Weighted AUC', pad=12)
    ax.set_ylim(0.5, 1.05)
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))
    ax.grid(axis='y', which='major', alpha=0.3, linewidth=0.6, zorder=0)
    ax.grid(axis='y', which='minor', alpha=0.15, linewidth=0.4, zorder=0)
    ax.legend(frameon=True, framealpha=0.9, edgecolor='#cccccc')

    plt.tight_layout()
    out = os.path.join(output_dir, 'resnet50_dr_auc_bar_comparison.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


def plot_resnet50_dr_per_class_auc(json_paths, model_names, output_dir, MODE):
    """per-class auc bar chart across all five DR grades."""
    _apply_rc()

    dr_class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
    x       = np.arange(len(dr_class_names))
    width   = 0.22
    n_models = len(model_names)

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (path, model) in enumerate(zip(json_paths, model_names)):
        data   = _load_json(path)
        # per_class_auc is a dict keyed by class name
        vals   = [data['per_class_auc'].get(c) or 0.0 for c in dr_class_names]
        offset = (i - (n_models - 1) / 2) * width
        color  = COLOR_MAP.get(model, f'C{i}')
        bars   = ax.bar(x + offset, vals, width,
                        label=model, color=color,
                        edgecolor='black', linewidth=0.6, zorder=3)
        _label_bars(ax, bars)

    ax.set_xticks(x)
    ax.set_xticklabels(dr_class_names, rotation=30, ha='right')
    ax.set_ylabel('AUC Score')
    ax.set_title(f'{MODE} DR Grading — Per-Class AUC (One-vs-Rest)', pad=12)
    ax.set_ylim(0.5, 1.05)
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))
    ax.grid(axis='y', which='major', alpha=0.3, linewidth=0.6, zorder=0)
    ax.grid(axis='y', which='minor', alpha=0.15, linewidth=0.4, zorder=0)
    ax.legend(frameon=True, framealpha=0.9, edgecolor='#cccccc')

    plt.tight_layout()
    out = os.path.join(output_dir, 'resnet50_dr_per_class_auc.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


def plot_resnet50_dr_balanced_accuracy(json_paths, model_names, output_dir, MODE):
    _plot_single_metric(json_paths, model_names, output_dir, MODE,
                        'balanced_accuracy', 'Balanced Accuracy',
                        'resnet50_dr_balanced_accuracy.png')


def plot_resnet50_dr_macro_f1(json_paths, model_names, output_dir, MODE):
    _plot_single_metric(json_paths, model_names, output_dir, MODE,
                        'macro_f1', 'Macro F1',
                        'resnet50_dr_macro_f1.png')


def plot_resnet50_dr_qwk(json_paths, model_names, output_dir, MODE):
    _plot_single_metric(json_paths, model_names, output_dir, MODE,
                        'qwk', 'Quadratic Weighted Kappa',
                        'resnet50_dr_qwk.png')


def plot_all_metrics_resnet50_dr(json_paths, model_names, output_dir, MODE):
    os.makedirs(output_dir, exist_ok=True)

    plot_resnet50_dr_auc_bars(json_paths, model_names, output_dir, MODE)
    plot_resnet50_dr_per_class_auc(json_paths, model_names, output_dir, MODE)
    plot_resnet50_dr_balanced_accuracy(json_paths, model_names, output_dir, MODE)
    plot_resnet50_dr_macro_f1(json_paths, model_names, output_dir, MODE)
    plot_resnet50_dr_qwk(json_paths, model_names, output_dir, MODE)

    print(f'\nAll ResNet50 DR plots saved to: {output_dir}')


# -------------------------
# Jensen-Shannon Divergence
# -------------------------

JS_DIVERGENCE_EPSILON = 1e-10

def js_divergence(p, q):
    p = np.clip(p, JS_DIVERGENCE_EPSILON, 1.0)
    q = np.clip(q, JS_DIVERGENCE_EPSILON, 1.0)
    m = 0.5 * (p + q)
    return float(np.mean(0.5 * np.sum(p * np.log(p / m) + q * np.log(q / m), axis=1)))


def compute_pairwise_js(probs_dict):
    names  = list(probs_dict.keys())
    result = {}
    for a in names:
        for b in names:
            result[f"{a}_vs_{b}"] = 0.0 if a == b else js_divergence(
                probs_dict[a], probs_dict[b]
            )
    return result


def plot_js_heatmap(js_dict, model_names, title, output_path):
    plt.rcParams.update(PUBLICATION_RC)
    n      = len(model_names)
    matrix = np.zeros((n, n))
    for i, a in enumerate(model_names):
        for j, b in enumerate(model_names):
            matrix[i, j] = js_dict[f"{a}_vs_{b}"]

    cell_size = 1.1
    fig_size  = max(6, n * cell_size)
    fig, ax   = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    sns.heatmap(
        matrix, annot=True, fmt=".4f",
        xticklabels=model_names, yticklabels=model_names,
        cmap="YlOrRd", ax=ax, linewidths=0.5,
        annot_kws={"size": max(7, 10 - n)},
        cbar_kws={"label": "JS Divergence"}
    )
    ax.set_title(title, pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":

    DR_NONLORA_TEST_RESULTS_DIR       = "./testing/non-lora/results"
    DR_LORA_TEST_RESULTS_DIR          = "./testing/lora-based/results"
    GLAUCOMA_RESNET50_TEST_RESULTS_DIR = "./testing/results/resnet50-glaucoma"
    GLAUCOMA_RESNET50_TEST_PLOTS_DIR  = "../plots/baseline-plots/resnet50-glaucoma-testing-plots"
    DR_RESNET50_TEST_RESULTS_DIR      = "./testing/results/resnet50-dr"
    DR_RESNET50_TEST_PLOTS_DIR        = "../plots/baseline-plots/resnet50-dr-testing-plots"

    PROBS_DIR      = "./testing/probs_numpy"
    JS_PLOT_DIR    = "../plots/js-divergence"
    JS_RESULTS_DIR = "./testing/results/js-divergence"

    dr_classes       = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
    glaucoma_classes = ["Healthy", "Glaucoma"]

    # -------------------------
    # ResNet50 DR plots
    # -------------------------
    resnet50_dr_jsons = [
        f"{DR_RESNET50_TEST_RESULTS_DIR}/resnet50_dr_test_results.json"
    ]
    resnet50_dr_model_names = ["ResNet50"]

    if all(os.path.exists(p) for p in resnet50_dr_jsons):
        plot_all_metrics_resnet50_dr(
            resnet50_dr_jsons,
            resnet50_dr_model_names,
            DR_RESNET50_TEST_PLOTS_DIR,
            "BASELINE"
        )
    else:
        print("Skipping ResNet50 DR plots — results JSON not found.")

    # -------------------------
    # ResNet50 Glaucoma plots
    # -------------------------
    # resnet_50_glaucoma_jsons = [
    #     f"{GLAUCOMA_RESNET50_TEST_RESULTS_DIR}/resnet50_glaucoma_test_results.json"
    # ]
    # plot_all_metrics_glaucoma(resnet_50_glaucoma_jsons, ["ResNet50"], GLAUCOMA_RESNET50_TEST_PLOTS_DIR, "BASELINE")

    # -------------------------
    # JS divergence
    # -------------------------
    dr_prob_files = {
        "RETFound_LORA_DR":    f"{PROBS_DIR}/retfound_dr_lora_probs.npy",
        "RETFound_NONLORA_DR": f"{PROBS_DIR}/retfound_dr_nonlora_probs.npy",
        "UrFound_LORA_DR":     f"{PROBS_DIR}/urfound_lora_dr_probs.npy",
        "UrFound_NONLORA_DR":  f"{PROBS_DIR}/urfound_nonlora_dr_probs.npy",
        "CLIP_LORA_DR":        f"{PROBS_DIR}/clip_dr_lora_probs.npy",
        "CLIP_NONLORA_DR":     f"{PROBS_DIR}/clip_dr_nonlora_probs.npy",
        "ResNet50_DR":         f"{PROBS_DIR}/resnet50-dr-testing.npy",
    }

    glaucoma_prob_files = {
        "RETFound_LORA_GLAUCOMA":    f"{PROBS_DIR}/retfound_lora_glaucoma_probs.npy",
        "RETFound_NONLORA_GLAUCOMA": f"{PROBS_DIR}/retfound_glaucoma_nonlora_probs.npy",
        "UrFound_LORA_GLAUCOMA":     f"{PROBS_DIR}/urfound_lora_glaucoma_probs.npy",
        "UrFound_NONLORA_GLAUCOMA":  f"{PROBS_DIR}/urfound_nonlora_glaucoma_probs.npy",
        "CLIP_LORA_GLAUCOMA":        f"{PROBS_DIR}/clip_lora_glaucoma_probs.npy",
        "CLIP_NONLORA_GLAUCOMA":     f"{PROBS_DIR}/clip_glaucoma_nonlora_probs.npy",
        "ResNet50_GLAUCOMA":         f"{PROBS_DIR}/resnet50-glaucoma-testing-probs.npy",
    }

    # DR JS divergence
    if all(os.path.exists(p) for p in dr_prob_files.values()):
        probs_dict     = {name: np.load(path) for name, path in dr_prob_files.items()}
        js_results     = compute_pairwise_js(probs_dict)
        model_names_js = list(probs_dict.keys())

        os.makedirs(JS_RESULTS_DIR, exist_ok=True)
        with open(f"{JS_RESULTS_DIR}/dr_js_divergence.json", "w") as f:
            json.dump({"task": "DR severity grading", "js_divergence": js_results}, f, indent=4)

        os.makedirs(JS_PLOT_DIR, exist_ok=True)
        plot_js_heatmap(
            js_results, model_names_js,
            "DR Severity Grading — Pairwise Jensen-Shannon Divergence\n(0 = identical distributions, 1 = non-overlapping)",
            f"{JS_PLOT_DIR}/dr_js_heatmap.png"
        )
    else:
        missing = [name for name, path in dr_prob_files.items() if not os.path.exists(path)]
        print(f"Skipping DR JS divergence — missing prob files for: {missing}")

    # Glaucoma JS divergence
    if all(os.path.exists(p) for p in glaucoma_prob_files.values()):
        probs_dict     = {name: np.load(path) for name, path in glaucoma_prob_files.items()}
        js_results     = compute_pairwise_js(probs_dict)
        model_names_js = list(probs_dict.keys())

        os.makedirs(JS_RESULTS_DIR, exist_ok=True)
        with open(f"{JS_RESULTS_DIR}/glaucoma_js_divergence.json", "w") as f:
            json.dump({"task": "glaucoma detection", "js_divergence": js_results}, f, indent=4)

        os.makedirs(JS_PLOT_DIR, exist_ok=True)
        plot_js_heatmap(
            js_results, model_names_js,
            "Glaucoma Detection — Pairwise Jensen-Shannon Divergence\n(0 = identical distributions, 1 = non-overlapping)",
            f"{JS_PLOT_DIR}/glaucoma_js_heatmap.png"
        )
    else:
        missing = [name for name, path in glaucoma_prob_files.items() if not os.path.exists(path)]
        print(f"Skipping glaucoma JS divergence — missing prob files for: {missing}")

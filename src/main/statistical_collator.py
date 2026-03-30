import os, json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy.stats import chi2
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, roc_auc_score, cohen_kappa_score
)


PROBS_DIR = "./testing/probs_numpy"
OUT_DIR   = "./testing/results/statistical-tests"
PLOT_DIR  = "../plots/statistical-tests"
N_BOOT    = 1000
ALPHA     = 0.05
CI        = 95
THRESHOLD = 0.6    # glaucoma decision threshold matches what is in testing scripts
rng       = np.random.default_rng(42)

RC = {
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.labelsize': 11, 'axes.titlesize': 12,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'figure.dpi': 150,
}

GLAUCOMA_MODELS = {
    "RetFound-LoRA-Glaucoma":     "retfound_lora_glaucoma",
    "RetFound-NoLoRA-Glaucoma":   "retfound_glaucoma_nonlora",
    "UrFound-LoRA-Glaucoma":     "urfound_lora_glaucoma",
    "UrFound-NoLoRA-Glaucoma":   "urfound_nonlora_glaucoma",
    "CLIP-LoRA-Glaucoma":   "clip_lora_glaucoma",
    "CLIP-NoLoRA-Glaucoma": "clip_nonlora_glaucoma",
    "ResNet50-Glaucoma":    "resnet50_glaucoma",
}
DR_MODELS = {
    "RetFound-LoRA-DR":     "retfound_dr_lora",
    "RetFound-NoLoRA-DR":   "retfound_dr_nonlora",
    "UrFound-LoRA-DR":     "urfound_lora_dr",
    "UrFound-NoLoRA-DR":   "urfound-dr-nonlora",
    "CLIP-LoRA-DR":   "clip_dr_lora",
    "CLIP-NoLoRA-DR": "clip-dr-nonlora",
    "ResNet50-DR":    "resnet50-dr",
}


# loading — y_pred derived from y_probs

def load(prefix, task):
    yt  = np.load(f"{PROBS_DIR}/{prefix}_true.npy")
    ypr = np.load(f"{PROBS_DIR}/{prefix}_probs.npy")
    yp  = (ypr[:, 1] >= THRESHOLD).astype(int) if task == "glaucoma" \
          else ypr.argmax(axis=1)
    return yt, yp, ypr

def exists(prefix):
    return all(os.path.exists(f"{PROBS_DIR}/{prefix}_{s}.npy")
               for s in ["true", "probs"])


# metrics

def compute_metrics(yt, yp, ypr, task):
    m = {
        "Balanced Acc": 100.0 * balanced_accuracy_score(yt, yp),
        "Macro F1":     100.0 * f1_score(yt, yp, average="macro", zero_division=0),
    }
    try:
        m["Macro AUC"] = 100.0 * (
            roc_auc_score(yt, ypr[:, 1]) if task == "glaucoma"
            else roc_auc_score(yt, ypr, multi_class="ovr", average="macro")
        )
    except ValueError:
        m["Macro AUC"] = 0.0
    if task == "dr":
        try:    m["QWK"] = cohen_kappa_score(yt, yp, weights="quadratic")
        except: m["QWK"] = 0.0
    return m


# bootstrapping CI

def bootstrap(yt, yp, ypr, task):
    keys   = list(compute_metrics(yt, yp, ypr, task).keys())
    scores = {k: [] for k in keys}
    n      = len(yt)
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        for k, v in compute_metrics(yt[idx], yp[idx], ypr[idx], task).items():
            scores[k].append(v)
    lo = (100 - CI) / 2
    return {k: (float(np.mean(v)),
                float(np.percentile(v, lo)),
                float(np.percentile(v, 100 - lo)))
            for k, v in scores.items()}


# McNemar's test

def mcnemar(yt, yp_a, yp_b):
    b01 = int(np.sum((yp_a != yt) & (yp_b == yt)))
    b10 = int(np.sum((yp_a == yt) & (yp_b != yt)))
    d   = b01 + b10
    if d == 0: return 0.0, 1.0, b01, b10
    stat = (abs(b01 - b10) - 1.0) ** 2 / d
    return float(stat), float(1.0 - chi2.cdf(stat, df=1)), b01, b10

def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


# plot: horizontal bar chart with CIs

def plot_ci(ci_all, title, path):
    plt.rcParams.update(RC)
    models = list(ci_all.keys())
    met    = list(next(iter(ci_all.values())).keys())
    n_m    = len(models)

    fig, axes = plt.subplots(1, len(met),
                             figsize=(4.5 * len(met), max(3, n_m * 0.55 + 1.5)))
    if len(met) == 1: axes = [axes]

    for ax, metric in zip(axes, met):
        y      = np.arange(n_m)
        means  = [ci_all[m][metric][0] for m in models]
        lowers = [ci_all[m][metric][0] - ci_all[m][metric][1] for m in models]
        uppers = [ci_all[m][metric][2] - ci_all[m][metric][0] for m in models]

        bars = ax.barh(y, means, xerr=[lowers, uppers], align='center',
                       color='#4c72b0', ecolor='#c0392b', capsize=4,
                       height=0.55, error_kw={"linewidth": 1.5, "zorder": 5})

        for bar, mean in zip(bars, means):
            ax.text(bar.get_width() * 0.98, bar.get_y() + bar.get_height() / 2,
                    f'{mean:.2f}', va='center', ha='right',
                    fontsize=8, color='white', fontweight='bold')

        ax.set_yticks(y)
        ax.set_yticklabels(models)
        ax.set_xlabel(f"{metric} ({CI}% CI)")
        ax.set_title(metric)
        ax.grid(axis='x', alpha=0.3, linewidth=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# plot: McNemar p-value heatmap with significance stars

def plot_mcnemar(mc, model_names, title, path):
    plt.rcParams.update(RC)
    n     = len(model_names)
    p_mat = np.ones((n, n))
    idx   = {m: i for i, m in enumerate(model_names)}

    for pair, (_, p, _, _) in mc.items():
        a, b = pair.split("_vs_")
        if a in idx and b in idx:
            p_mat[idx[a], idx[b]] = p
            p_mat[idx[b], idx[a]] = p

    annot = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            annot[i, j] = "" if i == j else f"{p_mat[i,j]:.3f}\n{stars(p_mat[i,j])}"

    cell  = max(1.2, n * 0.85)
    fig, ax = plt.subplots(figsize=(cell, cell))
    sns.heatmap(p_mat, annot=annot, fmt="", mask=np.eye(n, dtype=bool),
                xticklabels=model_names, yticklabels=model_names,
                cmap="RdYlGn_r", vmin=0, vmax=0.1,
                ax=ax, linewidths=0.5,
                cbar_kws={"label": "p-value", "shrink": 0.8},
                annot_kws={"size": max(6, 9 - n)})
    ax.set_title(f"{title}\n* p<0.05  ** p<0.01  *** p<0.001  ns = not significant",
                 pad=12, fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def run(model_map, task, label):
    print(f"\n{'='*60}\n{label}\n{'='*60}")
    ci_all, pred_map = {}, {}

    for name, prefix in model_map.items():
        if not exists(prefix):
            print(f"  Skipping {name} — {prefix}_true/probs.npy not found")
            continue
        yt, yp, ypr    = load(prefix, task)
        ci_all[name]   = bootstrap(yt, yp, ypr, task)
        pred_map[name] = (yt, yp)
        for k, (mean, lo, hi) in ci_all[name].items():
            print(f"  {name:<14} {k:<14}: {mean:.2f}  ({CI}% CI {lo:.2f}–{hi:.2f})")

    print(f"\n  McNemar's Test")
    mc = {}
    for a, b in combinations(list(pred_map.keys()), 2):
        yta, ypa = pred_map[a]
        ytb, ypb = pred_map[b]
        if len(yta) != len(ytb): continue
        stat, p, b01, b10 = mcnemar(yta, ypa, ypb)
        mc[f"{a}_vs_{b}"] = (stat, p, b01, b10)
        print(f"  {a:<14} vs {b:<14}: chi2={stat:.3f}  p={p:.4f}  "
              f"{stars(p)}  (b01={b01} b10={b10})")

    os.makedirs(PLOT_DIR, exist_ok=True)
    slug = task.lower()
    if ci_all:
        plot_ci(ci_all, f"{label} — Bootstrap {CI}% Confidence Intervals",
                f"{PLOT_DIR}/{slug}_bootstrap_ci.png")
    if mc:
        plot_mcnemar(mc, list(pred_map.keys()),
                     f"{label} — McNemar's Test",
                     f"{PLOT_DIR}/{slug}_mcnemar.png")

    return ci_all, mc



def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    g_ci, g_mc = run(GLAUCOMA_MODELS, "glaucoma", "Glaucoma Detection")
    d_ci, d_mc = run(DR_MODELS,       "dr",       "DR Severity Grading")

    def serialise(ci, mc):
        return {
            "bootstrap_ci": {
                m: {k: {"mean": v[0], "ci_lo": v[1], "ci_hi": v[2]}
                    for k, v in met.items()}
                for m, met in ci.items()
            },
            "mcnemar": {
                pair: {"chi2": v[0], "p_value": v[1],
                       "b01": v[2], "b10": v[3],
                       "significant": v[1] < ALPHA}
                for pair, v in mc.items()
            }
        }

    with open(f"{OUT_DIR}/statistical_tests.json", "w") as f:
        json.dump({
            "glaucoma":   serialise(g_ci, g_mc),
            "dr_grading": serialise(d_ci, d_mc),
            "config":     {"n_bootstrap": N_BOOT, "ci": CI,
                           "alpha": ALPHA, "glaucoma_threshold": THRESHOLD}
        }, f, indent=4)

    print(f"\nAll results saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()

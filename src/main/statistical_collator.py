import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy.stats import chi2
from sklearn.metrics import f1_score, roc_auc_score

PROBS_DIR = "./testing/probs_numpy"
OUT_DIR = "./testing/results/statistical-tests"
PLOT_DIR = "../plots/statistical-tests"

N_BOOT = 1000
N_PERM = 5000
ALPHA = 0.05
CI = 95

rng = np.random.default_rng(42)

RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150
}

ODIR_MODELS = {
    "RETFound-ODIR": "retfound_mixed_disease",
    "ResNet50-ODIR": "resnet50_mixed_disease"
}

ODIR_CLASS_NAMES = [
    "Normal", "Diabetes", "Glaucoma", "Cataract",
    "AMD", "Hypertension", "Myopia", "Other"
]

def exists(prefix):
    return all(
        os.path.exists(f"{PROBS_DIR}/{prefix}_{s}.npy")
        for s in ["true", "probs"]
    )

def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

def load_odir(prefix):
    yt = np.load(f"{PROBS_DIR}/{prefix}_true.npy").astype(int)
    ypr = np.load(f"{PROBS_DIR}/{prefix}_probs.npy").astype(float)
    yp = (ypr >= 0.5).astype(int)
    return yt, yp, ypr

def compute_metrics(yt, yp, ypr):
    m = {}
    m["Exact Match"] = 100.0 * float((yp == yt).all(axis=1).mean())
    m["Macro F1"] = 100.0 * float(
        f1_score(yt, yp, average="macro", zero_division=0)
    )
    try:
        m["Macro AUC"] = 100.0 * float(
            roc_auc_score(yt, ypr, average="macro")
        )
    except ValueError:
        m["Macro AUC"] = 0.0
    return m

def bootstrap_ci(yt, yp, ypr):
    keys = list(compute_metrics(yt, yp, ypr).keys())
    vals = {k: [] for k in keys}
    n = len(yt)

    for _ in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        res = compute_metrics(yt[idx], yp[idx], ypr[idx])
        for k, v in res.items():
            vals[k].append(v)

    lo = (100 - CI) / 2

    return {
        k: (
            float(np.mean(v)),
            float(np.percentile(v, lo)),
            float(np.percentile(v, 100 - lo))
        )
        for k, v in vals.items()
    }

def mcnemar_test(y_true, pred_a, pred_b):
    a_ok = pred_a == y_true
    b_ok = pred_b == y_true

    b01 = int(np.sum((~a_ok) & b_ok))
    b10 = int(np.sum(a_ok & (~b_ok)))

    d = b01 + b10

    if d == 0:
        return 0.0, 1.0, b01, b10

    stat = (abs(b01 - b10) - 1.0) ** 2 / d
    p = 1.0 - chi2.cdf(stat, 1)

    return float(stat), float(p), b01, b10

def permutation_macro_f1(yt, pa, pb):
    obs = (
        f1_score(yt, pa, average="macro", zero_division=0)
        -
        f1_score(yt, pb, average="macro", zero_division=0)
    )

    n = len(yt)
    count = 0

    for _ in range(N_PERM):
        mask = rng.integers(0, 2, size=n).astype(bool)

        xa = pa.copy()
        xb = pb.copy()

        xa[mask], xb[mask] = pb[mask], pa[mask].copy()

        diff = (
            f1_score(yt, xa, average="macro", zero_division=0)
            -
            f1_score(yt, xb, average="macro", zero_division=0)
        )

        if abs(diff) >= abs(obs):
            count += 1

    p = (count + 1) / (N_PERM + 1)

    return float(obs * 100.0), float(p)

def plot_ci(ci_all, path):
    plt.rcParams.update(RC)

    models = list(ci_all.keys())
    metrics = list(next(iter(ci_all.values())).keys())

    fig, axes = plt.subplots(
        1, len(metrics),
        figsize=(4.5 * len(metrics), max(3, len(models) * 0.8 + 1.5))
    )

    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        y = np.arange(len(models))

        means = [ci_all[m][metric][0] for m in models]
        lowers = [ci_all[m][metric][0] - ci_all[m][metric][1] for m in models]
        uppers = [ci_all[m][metric][2] - ci_all[m][metric][0] for m in models]

        bars = ax.barh(
            y, means,
            xerr=[lowers, uppers],
            capsize=4,
            height=0.55
        )

        for bar, val in zip(bars, means):
            ax.text(
                bar.get_width() * 0.98,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}",
                ha="right",
                va="center",
                fontsize=8,
                color="white",
                fontweight="bold"
            )

        ax.set_yticks(y)
        ax.set_yticklabels(models)
        ax.set_title(metric)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle(f"ODIR-5K Bootstrap {CI}% Confidence Intervals")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_mcnemar_heatmap(results, pair_name, path):
    plt.rcParams.update(RC)

    pvals = np.array([[results[c]["p_value"]] for c in ODIR_CLASS_NAMES])
    annot = np.array([
        [
            f"p={results[c]['p_value']:.3f} {stars(results[c]['p_value'])}\n"
            f"χ²={results[c]['chi2']:.2f}\n"
            f"b01={results[c]['b01']} b10={results[c]['b10']}"
        ]
        for c in ODIR_CLASS_NAMES
    ])

    fig, ax = plt.subplots(figsize=(5, 6))

    sns.heatmap(
        pvals,
        annot=annot,
        fmt="",
        cmap="RdYlGn_r",
        vmin=0,
        vmax=0.1,
        cbar_kws={"label": "p-value"},
        yticklabels=ODIR_CLASS_NAMES,
        xticklabels=[pair_name],
        linewidths=0.5,
        ax=ax
    )

    ax.set_title("Per-label McNemar Test")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_permutation(results, path):
    plt.rcParams.update(RC)

    names = list(results.keys())
    diffs = [results[k]["macro_f1_difference"] for k in names]
    pvals = [results[k]["p_value"] for k in names]

    fig, ax = plt.subplots(figsize=(7, 4))

    bars = ax.barh(np.arange(len(names)), diffs, height=0.55)

    for i, bar in enumerate(bars):
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f" {diffs[i]:.2f} ({stars(pvals[i])})",
            va="center"
        )

    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Δ Macro F1 (%)")
    ax.set_title("Permutation Test")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    ci_all = {}
    pred_map = {}

    for name, prefix in ODIR_MODELS.items():
        if not exists(prefix):
            continue

        yt, yp, ypr = load_odir(prefix)
        ci_all[name] = bootstrap_ci(yt, yp, ypr)
        pred_map[name] = (yt, yp, ypr)

        for k, v in ci_all[name].items():
            print(
                f"{name:<16} {k:<12}: "
                f"{v[0]:.2f} ({CI}% CI {v[1]:.2f}-{v[2]:.2f})"
            )

    pair_results = {}
    perm_results = {}

    for a, b in combinations(pred_map.keys(), 2):
        yt, pa, _ = pred_map[a]
        _, pb, _ = pred_map[b]

        pair = f"{a}_vs_{b}"
        pair_results[pair] = {}

        for j, cname in enumerate(ODIR_CLASS_NAMES):
            stat, p, b01, b10 = mcnemar_test(
                yt[:, j],
                pa[:, j],
                pb[:, j]
            )

            pair_results[pair][cname] = {
                "chi2": stat,
                "p_value": p,
                "b01": b01,
                "b10": b10,
                "significant": p < ALPHA
            }

        diff, p = permutation_macro_f1(yt, pa, pb)

        perm_results[pair] = {
            "macro_f1_difference": diff,
            "p_value": p,
            "significant": p < ALPHA
        }

    plot_ci(ci_all, f"{PLOT_DIR}/odir_bootstrap_ci.png")

    for pair, res in pair_results.items():
        plot_mcnemar_heatmap(
            res,
            pair,
            f"{PLOT_DIR}/{pair.lower()}_mcnemar.png"
        )

    plot_permutation(
        perm_results,
        f"{PLOT_DIR}/odir_permutation_macro_f1.png"
    )

    out = {
        "bootstrap_ci": {
            m: {
                k: {
                    "mean": v[0],
                    "ci_lo": v[1],
                    "ci_hi": v[2]
                }
                for k, v in met.items()
            }
            for m, met in ci_all.items()
        },
        "per_label_mcnemar": pair_results,
        "permutation_macro_f1": perm_results,
        "config": {
            "n_bootstrap": N_BOOT,
            "n_permutations": N_PERM,
            "alpha": ALPHA
        }
    }

    with open(f"{OUT_DIR}/odir_statistical_tests.json", "w") as f:
        json.dump(out, f, indent=4)

if __name__ == "__main__":
    run()

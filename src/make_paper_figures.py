#!/usr/bin/env python3
"""
make_paper_figures.py

Generates the core figure set for:
"FiLM-Conditioned Attention-Based MIL Reveals Subtype-Specific Morphological
Encoding of Antigen Presentation and T-Cell Inflammation in NSCLC"

Usage:
    python src/make_paper_figures.py --artifacts-dir results/paper_artifacts --out-dir figures

Requires: numpy, pandas, matplotlib, scikit-learn, scipy, statsmodels
    pip install numpy pandas matplotlib scikit-learn scipy statsmodels
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import pearsonr, linregress
from statsmodels.stats.multitest import multipletests  # falls back below if unavailable

# Style settings
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 9,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "svg.fonttype": "none",  # keep text editable in vector output
})

SUBTYPE_COLORS = {"LUAD": "#2166AC", "LUSC": "#B2182B"}
PANEL_COLORS   = {"APM": "#4DAF4A", "TIS": "#984EA3", "other": "#999999"}
FOLD_CMAP      = plt.cm.viridis


# Loading helpers
def load_artifacts(artifacts_dir: Path):
    with open(artifacts_dir / "model_config.json") as f:
        config = json.load(f)
    with open(artifacts_dir / "bootstrap_permutation_stats.json") as f:
        boot_stats = json.load(f)
    with open(artifacts_dir / "results.json") as f:
        results = json.load(f)

    fold_npz = {}
    for p in sorted(artifacts_dir.glob("fold*_test_predictions.npz")):
        fold = int(p.stem.split("fold")[1].split("_")[0])
        fold_npz[fold] = np.load(p, allow_pickle=True)

    ensemble = np.load(artifacts_dir / "ensemble_test_predictions.npz", allow_pickle=True)

    return config, boot_stats, results, fold_npz, ensemble


def panel_score(preds_or_labels, gene_cols, panel_genes):
    """Mean expression across panel genes (matches compute_panel_pcc definition)."""
    gene_cols = [g.replace("_fpkm_uq", "") for g in gene_cols]
    idxs = [gene_cols.index(g) for g in panel_genes if g in gene_cols]
    return preds_or_labels[:, idxs].mean(axis=1)


def fdr_correct(pvals):
    try:
        return multipletests(pvals, method="fdr_bh")[1]
    except NameError:
        # statsmodels not installed - fallback
        pvals = np.asarray(pvals)
        n = len(pvals)
        order = np.argsort(pvals)
        ranked = pvals[order] * n / (np.arange(n) + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        out = np.empty(n)
        out[order] = np.clip(ranked, 0, 1)
        return out


# Figure 1: predicted vs actual scatter
def fig_scatter(ensemble, config, boot_stats, out_dir):
    gene_cols = list(ensemble["gene_cols"])
    subtypes = ensemble["subtype"]
    preds, labels = ensemble["preds"], ensemble["labels"]

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.4))
    for ax, panel in zip(axes, ["APM", "TIS"]):
        genes = config[f"{panel}_GENES"]
        x = panel_score(labels, gene_cols, genes)
        y = panel_score(preds, gene_cols, genes)

        for subtype, color in SUBTYPE_COLORS.items():
            mask = subtypes == subtype
            ax.scatter(x[mask], y[mask], s=16, alpha=0.65, color=color,
                       edgecolor="white", linewidth=0.3, label=subtype)

        slope, intercept, r, p, _ = linregress(x, y)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, slope * xs + intercept, color="black", linewidth=1, linestyle="--")

        stats = boot_stats["ensemble"][panel]
        p_str = "<0.001" if stats["perm_p"] < 0.001 else f"{stats['perm_p']:.3f}"
        ax.set_title(
            f"{panel}: r={stats['PCC']:.2f} "
            f"(95% CI {stats['CI95_low']:.2f}\u2013{stats['CI95_high']:.2f}), "
            f"p={p_str}",
            fontsize=8,
        )
        ax.set_xlabel(f"Actual {panel} score")
        ax.set_ylabel(f"Predicted {panel} score")

    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Predicted vs. actual signature scores (ensemble, held-out test set)", fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir, "fig1_predicted_vs_actual")


# Figure 2: ROC curves
def fig_roc(fold_npz, ensemble, config, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.4))
    for ax, panel in zip(axes, ["APM", "TIS"]):
        genes = config[f"{panel}_GENES"]

        for fold, data in sorted(fold_npz.items()):
            gene_cols = list(data["gene_cols"])
            pred_score = panel_score(data["preds"], gene_cols, genes)
            label_score = panel_score(data["labels"], gene_cols, genes)
            thresh = np.percentile(label_score, 75)
            y_true = (label_score >= thresh).astype(int)
            if y_true.sum() in (0, len(y_true)):
                continue
            fpr, tpr, _ = roc_curve(y_true, pred_score)
            auc = roc_auc_score(y_true, pred_score)
            ax.plot(fpr, tpr, color=FOLD_CMAP(fold / max(1, len(fold_npz) - 1)),
                     linewidth=0.9, alpha=0.6, label=f"Fold {fold} (AUC={auc:.2f})")

        gene_cols = list(ensemble["gene_cols"])
        pred_score = panel_score(ensemble["preds"], gene_cols, genes)
        label_score = panel_score(ensemble["labels"], gene_cols, genes)
        thresh = np.percentile(label_score, 75)
        y_true = (label_score >= thresh).astype(int)
        fpr, tpr, _ = roc_curve(y_true, pred_score)
        auc = roc_auc_score(y_true, pred_score)
        ax.plot(fpr, tpr, color="black", linewidth=2, label=f"Ensemble (AUC={auc:.2f})")

        ax.plot([0, 1], [0, 1], color="gray", linewidth=0.7, linestyle=":")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"{panel} (upper-quartile binarized)", fontsize=9)
        ax.legend(frameon=False, fontsize=6, loc="lower right")

    fig.suptitle("ROC curves: per-fold models and ensemble", fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir, "fig2_roc_curves")


# Figure 3: fold-to-fold stability
def fig_fold_stability(results, out_dir):
    rows = []
    for r in results:
        rows.append({"fold": r["fold"], "panel": "APM", "PCC": r["APM_PCC"]})
        rows.append({"fold": r["fold"], "panel": "TIS", "PCC": r["TIS_PCC"]})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(4, 3.4))
    for i, panel in enumerate(["APM", "TIS"]):
        vals = df[df.panel == panel]["PCC"].values
        x = np.full(len(vals), i) + np.random.default_rng(0).uniform(-0.08, 0.08, len(vals))
        ax.scatter(x, vals, color=PANEL_COLORS[panel], s=40, zorder=3, edgecolor="white", linewidth=0.5)
        ax.boxplot([vals], positions=[i], widths=0.4, showfliers=False,
                   boxprops=dict(color="gray"), whiskerprops=dict(color="gray"),
                   medianprops=dict(color="black"))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["APM", "TIS"])
    ax.set_ylabel("Test-set PCC")
    ax.set_title(f"Stability across {df.fold.nunique()} independently-trained folds\n(same fixed held-out test set)", fontsize=8.5)
    fig.tight_layout()
    _save(fig, out_dir, "fig3_fold_stability")


# Figure 4: subtype-stratified comparison
def fig_subtype_comparison(ensemble, config, out_dir, n_boot=2000):
    gene_cols = list(ensemble["gene_cols"])
    subtypes = ensemble["subtype"]
    preds, labels = ensemble["preds"], ensemble["labels"]
    rng = np.random.default_rng(1)

    bars = []
    for panel in ["APM", "TIS"]:
        genes = config[f"{panel}_GENES"]
        for subtype in ["LUAD", "LUSC"]:
            mask = subtypes == subtype
            p, l = preds[mask], labels[mask]
            x = panel_score(l, gene_cols, genes)
            y = panel_score(p, gene_cols, genes)
            r_obs, _ = pearsonr(x, y)

            boot = []
            n = len(x)
            for _ in range(n_boot):
                idx = rng.integers(0, n, n)
                bx, by = x[idx], y[idx]
                if bx.std() == 0 or by.std() == 0:
                    continue
                boot.append(pearsonr(bx, by)[0])
            lo, hi = np.percentile(boot, [2.5, 97.5])
            bars.append({"panel": panel, "subtype": subtype, "PCC": r_obs,
                         "err_lo": r_obs - lo, "err_hi": hi - r_obs, "n": mask.sum()})

    df = pd.DataFrame(bars)
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    width = 0.35
    x_pos = np.arange(2)
    for i, subtype in enumerate(["LUAD", "LUSC"]):
        sub = df[df.subtype == subtype].set_index("panel").loc[["APM", "TIS"]]
        offset = (i - 0.5) * width
        ax.bar(x_pos + offset, sub["PCC"], width, color=SUBTYPE_COLORS[subtype],
               label=f"{subtype} (n={sub['n'].iloc[0]})",
               yerr=[sub["err_lo"], sub["err_hi"]], capsize=3, error_kw=dict(linewidth=0.8))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(["APM", "TIS"])
    ax.set_ylabel("PCC (bootstrap 95% CI)")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Subtype-specific prediction performance", fontsize=9)
    fig.tight_layout()
    _save(fig, out_dir, "fig4_subtype_comparison")
    df.to_csv(out_dir / "fig4_subtype_comparison_data.csv", index=False)


# Figure 5: per-gene PCC
def fig_gene_pcc(ensemble, config, out_dir):
    gene_cols = [g.replace("_fpkm_uq", "") for g in ensemble["gene_cols"]]
    preds, labels = ensemble["preds"], ensemble["labels"]
    apm_set = set(config["APM_GENES"])
    tis_set = set(config["TIS_GENES"])

    rows = []
    for i, g in enumerate(gene_cols):
        r, p = pearsonr(preds[:, i], labels[:, i])
        panel = "APM" if g in apm_set else ("TIS" if g in tis_set else "other")
        rows.append({"gene": g, "PCC": r, "p": p, "panel": panel})
    df = pd.DataFrame(rows).sort_values("PCC", ascending=True)
    df["q"] = fdr_correct(df["p"].values)

    fig, ax = plt.subplots(figsize=(4.5, max(3.5, 0.22 * len(df))))
    colors = df["panel"].map(PANEL_COLORS)
    ax.hlines(df["gene"], 0, df["PCC"], color=colors, linewidth=1.5)
    ax.scatter(df["PCC"], df["gene"], color=colors, s=25, zorder=3)
    for y, (sig, q) in enumerate(zip(df["PCC"], df["q"])):
        if q < 0.05:
            ax.text(sig + 0.02 * np.sign(sig if sig != 0 else 1), y, "*", va="center", fontsize=10)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Gene-level PCC (ensemble, test set)")
    handles = [plt.Line2D([0], [0], color=c, lw=3) for c in PANEL_COLORS.values()]
    ax.legend(handles, PANEL_COLORS.keys(), frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Per-gene prediction accuracy (* = FDR q<0.05)", fontsize=9)
    fig.tight_layout()
    _save(fig, out_dir, "fig5_gene_level_pcc")
    df.to_csv(out_dir / "fig5_gene_level_pcc_data.csv", index=False)


# Figure 6: forest plot of bootstrap CIs
def fig_forest(boot_stats, out_dir):
    rows = []
    for name, panels in boot_stats.items():
        for panel, stats in panels.items():
            rows.append({"source": name, "panel": panel, **stats})
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.6), sharey=True)
    for ax, panel in zip(axes, ["APM", "TIS"]):
        sub = df[df.panel == panel].copy()
        sub["order"] = sub["source"].apply(lambda s: -1 if s == "ensemble" else int(s.replace("fold", "")))
        sub = sub.sort_values("order")
        y = np.arange(len(sub))
        colors = ["black" if s == "ensemble" else FOLD_CMAP(i / max(1, len(sub) - 2))
                  for i, s in enumerate(sub["source"])]
        ax.errorbar(sub["PCC"], y,
                    xerr=[sub["PCC"] - sub["CI95_low"], sub["CI95_high"] - sub["PCC"]],
                    fmt="o", color="black", ecolor="gray", capsize=3, markersize=5)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["source"])
        ax.axvline(0, color="gray", linewidth=0.6, linestyle=":")
        ax.set_xlabel("PCC (95% bootstrap CI)")
        ax.set_title(panel, fontsize=9)

    fig.suptitle("Bootstrap confidence intervals: per-fold models vs. ensemble", fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir, "fig6_forest_plot")


# Utility
def _save(fig, out_dir, name):
    fig.savefig(out_dir / f"{name}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")  # vector, for submission
    plt.close(fig)
    print(f"  saved {name}.png / .pdf")


ap = argparse.ArgumentParser()
ap.add_argument("--artifacts-dir", type=Path, required=True,
                    help="Path to unzipped paper_artifacts folder")
ap.add_argument("--out-dir", type=Path, default=Path("figures"))
args = ap.parse_args()
args.out_dir.mkdir(parents=True, exist_ok=True)

print(f"Loading from {args.artifacts_dir} ...")
config, boot_stats, results, fold_npz, ensemble = load_artifacts(args.artifacts_dir)

print("Figure 1: predicted vs actual scatter")
fig_scatter(ensemble, config, boot_stats, args.out_dir)

print("Figure 2: ROC curves")
fig_roc(fold_npz, ensemble, config, args.out_dir)

print("Figure 3: fold-to-fold stability")
fig_fold_stability(results, args.out_dir)

print("Figure 4: subtype-stratified comparison")
fig_subtype_comparison(ensemble, config, args.out_dir)

print("Figure 5: per-gene PCC")
fig_gene_pcc(ensemble, config, args.out_dir)

print("Figure 6: forest plot")
fig_forest(boot_stats, args.out_dir)

print(f"\nAll figures saved to {args.out_dir.resolve()}")
"""
compare_film_vs_nofilm.py

Compares the FiLM and no-FiLM CPTAC external-validation runs produced by `kaggle-external-validation.ipynb`, and produces the statistics + figures

WHAT IT EXPECTS:
Two directories, one per run (i.e. two runs of the notebook with FILM_ENABLED = True / False), each containing the files the notebook saves:

    ensemble_cptac_predictions.npz
    fold0_cptac_predictions.npz ... fold4_cptac_predictions.npz
    model_config.json
    cptac_gene_level_pcc.csv          (optional, recomputed here anyway)

Download the whole `results/` output folder from each Kaggle run and point FILM_DIR / NOFILM_DIR at them below

Optionally, if you also have fold-level TCGA held-out predictions saved the
same way (fold0_tcga_predictions.npz ... foldN_tcga_predictions.npz,
ensemble_tcga_predictions.npz, model_config_used.json), point
TCGA_FILM_DIR / TCGA_NOFILM_DIR at those directories to include TCGA rows in
the fold-level subtype gap forest plot below. Leave them as None to skip TCGA
and get CPTAC-only rows.

WHAT IT PRODUCES (in OUT_DIR):
Stats (CSV):
    paired_ensemble_diff.csv        FiLM-vs-noFiLM PCC diff, paired bootstrap CI, p
    delong_auc_diff.csv             paired AUC comparison (DeLong)
    fold_variance_test.csv          Levene/Bartlett test on 5-fold PCCs
    subtype_fisher_z.csv            LUAD vs LUSC PCC comparison, per model/panel
    subtype_fold_ttest.csv          LUAD vs LUSC PCC, paired t-test across 5 folds, per model/panel
    fold_subtype_gap.csv            LUSC-LUAD fold-level PCC gap, per model/cohort/panel, with 95% CI
    gene_rank_concordance.csv       Spearman rho between FiLM/no-FiLM gene PCCs
    gene_level_comparison.csv       merged per-gene PCC table (both models)

Figures (PNG, 300dpi):
    fig_forest_fold_pcc.png         per-fold + ensemble PCC, both models, with CI
    fig_forest_subtype_gap.png      forest plot of fold-level APM/TIS subtype (LUSC-LUAD) gaps, TCGA + CPTAC
    fig_subtype_panel_bars.png      LUAD/LUSC x APM/TIS x model, grouped bars + CI
    fig_gene_scatter.png            per-gene PCC, FiLM vs no-FiLM
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# CONFIG — edit these paths
FILM_DIR   = Path("results/FiLM/external-validation")     # FILM_ENABLED = True run
NOFILM_DIR = Path("results/no-FiLM/external-validation")   # FILM_ENABLED = False run

# Optional: per-patient TCGA predictions saved the same way as the CPTAC ones. "None" skips the formal generalization-gap test
TCGA_FILM_NPZ   = Path("results/FiLM/training")   # e.g. Path("tcga_held_out_film") / "ensemble_tcga_predictions.npz"
TCGA_NOFILM_NPZ = Path("results/no-FiLM/training")

# Optional: directories of fold-level TCGA held-out predictions, same layout as FILM_DIR/NOFILM_DIR
# (fold0_tcga_predictions.npz ... foldN_tcga_predictions.npz, ensemble_tcga_predictions.npz,
# model_config_used.json). "None" skips TCGA rows in the fold-level subtype gap forest plot.
TCGA_FILM_DIR   = Path("results/FiLM/training")   # e.g. Path("results/FiLM/tcga-held-out")
TCGA_NOFILM_DIR = Path("results/no-FiLM/training")   # e.g. Path("results/no-FiLM/tcga-held-out")

# TCGA scalar ensemble PCCs, used for the descriptive generalization-gap bar chart
TCGA_ENSEMBLE_PCC = {
    "FiLM":    {"APM": 0.5964, "TIS": 0.7260},
    "No-FiLM": {"APM": 0.5846, "TIS": 0.7179},
}

OUT_DIR = Path("figures/TEST_film_comparison_outputs")
N_BOOT = 5000
N_PERM = 5000
SEED = 0
rng = np.random.default_rng(SEED)

PANEL_COLORS = {"APM": "#4C72B0", "TIS": "#DD8452"}
MODEL_STYLE = {"FiLM": dict(color="#C44E52", marker="o"),
               "No-FiLM": dict(color="#55A868", marker="s")}

# LOADING
def load_run(run_dir: Path, cohort: str = "cptac"):
    """
    Load ensemble + per-fold predictions/labels and gene/panel config for one run.

    `cohort` selects the file naming convention ("cptac" or "tcga"), so this
    same loader works for the CPTAC external-validation files and for
    fold-level TCGA held-out files saved with the same layout
    (foldN_{cohort}_predictions.npz, ensemble_{cohort}_predictions.npz).
    """
    run_dir = Path(run_dir)
    cfg = json.load(open(run_dir / "model_config.json"))
    gene_cols = cfg["gene_cols"]
    apm_genes = cfg["APM_GENES"]
    tis_genes = cfg["TIS_GENES"]

    ens = np.load(run_dir / f"ensemble_{cohort}_predictions.npz", allow_pickle=True)
    ensemble_preds = ens["preds"]
    labels = ens["labels"]
    subtypes = ens["subtype"]
    sids = ens["submitter_id"]

    fold_preds = {}
    for f in sorted(run_dir.glob(f"fold*_{cohort}_predictions.npz")):
        fold_idx = int(f.stem.split("_")[0].replace("fold", ""))
        d = np.load(f, allow_pickle=True)
        fold_preds[fold_idx] = d["preds"]

    return dict(
        gene_cols=gene_cols, apm_genes=apm_genes, tis_genes=tis_genes,
        ensemble_preds=ensemble_preds, labels=labels,
        subtypes=subtypes, sids=sids, fold_preds=fold_preds,
    )


# CORE METRICS
def panel_score(arr, gene_cols, panel_genes):
    idx = [gene_cols.index(f"{g}_fpkm_uq") for g in panel_genes if f"{g}_fpkm_uq" in gene_cols]
    return arr[:, idx].mean(axis=1)


def panel_pcc(preds, labels, gene_cols, panel_genes):
    x = panel_score(preds, gene_cols, panel_genes)
    y = panel_score(labels, gene_cols, panel_genes)
    return float(np.corrcoef(x, y)[0, 1])


def panel_binary_labels(labels, gene_cols, panel_genes, pct=75):
    y = panel_score(labels, gene_cols, panel_genes)
    thresh = np.percentile(y, pct)
    return (y >= thresh).astype(int)


def panel_score_pred(preds, gene_cols, panel_genes):
    return panel_score(preds, gene_cols, panel_genes)


# 1. PAIRED BOOTSTRAP: FiLM - No-FiLM ensemble PCC difference
def paired_bootstrap_pcc_diff(film, nofilm, panel, genes, n_boot=N_BOOT):
    """
    Resamples the SAME patient indices for both models each iteration (paired), so the resulting CI is on the difference, not two marginals. Requires the two runs to share the same patient order/IDs.
    """
    assert list(film["sids"]) == list(nofilm["sids"]), \
        "FiLM and no-FiLM runs must have the same patients in the same order"

    n = len(film["sids"])
    obs_film = panel_pcc(film["ensemble_preds"], film["labels"], film["gene_cols"], genes)
    obs_nofilm = panel_pcc(nofilm["ensemble_preds"], nofilm["labels"], nofilm["gene_cols"], genes)
    obs_diff = obs_film - obs_nofilm

    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        pcc_f = panel_pcc(film["ensemble_preds"][idx], film["labels"][idx], film["gene_cols"], genes)
        pcc_n = panel_pcc(nofilm["ensemble_preds"][idx], nofilm["labels"][idx], nofilm["gene_cols"], genes)
        diffs[i] = pcc_f - pcc_n

    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # two-sided bootstrap p-value: proportion of the null-centered distribution as extreme as 0 being excluded, approximated via how much of the CI mass crosses zero
    p_boot = float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    p_boot = min(p_boot, 1.0)

    return dict(panel=panel, PCC_FiLM=obs_film, PCC_NoFiLM=obs_nofilm,
                diff=obs_diff, CI95_low=float(lo), CI95_high=float(hi), p_boot=p_boot)


# 2. DeLONG'S TEST for paired AUC comparison
def _delong_placements(pos_scores, neg_scores):
    """Fast DeLong via the structural components (Sun & Xu, 2014)."""
    m, n = len(pos_scores), len(neg_scores)
    all_scores = np.concatenate([pos_scores, neg_scores])
    order = np.argsort(all_scores)
    ranks = np.empty(len(all_scores))
    ranks[order] = stats.rankdata(all_scores[order])
    pos_ranks = ranks[:m]
    neg_ranks = ranks[m:]
    auc = (pos_ranks.sum() - m * (m + 1) / 2) / (m * n)

    v10 = (pos_ranks - stats.rankdata(pos_scores)) / n
    v01 = 1 - (neg_ranks - stats.rankdata(neg_scores)) / m
    return auc, v10, v01


def delong_paired_test(pred_a, pred_b, y_true):
    #Paired DeLong test comparing AUCs of two correlated classifiers (pred_a, pred_b) on the same binary outcome y_true. Returns AUCs, their variance-covariance-based difference SE, z, and two-sided p
    pos = y_true == 1
    neg = y_true == 0
    auc_a, v10_a, v01_a = _delong_placements(pred_a[pos], pred_a[neg])
    auc_b, v10_b, v01_b = _delong_placements(pred_b[pos], pred_b[neg])

    s10 = np.cov(np.vstack([v10_a, v10_b]))
    s01 = np.cov(np.vstack([v01_a, v01_b]))
    m, n = pos.sum(), neg.sum()
    var_cov = s10 / m + s01 / n

    diff = auc_a - auc_b
    se = np.sqrt(var_cov[0, 0] + var_cov[1, 1] - 2 * var_cov[0, 1])
    if se == 0 or np.isnan(se):
        return dict(auc_a=float(auc_a), auc_b=float(auc_b), diff=float(diff),
                     se=float(se), z=float("nan"), p=float("nan"))
    z = diff / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return dict(auc_a=float(auc_a), auc_b=float(auc_b), diff=float(diff),
                se=float(se), z=float(z), p=float(p))


# 3. FOLD-LEVEL VARIANCE TEST (assessing if FiLM is less stable across folds)
def fold_variance_test(film, nofilm, panel, genes):
    film_fold_pccs = [panel_pcc(p, film["labels"], film["gene_cols"], genes)
                       for _, p in sorted(film["fold_preds"].items())]
    nofilm_fold_pccs = [panel_pcc(p, nofilm["labels"], nofilm["gene_cols"], genes)
                         for _, p in sorted(nofilm["fold_preds"].items())]

    lev_stat, lev_p = stats.levene(film_fold_pccs, nofilm_fold_pccs)
    bart_stat, bart_p = stats.bartlett(film_fold_pccs, nofilm_fold_pccs)

    return dict(
        panel=panel,
        FiLM_fold_pccs=film_fold_pccs, NoFiLM_fold_pccs=nofilm_fold_pccs,
        FiLM_sd=float(np.std(film_fold_pccs, ddof=1)),
        NoFiLM_sd=float(np.std(nofilm_fold_pccs, ddof=1)),
        levene_stat=float(lev_stat), levene_p=float(lev_p),
        bartlett_stat=float(bart_stat), bartlett_p=float(bart_p),
    )


# 3b. FOLD-LEVEL PAIRED T-TEST: LUAD vs LUSC PCC, within a given model
def fold_subtype_ttest(run, model_name, panel, genes):
    """
    Computes panel PCC per fold, separately on the LUAD-only and LUSC-only
    subsets of that fold's predictions, then runs a paired t-test across the
    5 folds (paired because each fold is the same trained model evaluated on
    both subtypes). This tests whether LUAD/LUSC PCC differs consistently
    across folds, using fold-to-fold variability rather than a single-point
    Fisher r-to-z estimate.
    """
    subtypes_arr = np.array(run["subtypes"])
    mask_luad = subtypes_arr == "LUAD"
    mask_lusc = subtypes_arr == "LUSC"

    luad_fold_pccs, lusc_fold_pccs = [], []
    for _, preds in sorted(run["fold_preds"].items()):
        luad_fold_pccs.append(
            panel_pcc(preds[mask_luad], run["labels"][mask_luad], run["gene_cols"], genes))
        lusc_fold_pccs.append(
            panel_pcc(preds[mask_lusc], run["labels"][mask_lusc], run["gene_cols"], genes))

    luad_fold_pccs = np.array(luad_fold_pccs)
    lusc_fold_pccs = np.array(lusc_fold_pccs)

    t_stat, p_val = stats.ttest_rel(luad_fold_pccs, lusc_fold_pccs)

    return dict(
        model=model_name, panel=panel,
        LUAD_fold_pccs=luad_fold_pccs.tolist(), LUSC_fold_pccs=lusc_fold_pccs.tolist(),
        LUAD_mean=float(luad_fold_pccs.mean()), LUSC_mean=float(lusc_fold_pccs.mean()),
        mean_diff=float(luad_fold_pccs.mean() - lusc_fold_pccs.mean()),
        t_stat=float(t_stat), p=float(p_val),
    )


# 3c. FOLD-LEVEL SUBTYPE GAP (LUSC - LUAD), summarized across cohorts for the forest plot
def fold_subtype_gap(run, model_name, cohort_name, panel, genes):
    """
    Per-fold LUSC-minus-LUAD panel PCC (same per-fold computation as
    fold_subtype_ttest), summarized as the mean gap across folds with a
    t-based 95% CI (df = n_folds - 1). `cohort_name` is just a label
    ("CPTAC" / "TCGA") carried through into the output row so results from
    both cohorts can be combined into one forest plot.
    """
    subtypes_arr = np.array(run["subtypes"])
    mask_luad = subtypes_arr == "LUAD"
    mask_lusc = subtypes_arr == "LUSC"

    fold_gaps = []
    for _, preds in sorted(run["fold_preds"].items()):
        r_luad = panel_pcc(preds[mask_luad], run["labels"][mask_luad], run["gene_cols"], genes)
        r_lusc = panel_pcc(preds[mask_lusc], run["labels"][mask_lusc], run["gene_cols"], genes)
        fold_gaps.append(r_lusc - r_luad)
    fold_gaps = np.array(fold_gaps)

    n = len(fold_gaps)
    mean_gap = float(fold_gaps.mean())
    if n > 1:
        sem = float(fold_gaps.std(ddof=1) / np.sqrt(n))
        tcrit = float(stats.t.ppf(0.975, df=n - 1))
        ci_lo, ci_hi = mean_gap - tcrit * sem, mean_gap + tcrit * sem
    else:
        sem, ci_lo, ci_hi = float("nan"), float("nan"), float("nan")

    return dict(
        model=model_name, cohort=cohort_name, panel=panel, n_folds=n,
        fold_gaps=fold_gaps.tolist(),
        mean_gap_LUSC_minus_LUAD=mean_gap,
        sem=sem, CI95_low=ci_lo, CI95_high=ci_hi,
    )


# 4. FISHER r-to-z: LUAD vs LUSC PCC, within a given model
def fisher_r_to_z_test(r1, n1, r2, n2):
    z1 = np.arctanh(r1)
    z2 = np.arctanh(r2)
    se = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    z = (z1 - z2) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return dict(z=float(z), p=float(p))


def subtype_fisher_tests(run, model_name):
    rows = []
    subtypes = np.array(run["subtypes"])
    for panel, genes in [("APM", run["apm_genes"]), ("TIS", run["tis_genes"])]:
        mask_luad = subtypes == "LUAD"
        mask_lusc = subtypes == "LUSC"
        r_luad = panel_pcc(run["ensemble_preds"][mask_luad], run["labels"][mask_luad],
                            run["gene_cols"], genes)
        r_lusc = panel_pcc(run["ensemble_preds"][mask_lusc], run["labels"][mask_lusc],
                            run["gene_cols"], genes)
        test = fisher_r_to_z_test(r_lusc, mask_lusc.sum(), r_luad, mask_luad.sum())
        rows.append(dict(model=model_name, panel=panel,
                          PCC_LUAD=r_luad, n_LUAD=int(mask_luad.sum()),
                          PCC_LUSC=r_lusc, n_LUSC=int(mask_lusc.sum()),
                          z=test["z"], p=test["p"]))
    return rows

# 4.b. Direct subtype bootstrap + Permutation test
def subtype_pcc_bootstrap_permutation(
    run,
    panel,
    genes,
    n_boot=10000,
    n_perm=10000,
    seed=0,
):
    rng_local = np.random.default_rng(seed)
    subtypes = np.asarray(run["subtypes"])
    labels = np.asarray(run["labels"])
    preds = np.asarray(run["ensemble_preds"])
    mask_luad = subtypes == "LUAD"
    mask_lusc = subtypes == "LUSC"

    # Extract the two independent patient groups
    pred_luad = preds[mask_luad]
    lab_luad = labels[mask_luad]
    pred_lusc = preds[mask_lusc]
    lab_lusc = labels[mask_lusc]
    n_luad = len(pred_luad)
    n_lusc = len(pred_lusc)

    # Observed PCCs
    r_luad = panel_pcc(
        pred_luad,
        lab_luad,
        run["gene_cols"],
        genes,
    )
    r_lusc = panel_pcc(
        pred_lusc,
        lab_lusc,
        run["gene_cols"],
        genes,
    )

    obs_diff = r_lusc - r_luad

    # Bootstrap CI
    boot_diffs = np.empty(n_boot)

    for i in range(n_boot):

        # Resample patients WITH replacement within each subtype.
        idx_luad = rng_local.integers(0, n_luad, size=n_luad)
        idx_lusc = rng_local.integers(0, n_lusc, size=n_lusc)

        r_luad_boot = panel_pcc(
            pred_luad[idx_luad],
            lab_luad[idx_luad],
            run["gene_cols"],
            genes,
        )
        r_lusc_boot = panel_pcc(
            pred_lusc[idx_lusc],
            lab_lusc[idx_lusc],
            run["gene_cols"],
            genes,
        )

        boot_diffs[i] = r_lusc_boot - r_luad_boot

    # Percentile bootstrap 95% CI
    ci_low, ci_high = np.percentile(
        boot_diffs,
        [2.5, 97.5],
    )
    # Permutation test
    all_preds = np.concatenate([pred_luad, pred_lusc], axis=0)
    all_labels = np.concatenate([lab_luad, lab_lusc], axis=0)
    n_total = n_luad + n_lusc
    perm_diffs = np.empty(n_perm)

    for i in range(n_perm):
        perm_idx = rng_local.permutation(n_total)
        perm_luad_idx = perm_idx[:n_luad]
        perm_lusc_idx = perm_idx[n_luad:]
        r_luad_perm = panel_pcc(
            all_preds[perm_luad_idx],
            all_labels[perm_luad_idx],
            run["gene_cols"],
            genes,
        )
        r_lusc_perm = panel_pcc(
            all_preds[perm_lusc_idx],
            all_labels[perm_lusc_idx],
            run["gene_cols"],
            genes,
        )

        perm_diffs[i] = r_lusc_perm - r_luad_perm

    p_perm = (
        np.sum(np.abs(perm_diffs) >= abs(obs_diff)) + 1
    ) / (n_perm + 1)

    return dict(
        panel=panel,
        PCC_LUAD=float(r_luad),
        n_LUAD=int(n_luad),
        PCC_LUSC=float(r_lusc),
        n_LUSC=int(n_lusc),

        # Main effect size
        diff_LUSC_minus_LUAD=float(obs_diff),

        # Bootstrap uncertainty
        bootstrap_CI95_low=float(ci_low),
        bootstrap_CI95_high=float(ci_high),

        # Permutation inference
        permutation_p=float(p_perm),

        # Useful for diagnostics
        bootstrap_mean=float(np.mean(boot_diffs)),
        bootstrap_sd=float(np.std(boot_diffs, ddof=1)),
    )

# 5. GENE-LEVEL PCC + rank concordance between models
def gene_level_pccs(run):
    gene_symbols = [c.replace("_fpkm_uq", "") for c in run["gene_cols"] if c.endswith("_fpkm_uq")]
    rows = []
    for g in gene_symbols:
        i = run["gene_cols"].index(f"{g}_fpkm_uq")
        r, p = stats.pearsonr(run["ensemble_preds"][:, i], run["labels"][:, i])
        panel = "APM" if g in run["apm_genes"] else ("TIS" if g in run["tis_genes"] else "other")
        rows.append(dict(gene=g, PCC=r, p=p, panel=panel))
    return pd.DataFrame(rows)


# PLOTS
def plot_forest_fold_pcc(film, nofilm, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    for ax, (panel, genes) in zip(axes, [("APM", film["apm_genes"]), ("TIS", film["tis_genes"])]):
        labels_y, vals, cis, colors = [], [], [], []
        for model_name, run in [("FiLM", film), ("No-FiLM", nofilm)]:
            fold_items = sorted(run["fold_preds"].items())
            for fidx, preds in fold_items:
                pcc = panel_pcc(preds, run["labels"], run["gene_cols"], genes)
                labels_y.append(f"{model_name} fold{fidx}")
                vals.append(pcc)
                colors.append(MODEL_STYLE[model_name]["color"])
            ens_pcc = panel_pcc(run["ensemble_preds"], run["labels"], run["gene_cols"], genes)
            labels_y.append(f"{model_name} ENSEMBLE")
            vals.append(ens_pcc)
            colors.append(MODEL_STYLE[model_name]["color"])

        y_pos = np.arange(len(labels_y))[::-1]
        ax.scatter(vals, y_pos, c=colors, zorder=3, s=[80 if "ENSEMBLE" in l else 40 for l in labels_y])
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels_y, fontsize=8)
        ax.axvline(0, color="grey", lw=0.5)
        ax.set_xlabel("PCC")
        ax.set_title(panel)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Per-fold and ensemble PCC — FiLM vs No-FiLM (CPTAC external validation)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_forest_subtype_gap(gap_df, out_path):
    """
    Forest plot of the mean fold-level LUSC-LUAD panel-PCC gap: one row per
    model x cohort x panel combination, with whiskers at the t-based 95% CI
    (df = n_folds - 1) and a reference line at zero (no subtype gap).
    Rows are grouped by cohort (TCGA / CPTAC), with a separator line between
    cohort blocks; a row is skipped if that combination has < 2 folds (no CI).
    """
    df = gap_df.dropna(subset=["CI95_low", "CI95_high"]).copy()
    if df.empty:
        return

    cohort_order = sorted(df["cohort"].unique())
    model_order = ["FiLM", "No-FiLM"]
    panel_order = ["APM", "TIS"]
    df["_cohort_rank"] = df["cohort"].apply(cohort_order.index)
    df["_model_rank"] = df["model"].apply(lambda m: model_order.index(m) if m in model_order else 99)
    df["_panel_rank"] = df["panel"].apply(lambda p: panel_order.index(p) if p in panel_order else 99)
    df = df.sort_values(["_cohort_rank", "_model_rank", "_panel_rank"], ascending=False).reset_index(drop=True)
    df["row_label"] = df["cohort"] + " \u00b7 " + df["model"] + " \u00b7 " + df["panel"]

    y_pos = np.arange(len(df))
    colors = [MODEL_STYLE.get(m, dict(color="#333333"))["color"] for m in df["model"]]
    xerr = np.vstack([
        (df["mean_gap_LUSC_minus_LUAD"] - df["CI95_low"]).values,
        (df["CI95_high"] - df["mean_gap_LUSC_minus_LUAD"]).values,
    ])

    fig_h = max(3.0, 0.45 * len(df) + 1.5)
    fig, ax = plt.subplots(figsize=(8, fig_h))
    ax.errorbar(df["mean_gap_LUSC_minus_LUAD"], y_pos, xerr=xerr, fmt="none",
                ecolor="grey", elinewidth=1.2, capsize=3, zorder=2)
    ax.scatter(df["mean_gap_LUSC_minus_LUAD"], y_pos, c=colors, s=70, zorder=3,
               edgecolor="k", linewidth=0.4)
    ax.axvline(0, color="grey", lw=1, ls="--", zorder=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["row_label"], fontsize=9)
    ax.set_xlabel("Fold-level PCC gap, LUSC \u2212 LUAD (mean \u00b1 95% CI across folds)")
    ax.set_title("Subtype gap by fold: APM/TIS panels, FiLM vs No-FiLM")
    ax.grid(axis="x", alpha=0.3)

    # separator line between cohort blocks
    for rank in range(1, len(cohort_order)):
        boundary_cohort = cohort_order[rank]
        first_idx = df.index[df["cohort"] == boundary_cohort].max()
        ax.axhline(first_idx + 0.5, color="black", lw=0.8, alpha=0.4)

    handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=MODEL_STYLE[m]["color"],
                           label=m, markeredgecolor="k", markeredgewidth=0.4)
               for m in model_order if m in df["model"].unique()]
    ax.legend(handles=handles, loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_subtype_panel_bars(film, nofilm, out_path):
    """
    Bar height = ensemble PCC (subtype x panel). Error bars = fold-consistency
    spread, i.e. the SD of the per-fold PCCs (each fold's model evaluated on
    the same subtype/panel subset), rather than a bootstrap CI. This reflects
    how much the metric varies across the 5 trained folds instead of sampling
    uncertainty from resampling patients.
    """
    panels = ["APM", "TIS"]
    subtypes = ["LUAD", "LUSC"]
    models = [("FiLM", film), ("No-FiLM", nofilm)]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bar_width = 0.18
    group_gap = 1.0
    x_base = np.arange(len(panels) * len(subtypes)) * group_gap
    labels = [f"{s}\n{p}" for p in panels for s in subtypes]

    for m_i, (model_name, run) in enumerate(models):
        genes_map = {"APM": run["apm_genes"], "TIS": run["tis_genes"]}
        subtypes_arr = np.array(run["subtypes"])
        fold_items = sorted(run["fold_preds"].items())
        vals, err = [], []
        for panel in panels:
            for subtype in subtypes:
                mask = subtypes_arr == subtype
                genes = genes_map[panel]
                labs = run["labels"][mask]

                obs = panel_pcc(run["ensemble_preds"][mask], labs, run["gene_cols"], genes)

                fold_pccs = np.array([
                    panel_pcc(preds[mask], labs, run["gene_cols"], genes)
                    for _, preds in fold_items
                ])
                fold_sd = float(np.std(fold_pccs, ddof=1))

                vals.append(obs)
                err.append(fold_sd)

        offset = (m_i - 0.5) * bar_width
        ax.bar(x_base + offset, vals, width=bar_width, label=model_name,
               color=MODEL_STYLE[model_name]["color"], alpha=0.85,
               yerr=err, capsize=3)

    ax.set_xticks(x_base)
    ax.set_xticklabels(labels)
    ax.set_ylabel("PCC (\u00b1 SD across 5 folds)")
    ax.set_title("CPTAC PCC by subtype and panel: FiLM vs No-FiLM")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_gene_scatter(gene_df_film, gene_df_nofilm, out_path):
    merged = gene_df_film.merge(gene_df_nofilm, on="gene", suffixes=("_FiLM", "_NoFiLM"))
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for panel, color in PANEL_COLORS.items():
        sub = merged[merged["panel_FiLM"] == panel]
        ax.scatter(sub["PCC_NoFiLM"], sub["PCC_FiLM"], c=color, label=panel, alpha=0.8, edgecolor="k", linewidth=0.3)
    lims = [merged[["PCC_FiLM", "PCC_NoFiLM"]].min().min() - 0.02,
            merged[["PCC_FiLM", "PCC_NoFiLM"]].max().max() + 0.02]
    ax.plot(lims, lims, "--", color="grey", lw=1, label="y = x")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("No-FiLM per-gene PCC")
    ax.set_ylabel("FiLM per-gene PCC")
    ax.set_title("Per-gene PCC: FiLM vs No-FiLM (CPTAC)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return merged


def plot_generalization_gap(out_path):
    panels = ["APM", "TIS"]
    models = ["FiLM", "No-FiLM"]
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(panels))
    width = 0.2
    offsets = {"TCGA-FiLM": -1.5, "CPTAC-FiLM": -0.5, "TCGA-No-FiLM": 0.5, "CPTAC-No-FiLM": 1.5}
    colors = {"TCGA-FiLM": "#C44E52", "CPTAC-FiLM": "#F1948A",
              "TCGA-No-FiLM": "#55A868", "CPTAC-No-FiLM": "#A9DFBF"}

    for label, off in offsets.items():
        cohort, model = label.split("-", 1)
        vals = []
        for panel in panels:
            if cohort == "TCGA":
                vals.append(TCGA_ENSEMBLE_PCC[model][panel])
            else:
                vals.append(None)  # filled by caller if CPTAC ensemble PCC provided
        ax.bar(x + off * width, vals, width=width, label=label, color=colors[label])

    ax.set_xticks(x)
    ax.set_xticklabels(panels)
    ax.set_ylabel("Ensemble PCC")
    ax.set_title("Internal (TCGA) vs External (CPTAC) generalization")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


OUT_DIR.mkdir(exist_ok=True, parents=True)

film = load_run(FILM_DIR)
nofilm = load_run(NOFILM_DIR)

# 1. Paired bootstrap ensemble PCC diff
diff_rows = []
for panel, genes in [("APM", film["apm_genes"]), ("TIS", film["tis_genes"])]:
    diff_rows.append(paired_bootstrap_pcc_diff(film, nofilm, panel, genes))
diff_df = pd.DataFrame(diff_rows)
diff_df.to_csv(OUT_DIR / "paired_ensemble_diff.csv", index=False)
print("Paired bootstrap: FiLM - No-FiLM ensemble PCC")
print(diff_df.to_string(index=False))

# 2. DeLong paired AUC test
delong_rows = []
for panel, genes in [("APM", film["apm_genes"]), ("TIS", film["tis_genes"])]:
    y_true = panel_binary_labels(film["labels"], film["gene_cols"], genes)  # same labels both runs
    pred_film = panel_score_pred(film["ensemble_preds"], film["gene_cols"], genes)
    pred_nofilm = panel_score_pred(nofilm["ensemble_preds"], nofilm["gene_cols"], genes)
    res = delong_paired_test(pred_film, pred_nofilm, y_true)
    res["panel"] = panel
    delong_rows.append(res)
delong_df = pd.DataFrame(delong_rows)
delong_df.to_csv(OUT_DIR / "delong_auc_diff.csv", index=False)
print("DeLong paired AUC test (FiLM vs No-FiLM")
print(delong_df.to_string(index=False))

# 3. Fold variance test
var_rows = []
for panel, genes in [("APM", film["apm_genes"]), ("TIS", film["tis_genes"])]:
    var_rows.append(fold_variance_test(film, nofilm, panel, genes))
var_df = pd.DataFrame(var_rows)
var_df.to_csv(OUT_DIR / "fold_variance_test.csv", index=False)
print("Fold-level variance test (Levene/Bartlett)")
print(var_df[["panel", "FiLM_sd", "NoFiLM_sd", "levene_p", "bartlett_p"]].to_string(index=False))

# 3b. Fold-level paired t-test: LUAD vs LUSC PCC, per panel, per model
ttest_rows = []
for model_name, run in [("FiLM", film), ("No-FiLM", nofilm)]:
    for panel, genes in [("APM", run["apm_genes"]), ("TIS", run["tis_genes"])]:
        ttest_rows.append(fold_subtype_ttest(run, model_name, panel, genes))
ttest_df = pd.DataFrame(ttest_rows)
ttest_df.to_csv(OUT_DIR / "subtype_fold_ttest.csv", index=False)
print("\nFold-level paired t-test: LUAD vs LUSC PCC (5 folds per model/panel)")
print(ttest_df[["model", "panel", "LUAD_mean", "LUSC_mean", "mean_diff", "t_stat", "p"]].to_string(index=False))

# 3c. Fold-level subtype (LUSC-LUAD) gap forest plot — CPTAC, plus TCGA if configured
gap_rows = []
for model_name, run in [("FiLM", film), ("No-FiLM", nofilm)]:
    for panel, genes in [("APM", run["apm_genes"]), ("TIS", run["tis_genes"])]:
        gap_rows.append(fold_subtype_gap(run, model_name, "CPTAC", panel, genes))

if TCGA_FILM_DIR is not None and TCGA_NOFILM_DIR is not None:
    tcga_film = load_run(TCGA_FILM_DIR, cohort="tcga")
    tcga_nofilm = load_run(TCGA_NOFILM_DIR, cohort="tcga")
    for model_name, run in [("FiLM", tcga_film), ("No-FiLM", tcga_nofilm)]:
        for panel, genes in [("APM", run["apm_genes"]), ("TIS", run["tis_genes"])]:
            gap_rows.append(fold_subtype_gap(run, model_name, "TCGA", panel, genes))
else:
    print("\n[fold_subtype_gap] TCGA_FILM_DIR / TCGA_NOFILM_DIR not set — "
          "forest plot will show CPTAC only. Point them at directories of "
          "fold-level TCGA held-out predictions to add TCGA rows.")

gap_df = pd.DataFrame(gap_rows)
gap_df.drop(columns=["fold_gaps"]).to_csv(OUT_DIR / "fold_subtype_gap.csv", index=False)
plot_forest_subtype_gap(gap_df, OUT_DIR / "fig_forest_subtype_gap.png")
print("\nFold-level subtype gap, LUSC-LUAD (mean +/- 95% CI across folds)")
print(gap_df.drop(columns=["fold_gaps"]).to_string(index=False))

# 4. Subtype Fisher r-to-z tests, per model
fisher_rows = subtype_fisher_tests(film, "FiLM") + subtype_fisher_tests(nofilm, "No-FiLM")
fisher_df = pd.DataFrame(fisher_rows)
fisher_df.to_csv(OUT_DIR / "subtype_fisher_z.csv", index=False)
print("LUAD vs LUSC PCC (Fisher r-to-z), per model")
print(fisher_df.to_string(index=False))

# 4b. Direct subtype bootstrap + permutation test
subtype_resampling_rows = []
for panel, genes in [
    ("APM", film["apm_genes"]),
    ("TIS", film["tis_genes"]),
]:
    res = subtype_pcc_bootstrap_permutation(
        film,
        panel,
        genes,
        n_boot=10000,
        n_perm=10000,
        seed=0,
    )
    subtype_resampling_rows.append(res)

subtype_resampling_df = pd.DataFrame(subtype_resampling_rows)
subtype_resampling_df.to_csv(
    OUT_DIR / "subtype_pcc_bootstrap_permutation.csv",
    index=False,
)
print("\nDirect subtype PCC bootstrap + permutation test")
print(subtype_resampling_df.to_string(index=False))

# 5. Gene-level comparison + rank concordance
gene_df_film = gene_level_pccs(film)
gene_df_nofilm = gene_level_pccs(nofilm)
merged_genes = plot_gene_scatter(gene_df_film, gene_df_nofilm, OUT_DIR / "fig_gene_scatter.png")
merged_genes.to_csv(OUT_DIR / "gene_level_comparison.csv", index=False)
rho, rho_p = stats.spearmanr(merged_genes["PCC_FiLM"], merged_genes["PCC_NoFiLM"])
pd.DataFrame([{"spearman_rho": rho, "p": rho_p, "n_genes": len(merged_genes)}]).to_csv(
    OUT_DIR / "gene_rank_concordance.csv", index=False)
print(f"\nGene-rank concordance (FiLM vs No-FiLM)\nSpearman rho={rho:.4f}, p={rho_p:.2e}")

# Figures
plot_forest_fold_pcc(film, nofilm, OUT_DIR / "fig_forest_fold_pcc.png")
plot_subtype_panel_bars(film, nofilm, OUT_DIR / "fig_subtype_panel_bars.png")

print(f"All stats + figures written to {OUT_DIR.resolve()}")
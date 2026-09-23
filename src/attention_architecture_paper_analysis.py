#!/usr/bin/env python3
"""
Comprehensive TCGA-held-out vs CPTAC external validation analysis
for identical histology -> immune-gene models differing only in the
MIL/attention aggregation mechanism.

Expected files (recursively searched under --input-dir):
    abmil_fold0_test_predictions.npz
    abmil_fold1_test_predictions.npz
    ...
    abmil_ensemble_test_predictions.npz

    abmil_fold0_cptac_predictions.npz
    abmil_fold1_cptac_predictions.npz
    ...
    abmil_ensemble_cptac_predictions.npz

Each NPZ should contain:
    preds         shape (n_samples, n_genes)
    labels        shape (n_samples, n_genes)
    submitter_id  shape (n_samples,)
    gene_cols     shape (n_genes,)

The script intentionally does NOT require the training checkpoints. It uses
only saved predictions/labels, which is appropriate for post-hoc evaluation.

Important:
- Panel definitions should match the original training notebook exactly.
- Panel AUC uses the same 75th-percentile label threshold as the supplied
  external-validation notebook.
- Statistical analyses are exploratory unless prespecified; genes within
  immune panels are correlated, so effect sizes and CIs should be emphasized
  over raw p-values.
- This script does not analyze tile-level attention maps. Those require
  attention/importance weights and tile coordinates/IDs, which are not in
  the prediction NPZs.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests


# ============================================================================
# USER CONFIGURATION
# ============================================================================

# Paste the exact gene lists used in your original notebook here if you want
# panel-level APM/TIS analysis. The names should match the base gene symbol
# appearing before "_fpkm_uq" in gene_cols.
APM_GENES: List[str] = [
    "TAP2", "TAP1", "PSMB9", "B2M", "HLA-B", "PSMB6",
    "HLA-A", "PSMB8", "PSMB7", "ERAP1", "PDIA3", "PSMB10",
    "HLA-C", "CANX", "ERAP2", "TAPBP", "PSMB5", "CALR",
]
TIS_GENES: List[str] = [
    "PDCD1LG2", "LAG3", "CCL5", "STAT1", "CD274", "CXCR6",
    "CD8A", "TIGIT", "CXCL9", "HLA-DQA1", "HLA-DRB1", "CD276",
    "CMKLR1", "NKG7", "IDO1", "HLA-E", "CD27",
]

# Number of bootstrap/permutation iterations. 2000 is usually fine for a
# final run; 5000 gives a little more stable tails for CIs/p-values.
N_BOOT = 5000
N_PERM = 5000
# Additional patient-level bootstrap for architecture x cohort stability interaction.
# This is deliberately separate from the gene-level exploratory bootstrap above.
N_STAB_BOOT = 2000
RANDOM_SEED = 99

# Minimum number of finite paired observations required for PCC.
MIN_N = 5


# ============================================================================
# DATA STRUCTURES / IO
# ============================================================================

@dataclass
class PredictionBundle:
    preds: np.ndarray
    labels: np.ndarray
    sids: np.ndarray
    gene_cols: List[str]
    path: str


FILENAME_RE = re.compile(
    r"^(?P<arch>.+)_(?P<kind>fold\d+|ensemble)_(?P<cohort>test|cptac)_predictions\.npz$"
)


def load_npz(path: Path) -> PredictionBundle:
    z = np.load(path, allow_pickle=True)
    required = {"preds", "labels", "submitter_id", "gene_cols"}
    missing = required.difference(z.files)
    if missing:
        raise ValueError(f"{path} missing keys: {sorted(missing)}")

    preds = np.asarray(z["preds"], dtype=float)
    labels = np.asarray(z["labels"], dtype=float)
    sids = np.asarray(z["submitter_id"]).astype(str)
    gene_cols = [str(x) for x in np.asarray(z["gene_cols"]).tolist()]

    if preds.ndim != 2 or labels.ndim != 2:
        raise ValueError(f"{path}: preds/labels must be 2D")
    if preds.shape != labels.shape:
        raise ValueError(f"{path}: preds shape {preds.shape} != labels shape {labels.shape}")
    if preds.shape[0] != len(sids):
        raise ValueError(f"{path}: n samples {preds.shape[0]} != len(sids) {len(sids)}")
    if preds.shape[1] != len(gene_cols):
        raise ValueError(f"{path}: n genes {preds.shape[1]} != len(gene_cols) {len(gene_cols)}")
    if len(set(sids)) != len(sids):
        raise ValueError(f"{path}: duplicate submitter_id values")
    if len(set(gene_cols)) != len(gene_cols):
        raise ValueError(f"{path}: duplicate gene_cols values")

    return PredictionBundle(preds, labels, sids, gene_cols, str(path))


def discover_files(root: Path) -> Dict[str, Dict[str, Dict[str, Path]]]:
    found: Dict[str, Dict[str, Dict[str, Path]]] = {}
    for p in root.rglob("*.npz"):
        m = FILENAME_RE.match(p.name)
        if not m:
            continue
        arch = m.group("arch")
        kind = m.group("kind")
        cohort = m.group("cohort")
        found.setdefault(arch, {}).setdefault(cohort, {})[kind] = p
    if not found:
        raise FileNotFoundError(
            f"No matching *_fold*_test/cptac_predictions.npz files found under {root}"
        )
    return found


def make_gene_index(gene_cols: Sequence[str]) -> Dict[str, int]:
    return {g: i for i, g in enumerate(gene_cols)}


def base_gene_name(gene_col: str) -> str:
    return gene_col[:-len("_fpkm_uq")] if gene_col.endswith("_fpkm_uq") else gene_col


def resolve_gene_columns(bundle: PredictionBundle) -> Dict[str, int]:
    return {base_gene_name(g): i for i, g in enumerate(bundle.gene_cols)}


def align_pair(
    a: PredictionBundle,
    b: PredictionBundle,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return row indices aligning b to a by submitter_id."""
    b_pos = {sid: i for i, sid in enumerate(b.sids)}
    common = [sid for sid in a.sids if sid in b_pos]
    if len(common) < MIN_N:
        raise ValueError(f"Only {len(common)} common samples between {a.path} and {b.path}")
    ia = np.asarray([np.where(a.sids == sid)[0][0] for sid in common], dtype=int)
    ib = np.asarray([b_pos[sid] for sid in common], dtype=int)
    return ia, ib, common


def ensure_common_genes(
    a: PredictionBundle,
    b: PredictionBundle,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    ai = resolve_gene_columns(a)
    bi = resolve_gene_columns(b)
    common = [g for g in ai if g in bi]
    if not common:
        raise ValueError(f"No common genes between {a.path} and {b.path}")
    ia = np.asarray([ai[g] for g in common], dtype=int)
    ib = np.asarray([bi[g] for g in common], dtype=int)
    return common, ia, ib


# ============================================================================
# METRICS
# ============================================================================


def finite_pair(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x, y = finite_pair(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    if len(x) < MIN_N or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(pearsonr(x, y).statistic)


def panel_score(arr: np.ndarray, gene_cols: Sequence[str], panel_genes: Sequence[str]) -> np.ndarray:
    idx = make_gene_index(gene_cols)
    base_idx = resolve_gene_columns(
        PredictionBundle(
            preds=np.empty((0, len(gene_cols))),
            labels=np.empty((0, len(gene_cols))),
            sids=np.empty((0,), dtype=str),
            gene_cols=list(gene_cols),
            path="<in-memory>",
        )
    )
    cols = [
        base_idx[g] if g in base_idx else idx[g]
        for g in panel_genes
        if g in base_idx or g in idx
    ]
    if not cols:
        raise ValueError(
            "No panel genes found in gene_cols. Expected either base symbols "
            "or names with the _fpkm_uq suffix."
        )
    return np.nanmean(arr[:, cols], axis=1)


def panel_pcc(
    preds: np.ndarray,
    labels: np.ndarray,
    gene_cols: Sequence[str],
    panel_genes: Sequence[str],
) -> float:
    x = panel_score(preds, gene_cols, panel_genes)
    y = panel_score(labels, gene_cols, panel_genes)
    return safe_pearson(x, y)


def panel_auc(
    preds: np.ndarray,
    labels: np.ndarray,
    gene_cols: Sequence[str],
    panel_genes: Sequence[str],
) -> float:
    pred_score = panel_score(preds, gene_cols, panel_genes)
    label_score = panel_score(labels, gene_cols, panel_genes)
    m = np.isfinite(pred_score) & np.isfinite(label_score)
    pred_score = pred_score[m]
    label_score = label_score[m]
    if len(label_score) < MIN_N:
        return np.nan
    thresh = np.percentile(label_score, 75)
    y_true = (label_score >= thresh).astype(int)
    if y_true.sum() in (0, len(y_true)):
        return np.nan
    return float(roc_auc_score(y_true, pred_score))


def gene_level_stats(
    bundle: PredictionBundle,
    cohort: str,
    architecture: str,
) -> pd.DataFrame:
    rows = []
    panels = {g: "APM" for g in APM_GENES}
    panels.update({g: "TIS" for g in TIS_GENES})
    for j, gc in enumerate(bundle.gene_cols):
        g = base_gene_name(gc)
        x, y = finite_pair(bundle.preds[:, j], bundle.labels[:, j])
        if len(x) < MIN_N or np.std(x) == 0 or np.std(y) == 0:
            r, p = np.nan, np.nan
        else:
            pr = pearsonr(x, y)
            r, p = float(pr.statistic), float(pr.pvalue)
        rows.append({
            "architecture": architecture,
            "cohort": cohort,
            "gene": g,
            "panel": panels.get(g, "other"),
            "n": len(x),
            "PCC": r,
            "p": p,
        })
    out = pd.DataFrame(rows)
    # BH within architecture x cohort x panel where panel is APM/TIS; keep other separate.
    out["q"] = np.nan
    for panel in out["panel"].dropna().unique():
        m = out["panel"] == panel
        valid = m & out["p"].notna()
        if valid.sum():
            out.loc[valid, "q"] = multipletests(out.loc[valid, "p"].values, method="fdr_bh")[1]
    return out


# ============================================================================
# BOOTSTRAP / PERMUTATION
# ============================================================================


def bootstrap_panel_ci(
    preds: np.ndarray,
    labels: np.ndarray,
    gene_cols: Sequence[str],
    panel_genes: Sequence[str],
    n_boot: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    obs = panel_pcc(preds, labels, gene_cols, panel_genes)
    n = preds.shape[0]
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[i] = panel_pcc(preds[idx], labels[idx], gene_cols, panel_genes)
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return obs, float(lo), float(hi)


def bootstrap_diff_of_gaps(
    tcga_a: PredictionBundle,
    tcga_b: PredictionBundle,
    cptac_a: PredictionBundle,
    cptac_b: PredictionBundle,
    panel_genes: Sequence[str],
    n_boot: int,
    seed: int,
) -> Dict[str, float]:
    """
    D = (CPTAC_A - CPTAC_B) - (TCGA_A - TCGA_B), where each term is panel PCC.

    TCGA and CPTAC are independent cohorts, so patient resampling is performed
    independently within each cohort, while A and B are paired within cohort.
    """
    # Align architectures within each cohort.
    t_ia, t_ib, _ = align_pair(tcga_a, tcga_b)
    c_ia, c_ib, _ = align_pair(cptac_a, cptac_b)

    # Ensure labels are the same for paired architecture files.
    if not np.allclose(
        tcga_a.labels[t_ia], tcga_b.labels[t_ib], equal_nan=True
    ):
        raise ValueError("TCGA labels differ between paired architecture files")
    if not np.allclose(
        cptac_a.labels[c_ia], cptac_b.labels[c_ib], equal_nan=True
    ):
        raise ValueError("CPTAC labels differ between paired architecture files")

    tA = tcga_a.preds[t_ia]
    tB = tcga_b.preds[t_ib]
    tY = tcga_a.labels[t_ia]

    cA = cptac_a.preds[c_ia]
    cB = cptac_b.preds[c_ib]
    cY = cptac_a.labels[c_ia]

    obs_t = panel_pcc(tA, tY, tcga_a.gene_cols, panel_genes) - panel_pcc(
        tB, tY, tcga_b.gene_cols, panel_genes
    )
    obs_c = panel_pcc(cA, cY, cptac_a.gene_cols, panel_genes) - panel_pcc(
        cB, cY, cptac_b.gene_cols, panel_genes
    )
    observed = obs_c - obs_t

    rng = np.random.default_rng(seed)
    bt = np.empty(n_boot)
    for i in range(n_boot):
        it = rng.integers(0, len(tY), size=len(tY))
        ic = rng.integers(0, len(cY), size=len(cY))
        gt = panel_pcc(tA[it], tY[it], tcga_a.gene_cols, panel_genes) - panel_pcc(
            tB[it], tY[it], tcga_b.gene_cols, panel_genes
        )
        gc = panel_pcc(cA[ic], cY[ic], cptac_a.gene_cols, panel_genes) - panel_pcc(
            cB[ic], cY[ic], cptac_b.gene_cols, panel_genes
        )
        bt[i] = gc - gt

    lo, hi = np.percentile(bt, [2.5, 97.5])
    # Two-sided bootstrap sign probability around zero.
    p = min(1.0, 2.0 * min(np.mean(bt <= 0), np.mean(bt >= 0)))
    return {
        "arch_A": tcga_a.path.split("/")[-1].split("_fold")[0],
        "arch_B": tcga_b.path.split("/")[-1].split("_fold")[0],
        "TCGA_gap_A_minus_B": float(obs_t),
        "CPTAC_gap_A_minus_B": float(obs_c),
        "delta_of_gaps_CPTAC_minus_TCGA": float(observed),
        "CI95_low": float(lo),
        "CI95_high": float(hi),
        "p_boot": float(p),
        "robust": bool(lo > 0 or hi < 0),
    }


def gene_stability(
    fold_bundles: Sequence[PredictionBundle],
) -> pd.DataFrame:
    rows = []
    # Use first fold as row-order reference, align all later folds by SID.
    ref = fold_bundles[0]
    gene_maps = [resolve_gene_columns(b) for b in fold_bundles]
    common_genes = [g for g in gene_maps[0] if all(g in gm for gm in gene_maps)]

    aligned_preds = []
    for b in fold_bundles:
        ia, _, _ = align_pair(ref, b)
        # ia indexes ref; need row indices in b too
        _, ib, _ = align_pair(ref, b)
        pred = b.preds[ib]
        aligned_preds.append(pred)

    fold_pairs = list(itertools.combinations(range(len(fold_bundles)), 2))
    for g in common_genes:
        per_pair = []
        for i, j in fold_pairs:
            ai = gene_maps[i][g]
            aj = gene_maps[j][g]
            r = safe_pearson(aligned_preds[i][:, ai], aligned_preds[j][:, aj])
            per_pair.append(r)
        per_pair = np.asarray(per_pair, dtype=float)
        panel = "APM" if g in APM_GENES else ("TIS" if g in TIS_GENES else "other")
        rows.append({
            "gene": g,
            "panel": panel,
            "n_fold_pairs": len(per_pair),
            "mean_fold_stability": float(np.nanmean(per_pair)),
            "median_fold_stability": float(np.nanmedian(per_pair)),
            "std_fold_stability": float(np.nanstd(per_pair, ddof=1)),
            "min_fold_stability": float(np.nanmin(per_pair)),
            "max_fold_stability": float(np.nanmax(per_pair)),
        })
    return pd.DataFrame(rows)


def paired_gene_stability_test(
    stability_df: pd.DataFrame,
    arch_a: str,
    arch_b: str,
    panel: Optional[str],
    n_boot: int,
    seed: int,
) -> Dict[str, float]:
    d = stability_df.copy()
    if panel is not None:
        d = d[d["panel"] == panel]
    wa = d[d.architecture == arch_a].set_index("gene")["mean_fold_stability"]
    wb = d[d.architecture == arch_b].set_index("gene")["mean_fold_stability"]
    common = wa.index.intersection(wb.index)
    x = wa.loc[common].to_numpy(float)
    y = wb.loc[common].to_numpy(float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    diff = x - y
    obs = float(np.mean(diff))
    n = len(diff)
    rng = np.random.default_rng(seed)

    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = np.mean(diff[idx])
    lo, hi = np.percentile(boot, [2.5, 97.5])

    perm = np.empty(n_boot)
    for i in range(n_boot):
        signs = rng.choice([-1.0, 1.0], size=n)
        perm[i] = np.mean(diff * signs)
    p = (np.sum(np.abs(perm) >= abs(obs)) + 1) / (n_boot + 1)
    return {
        "arch_A": arch_a,
        "arch_B": arch_b,
        "panel": panel or "ALL",
        "n_genes": n,
        "mean_stability_A": float(np.mean(x)),
        "mean_stability_B": float(np.mean(y)),
        "delta_A_minus_B": obs,
        "CI95_low": float(lo),
        "CI95_high": float(hi),
        "perm_p": float(p),
        "robust": bool(lo > 0 or hi < 0),
    }


def spearman_bootstrap_permutation(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int,
    n_perm: int,
    seed: int,
) -> Dict[str, float]:
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    rho = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)

    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(x), size=len(x))
        boot[i] = spearmanr(x[idx], y[idx]).statistic
    lo, hi = np.percentile(boot, [2.5, 97.5])

    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = spearmanr(x, rng.permutation(y)).statistic
    p = (np.sum(np.abs(null) >= abs(rho)) + 1) / (n_perm + 1)
    return {
        "n": len(x),
        "spearman_rho": rho,
        "CI95_low": float(lo),
        "CI95_high": float(hi),
        "permutation_p": float(p),
    }


# ============================================================================
# CROSS-ARCHITECTURE AGREEMENT / ENSEMBLES
# ============================================================================


def cross_architecture_agreement(
    bundles: Dict[str, PredictionBundle]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    gene_rows = []
    archs = sorted(bundles)
    for a, b in itertools.combinations(archs, 2):
        ba, bb = bundles[a], bundles[b]
        ia, ib, common_sids = align_pair(ba, bb)
        genes, ga, gb = ensure_common_genes(ba, bb)
        rs = []
        for g, i, j in zip(genes, ga, gb):
            r = safe_pearson(ba.preds[ia, i], bb.preds[ib, j])
            rs.append(r)
            gene_rows.append({
                "arch_A": a,
                "arch_B": b,
                "gene": g,
                "panel": "APM" if g in APM_GENES else ("TIS" if g in TIS_GENES else "other"),
                "prediction_r": r,
            })
        rs = np.asarray(rs, float)
        summary_rows.append({
            "arch_A": a,
            "arch_B": b,
            "n_common_patients": len(common_sids),
            "mean_prediction_r": float(np.nanmean(rs)),
            "median_prediction_r": float(np.nanmedian(rs)),
            "min_prediction_r": float(np.nanmin(rs)),
            "max_prediction_r": float(np.nanmax(rs)),
        })
    return pd.DataFrame(summary_rows), pd.DataFrame(gene_rows)


def correlation_of_gene_pcc_vectors(
    gene_pcc_df: pd.DataFrame,
    cohort: str,
) -> pd.DataFrame:
    """
    Correlate the per-gene PCC profiles of architecture ensembles.

    Important:
    gene_pcc_df contains fold and ensemble rows. For this analysis we
    explicitly use ensemble rows only so that each architecture contributes
    exactly one PCC per gene.
    """

    d = gene_pcc_df[
        (gene_pcc_df["cohort"] == cohort)
        & (gene_pcc_df["kind"] == "ensemble")
    ].copy()

    out = []

    archs = sorted(d["architecture"].unique())

    for a, b in itertools.combinations(archs, 2):

        da = (
            d[d["architecture"] == a]
            .drop_duplicates(subset=["gene"], keep="first")
            .set_index("gene")["PCC"]
        )

        db = (
            d[d["architecture"] == b]
            .drop_duplicates(subset=["gene"], keep="first")
            .set_index("gene")["PCC"]
        )

        common = da.index.intersection(db.index)

        x = da.loc[common].to_numpy(dtype=float)
        y = db.loc[common].to_numpy(dtype=float)

        m = np.isfinite(x) & np.isfinite(y)

        x = x[m]
        y = y[m]

        if len(x) >= MIN_N:
            stat = pearsonr(x, y)

            out.append({
                "cohort": cohort,
                "arch_A": a,
                "arch_B": b,
                "n_genes": len(x),
                "r_per_gene_PCC": float(stat.statistic),
                "p": float(stat.pvalue),
            })

    return pd.DataFrame(out)


def equal_weight_arch_ensembles(
    bundles: Dict[str, PredictionBundle]
) -> pd.DataFrame:
    """Evaluate equal-weight ensembles using the intersection of patients/genes."""
    archs = sorted(bundles)
    rows = []
    for r in range(2, len(archs) + 1):
        for combo in itertools.combinations(archs, r):
            ref = bundles[combo[0]]
            common_sids = set(ref.sids)
            common_genes = set(base_gene_name(g) for g in ref.gene_cols)
            for arch in combo[1:]:
                common_sids &= set(bundles[arch].sids)
                common_genes &= set(base_gene_name(g) for g in bundles[arch].gene_cols)
            common_sids = [sid for sid in ref.sids if sid in common_sids]
            common_genes = [g for g in [base_gene_name(gc) for gc in ref.gene_cols] if g in common_genes]
            if len(common_sids) < MIN_N or not common_genes:
                continue

            pred_arrays = []
            label_ref = None
            ref_gene_map = resolve_gene_columns(ref)
            ref_sid_map = {sid: i for i, sid in enumerate(ref.sids)}
            ref_rows = np.asarray([ref_sid_map[sid] for sid in common_sids], dtype=int)
            label_ref = ref.labels[ref_rows]

            for arch in combo:
                b = bundles[arch]
                sid_map = {sid: i for i, sid in enumerate(b.sids)}
                gm = resolve_gene_columns(b)
                rows_i = np.asarray([sid_map[sid] for sid in common_sids], dtype=int)
                cols_i = np.asarray([gm[g] for g in common_genes], dtype=int)
                pred_arrays.append(b.preds[np.ix_(rows_i, cols_i)])

            preds = np.mean(pred_arrays, axis=0)
            labels = label_ref[:, [ref_gene_map[g] for g in common_genes]]

            gene_cols_common = common_genes
            for panel, genes in [("APM", APM_GENES), ("TIS", TIS_GENES)]:
                if not genes:
                    continue
                genes_present = [g for g in genes if g in common_genes]
                if not genes_present:
                    continue
                rows.append({
                    "ensemble": "+".join(combo),
                    "n_architectures": len(combo),
                    "n_common_patients": len(common_sids),
                    "panel": panel,
                    "PCC": panel_pcc(preds, labels, gene_cols_common, genes_present),
                    "AUC": panel_auc(preds, labels, gene_cols_common, genes_present),
                })
    return pd.DataFrame(rows)


# ============================================================================
# PLOTTING
# ============================================================================


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=250, bbox_inches="tight")
    plt.close()


def plot_gene_domain_retention(per_gene_domain: pd.DataFrame, out_dir: Path) -> None:
    for arch in sorted(per_gene_domain.architecture.unique()):
        d = per_gene_domain[per_gene_domain.architecture == arch]
        fig, ax = plt.subplots(figsize=(6.5, 6.0))
        for panel, marker in [("APM", "o"), ("TIS", "^")]:
            s = d[d.panel == panel]
            if len(s):
                ax.scatter(s.TCGA_PCC, s.CPTAC_PCC, marker=marker, alpha=0.8, label=panel)
        vals = d[["TCGA_PCC", "CPTAC_PCC"]].to_numpy().ravel()
        vals = vals[np.isfinite(vals)]
        lo = min(-0.05, float(vals.min()) - 0.05)
        hi = max(0.75, float(vals.max()) + 0.05)
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.5, label="y=x")
        ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="TCGA held-out test PCC", ylabel="CPTAC external PCC")
        ax.set_title(f"Per-gene domain retention: {arch}")
        ax.legend()
        ax.grid(alpha=0.2)
        savefig(out_dir / f"{arch}_per_gene_domain_retention.png")


def plot_stability_vs_drop(
    trans_df: pd.DataFrame,
    out_dir: Path,
    arch: str,
    panel: str = "ALL",
) -> None:
    """
    Plot internal fold-to-fold stability against TCGA->CPTAC
    domain change in per-gene PCC.

    trans_df should already contain rows for one architecture.
    Expected columns:
        gene
        panel
        mean_fold_stability
        TCGA_PCC
        CPTAC_PCC
        domain_drop
    """

    d = trans_df.copy()

    if panel != "ALL":
        d = d[d["panel"] == panel].copy()

    d = d[
        [
            "gene",
            "panel",
            "mean_fold_stability",
            "TCGA_PCC",
            "CPTAC_PCC",
            "domain_drop",
        ]
    ].dropna()

    if len(d) < 3:
        return

    fig, ax = plt.subplots(figsize=(7, 6))

    # Plot APM and TIS separately when ALL is requested.
    if panel == "ALL":
        for p_name, marker in [
            ("APM", "o"),
            ("TIS", "^"),
        ]:
            s = d[d["panel"] == p_name]

            if len(s):
                ax.scatter(
                    s["mean_fold_stability"],
                    s["domain_drop"],
                    marker=marker,
                    alpha=0.8,
                    label=p_name,
                )
    else:
        marker = "o" if panel == "APM" else "^"

        ax.scatter(
            d["mean_fold_stability"],
            d["domain_drop"],
            marker=marker,
            alpha=0.8,
            label=panel,
        )

    # Linear trend line.
    x = d["mean_fold_stability"].to_numpy(dtype=float)
    y = d["domain_drop"].to_numpy(dtype=float)

    if len(x) >= 2:
        coef = np.polyfit(x, y, 1)

        xx = np.linspace(
            np.nanmin(x),
            np.nanmax(x),
            100,
        )

        ax.plot(
            xx,
            coef[0] * xx + coef[1],
            linestyle="--",
            linewidth=1.5,
        )

    ax.axhline(
        0,
        linestyle=":",
        linewidth=1,
    )

    ax.set_xlabel(
        f"{arch} fold-to-fold prediction stability (TCGA)"
    )

    ax.set_ylabel(
        "External domain change in PCC\n"
        "(CPTAC - TCGA)"
    )

    title_panel = (
        "APM + TIS"
        if panel == "ALL"
        else panel
    )

    ax.set_title(
        f"{arch}: stability vs external generalization ({title_panel})"
    )

    ax.legend()
    ax.grid(alpha=0.2)

    savefig(
        out_dir
        / f"{arch}_stability_vs_external_domain_change_{panel}.png"
    )


def plot_stability_summary(stability_summary: pd.DataFrame, out_dir: Path) -> None:
    d = stability_summary.groupby("architecture", as_index=False)["mean_fold_stability"].mean()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(d.architecture, d.mean_fold_stability)
    ax.set_ylabel("Mean per-gene fold-to-fold prediction stability")
    ax.set_title("Prediction stability across independently trained folds")
    ax.set_ylim(0, 1)
    savefig(out_dir / "architecture_fold_stability_summary.png")


def _pairwise_fold_stability_patient_bootstrap(
    fold_bundles: Sequence[PredictionBundle],
    panel_genes: Sequence[str],
    idx: np.ndarray,
) -> float:
    """Patient-resampled grand mean fold-to-fold prediction stability.

    For each fold pair and each requested gene, compute Pearson r across the
    resampled patients. Aggregate correlations on the Fisher-z scale and
    transform back. This preserves the patient as the resampling unit and
    averages over all fold-pair/gene prediction relationships.
    """
    if len(fold_bundles) < 2:
        return np.nan
    ref = fold_bundles[0]
    maps = [resolve_gene_columns(b) for b in fold_bundles]
    common_genes = [g for g in panel_genes if all(g in m for m in maps)]
    if not common_genes:
        return np.nan

    # Align rows by submitter_id once.
    aligned = []
    for b in fold_bundles:
        _, ib, _ = align_pair(ref, b)
        aligned.append(b.preds[ib]
)

    zs = []
    for i, j in itertools.combinations(range(len(aligned)), 2):
        ai = aligned[i][idx]
        aj = aligned[j][idx]
        mi = np.asarray([maps[i][g] for g in common_genes], dtype=int)
        mj = np.asarray([maps[j][g] for g in common_genes], dtype=int)
        ai = ai[:, mi]
        aj = aj[:, mj]
        ai = ai - np.nanmean(ai, axis=0, keepdims=True)
        aj = aj - np.nanmean(aj, axis=0, keepdims=True)
        num = np.nansum(ai * aj, axis=0)
        den = np.sqrt(np.nansum(ai * ai, axis=0) * np.nansum(aj * aj, axis=0))
        rs = num / den
        rs = rs[np.isfinite(rs)]
        if rs.size:
            rs = np.clip(rs, -0.999999, 0.999999)
            zs.extend(np.arctanh(rs).tolist())
    if not zs:
        return np.nan
    return float(np.tanh(np.mean(zs)))


def bootstrap_stability_cohort_interactions(
    bundles: Dict[str, Dict[str, Dict[str, PredictionBundle]]],
    architectures: Sequence[str],
    panel_genes: Sequence[str],
    n_boot: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate architecture x cohort interaction for fold stability.

    Primary statistic is the change in aggregate fold-to-fold predictive
    stability from TCGA to CPTAC. A patient-level bootstrap is performed
    independently within each cohort. Architecture contrasts then compare
    these stability changes, analogous to the performance difference-in-differences.
    """
    cohort_folds = {}
    for arch in architectures:
        cohort_folds[arch] = {}
        for cohort in ["test", "cptac"]:
            fks = sorted(
                [k for k in bundles[arch].get(cohort, {}) if k.startswith("fold")],
                key=lambda x: int(x.replace("fold", "")),
            )
            if len(fks) < 2:
                raise ValueError(f"Need at least 2 folds for {arch}/{cohort}")
            cohort_folds[arch][cohort] = [bundles[arch][cohort][k] for k in fks]

    # Point estimates.
    point = []
    for arch in architectures:
        vals = {}
        for cohort in ["test", "cptac"]:
            n = len(cohort_folds[arch][cohort][0].sids)
            idx = np.arange(n)
            vals[cohort] = _pairwise_fold_stability_patient_bootstrap(
                cohort_folds[arch][cohort], panel_genes, idx
            )
            point.append({
                "architecture": arch,
                "cohort": cohort,
                "stability": vals[cohort],
                "n_folds": len(cohort_folds[arch][cohort]),
                "n_patients": n,
            })

    rng = np.random.default_rng(seed)
    boot_by_arch = {a: [] for a in architectures}
    for _ in range(n_boot):
        tcga_idx = rng.integers(0, len(cohort_folds[architectures[0]]["test"][0].sids), size=len(cohort_folds[architectures[0]]["test"][0].sids))
        cptac_idx = rng.integers(0, len(cohort_folds[architectures[0]]["cptac"][0].sids), size=len(cohort_folds[architectures[0]]["cptac"][0].sids))
        for arch in architectures:
            st = _pairwise_fold_stability_patient_bootstrap(cohort_folds[arch]["test"], panel_genes, tcga_idx)
            sc = _pairwise_fold_stability_patient_bootstrap(cohort_folds[arch]["cptac"], panel_genes, cptac_idx)
            boot_by_arch[arch].append((st, sc, sc - st))

    summary_rows = []
    for arch in architectures:
        arr = np.asarray(boot_by_arch[arch], dtype=float)
        st_lo, st_hi = np.nanpercentile(arr[:, 0], [2.5, 97.5])
        sc_lo, sc_hi = np.nanpercentile(arr[:, 1], [2.5, 97.5])
        shift_lo, shift_hi = np.nanpercentile(arr[:, 2], [2.5, 97.5])
        p_shift = min(1.0, 2.0 * min(np.nanmean(arr[:, 2] <= 0), np.nanmean(arr[:, 2] >= 0)))
        summary_rows.append({
            "architecture": arch,
            "TCGA_stability": next(r["stability"] for r in point if r["architecture"] == arch and r["cohort"] == "test"),
            "TCGA_CI_low": st_lo,
            "TCGA_CI_high": st_hi,
            "CPTAC_stability": next(r["stability"] for r in point if r["architecture"] == arch and r["cohort"] == "cptac"),
            "CPTAC_CI_low": sc_lo,
            "CPTAC_CI_high": sc_hi,
            "CPTAC_minus_TCGA": next(r["stability"] for r in point if r["architecture"] == arch and r["cohort"] == "cptac") - next(r["stability"] for r in point if r["architecture"] == arch and r["cohort"] == "test"),
            "shift_CI_low": shift_lo,
            "shift_CI_high": shift_hi,
            "shift_boot_p": p_shift,
        })

    interaction_rows = []
    for a, b in itertools.combinations(architectures, 2):
        aa = np.asarray(boot_by_arch[a], dtype=float)
        bb = np.asarray(boot_by_arch[b], dtype=float)
        # A = architecture a; B = architecture b. Positive means a has a larger
        # stability change (CPTAC-TCGA) than b.
        d = aa[:, 2] - bb[:, 2]
        obs_a = next(r["stability"] for r in point if r["architecture"] == a and r["cohort"] == "cptac") - next(r["stability"] for r in point if r["architecture"] == a and r["cohort"] == "test")
        obs_b = next(r["stability"] for r in point if r["architecture"] == b and r["cohort"] == "cptac") - next(r["stability"] for r in point if r["architecture"] == b and r["cohort"] == "test")
        lo, hi = np.nanpercentile(d, [2.5, 97.5])
        p = min(1.0, 2.0 * min(np.nanmean(d <= 0), np.nanmean(d >= 0)))
        interaction_rows.append({
            "architecture_A": a,
            "architecture_B": b,
            "A_stability_shift_CPTAC_minus_TCGA": obs_a,
            "B_stability_shift_CPTAC_minus_TCGA": obs_b,
            "interaction_A_minus_B": obs_a - obs_b,
            "CI95_low": float(lo),
            "CI95_high": float(hi),
            "p_boot": float(p),
            "robust": bool(lo > 0 or hi < 0),
        })

    return pd.DataFrame(summary_rows), pd.DataFrame(interaction_rows)


def plot_architecture_performance_panel(metrics_df: pd.DataFrame, out_dir: Path) -> None:
    d = metrics_df[metrics_df.kind == "ensemble"].copy()
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    archs = [a for a in ["abmil", "clam", "transmil"] if a in d.architecture.unique()]
    offsets = {"abmil": -0.22, "clam": 0.0, "transmil": 0.22}
    x0, x1 = 0, 1
    for arch in archs:
        for panel, marker in [("APM", "o"), ("TIS", "^")]:
            q = d[(d.architecture == arch) & (d.panel == panel)].set_index("cohort")
            if not {"test", "cptac"}.issubset(q.index):
                continue
            xs = [x0 + offsets[arch], x1 + offsets[arch]]
            ys = [float(q.loc["test", "PCC"]), float(q.loc["cptac", "PCC"])]
            ax.plot(xs, ys, marker=marker, linewidth=2, label=f"{arch.upper()} {panel}")
    ax.axhline(0, linestyle=":", linewidth=1)
    ax.set_xticks([0, 1], ["TCGA", "CPTAC"])
    ax.set_ylabel("Panel Pearson correlation (PCC)")
    ax.set_title("External performance exposes architecture-dependent degradation")
    ax.legend(ncol=2, frameon=False, fontsize=9)
    ax.grid(alpha=0.2)
    savefig(out_dir / "architecture_performance_tcga_cptac_panel.png")


def plot_architecture_stability_panel(stability_summary: pd.DataFrame, out_dir: Path) -> None:
    d = stability_summary.copy()
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    archs = [a for a in ["abmil", "clam", "transmil"] if a in d.architecture.unique()]
    offsets = {"abmil": -0.22, "clam": 0.0, "transmil": 0.22}
    for arch in archs:
        q = d[d.architecture == arch].iloc[0]
        xs = [0 + offsets[arch], 1 + offsets[arch]]
        ys = [q["TCGA_stability"], q["CPTAC_stability"]]
        ax.plot(xs, ys, marker="o", linewidth=2, label=arch.upper())
    ax.set_xticks([0, 1], ["TCGA", "CPTAC"])
    ax.set_ylim(0.65, 1.0)
    ax.set_ylabel("Mean fold-to-fold predictive stability")
    ax.set_title("TransMIL instability is visible internally and amplified externally")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    savefig(out_dir / "architecture_stability_tcga_cptac_panel.png")


def plot_combined_two_panel(metrics_df: pd.DataFrame, stability_summary: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    # Panel A: performance
    d = metrics_df[metrics_df.kind == "ensemble"].copy()
    archs = [a for a in ["abmil", "clam", "transmil"] if a in d.architecture.unique()]
    offsets = {"abmil": -0.18, "clam": 0.0, "transmil": 0.18}
    for arch in archs:
        for panel, marker in [("APM", "o"), ("TIS", "^")]:
            q = d[(d.architecture == arch) & (d.panel == panel)].set_index("cohort")
            if {"test", "cptac"}.issubset(q.index):
                axes[0].plot([offsets[arch], 1 + offsets[arch]], [q.loc["test", "PCC"], q.loc["cptac", "PCC"]], marker=marker, linewidth=2, label=f"{arch.upper()} {panel}")
    axes[0].set_xticks([0, 1], ["TCGA", "CPTAC"])
    axes[0].set_ylabel("Panel PCC")
    axes[0].set_title("A. Predictive performance")
    axes[0].grid(alpha=0.2)
    axes[0].legend(ncol=2, frameon=False, fontsize=8)
    # Panel B: stability
    for arch in archs:
        q = stability_summary[stability_summary.architecture == arch].iloc[0]
        axes[1].plot([offsets[arch], 1 + offsets[arch]], [q.TCGA_stability, q.CPTAC_stability], marker="o", linewidth=2, label=arch.upper())
    axes[1].set_xticks([0, 1], ["TCGA", "CPTAC"])
    axes[1].set_ylim(0.65, 1.0)
    axes[1].set_ylabel("Mean fold-to-fold predictive stability")
    axes[1].set_title("B. Predictive stability")
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False)
    fig.suptitle("Internal performance vs. stability under external cohort shift", y=1.02, fontsize=15)
    savefig(out_dir / "architecture_performance_and_stability_two_panel.png")


# ============================================================================
# MAIN ANALYSIS
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Directory containing prediction NPZ files")
    parser.add_argument("--output-dir", required=True, help="Directory for analysis results")
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--n-perm", type=int, default=N_PERM)
    parser.add_argument("--n-stab-boot", type=int, default=N_STAB_BOOT, help="Patient-level bootstrap iterations for architecture x cohort stability analysis")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    root = Path(args.input_dir).expanduser().resolve()
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)

    if not APM_GENES or not TIS_GENES:
        print("WARNING: APM_GENES/TIS_GENES are empty. Panel-level analyses will be skipped.")

    discovered = discover_files(root)
    architectures = sorted(discovered)
    print("Architectures discovered:", architectures)

    bundles: Dict[str, Dict[str, Dict[str, PredictionBundle]]] = {}
    inventory_rows = []
    for arch in architectures:
        bundles[arch] = {}
        for cohort in ["test", "cptac"]:
            bundles[arch][cohort] = {}
            for kind, path in discovered[arch].get(cohort, {}).items():
                b = load_npz(path)
                bundles[arch][cohort][kind] = b
                inventory_rows.append({
                    "architecture": arch,
                    "cohort": cohort,
                    "kind": kind,
                    "path": str(path),
                    "n_samples": b.preds.shape[0],
                    "n_genes": b.preds.shape[1],
                })
    inventory = pd.DataFrame(inventory_rows)
    inventory.to_csv(out / "file_inventory.csv", index=False)

    # ----------------------------------------------------------------------
    # QA: check ensemble consistency and fold IDs.
    # ----------------------------------------------------------------------
    qa_rows = []
    for arch in architectures:
        for cohort in ["test", "cptac"]:
            fks = sorted(
                [k for k in bundles[arch].get(cohort, {}) if k.startswith("fold")],
                key=lambda x: int(x.replace("fold", ""))
            )
            if not fks:
                continue
            ref = bundles[arch][cohort][fks[0]]
            same_ids = True
            same_genes = True
            for fk in fks[1:]:
                b = bundles[arch][cohort][fk]
                same_ids &= np.array_equal(ref.sids, b.sids)
                same_genes &= ref.gene_cols == b.gene_cols
            qa_rows.append({
                "architecture": arch,
                "cohort": cohort,
                "n_folds": len(fks),
                "folds": ",".join(fks),
                "same_patient_order_across_folds": same_ids,
                "same_gene_order_across_folds": same_genes,
                "ensemble_present": "ensemble" in bundles[arch].get(cohort, {}),
            })
    pd.DataFrame(qa_rows).to_csv(out / "fold_qa.csv", index=False)

    # ----------------------------------------------------------------------
    # Panel metrics for each fold and ensemble, both cohorts.
    # ----------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    metric_rows = []
    for arch in architectures:
        for cohort in ["test", "cptac"]:
            for kind, b in bundles[arch].get(cohort, {}).items():
                for panel, genes in [("APM", APM_GENES), ("TIS", TIS_GENES)]:
                    if not genes:
                        continue
                    pcc = panel_pcc(b.preds, b.labels, b.gene_cols, genes)
                    auc = panel_auc(b.preds, b.labels, b.gene_cols, genes)
                    _, lo, hi = bootstrap_panel_ci(b.preds, b.labels, b.gene_cols, genes, args.n_boot, rng)
                    metric_rows.append({
                        "architecture": arch,
                        "cohort": cohort,
                        "kind": kind,
                        "panel": panel,
                        "PCC": pcc,
                        "CI95_low": lo,
                        "CI95_high": hi,
                        "AUC": auc,
                        "n_samples": len(b.sids),
                    })
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(out / "panel_metrics_all_folds_and_ensembles.csv", index=False)
    plot_architecture_performance_panel(metrics_df, fig_dir)

    # ----------------------------------------------------------------------
    # Per-gene PCC, p, q in both cohorts; ensemble and every fold.
    # ----------------------------------------------------------------------
    gene_stats = []
    for arch in architectures:
        for cohort in ["test", "cptac"]:
            for kind, b in bundles[arch].get(cohort, {}).items():
                g = gene_level_stats(b, cohort, arch)
                g["kind"] = kind
                gene_stats.append(g)
    gene_stats_df = pd.concat(gene_stats, ignore_index=True) if gene_stats else pd.DataFrame()
    gene_stats_df.to_csv(out / "per_gene_pcc_p_q_all_models.csv", index=False)

    # Ensemble per-gene domain table.
    ensemble_gene = gene_stats_df[gene_stats_df.kind == "ensemble"].copy() if len(gene_stats_df) else pd.DataFrame()
    if len(ensemble_gene):
        piv = ensemble_gene.pivot_table(index=["architecture", "gene", "panel"], columns="cohort", values="PCC", aggfunc="first").reset_index()
        if "test" in piv.columns and "cptac" in piv.columns:
            piv["TCGA_PCC"] = piv["test"]
            piv["CPTAC_PCC"] = piv["cptac"]
            piv["domain_drop_CPTAC_minus_TCGA"] = piv["CPTAC_PCC"] - piv["TCGA_PCC"]
            piv["absolute_drop_TCGA_minus_CPTAC"] = piv["TCGA_PCC"] - piv["CPTAC_PCC"]
            piv.to_csv(out / "per_gene_tcga_cptac_ensemble_domain_shift.csv", index=False)
            plot_gene_domain_retention(piv.rename(columns={"domain_drop_CPTAC_minus_TCGA": "domain_drop"}), fig_dir)

    # ----------------------------------------------------------------------
    # Fold-to-fold stability on TCGA and CPTAC.
    # ----------------------------------------------------------------------
    stab_rows = []
    stab_gene_rows = []
    for arch in architectures:
        for cohort in ["test", "cptac"]:
            fks = sorted(
                [k for k in bundles[arch].get(cohort, {}) if k.startswith("fold")],
                key=lambda x: int(x.replace("fold", ""))
            )
            if len(fks) < 2:
                continue
            sb = gene_stability([bundles[arch][cohort][fk] for fk in fks])
            sb["architecture"] = arch
            sb["cohort"] = cohort
            stab_gene_rows.append(sb)
            for _, r in sb.iterrows():
                stab_rows.append({
                    "architecture": arch,
                    "cohort": cohort,
                    **r.to_dict(),
                })
    stability_gene_df = pd.DataFrame(stab_rows)
    stability_gene_df.to_csv(out / "fold_to_fold_stability_per_gene_all_architectures.csv", index=False)
    if len(stability_gene_df):
        plot_stability_summary(stability_gene_df[stability_gene_df.cohort == "test"], fig_dir)

        # Statistical stability comparisons: TCGA and CPTAC, all/APM/TIS.
        stability_tests = []
        arch_pairs = list(itertools.combinations(architectures, 2))
        for cohort in ["test", "cptac"]:
            d = stability_gene_df[stability_gene_df.cohort == cohort].copy()
            for a, b in arch_pairs:
                for panel in [None, "APM", "TIS"]:
                    res = paired_gene_stability_test(
                        d, a, b, panel, args.n_boot, args.seed
                    ) if a in d.architecture.values and b in d.architecture.values else None
                    if res:
                        res["cohort"] = cohort
                        stability_tests.append(res)
        pd.DataFrame(stability_tests).to_csv(out / "fold_stability_architecture_pairwise_statistics.csv", index=False)

        # Patient-level architecture x cohort interaction for predictive stability.
        try:
            all_panel_genes = APM_GENES + TIS_GENES
            stab_cohort_summary, stab_interactions = bootstrap_stability_cohort_interactions(
                bundles, architectures, all_panel_genes, args.n_stab_boot, args.seed
            )
            stab_cohort_summary.to_csv(out / "architecture_cohort_stability_summary.csv", index=False)
            stab_interactions.to_csv(out / "architecture_cohort_stability_interactions.csv", index=False)
            plot_architecture_stability_panel(stab_cohort_summary, fig_dir)
            plot_combined_two_panel(metrics_df, stab_cohort_summary, fig_dir)
        except Exception as exc:
            print(f"WARNING: stability interaction analysis skipped: {exc}")

    # ----------------------------------------------------------------------
    # Cross-architecture agreement and per-gene PCC-vector agreement.
    # ----------------------------------------------------------------------
    for cohort in ["test", "cptac"]:
        ens = {
            a: bundles[a][cohort]["ensemble"]
            for a in architectures
            if "ensemble" in bundles[a].get(cohort, {})
        }
        if len(ens) >= 2:
            agree_summary, agree_gene = cross_architecture_agreement(ens)
            agree_summary["cohort"] = cohort
            agree_gene["cohort"] = cohort
            agree_summary.to_csv(out / f"{cohort}_architecture_prediction_agreement_summary.csv", index=False)
            agree_gene.to_csv(out / f"{cohort}_architecture_prediction_agreement_per_gene.csv", index=False)

            c = correlation_of_gene_pcc_vectors(gene_stats_df, cohort)
            c.to_csv(out / f"{cohort}_architecture_per_gene_PCC_vector_correlation.csv", index=False)

    # ----------------------------------------------------------------------
    # Equal-weight cross-architecture ensembles (external + internal).
    # ----------------------------------------------------------------------
    for cohort in ["test", "cptac"]:
        ens = {
            a: bundles[a][cohort]["ensemble"]
            for a in architectures
            if "ensemble" in bundles[a].get(cohort, {})
        }
        if len(ens) >= 2 and APM_GENES and TIS_GENES:
            combo_df = equal_weight_arch_ensembles(ens)
            combo_df.to_csv(out / f"{cohort}_cross_architecture_equal_weight_ensembles.csv", index=False)

    # ----------------------------------------------------------------------
    # Difference-in-differences / domain-shift architecture comparisons.
    # ----------------------------------------------------------------------
    if APM_GENES and TIS_GENES:
        did_rows = []
        arch_pairs = list(itertools.combinations(architectures, 2))
        for a, b in arch_pairs:
            required = (
                a in bundles and b in bundles
                and "ensemble" in bundles[a].get("test", {})
                and "ensemble" in bundles[b].get("test", {})
                and "ensemble" in bundles[a].get("cptac", {})
                and "ensemble" in bundles[b].get("cptac", {})
            )
            if not required:
                continue
            for panel, genes in [("APM", APM_GENES), ("TIS", TIS_GENES)]:
                res = bootstrap_diff_of_gaps(
                    bundles[a]["test"]["ensemble"],
                    bundles[b]["test"]["ensemble"],
                    bundles[a]["cptac"]["ensemble"],
                    bundles[b]["cptac"]["ensemble"],
                    genes,
                    args.n_boot,
                    args.seed,
                )
                res["panel"] = panel
                did_rows.append(res)
        pd.DataFrame(did_rows).to_csv(out / "architecture_domain_shift_difference_in_differences.csv", index=False)

    # ----------------------------------------------------------------------
    # Stability -> external generalization.
    # The principal predictor is INTERNAL (TCGA) fold stability, because that
    # does not use CPTAC labels to define the stability measure.
    # ----------------------------------------------------------------------
    if len(stability_gene_df) and len(ensemble_gene):
        tcga_stab = stability_gene_df[stability_gene_df.cohort == "test"].copy()
        ext = ensemble_gene.pivot_table(index=["architecture", "gene", "panel"], columns="cohort", values="PCC", aggfunc="first").reset_index()
        if "test" in ext.columns and "cptac" in ext.columns:
            ext = ext.rename(columns={"test": "TCGA_PCC", "cptac": "CPTAC_PCC"})
            ext["domain_drop"] = ext["CPTAC_PCC"] - ext["TCGA_PCC"]

            rel_rows = []
            partial_rows = []
            for arch in architectures:
                s = tcga_stab[tcga_stab.architecture == arch][["gene", "panel", "mean_fold_stability"]]
                e = ext[ext.architecture == arch][["gene", "panel", "TCGA_PCC", "CPTAC_PCC", "domain_drop"]]
                d = s.merge(e, on=["gene", "panel"], how="inner")
                if len(d) < MIN_N:
                    continue
                for panel in ["ALL", "APM", "TIS"]:
                    q = d if panel == "ALL" else d[d.panel == panel]
                    if len(q) < MIN_N:
                        continue
                    rr = spearman_bootstrap_permutation(
                        q.mean_fold_stability.to_numpy(float),
                        q.domain_drop.to_numpy(float),
                        args.n_boot,
                        args.n_perm,
                        args.seed,
                    )
                    rr.update({"architecture": arch, "panel": panel})
                    rel_rows.append(rr)

                    # Partial Spearman: rank-transform, then residualise both
                    # stability and domain drop against TCGA baseline PCC.
                    x = q.mean_fold_stability.to_numpy(float)
                    y = q.domain_drop.to_numpy(float)
                    z = q.TCGA_PCC.to_numpy(float)
                    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
                    x, y, z = x[m], y[m], z[m]
                    if len(x) >= MIN_N:
                        rx = pd.Series(x).rank().to_numpy(float)
                        ry = pd.Series(y).rank().to_numpy(float)
                        rz = pd.Series(z).rank().to_numpy(float)
                        X = np.column_stack([np.ones(len(rz)), rz])
                        bx = np.linalg.lstsq(X, rx, rcond=None)[0]
                        by = np.linalg.lstsq(X, ry, rcond=None)[0]
                        resx = rx - X @ bx
                        resy = ry - X @ by
                        pr = pearsonr(resx, resy)
                        partial_rows.append({
                            "architecture": arch,
                            "panel": panel,
                            "n": len(x),
                            "partial_spearman_rho": float(pr.statistic),
                            "p_approx": float(pr.pvalue),
                        })

                    if panel == "ALL":
                        plot_stability_vs_drop(
                            d,
                            fig_dir / arch,
                            arch,
                            panel="ALL",
                        )
            pd.DataFrame(rel_rows).to_csv(out / "stability_vs_external_generalization.csv", index=False)
            pd.DataFrame(partial_rows).to_csv(out / "stability_vs_external_generalization_partial_spearman.csv", index=False)

    # ----------------------------------------------------------------------
    # Compact machine-readable manifest / README-style summary.
    # ----------------------------------------------------------------------
    manifest = {
        "input_dir": str(root),
        "output_dir": str(out),
        "architectures": architectures,
        "apm_n_genes": len(APM_GENES),
        "tis_n_genes": len(TIS_GENES),
        "n_boot": args.n_boot,
        "n_perm": args.n_perm,
        "seed": args.seed,
        "outputs": sorted(p.name for p in out.iterdir() if p.is_file()),
    }
    (out / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2))

    print("\nAnalysis complete.")
    print(f"Results: {out}")
    print("Key outputs:")
    for name in [
        "panel_metrics_all_folds_and_ensembles.csv",
        "per_gene_pcc_p_q_all_models.csv",
        "per_gene_tcga_cptac_ensemble_domain_shift.csv",
        "fold_to_fold_stability_per_gene_all_architectures.csv",
        "fold_stability_architecture_pairwise_statistics.csv",
        "architecture_domain_shift_difference_in_differences.csv",
        "stability_vs_external_generalization.csv",
        "stability_vs_external_generalization_partial_spearman.csv",
    ]:
        p = out / name
        print(f"  {'[OK]' if p.exists() else '[--]'} {name}")


if __name__ == "__main__":
    main()
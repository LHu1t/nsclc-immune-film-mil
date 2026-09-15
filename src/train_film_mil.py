"""
FiLM-Conditioned Attention-MIL Classifier
==========================================
Predicts 35 immune-related gene expression targets (APM + TIS panels)
from UNI2-h pre-extracted WSI features, conditioned on cancer subtype
(LUAD vs LUSC) via Feature-wise Linear Modulation (FiLM).

Architecture:
    UNI2-h features (frozen, 1536-dim)
        V
    Attention-MIL pooling  ->  slide embedding (512-dim)
        V
    FiLM conditioning (subtype modulates slide embedding channel-wise)
        V
    Concatenate clinical covariates (age_z, gender)
        V
    Regression head  ->  35 gene expression outputs

Novel contribution:
    FiLM conditioning allows the model to learn subtype-specific
    transformations of the shared morphological feature space, rather
    than training independent models or concatenating a one-hot label.
    This is the first application of FiLM to subtype-conditioned immune
    gene expression prediction from histopathology.

Aggregator-architecture robustness sweep:
    The tile-pooling stage below FiLM is pluggable (--aggregator), so the
    same frozen UNI2-h embeddings, split, and training schedule can be
    re-run through four different MIL pooling architectures:
        abmil    - additive Attention-MIL (Ilse et al., 2018)   [default]
        clam     - CLAM-style gated-attention pooling (Lu et al., 2021)
        transmil - compact self-attention / Transformer MIL, TransMIL-style
                   (Shao et al., 2021)
        graph    - k-NN graph-MIL with residual message passing
    This tests whether the FiLM / subtype-conditioning finding survives a
    change of aggregator, i.e. that it is a property of the modelling idea
    and not an artefact of one specific pooling architecture. Use
    --aggregator_sweep to run all of them in one call and produce a
    combined summary table (see run_sweep()).

Usage:
    # Single run with the default (additive Attention-MIL) aggregator:
    python train_film_mil.py \
        --luad_features /path/to/UNI2/Features/TCGA-LUAD \
        --lusc_features /path/to/UNI2/Features/TCGA-LUSC \
        --metadata      /path/to/metadata.csv \
        --output_dir    ./results \
        --n_folds       5 \
        --use_film      true

    # Single run with an alternative aggregator:
    python train_film_mil.py \
        --luad_features /path/to/UNI2/Features/TCGA-LUAD \
        --lusc_features /path/to/UNI2/Features/TCGA-LUSC \
        --metadata      /path/to/metadata.csv \
        --output_dir    ./results_transmil \
        --use_film      true \
        --aggregator    transmil \
        --agg_max_tiles 2000

    # Full aggregator-architecture robustness sweep (main technical
    # contribution): trains/evaluates all four aggregators back-to-back
    # under identical data/FiLM/training conditions and writes a combined
    # summary table to <output_dir>/aggregator_sweep_summary.{json,csv}:
    python train_film_mil.py \
        --luad_features /path/to/UNI2/Features/TCGA-LUAD \
        --lusc_features /path/to/UNI2/Features/TCGA-LUSC \
        --metadata      /path/to/metadata.csv \
        --output_dir    ./results_aggregator_sweep \
        --use_film      true \
        --aggregator_sweep abmil,clam,transmil,graph \
        --agg_max_tiles 2000

File structure expected in feature directories:
    <luad_features>/<submitter_id>.h5   (keys: features, coords)
    <lusc_features>/<submitter_id>.h5   (keys: features, coords)
"""

import argparse
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# Gene panels
APM_GENES = [
    "PSMB5","PSMB6","PSMB7","PSMB8","PSMB9","PSMB10",
    "TAP1","TAP2","ERAP1","ERAP2","TAPBP","CANX",
    "CALR","PDIA3","B2M","HLA-A","HLA-B","HLA-C",
]
TIS_GENES = [
    "PSMB10","HLA-DQA1","HLA-DRB1","CMKLR1","HLA-E","NKG7",
    "CD8A","CCL5","CXCL9","CD27","CXCR6","IDO1",
    "STAT1","CD274","CD276","LAG3","PDCD1LG2","TIGIT",
]
SUBTYPE_MAP = {"LUAD": 0, "LUSC": 1}

# Literal pre-computed panel-score columns in the metadata CSV (if present), used as ground truth for the direct APM/TIS head (see PanelHead below). These are distinct from APM_GENES / TIS_GENES, which are the individual genes used for the "mean-of-predicted-genes" panel PCC.
PANEL_TARGETS = ["APM", "TIS"]


# Metadata loading
def load_metadata(gene_csv: str):
    df = pd.read_csv(gene_csv)
    df.columns = [c.strip() for c in df.columns]

    # Find ID column
    id_col = None
    for c in df.columns:
        if any(k in c.lower() for k in ["sample", "submitter", "case", "id"]):
            id_col = c
            break
    if id_col is None:
        raise ValueError("No ID column found.")
    df["submitter_id"] = df[id_col].str.upper().str.strip()

    # Gene columns: _fpkm_uq + TMB + pre-computed APM and TIS panel scores
    gene_cols = [c for c in df.columns if c.endswith("_fpkm_uq") or c in ("TMB", "APM", "TIS")]
    if not gene_cols:
        raise ValueError("No _fpkm_uq columns found.")

    df[gene_cols] = df[gene_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=gene_cols).copy()

    # log1p + save raw copy
    df[gene_cols] = np.log1p(df[gene_cols])
    for g in gene_cols:
        df[g + "_raw"] = df[g].copy()

    # Z-score
    y_means = df[gene_cols].mean()
    y_stds  = df[gene_cols].std().replace(0, 1)
    df[gene_cols] = (df[gene_cols] - y_means) / y_stds

    # Clinical covariates
    if "age_years" not in df.columns or "demographic.gender" not in df.columns:
        raise ValueError("Expected 'age_years' and 'demographic.gender' in CSV.")
    df["age_years"] = pd.to_numeric(df["age_years"], errors="coerce")
    df["age_years"]   = df["age_years"].fillna(df["age_years"].median())
    df["age_years_z"] = (df["age_years"] - df["age_years"].mean()) / df["age_years"].std()
    df["gender_encoded"] = (
        df["demographic.gender"].str.lower()
        .map({"male": 0, "female": 1}).fillna(0).astype(np.float32)
    )

    clinical_cols = ["age_years_z", "gender_encoded"]

    # Locate the literal pre-computed APM/TIS panel-score columns (if the CSV provides them) within gene_cols, so the direct panel head can be supervised against and evaluated against the *true* panel scores rather than the mean of individually-predicted genes.
    panel_idx = {name: gene_cols.index(name) for name in PANEL_TARGETS if name in gene_cols}
    missing_panels = [name for name in PANEL_TARGETS if name not in panel_idx]
    if missing_panels:
        log.warning(
            f"  Metadata CSV has no literal column(s) {missing_panels} for the "
            f"direct panel head. Add pre-computed APM/TIS score columns to the "
            f"CSV to evaluate the head against true panel scores."
        )

    return df, gene_cols, clinical_cols, y_means, y_stds, panel_idx


# ---------------------------------------------------------------------------
# Process-wide feature cache
#
# FiLMDataset.__getitem__ previously opened and fully decompressed the
# relevant .h5 file from disk on every single access -- every epoch, every
# fold, and (in the aggregator sweep) every aggregator. That repeated disk
# IO + h5py decompression is CPU-bound work spread across the DataLoader
# workers, and it is almost always the real bottleneck behind the symptom
# pattern "CPU pinned near 100% across all cores, GPU utilization low,
# RAM/VRAM nowhere near their limits, and reducing n_tiles doesn't help" --
# because the FULL tile array was always read from disk before any
# tile-subsampling happened, so a tile cap only reduces GPU compute (already
# cheap for these aggregators), not the IO that was actually gating
# throughput.
#
# The fix: cache unique .h5 feature arrays in memory for the life of the
# process. When the dataset is small enough to fit in RAM outright, every
# file gets cached and every __getitem__ becomes a pure in-memory slice.
# When it doesn't (e.g. a 30GB-RAM Kaggle session against a 70GB combined
# LUAD+LUSC corpus), the cache is BUDGET-CAPPED via configure_feature_cache():
# it fills up to a byte ceiling and then stops caching new files -- it does
# NOT evict already-cached entries to make room for new ones, since with
# random per-epoch shuffling there's no useful temporal locality to exploit
# and LRU-style eviction would just thrash (constantly evicting and
# re-loading) without ever converging to a stable hit rate. Whatever doesn't
# fit in the budget is read from disk on every access, exactly as before
# caching existed -- no worse than the original behaviour, and the portion
# that DOES fit is a pure win. Optionally storing the cache in float16
# (roughly halving its footprint; values are upcast back to float32 per
# accessed tile, so training precision is unaffected) lets substantially
# more of a large corpus fit under a fixed RAM budget.
# ---------------------------------------------------------------------------
_FEATURE_CACHE: dict[str, np.ndarray] = {}
_FEATURE_CACHE_BYTES = 0
_FEATURE_CACHE_MAX_BYTES: int | None = None   # None = uncapped (original behaviour)
_FEATURE_CACHE_DTYPE = np.float32
_feature_cache_lock = threading.Lock()


def configure_feature_cache(max_gb: float | None = None, dtype: str = "float32") -> None:
    """
    Sets the memory budget and storage dtype for the process-wide feature
    cache. Call this ONCE, before preload_feature_cache() / before any
    DataLoader is constructed, whenever the full dataset does not comfortably
    fit in RAM alongside everything else (OS, PyTorch, pandas, DataLoader
    worker processes, pinned-memory buffers, ...).

    max_gb : cap on cache size in GiB. None (default) means uncapped -- only
        safe if the whole corpus comfortably fits in RAM on its own. Leave
        meaningful headroom below your box's total RAM: e.g. on a 30GB
        Kaggle session, something like max_gb=18-20 is a reasonable starting
        point, not max_gb=30.
    dtype : "float32" (default, no precision loss) or "float16" (roughly
        halves the cache footprint). Cached arrays are upcast to float32
        per accessed/subsampled tile before being handed to the model, so
        this only affects the resting in-memory copy, not the precision
        anything is trained/evaluated with.

    Example, for a ~70GB combined LUAD+LUSC corpus on a 30GB-RAM session:
        configure_feature_cache(max_gb=20, dtype="float16")
    caches up to 20GB of *float16* features (~40GB worth of original
    float32 data -- i.e. potentially the whole corpus), leaving ~10GB
    headroom for everything else.
    """
    global _FEATURE_CACHE_MAX_BYTES, _FEATURE_CACHE_DTYPE
    if dtype not in ("float32", "float16"):
        raise ValueError(f"dtype must be 'float32' or 'float16', got {dtype!r}")
    _FEATURE_CACHE_MAX_BYTES = None if max_gb is None else int(max_gb * (1024 ** 3))
    _FEATURE_CACHE_DTYPE = {"float32": np.float32, "float16": np.float16}[dtype]
    log.info(
        "Feature cache configured: budget="
        + ("unlimited" if max_gb is None else f"{max_gb:.1f} GB")
        + f", dtype={dtype}"
    )


def _load_features_cached(h5_path) -> np.ndarray:
    """
    Returns the (N_tiles, feat_dim) feature array for `h5_path`, from cache
    if present. If not cached and there's room left in the configured
    budget, reads it from disk, casts to the configured cache dtype, stores
    it, and returns it. If the budget is already full, reads it from disk
    and returns it WITHOUT caching it (no eviction of existing entries).
    """
    global _FEATURE_CACHE_BYTES
    key = str(h5_path)
    cached = _FEATURE_CACHE.get(key)
    if cached is not None:
        return cached

    with h5py.File(h5_path, "r") as f:
        arr = f["features"][:].astype(_FEATURE_CACHE_DTYPE, copy=False)

    with _feature_cache_lock:
        if key not in _FEATURE_CACHE and (
            _FEATURE_CACHE_MAX_BYTES is None
            or _FEATURE_CACHE_BYTES + arr.nbytes <= _FEATURE_CACHE_MAX_BYTES
        ):
            _FEATURE_CACHE[key] = arr
            _FEATURE_CACHE_BYTES += arr.nbytes
    return arr


def preload_feature_cache(feature_dirs: dict, max_workers: int = 8,
                           priority_paths: list | None = None) -> None:
    """
    Eagerly loads .h5 feature files found in `feature_dirs` into
    `_FEATURE_CACHE`, once, BEFORE any DataLoader workers are spawned, up to
    whatever budget was set via configure_feature_cache() (uncapped by
    default).

    priority_paths : optional list of h5 paths to attempt to cache FIRST,
        before the rest of the corpus. Pass the fixed test set's h5 paths
        here when the cache is budget-capped: the test set is re-read
        identically by every fold and (in the aggregator sweep) every
        aggregator, so it has the highest reuse value per cached byte and
        should be the first thing to survive a tight budget.

    Call this exactly once per script run (train() and the notebook sweep
    loop both do this); safe to call again later (e.g. once per aggregator
    in a sweep) since already-cached files are skipped instantly, and once
    the budget is full, remaining files are skipped without even being
    re-read.
    """
    all_paths = [p for fdir in feature_dirs.values() for p in Path(fdir).glob("*.h5")]
    if priority_paths:
        priority_set = {str(p) for p in priority_paths}
        ordered = ([p for p in all_paths if str(p) in priority_set]
                   + [p for p in all_paths if str(p) not in priority_set])
    else:
        ordered = all_paths

    to_consider = [p for p in ordered if str(p) not in _FEATURE_CACHE]
    if not to_consider:
        return

    budget_desc = ("unlimited" if _FEATURE_CACHE_MAX_BYTES is None
                   else f"{_FEATURE_CACHE_MAX_BYTES / 1e9:.1f} GB")
    log.info(f"Preloading feature cache (budget: {budget_desc}, "
             f"dtype: {_FEATURE_CACHE_DTYPE.__name__}, "
             f"{len(to_consider)} file(s) to consider)...")
    t0 = time.time()

    def _try_load(p):
        # Skip the disk read entirely once the budget is already full --
        # no point paying IO for an array we know we won't retain.
        if (_FEATURE_CACHE_MAX_BYTES is not None
                and _FEATURE_CACHE_BYTES >= _FEATURE_CACHE_MAX_BYTES):
            return
        _load_features_cached(p)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        list(ex.map(_try_load, to_consider))

    n_cached = sum(1 for p in to_consider if str(p) in _FEATURE_CACHE)
    n_skipped = len(to_consider) - n_cached
    log.info(
        f"Preload done in {time.time() - t0:.1f}s: {n_cached}/{len(to_consider)} "
        f"file(s) cached ({_FEATURE_CACHE_BYTES / 1e9:.2f} GB resident)."
        + (f" {n_skipped} file(s) did not fit the budget and will be read "
           f"from disk on each access (same cost as before caching)."
           if n_skipped else "")
    )


# Dataset
class FiLMDataset(Dataset):
    """
    Loads UNI2-h pre-extracted .h5 features and merges gene expression
    labels + clinical covariates from the metadata dataframe.

    Expects .h5 files with key 'features' (N_tiles, 1536).
    (MahmoodLab/UNI2-h-features format — no label inside h5.)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_dirs: dict, # {"LUAD": Path, "LUSC": Path}
        gene_cols: list,
        clinical_cols: list,
        n_tiles: int | None = None, # None = all tiles
        deterministic: bool = False,
        seed: int = 98,
    ):
        self.gene_cols     = gene_cols
        self.clinical_cols = clinical_cols
        self.n_tiles       = n_tiles
        self.deterministic = deterministic
        self.seed          = seed
        self.feature_dirs  = {k: Path(v) for k, v in feature_dirs.items()}

        # Build barcode index for each subtype directory.
        # UNI2-h filenames use full barcodes (e.g. TCGA-05-4244-01Z-00-DX1.h5)
        # but submitter_id in the CSV is the 12-char patient barcode (TCGA-05-4244).
        # We map first-12-chars -> full h5 path so the two formats match.
        self.barcode_index = {}
        for subtype, fdir in self.feature_dirs.items():
            index = {}
            for p in Path(fdir).glob("*.h5"):
                patient_barcode = p.stem[:12].upper()   # TCGA-XX-YYYY
                # If multiple slides per patient, keep the first found
                if patient_barcode not in index:
                    index[patient_barcode] = p
            self.barcode_index[subtype] = index
            log.info(f"  {subtype} feature index: {len(index)} unique patient barcodes")

        # Match rows to h5 files
        # Note: Subtype label is inferred from feature directory
        records = []
        raw_gene_cols = [g + "_raw" for g in gene_cols
                         if g + "_raw" in df.columns]
        n_ambiguous = 0
        for _, row in df.iterrows():
            sid = row["submitter_id"] # 12-char: TCGA-XX-YYYY

            found = {
                subtype: index[sid]
                for subtype, index in self.barcode_index.items()
                if sid in index
            }

            if len(found) == 0:
                continue
            if len(found) > 1:
                # Same patient barcode present in more than one feature directory, so skip
                n_ambiguous += 1
                continue

            subtype, h5_path = next(iter(found.items()))
            rec = {
                "sid":      sid,
                "subtype":  subtype,
                "h5_path":  h5_path,
                "label":    row[gene_cols].values.astype(np.float32),
                "clinical": row[clinical_cols].values.astype(np.float32),
            }
            # label_raw only exists for _fpkm_uq columns (not APM/TIS/TMB)
            if raw_gene_cols:
                rec["label_raw"] = row[raw_gene_cols].values.astype(np.float32)
            else:
                rec["label_raw"] = rec["label"].copy()
            records.append(rec)

        self.records = records
        if n_ambiguous:
            log.warning(f"  Skipped {n_ambiguous} patient(s) found in "
                        f"BOTH LUAD and LUSC feature directories "
                        f"(subtype could not be determined unambiguously)")
        log.info(f"Dataset: {len(records)} matched slides "
                 f"(LUAD: {sum(1 for r in records if r['subtype']=='LUAD')}, "
                 f"LUSC: {sum(1 for r in records if r['subtype']=='LUSC')})")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        # In-memory cache lookup (see preload_feature_cache) instead of a
        # fresh h5py.File(...).read() + decompress on every access.
        features_np = _load_features_cached(rec["h5_path"])

        if features_np.ndim == 3 and features_np.shape[0] == 1:
            features_np = features_np[0]
        elif features_np.ndim != 2:
            raise ValueError(
                f"Unexpected features shape {features_np.shape} in "
                f"{rec['h5_path']} - expected (N_tiles, feat_dim)."
            )

        # Tile sampling. Cast to float32 here regardless of the cache's
        # storage dtype (float16 when configure_feature_cache(dtype=...) is
        # used to fit a larger corpus in a limited RAM budget) -- casting
        # AFTER subsampling, not before, means we only upcast the tiles we
        # actually keep.
        n = features_np.shape[0]
        if self.n_tiles is not None and self.n_tiles < n:
            if self.deterministic:
                rng    = np.random.default_rng(self.seed + idx)
                chosen = sorted(rng.choice(n, size=self.n_tiles, replace=False))
            else:
                chosen = np.random.permutation(n)[: self.n_tiles]
            sub = features_np[chosen]
        else:
            sub = features_np
        features = torch.from_numpy(sub.astype(np.float32, copy=True))

        label      = torch.tensor(rec["label"],     dtype=torch.float32)
        label_raw  = torch.tensor(rec["label_raw"], dtype=torch.float32)
        clinical   = torch.tensor(rec["clinical"],  dtype=torch.float32)
        subtype_id = torch.tensor(SUBTYPE_MAP[rec["subtype"]], dtype=torch.long)

        return features, label, label_raw, clinical, subtype_id, rec["sid"]


# Attention-MIL pooling
class AttentionMIL(nn.Module):
    """
    Additive attention pooling (Ilse et al., 2018).
    Learns which tiles matter most for the prediction.
    Outputs a single slide-level embedding + attention weights for visualisation.
    """
    def __init__(self, feat_dim: int = 1536, hidden_dim: int = 256):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
        )

    def forward(self, features):
        """
        features : (N_tiles, feat_dim)
        returns  : slide_embed (512,), attn_weights (N_tiles,)
        """
        assert features.ndim == 2, (
            f"AttentionMIL expects (N_tiles, feat_dim), got shape "
            f"{tuple(features.shape)}"
        )
        projected = self.feat_proj(features)            # (N, 512)
        raw_attn  = self.attention(features)            # (N, 1)
        attn      = torch.softmax(raw_attn, dim=0)     # (N, 1)  sums to 1
        slide_embed = (attn * projected).sum(dim=0)    # (512,)
        return slide_embed, attn.squeeze(-1)            # (512,), (N,)


# ---------------------------------------------------------------------------
# Alternative MIL aggregators (aggregator-architecture robustness sweep)
#
# Motivation: the FiLM contribution above sits on top of one specific choice
# of tile-pooling architecture (additive Attention-MIL). To claim that the
# subtype-conditioned biological finding is a property of the *modelling
# idea* rather than an artefact of that one pooling architecture, the same
# frozen UNI2-h tile embeddings, the same train/val/test split, and the same
# loss/training schedule are re-used while only the aggregator below the
# FiLM/clinical/head stack is swapped out. See `build_aggregator()` and
# `run_sweep()` for the sweep driver.
# ---------------------------------------------------------------------------

class CLAMGatedAttentionMIL(nn.Module):
    """
    CLAM-style single-branch gated-attention pooling (Lu et al., 2021,
    "Data-efficient and weakly supervised computational pathology on
    whole-slide images").

    Difference from AttentionMIL above: the attention logit for each tile is
    computed from a *gated* combination of a tanh branch and a sigmoid gate
    branch (V(x) * sigmoid(U(x))) instead of a single tanh MLP, which lets
    the network suppress uninformative tiles more sharply.

    Scope note: this reproduces CLAM's attention-pooling backbone only.
    Full CLAM additionally trains an auxiliary instance-level clustering
    loss on the top/bottom-k attended patches per class, which assumes a
    discrete classification target and has no natural analogue for our
    continuous 35-gene regression targets, so it is intentionally omitted.
    This isolates the *aggregation* architecture for a fair comparison
    against AttentionMIL/TransMIL/graph-MIL — the axis this sweep tests.
    """
    def __init__(self, feat_dim: int = 1536, hidden_dim: int = 256,
                 embed_dim: int = 512, dropout: float = 0.25):
        super().__init__()
        self.attention_V = nn.Sequential(nn.Linear(feat_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(feat_dim, hidden_dim), nn.Sigmoid())
        self.attention_w = nn.Linear(hidden_dim, 1)
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, features):
        """
        features : (N_tiles, feat_dim)
        returns  : slide_embed (embed_dim,), attn_weights (N_tiles,)
        """
        assert features.ndim == 2, (
            f"CLAMGatedAttentionMIL expects (N_tiles, feat_dim), got shape "
            f"{tuple(features.shape)}"
        )
        projected = self.feat_proj(features)                       # (N, embed_dim)
        gated_logits = self.attention_w(
            self.attention_V(features) * self.attention_U(features)
        )                                                            # (N, 1)
        attn = torch.softmax(gated_logits, dim=0)                   # (N, 1) sums to 1
        slide_embed = (attn * projected).sum(dim=0)                 # (embed_dim,)
        return slide_embed, attn.squeeze(-1)


class TransMILAggregator(nn.Module):
    """
    Compact Transformer MIL aggregator, in the spirit of TransMIL
    (Shao et al., 2021, "TransMIL: Transformer based Correlated Multiple
    Instance Learning for Whole Slide Image Classification"): tiles attend
    to one another via self-attention (correlated MIL) instead of being
    pooled independently of each other, and a learnable [CLS] token
    summarises the bag into a single slide embedding.

    Simplifications relative to the original TransMIL, made for tractability
    on single-GPU (Kaggle T4) training of individual WSI bags that can
    contain thousands of tiles:
      1. Standard scaled dot-product self-attention is used in place of the
         Nystrom approximation, so this module is O(N^2) in tile count.
         Use --agg_max_tiles to cap N for large bags.
      2. The Pyramid Position Encoding Generator (PPEG), which requires a
         registered 2D tile grid, is omitted, since tile spatial coordinates
         are not consumed elsewhere in this pipeline — tiles are treated as
         an unordered set, consistent with the other three aggregators in
         this sweep.
    """
    def __init__(self, feat_dim: int = 1536, embed_dim: int = 512,
                 n_heads: int = 8, n_layers: int = 2,
                 dim_feedforward: int = 512, dropout: float = 0.25):
        super().__init__()
        self.input_proj = nn.Linear(feat_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, features):
        """
        features : (N_tiles, feat_dim)
        returns  : slide_embed (embed_dim,), attn_weights (N_tiles,)
                   (attn_weights here is a cosine-similarity-to-CLS proxy for
                   visualisation, since nn.TransformerEncoder does not
                   expose its internal attention maps by default — it is not
                   used anywhere in training/loss.)
        """
        assert features.ndim == 2, (
            f"TransMILAggregator expects (N_tiles, feat_dim), got shape "
            f"{tuple(features.shape)}"
        )
        tokens = self.input_proj(features)                          # (N, embed_dim)
        tokens = torch.cat([self.cls_token, tokens], dim=0).unsqueeze(0)  # (1, N+1, embed_dim)
        encoded = self.encoder(tokens).squeeze(0)                    # (N+1, embed_dim)
        slide_embed = self.norm(encoded[0])                          # CLS token -> (embed_dim,)
        with torch.no_grad():
            tile_repr = encoded[1:]
            attn_proxy = F.cosine_similarity(
                tile_repr, slide_embed.unsqueeze(0).expand_as(tile_repr), dim=-1
            )
            attn_proxy = torch.softmax(attn_proxy, dim=0)
        return slide_embed, attn_proxy


class GraphMILAggregator(nn.Module):
    """
    Lightweight Graph-MIL aggregator. Tiles are treated as nodes of a
    feature-space k-nearest-neighbour graph (built by cosine similarity on
    the projected tile embeddings, since tile spatial coordinates are not
    consumed elsewhere in this pipeline), messages are passed with a small
    stack of residual mean-aggregation graph-conv layers, and the resulting
    node embeddings are pooled to a single slide embedding via an
    attention-pooling readout (same softmax-attention readout family as
    AttentionMIL/CLAM above, applied on top of the graph-propagated node
    features rather than the raw tile features).

    Implemented in plain PyTorch (no torch_geometric dependency) for
    reliability inside a Kaggle notebook environment. This is intentionally
    simpler than slide-graph literature such as Patch-GCN, which uses true
    spatial (not feature-space) adjacency and richer GNN layers; the goal
    here is to add a genuinely different message-passing inductive bias to
    the robustness sweep, not to reproduce a specific published model.
    """
    def __init__(self, feat_dim: int = 1536, embed_dim: int = 512,
                 hidden_dim: int = 256, k: int = 8, n_layers: int = 2,
                 dropout: float = 0.25):
        super().__init__()
        self.k = k
        self.input_proj = nn.Sequential(nn.Linear(feat_dim, hidden_dim), nn.ReLU())
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for _ in range(n_layers)
        ])
        self.out_proj = nn.Linear(hidden_dim, embed_dim)
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )

    def _knn_indices(self, x: torch.Tensor):
        """x : (N, hidden_dim). Returns (N, k) neighbour indices by cosine
        similarity (self excluded), or None if N <= 1."""
        n = x.shape[0]
        if n <= 1:
            return None
        with torch.no_grad():
            x_norm = F.normalize(x, dim=-1)
            sim = x_norm @ x_norm.t()                 # (N, N)
            sim.fill_diagonal_(-float("inf"))
            k = min(self.k, n - 1)
            _, idx = sim.topk(k, dim=-1)               # (N, k)
        return idx

    def forward(self, features):
        """
        features : (N_tiles, feat_dim)
        returns  : slide_embed (embed_dim,), attn_weights (N_tiles,)
        """
        assert features.ndim == 2, (
            f"GraphMILAggregator expects (N_tiles, feat_dim), got shape "
            f"{tuple(features.shape)}"
        )
        h = self.input_proj(features)                  # (N, hidden_dim)
        n = h.shape[0]
        idx = self._knn_indices(h)
        for layer in self.gnn_layers:
            neighbor_mean = h[idx].mean(dim=1) if idx is not None else h  # (N, hidden_dim)
            h = layer(torch.cat([h, neighbor_mean], dim=-1)) + h          # residual message passing
        node_embed = self.out_proj(h)                    # (N, embed_dim)
        attn_logits = self.attention(node_embed)          # (N, 1)
        attn = torch.softmax(attn_logits, dim=0)
        slide_embed = (attn * node_embed).sum(dim=0)      # (embed_dim,)
        return slide_embed, attn.squeeze(-1)


AGGREGATOR_CHOICES = ["abmil", "clam", "transmil", "graph"]


def build_aggregator(name: str, feat_dim: int, embed_dim: int,
                      hidden_dim: int = 256, dropout: float = 0.25,
                      **kwargs) -> nn.Module:
    """
    Factory for the aggregator-architecture robustness sweep. Every
    aggregator below shares the exact same interface:
        forward(features: (N_tiles, feat_dim)) -> (slide_embed: (embed_dim,),
                                                     attn_weights: (N_tiles,))
    so FiLMMILModel (and everything downstream of it — FiLM conditioning,
    clinical concatenation, the gene/panel heads, the loss, run_epoch) is
    completely agnostic to which aggregator is plugged in.

    name : one of AGGREGATOR_CHOICES
        "abmil"    - baseline additive Attention-MIL (Ilse et al., 2018)
        "clam"     - CLAM-style gated-attention pooling (Lu et al., 2021)
        "transmil" - compact Transformer / self-attention MIL, TransMIL-style
                     (Shao et al., 2021)
        "graph"    - k-NN graph-MIL with residual message passing + attention
                     readout
    """
    name = name.lower()
    if name == "abmil":
        return AttentionMIL(feat_dim=feat_dim, hidden_dim=hidden_dim)
    elif name == "clam":
        return CLAMGatedAttentionMIL(
            feat_dim=feat_dim, hidden_dim=hidden_dim, embed_dim=embed_dim, dropout=dropout,
        )
    elif name == "transmil":
        return TransMILAggregator(
            feat_dim=feat_dim,
            embed_dim=embed_dim,
            n_heads=kwargs.get("transmil_heads", 8),
            n_layers=kwargs.get("transmil_layers", 2),
            dim_feedforward=kwargs.get("transmil_dim_feedforward", 512),
            dropout=dropout,
        )
    elif name == "graph":
        return GraphMILAggregator(
            feat_dim=feat_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            k=kwargs.get("graph_k", 8),
            n_layers=kwargs.get("graph_layers", 2),
            dropout=dropout,
        )
    else:
        raise ValueError(
            f"Unknown aggregator '{name}'. Choose from: {AGGREGATOR_CHOICES}"
        )


# FiLM conditioning layer
class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (Perez et al., 2018).
    Conditions slide embeddings on cancer subtype via learned affine transform:
        output = gamma(subtype) * embedding + beta(subtype)

    This allows each feature channel to be independently scaled and shifted
    based on the subtype signal, giving the model a fine-grained pathway
    to learn subtype-specific immune morphology patterns without requiring
    fully separate models per subtype.

    Novelty in this context: first application of FiLM conditioning to
    subtype-aware immune gene expression prediction from histopathology.
    """
    def __init__(self, embed_dim: int = 512, n_subtypes: int = 2):
        super().__init__()
        self.subtype_embed = nn.Embedding(n_subtypes, embed_dim)
        # gamma and beta project the subtype embedding to scale/shift vectors
        self.gamma = nn.Linear(embed_dim, embed_dim)
        self.beta  = nn.Linear(embed_dim, embed_dim)
        # Initialise close to identity: gamma≈1, beta≈0
        nn.init.ones_(self.gamma.weight.data.fill_diagonal_(1))
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, slide_embed: torch.Tensor, subtype_id: torch.Tensor):
        """
        slide_embed : (512,) or (B, 512)
        subtype_id  : scalar long or (B,) long
        returns     : modulated embedding, same shape as slide_embed
        """
        s     = self.subtype_embed(subtype_id)   # (512,) or (B, 512)
        gamma = self.gamma(s)                    # learned scale per channel
        beta  = self.beta(s)                     # learned shift per channel
        return gamma * slide_embed + beta        # element-wise affine


# Full model
class FiLMMILModel(nn.Module):
    """
    End-to-end FiLM-conditioned MIL model.

    Pipeline:
        1. Check if FiLM conditioning is enabled
        2. Aggregator:    (N_tiles, 1536) -> (512,) slide embedding. The
                           pooling architecture is pluggable — see
                           `aggregator` / `build_aggregator()` — and
                           defaults to the original additive Attention-MIL
                           for full backward compatibility with existing
                           FiLM-vs-no-FiLM checkpoints and results.
        3. FiLM:          condition slide embedding on subtype (LUAD/LUSC)
        4. Clinical:      project [age_z, gender] -> (64,), concatenate
        5. Regression:    (512+64,) -> (n_genes,)
        6. Panel head:    (512+64,) -> (2,)  [APM, TIS] direct prediction, a separate small head from the 35-gene head above, so APM/TIS can be evaluated either as the mean of the individually-predicted genes (existing metric) or as this head's direct output (new metric).
    """
    def __init__(
        self,
        feat_dim:     int = 1536,
        embed_dim:    int = 512,
        clinical_dim: int = 2,
        n_clinical:   int = 64,
        n_genes:      int = 35,
        n_subtypes:   int = 2,
        dropout:      float = 0.25,
        *,
        use_film:      bool,
        use_panel_head: bool = True,
        aggregator:    str = "abmil",
        agg_hidden_dim: int = 256,
        graph_k:       int = 8,
        graph_layers:  int = 2,
        transmil_heads: int = 8,
        transmil_layers: int = 2,
        transmil_dim_feedforward: int = 512,
    ):
        super().__init__()

        #1. Check if FiLM conditioning is enabled
        self.use_film = use_film
        self.use_panel_head = use_panel_head
        self.aggregator_name = aggregator

        # 2. MIL aggregator / pooling architecture (aggregator-architecture
        # robustness sweep — see build_aggregator() docstring for the full
        # list and the scope/simplification notes for clam/transmil/graph).
        # Kept under the historical attribute name `attention_mil` so that
        # aggregator="abmil" (the default) remains checkpoint-compatible
        # with results produced before this option existed.
        self.attention_mil = build_aggregator(
            aggregator,
            feat_dim=feat_dim,
            embed_dim=embed_dim,
            hidden_dim=agg_hidden_dim,
            dropout=dropout,
            graph_k=graph_k,
            graph_layers=graph_layers,
            transmil_heads=transmil_heads,
            transmil_layers=transmil_layers,
            transmil_dim_feedforward=transmil_dim_feedforward,
        )

        # 3. FiLM subtype conditioning
        if self.use_film:
            self.film = FiLMLayer(embed_dim=embed_dim, n_subtypes=n_subtypes)

        # 4. Clinical projection
        self.clinical_proj = nn.Sequential(
            nn.Linear(clinical_dim, n_clinical),
            nn.ReLU(),
        )

        # 5. Regression head (35 individual gene/TMB targets)
        self.head = nn.Sequential(
            nn.Linear(embed_dim + n_clinical, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_genes),
        )

        # 6. Direct APM/TIS panel head — a separate, smaller MLP off the same slide embedding, trained to directly regress the two panel scores rather than deriving them from individual gene outputs.
        if self.use_panel_head:
            self.panel_head = nn.Sequential(
                nn.Linear(embed_dim + n_clinical, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, len(PANEL_TARGETS)),  # [APM, TIS]
            )

    def forward(self, features, clinical, subtype_id):
        """
        features   : (N_tiles, feat_dim)
        clinical   : (clinical_dim,)
        subtype_id : scalar long tensor (0=LUAD, 1=LUSC)

        returns    : predictions (n_genes,), attn_weights (N_tiles,),
                     panel_preds (2,) or None if use_panel_head=False
        """
        # 1. Attention pooling -> slide embedding
        slide_embed, attn_weights = self.attention_mil(features)  # (512,), (N,)

        # 2. FiLM conditioning on subtype
        if self.use_film:
            slide_embed = self.film(slide_embed, subtype_id)          # (512,)

        # 3. Clinical covariates
        clin_embed  = self.clinical_proj(clinical)                # (64,)

        # 4. Concatenate and predict
        combined = torch.cat([slide_embed, clin_embed], dim=-1)  # (576,)
        preds    = self.head(combined)                            # (35,)

        # 5. Direct panel-head prediction (independent of the 35-gene head)
        panel_preds = self.panel_head(combined) if self.use_panel_head else None  # (2,)

        return preds, attn_weights, panel_preds


# Loss function (MSE + PCC + Var)
class CompositeLoss(nn.Module):
    """
    L = alpha * MSE + beta * (1 - PCC) + gamma * (-Var)
    Two-phase training schedule:
        Epochs  1-25: alpha=1,   beta=0,   gamma=0  (MSE only, stabilise)
        Epochs 26+:   alpha=0.4, beta=0.2, gamma=0.4 (full composite, weights adjusted empirically)
    """
    def __init__(self):
        super().__init__()

    def pearson_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """1 - PCC, computed across the batch for each gene, then averaged."""
        # pred, target: (B, n_genes)
        pred_m   = pred   - pred.mean(dim=0)
        target_m = target - target.mean(dim=0)
        num  = (pred_m * target_m).sum(dim=0)
        denom = (pred_m.pow(2).sum(dim=0) * target_m.pow(2).sum(dim=0)).sqrt() + 1e-8
        pcc  = num / denom
        return (1 - pcc).mean()

    def forward(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
        epoch:  int,
        panel_pred:   torch.Tensor | None = None,
        panel_target: torch.Tensor | None = None,
        panel_weight: float = 0.5,
    ) -> tuple[torch.Tensor, dict]:

        mse = F.mse_loss(pred, target)
        pcc_loss = self.pearson_loss(pred, target)
        var_loss = -pred.var(dim=0).mean()   # penalise low-variance predictions

        if epoch <= 25:
            alpha, beta, gamma = 1.0, 0.0, 0.0
        else:
            alpha, beta, gamma = 0.4, 0.2, 0.4

        loss = alpha * mse + beta * pcc_loss + gamma * var_loss

        panel_mse = None
        if panel_pred is not None and panel_target is not None:
            # Same MSE + (1-PCC) composite as the main head, applied to the direct 2-value [APM, TIS] output. Added on top of the main gene-head loss so both heads are trained jointly.
            panel_mse = F.mse_loss(panel_pred, panel_target)
            panel_pcc_loss = self.pearson_loss(panel_pred, panel_target)
            if epoch <= 25:
                panel_loss = panel_mse
            else:
                panel_loss = 0.7 * panel_mse + 0.3 * panel_pcc_loss
            loss = loss + panel_weight * panel_loss

        return loss, {
            "mse":     mse.item(),
            "pcc":     (1 - pcc_loss).item(),
            "var":     (-var_loss).item(),
            "panel_mse": panel_mse.item() if panel_mse is not None else None,
            "total":   loss.item(),
        }


# Evaluation helpers
def compute_gene_pccs(all_preds: np.ndarray, all_labels: np.ndarray) -> np.ndarray:
    """PCC for each gene across samples. Returns (n_genes,)."""
    n_genes = all_preds.shape[1]
    pccs = []
    for g in range(n_genes):
        r, _ = pearsonr(all_preds[:, g], all_labels[:, g])
        pccs.append(r if not np.isnan(r) else 0.0)
    return np.array(pccs)


def compute_panel_pcc(all_preds: np.ndarray, all_labels: np.ndarray,
                      gene_cols: list, panel_genes: list,
                      gene_symbol_to_idx: dict) -> float:
    """
    Signature-level PCC:
    average expression across panel genes first, then correlate.
    """
    idxs = [gene_symbol_to_idx[g] for g in panel_genes if g in gene_symbol_to_idx]
    if not idxs:
        return 0.0
    pred_score  = all_preds[:, idxs].mean(axis=1)
    label_score = all_labels[:, idxs].mean(axis=1)
    r, _ = pearsonr(pred_score, label_score)
    return r if not np.isnan(r) else 0.0


def compute_panel_head_pcc(all_panel_preds: np.ndarray, all_panel_labels: np.ndarray) -> dict:
    """
    Direct-head PCC: correlate the panel head's own [APM, TIS] output
    against the true APM/TIS values (no averaging of individual genes).

    all_panel_preds, all_panel_labels : (n_samples, 2), columns ordered
    to match PANEL_TARGETS = ["APM", "TIS"].
    """
    results = {}
    for i, name in enumerate(PANEL_TARGETS):
        r, _ = pearsonr(all_panel_preds[:, i], all_panel_labels[:, i])
        # Cast away from numpy float32/float64 (pearsonr preserves input
        # dtype, and predictions/labels here originate from float32 torch
        # tensors) so downstream json.dump of fold_results never chokes on
        # a non-native-float scalar.
        results[name] = float(r) if not np.isnan(r) else 0.0
    return results


def compute_auc(all_preds: np.ndarray, all_labels: np.ndarray,
                gene_cols: list, panel_genes: list,
                gene_symbol_to_idx: dict) -> float:
    """Upper-quartile binary AUC for a panel."""
    idxs = [gene_symbol_to_idx[g] for g in panel_genes if g in gene_symbol_to_idx]
    if not idxs:
        return 0.5
    pred_score  = all_preds[:, idxs].mean(axis=1)
    label_score = all_labels[:, idxs].mean(axis=1)
    threshold   = np.percentile(label_score, 75)
    binary_true = (label_score >= threshold).astype(int)
    if binary_true.sum() == 0 or binary_true.sum() == len(binary_true):
        return 0.5
    return roc_auc_score(binary_true, pred_score)


# Training and evaluation loops
def run_epoch(
    model, loader, loss_fn, optimizer, epoch, device, training=True,
    panel_idx: dict | None = None,
    panel_gene_fallback_idx: dict | None = None,
):
    """
    panel_idx               : {"APM": idx_in_label, "TIS": idx_in_label} for whichever literal panel columns exist in the label vector. Used as the ground truth for the direct panel head.
    panel_gene_fallback_idx  : {"APM": [gene indices...], "TIS": [...]} used to build a mean-of-genes ground truth for any panel missing from panel_idx.
    """
    model.train() if training else model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []      # detached copies, for metrics/return only
    all_panel_preds, all_panel_labels = [], []
    batch_preds, batch_labels = [], []
    batch_panel_preds, batch_panel_labels = [], []

    panel_idx = panel_idx or {}
    panel_gene_fallback_idx = panel_gene_fallback_idx or {}
    use_panel_head = getattr(model, "use_panel_head", False)

    def _panel_target(label: torch.Tensor) -> torch.Tensor:
        """Build the (2,) [APM, TIS] ground-truth vector for one sample."""
        vals = []
        for name in PANEL_TARGETS:
            if name in panel_idx:
                vals.append(label[panel_idx[name]])
            elif name in panel_gene_fallback_idx and panel_gene_fallback_idx[name]:
                vals.append(label[panel_gene_fallback_idx[name]].mean())
            else:
                vals.append(torch.tensor(0.0, device=label.device))
        return torch.stack(vals)

    def _step(preds_list, labels_list, panel_preds_list, panel_labels_list):
        pred_batch  = torch.stack(preds_list)
        label_batch = torch.stack(labels_list)
        panel_pred_batch  = torch.stack(panel_preds_list)  if panel_preds_list  else None
        panel_label_batch = torch.stack(panel_labels_list) if panel_labels_list else None
        loss, _ = loss_fn(pred_batch, label_batch, epoch,
                           panel_pred=panel_pred_batch, panel_target=panel_label_batch)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in loader:
            features, label, _, clinical, subtype_id, _ = batch

            # All inputs: single slide (batch_size=1), squeeze batch dim
            features   = features.squeeze(0).to(device, non_blocking=True)    # (N_tiles, 1536)
            label      = label.squeeze(0).to(device, non_blocking=True)       # (35,)
            clinical   = clinical.squeeze(0).to(device, non_blocking=True)    # (2,)
            subtype_id = subtype_id.squeeze(0).to(device, non_blocking=True)  # scalar

            pred, _, panel_pred = model(features, clinical, subtype_id)  # (35,), (2,) or None

            # Detached copies for logging / PCC metrics
            all_preds.append(pred.detach().cpu())
            all_labels.append(label.detach().cpu())

            panel_target = None
            if use_panel_head and panel_pred is not None:
                panel_target = _panel_target(label)
                all_panel_preds.append(panel_pred.detach().cpu())
                all_panel_labels.append(panel_target.detach().cpu())

            # Per-sample MSE contribution to total loss (metrics only)
            mse = F.mse_loss(pred, label)
            total_loss += mse.item()

            if training:
                batch_preds.append(pred)
                batch_labels.append(label)
                if use_panel_head and panel_pred is not None:
                    batch_panel_preds.append(panel_pred)
                    batch_panel_labels.append(panel_target)

                if len(batch_preds) % 16 == 0:
                    _step(batch_preds, batch_labels, batch_panel_preds, batch_panel_labels)
                    batch_preds, batch_labels = [], []
                    batch_panel_preds, batch_panel_labels = [], []

        # Note: This flushes any leftover slides (< 16) so the tail of the epoch still contributes a gradient update instead of being ignored
        if training and batch_preds:
            _step(batch_preds, batch_labels, batch_panel_preds, batch_panel_labels)

    preds  = torch.stack(all_preds).numpy()
    labels = torch.stack(all_labels).numpy()
    mean_pcc = compute_gene_pccs(preds, labels).mean()

    panel_preds_arr  = torch.stack(all_panel_preds).numpy()  if all_panel_preds  else None
    panel_labels_arr = torch.stack(all_panel_labels).numpy() if all_panel_labels else None

    return total_loss / len(loader), mean_pcc, preds, labels, panel_preds_arr, panel_labels_arr


# Main training script
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    df, gene_cols, clinical_cols, _, _, panel_idx = load_metadata(args.metadata)

    # Build gene symbol -> index mapping for panel evaluation.
    # Strips _fpkm_uq suffix and leaves TMB, APM, and TIS unaltered.
    gene_symbol_to_idx = {}
    for i, g in enumerate(gene_cols):
        symbol = g.replace("_fpkm_uq", "")   # e.g. HLA-A_fpkm_uq -> HLA-A
        gene_symbol_to_idx[symbol] = i        # TMB, APM, TIS kept unchanged
    log.info(f"Predicting {len(gene_cols)} targets: "
             f"{len([g for g in gene_cols if g.endswith('_fpkm_uq')])} genes "
             f"+ TMB + APM + TIS")

    # Fallback ground truth (mean of individual panel genes) for the direct panel head, used only for whichever of APM/TIS has no literal column.
    panel_gene_fallback_idx = {
        "APM": [gene_symbol_to_idx[g] for g in APM_GENES if g in gene_symbol_to_idx],
        "TIS": [gene_symbol_to_idx[g] for g in TIS_GENES if g in gene_symbol_to_idx],
    }

    # Feature directories
    feature_dirs = {
        "LUAD": args.luad_features,
        "LUSC": args.lusc_features,
    }

    # Fixed test set (20%)
    rng = np.random.default_rng(98)
    all_sids = df["submitter_id"].unique()
    test_sids = set(rng.choice(all_sids, size=int(0.2 * len(all_sids)), replace=False))
    dev_sids  = [s for s in all_sids if s not in test_sids]

    df_test = df[df["submitter_id"].isin(test_sids)].reset_index(drop=True)
    df_dev  = df[df["submitter_id"].isin(dev_sids)].reset_index(drop=True)

    log.info(f"Dev: {len(df_dev)} slides | Test: {len(df_test)} slides")

    # Configure + preload the feature cache. If --feature_cache_max_gb is
    # unset, this preserves the original uncapped behaviour (only safe if
    # the corpus comfortably fits in RAM). If set (e.g. on a 30GB-RAM
    # session against a much larger combined LUAD+LUSC corpus), the cache
    # fills up to that budget and stops -- no eviction/thrashing, whatever
    # doesn't fit is just read from disk as before. The fixed test set is
    # prioritised first since it's re-read identically by every fold and
    # (in the aggregator sweep) every aggregator.
    configure_feature_cache(
        max_gb=getattr(args, "feature_cache_max_gb", None),
        dtype=getattr(args, "feature_cache_dtype", "float32"),
    )
    _test_paths_for_priority = FiLMDataset(
        df_test, feature_dirs, gene_cols, clinical_cols
    ).records
    preload_feature_cache(
        feature_dirs,
        priority_paths=[r["h5_path"] for r in _test_paths_for_priority],
    )

    # Aggregator configuration for this run (defaults preserve the original
    # AttentionMIL behaviour when these CLI flags are absent, e.g. when
    # `args` is a plain argparse.Namespace built before this option existed).
    aggregator = getattr(args, "aggregator", "abmil")
    agg_hidden_dim = getattr(args, "agg_hidden_dim", 256)
    agg_max_tiles = getattr(args, "agg_max_tiles", None)
    graph_k = getattr(args, "graph_k", 8)
    graph_layers = getattr(args, "graph_layers", 2)
    transmil_heads = getattr(args, "transmil_heads", 8)
    transmil_layers = getattr(args, "transmil_layers", 2)
    transmil_dim_feedforward = getattr(args, "transmil_dim_feedforward", 512)
    log.info(f"Aggregator: {aggregator}"
             + (f" (agg_max_tiles={agg_max_tiles})" if agg_max_tiles else ""))

    # 5-fold cross validation
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=98)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(df_dev)):
        log.info(f"\n{'='*60}")
        log.info(f"FOLD {fold} / {args.n_folds - 1}")
        log.info(f"{'='*60}")

        df_train = df_dev.iloc[train_idx].reset_index(drop=True)
        df_val   = df_dev.iloc[val_idx].reset_index(drop=True)

        # Datasets. agg_max_tiles caps tiles/slide — strongly recommended
        # for the transmil/graph aggregators, whose cost scales ~O(N^2) in
        # tile count, but applied uniformly across aggregators when set so
        # the comparison stays apples-to-apples on the same input bags.
        train_ds = FiLMDataset(df_train, feature_dirs, gene_cols, clinical_cols,
                               n_tiles=agg_max_tiles, deterministic=False)
        val_ds   = FiLMDataset(df_val,   feature_dirs, gene_cols, clinical_cols,
                               n_tiles=agg_max_tiles, deterministic=True)
        test_ds  = FiLMDataset(df_test,  feature_dirs, gene_cols, clinical_cols,
                               n_tiles=agg_max_tiles, deterministic=True)

        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,  num_workers=4,
                                  pin_memory=True, persistent_workers=True, prefetch_factor=4)
        val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, num_workers=2,
                                  pin_memory=True, persistent_workers=True, prefetch_factor=4)
        test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False, num_workers=2,
                                  pin_memory=True, persistent_workers=True, prefetch_factor=4)

        # Model
        model    = FiLMMILModel(
            feat_dim=1536, n_genes=len(gene_cols), use_film=args.use_film,
            aggregator=aggregator, agg_hidden_dim=agg_hidden_dim,
            graph_k=graph_k, graph_layers=graph_layers,
            transmil_heads=transmil_heads, transmil_layers=transmil_layers,
            transmil_dim_feedforward=transmil_dim_feedforward,
        ).to(device)
        loss_fn  = CompositeLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        try:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=5, factor=0.5, verbose=True
            )
        except TypeError:
            # `verbose` was removed from ReduceLROnPlateau in newer PyTorch
            # releases; fall back silently so the sweep still runs.
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=5, factor=0.5
            )

        # Training loop with early stopping
        best_val_pcc  = -np.inf
        best_weights  = None
        patience_ctr  = 0

        for epoch in range(1, args.max_epochs + 1):
            train_loss, train_pcc, _, _, _, _ = run_epoch(
                model, train_loader, loss_fn, optimizer, epoch, device, training=True,
                panel_idx=panel_idx, panel_gene_fallback_idx=panel_gene_fallback_idx,
            )
            val_loss, val_pcc, _, _, _, _ = run_epoch(
                model, val_loader, loss_fn, optimizer, epoch, device, training=False,
                panel_idx=panel_idx, panel_gene_fallback_idx=panel_gene_fallback_idx,
            )
            scheduler.step(val_loss)

            log.info(
                f"Epoch {epoch:3d} | "
                f"Train loss {train_loss:.4f}  PCC {train_pcc:.4f} | "
                f"Val loss {val_loss:.4f}  PCC {val_pcc:.4f}"
            )

            if val_pcc > best_val_pcc:
                best_val_pcc = val_pcc
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= args.patience:
                    log.info(f"Early stopping at epoch {epoch}")
                    break

        # Evaluate on held-out test set
        model.load_state_dict(best_weights)
        model.to(device)
        _, _, test_preds, test_labels, test_panel_preds, test_panel_labels = run_epoch(
            model, test_loader, loss_fn, optimizer, 999, device, training=False,
            panel_idx=panel_idx, panel_gene_fallback_idx=panel_gene_fallback_idx,
        )

        # Gene-level PCCs
        gene_pccs = compute_gene_pccs(test_preds, test_labels)

        # Panel-level metrics (LUAD + LUSC combined, then subtype-split below)
        # (A) Mean-of-predicted-genes PCC — existing method: average the model's 18 individual gene predictions, correlate against the average of the 18 true (z-scored) gene values.
        apm_pcc  = compute_panel_pcc(test_preds, test_labels, gene_cols, APM_GENES, gene_symbol_to_idx)
        tis_pcc  = compute_panel_pcc(test_preds, test_labels, gene_cols, TIS_GENES, gene_symbol_to_idx)
        apm_auc  = compute_auc(test_preds, test_labels, gene_cols, APM_GENES, gene_symbol_to_idx)
        tis_auc  = compute_auc(test_preds, test_labels, gene_cols, TIS_GENES, gene_symbol_to_idx)

        # (B) Direct panel-head PCC: the model's own dedicated APM/TIS head, correlated against the true APM/TIS values (literal CSV columns when available, else mean-of-genes as the fallback ground truth — see panel_gene_fallback_idx above).
        if test_panel_preds is not None:
            head_pccs = compute_panel_head_pcc(test_panel_preds, test_panel_labels)
            apm_head_pcc = head_pccs["APM"]
            tis_head_pcc = head_pccs["TIS"]
        else:
            apm_head_pcc = tis_head_pcc = None

        # Subtype-split evaluation (your core biological finding)
        luad_mask = np.array([
            r["subtype"] == "LUAD" for r in test_ds.records
        ])
        lusc_mask = ~luad_mask

        subtype_results = {}
        for name, mask in [("LUAD", luad_mask), ("LUSC", lusc_mask)]:
            if mask.sum() == 0:
                continue
            p, l = test_preds[mask], test_labels[mask]
            # float(...) here (not just at the top-level fold_result below)
            # because pearsonr/roc_auc_score preserve the float32 dtype of
            # our torch-derived arrays, which json.dump cannot serialize.
            res = {
                "APM_PCC": float(compute_panel_pcc(p, l, gene_cols, APM_GENES, gene_symbol_to_idx)),
                "TIS_PCC": float(compute_panel_pcc(p, l, gene_cols, TIS_GENES, gene_symbol_to_idx)),
                "APM_AUC": float(compute_auc(p, l, gene_cols, APM_GENES, gene_symbol_to_idx)),
                "TIS_AUC": float(compute_auc(p, l, gene_cols, TIS_GENES, gene_symbol_to_idx)),
                "n":       int(mask.sum()),
            }
            if test_panel_preds is not None:
                head_res = compute_panel_head_pcc(test_panel_preds[mask], test_panel_labels[mask])
                res["APM_head_PCC"] = head_res["APM"]
                res["TIS_head_PCC"] = head_res["TIS"]
            subtype_results[name] = res

        fold_result = {
            "fold":           fold,
            "best_val_pcc":   float(best_val_pcc),
            # Mean-of-predicted-genes panel PCC (existing method)
            "APM_PCC":        float(apm_pcc),
            "TIS_PCC":        float(tis_pcc),
            "APM_AUC":        float(apm_auc),
            "TIS_AUC":        float(tis_auc),
            # Direct panel-head PCC: head output vs true APM/TIS
            "APM_head_PCC":   float(apm_head_pcc) if apm_head_pcc is not None else None,
            "TIS_head_PCC":   float(tis_head_pcc) if tis_head_pcc is not None else None,
            "gene_pccs":      {
                g.replace("_fpkm_uq", ""): float(gene_pccs[i])
                for i, g in enumerate(gene_cols)
            },
            "subtype":        subtype_results,
            "use_film": args.use_film,
            "aggregator": aggregator,
        }
        fold_results.append(fold_result)

        log.info(f"\nFold {fold} Test Results:")
        log.info(f"  APM  meanGenePCC={apm_pcc:.4f}  headPCC={apm_head_pcc if apm_head_pcc is not None else float('nan'):.4f}  AUC={apm_auc:.4f}")
        log.info(f"  TIS  meanGenePCC={tis_pcc:.4f}  headPCC={tis_head_pcc if tis_head_pcc is not None else float('nan'):.4f}  AUC={tis_auc:.4f}")
        for name, res in subtype_results.items():
            log.info(f"  {name} (n={res['n']}): APM meanGenePCC={res['APM_PCC']:.4f}"
                     f" headPCC={res.get('APM_head_PCC', float('nan')):.4f}, "
                     f"TIS meanGenePCC={res['TIS_PCC']:.4f} headPCC={res.get('TIS_head_PCC', float('nan')):.4f}")

        # Save model weights for this fold
        torch.save(
            best_weights,
            output_dir / f"fold{fold}_best_model.pt"
        )

    # Summary across folds
    log.info(f"\n{'='*60}")
    log.info("CROSS-VALIDATION SUMMARY")
    log.info(f"{'='*60}")
    for metric in ["APM_PCC", "TIS_PCC", "APM_AUC", "TIS_AUC", "APM_head_PCC", "TIS_head_PCC"]:
        vals = [r[metric] for r in fold_results if r.get(metric) is not None]
        if vals:
            log.info(f"  {metric}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    for subtype in ["LUAD", "LUSC"]:
        for metric in ["APM_PCC", "TIS_PCC", "APM_AUC", "TIS_AUC", "APM_head_PCC", "TIS_head_PCC"]:
            vals = [r["subtype"][subtype][metric]
                    for r in fold_results if subtype in r["subtype"]]
            if vals:
                log.info(f"  {subtype} {metric}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # Save all results
    with open(output_dir / "results.json", "w") as f:
        json.dump(fold_results, f, indent=2)
    log.info(f"\nResults saved to {output_dir / 'results.json'}")

    return fold_results


# ---------------------------------------------------------------------------
# Aggregator-architecture robustness sweep driver
# ---------------------------------------------------------------------------
def run_sweep(args):
    """
    Runs the full n_folds x train/val/test pipeline once per aggregator in
    `args.aggregator_sweep` (comma-separated, e.g. "abmil,clam,transmil,graph"),
    holding everything else fixed — data split, FiLM setting, loss, training
    schedule, epochs/patience — so the only variable across runs is the
    pooling/aggregation architecture applied to the same frozen UNI2-h tile
    embeddings. Each aggregator's full results (per-fold, per-gene, per-
    subtype) are written to `<output_dir>/<aggregator>/results.json` exactly
    as `train()` normally would, and a combined summary table is written to
    `<output_dir>/aggregator_sweep_summary.{json,csv}` for direct use as a
    robustness table/figure in the paper.
    """
    import copy
    import csv

    aggregators = [a.strip() for a in args.aggregator_sweep.split(",") if a.strip()]
    unknown = [a for a in aggregators if a not in AGGREGATOR_CHOICES]
    if unknown:
        raise ValueError(
            f"Unknown aggregator(s) in --aggregator_sweep: {unknown}. "
            f"Choose from: {AGGREGATOR_CHOICES}"
        )

    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    metrics = ["APM_PCC", "TIS_PCC", "APM_AUC", "TIS_AUC", "APM_head_PCC", "TIS_head_PCC"]
    sweep_summary = []
    all_fold_results = {}

    for agg in aggregators:
        log.info(f"\n{'#'*70}")
        log.info(f"# AGGREGATOR SWEEP: {agg}")
        log.info(f"{'#'*70}")

        run_args = copy.deepcopy(args)
        run_args.aggregator = agg
        run_args.output_dir = str(base_output_dir / agg)

        fold_results = train(run_args)
        all_fold_results[agg] = fold_results

        row = {"aggregator": agg, "n_folds": len(fold_results)}
        for metric in metrics:
            vals = [r[metric] for r in fold_results if r.get(metric) is not None]
            if vals:
                row[f"{metric}_mean"] = float(np.mean(vals))
                row[f"{metric}_std"] = float(np.std(vals))
        n_params = sum(
            p.numel() for p in
            FiLMMILModel(
                feat_dim=1536, n_genes=1, use_film=args.use_film, aggregator=agg,
                agg_hidden_dim=getattr(args, "agg_hidden_dim", 256),
                graph_k=getattr(args, "graph_k", 8),
                graph_layers=getattr(args, "graph_layers", 2),
                transmil_heads=getattr(args, "transmil_heads", 8),
                transmil_layers=getattr(args, "transmil_layers", 2),
                transmil_dim_feedforward=getattr(args, "transmil_dim_feedforward", 512),
            ).parameters()
        )
        row["n_params"] = n_params
        sweep_summary.append(row)

    with open(base_output_dir / "aggregator_sweep_summary.json", "w") as f:
        json.dump(sweep_summary, f, indent=2)

    if sweep_summary:
        fieldnames = list(sweep_summary[0].keys())
        with open(base_output_dir / "aggregator_sweep_summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sweep_summary)

    log.info(f"\n{'='*70}")
    log.info("AGGREGATOR ROBUSTNESS SWEEP SUMMARY")
    log.info(f"{'='*70}")
    for row in sweep_summary:
        log.info(
            f"  {row['aggregator']:>10s} | "
            f"APM_PCC={row.get('APM_PCC_mean', float('nan')):.4f}±{row.get('APM_PCC_std', 0.0):.4f} | "
            f"TIS_PCC={row.get('TIS_PCC_mean', float('nan')):.4f}±{row.get('TIS_PCC_std', 0.0):.4f} | "
            f"params={row.get('n_params', 'NA')}"
        )
    log.info(f"\nSummary saved to {base_output_dir / 'aggregator_sweep_summary.json'} and .csv")

    return sweep_summary, all_fold_results


# CLI
def parse_args():
    p = argparse.ArgumentParser(description="FiLM-conditioned MIL for immune gene prediction")
    p.add_argument("--luad_features", required=True,
                   help="Directory of LUAD UNI2-h .h5 feature files")
    p.add_argument("--lusc_features", required=True,
                   help="Directory of LUSC UNI2-h .h5 feature files")
    p.add_argument("--metadata",      required=True,
                   help="Combined expression + clinical CSV (same as your classifier)")
    p.add_argument("--output_dir",    default="./results",
                   help="Where to save model weights + results")
    p.add_argument("--n_folds",       type=int, default=5)
    p.add_argument("--max_epochs",    type=int, default=200)
    p.add_argument("--patience",      type=int, default=10,
                   help="Early stopping patience (epochs without val PCC improvement)")
    p.add_argument("--use_film",
                   help="Enable FiLM subtype conditioning.")

    # Aggregator-architecture robustness sweep
    p.add_argument("--aggregator", choices=AGGREGATOR_CHOICES, default="abmil",
                   help="MIL pooling/aggregation architecture applied to the frozen "
                        "UNI2-h tile embeddings, below FiLM/clinical/heads. "
                        f"Choices: {AGGREGATOR_CHOICES}.")
    p.add_argument("--aggregator_sweep", default=None,
                   help="Comma-separated list of aggregators (e.g. "
                        "'abmil,clam,transmil,graph') to run the full training "
                        "pipeline over, with FiLM/data/loss held fixed, producing "
                        "a combined robustness summary. Overrides --aggregator "
                        "and switches the script into sweep mode when set.")
    p.add_argument("--agg_hidden_dim", type=int, default=256,
                   help="Hidden width used inside the chosen aggregator "
                        "(attention MLP / gated-attention MLP / graph-conv width).")
    p.add_argument("--agg_max_tiles", type=int, default=None,
                   help="Optional cap on tiles sampled per slide before pooling. "
                        "Strongly recommended for --aggregator transmil/graph on "
                        "large bags, since both scale roughly O(N^2) in tile count.")
    p.add_argument("--graph_k", type=int, default=8,
                   help="[graph aggregator] number of nearest neighbours per tile node.")
    p.add_argument("--graph_layers", type=int, default=2,
                   help="[graph aggregator] number of residual graph-conv layers.")
    p.add_argument("--transmil_heads", type=int, default=8,
                   help="[transmil aggregator] number of self-attention heads.")
    p.add_argument("--transmil_layers", type=int, default=2,
                   help="[transmil aggregator] number of Transformer encoder layers.")
    p.add_argument("--transmil_dim_feedforward", type=int, default=512,
                   help="[transmil aggregator] Transformer feed-forward width.")

    # Feature cache (RAM-budget control for large corpora)
    p.add_argument("--feature_cache_max_gb", type=float, default=None,
                   help="Cap the in-memory feature cache at this many GiB. Leave "
                        "unset for the original uncapped behaviour (only safe if "
                        "the whole corpus comfortably fits in RAM). Once full, "
                        "additional files are read from disk on each access "
                        "instead of being cached (no eviction/thrashing). "
                        "The fixed test set is always prioritised into whatever "
                        "budget is available.")
    p.add_argument("--feature_cache_dtype", choices=["float32", "float16"], default="float32",
                   help="Storage dtype for the feature cache. float16 roughly "
                        "halves the cache footprint; values are upcast to "
                        "float32 per accessed tile before reaching the model, "
                        "so this only affects the resting in-memory copy.")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.aggregator_sweep:
        run_sweep(args)
    else:
        train(args)
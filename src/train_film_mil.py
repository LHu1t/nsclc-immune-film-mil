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

Usage:
    python train_film_mil.py \
        --luad_features /path/to/UNI2/Features/TCGA-LUAD \
        --lusc_features /path/to/UNI2/Features/TCGA-LUSC \
        --metadata      /path/to/metadata.csv \
        --output_dir    ./results \
        --n_folds       5

File structure expected in feature directories:
    <luad_features>/<submitter_id>.h5   (keys: features, coords)
    <lusc_features>/<submitter_id>.h5   (keys: features, coords)
"""

import argparse
import json
import logging
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

    return df, gene_cols, clinical_cols, y_means, y_stds


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

        with h5py.File(rec["h5_path"], "r") as f:
            features = torch.tensor(f["features"][:], dtype=torch.float32)  # expected (N, 1536)

        if features.ndim == 3 and features.shape[0] == 1:
            features = features.squeeze(0)
        elif features.ndim != 2:
            raise ValueError(
                f"Unexpected features shape {tuple(features.shape)} in "
                f"{rec['h5_path']} - expected (N_tiles, feat_dim)."
            )

        # Tile sampling
        n = features.shape[0]
        if self.n_tiles is not None and self.n_tiles < n:
            if self.deterministic:
                rng    = np.random.default_rng(self.seed + idx)
                chosen = torch.tensor(
                    sorted(rng.choice(n, size=self.n_tiles, replace=False))
                )
            else:
                chosen = torch.randperm(n)[: self.n_tiles]
            features = features[chosen]

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
    End-to-end FiLM-conditioned Attention-MIL model.

    Pipeline:
        1. Attention-MIL: (N_tiles, 1536) -> (512,) slide embedding
        2. FiLM:          condition slide embedding on subtype (LUAD/LUSC)
        3. Clinical:      project [age_z, gender] -> (64,), concatenate
        4. Regression:    (512+64,) -> (n_genes,)
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
    ):
        super().__init__()

        # 1. Attention-MIL pooling
        self.attention_mil = AttentionMIL(feat_dim=feat_dim, hidden_dim=256)

        # 2. FiLM subtype conditioning
        self.film = FiLMLayer(embed_dim=embed_dim, n_subtypes=n_subtypes)

        # 3. Clinical projection
        self.clinical_proj = nn.Sequential(
            nn.Linear(clinical_dim, n_clinical),
            nn.ReLU(),
        )

        # 4. Regression head
        self.head = nn.Sequential(
            nn.Linear(embed_dim + n_clinical, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_genes),
        )

    def forward(self, features, clinical, subtype_id):
        """
        features   : (N_tiles, feat_dim)
        clinical   : (clinical_dim,)
        subtype_id : scalar long tensor (0=LUAD, 1=LUSC)

        returns    : predictions (n_genes,), attn_weights (N_tiles,)
        """
        # 1. Attention pooling -> slide embedding
        slide_embed, attn_weights = self.attention_mil(features)  # (512,), (N,)

        # 2. FiLM conditioning on subtype
        slide_embed = self.film(slide_embed, subtype_id)          # (512,)

        # 3. Clinical covariates
        clin_embed  = self.clinical_proj(clinical)                # (64,)

        # 4. Concatenate and predict
        combined = torch.cat([slide_embed, clin_embed], dim=-1)  # (576,)
        preds    = self.head(combined)                            # (35,)

        return preds, attn_weights


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
    ) -> tuple[torch.Tensor, dict]:

        mse = F.mse_loss(pred, target)
        pcc_loss = self.pearson_loss(pred, target)
        var_loss = -pred.var(dim=0).mean()   # penalise low-variance predictions

        if epoch <= 25:
            alpha, beta, gamma = 1.0, 0.0, 0.0
        else:
            alpha, beta, gamma = 0.4, 0.2, 0.4

        loss = alpha * mse + beta * pcc_loss + gamma * var_loss

        return loss, {
            "mse":     mse.item(),
            "pcc":     (1 - pcc_loss).item(),
            "var":     (-var_loss).item(),
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
def run_epoch(model, loader, loss_fn, optimizer, epoch, device, training=True):
    model.train() if training else model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []      # detached copies, for metrics/return only
    batch_preds, batch_labels = [], []

    def _step(preds_list, labels_list):
        pred_batch  = torch.stack(preds_list)
        label_batch = torch.stack(labels_list)
        loss, _ = loss_fn(pred_batch, label_batch, epoch)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in loader:
            features, label, _, clinical, subtype_id, _ = batch

            # All inputs: single slide (batch_size=1), squeeze batch dim
            features   = features.squeeze(0).to(device)    # (N_tiles, 1536)
            label      = label.squeeze(0).to(device)       # (35,)
            clinical   = clinical.squeeze(0).to(device)    # (2,)
            subtype_id = subtype_id.squeeze(0).to(device)  # scalar

            pred, _ = model(features, clinical, subtype_id)  # (35,)

            # Detached copies for logging / PCC metrics
            all_preds.append(pred.detach().cpu())
            all_labels.append(label.detach().cpu())

            # Per-sample MSE contribution to total loss (metrics only)
            mse = F.mse_loss(pred, label)
            total_loss += mse.item()

            if training:
                batch_preds.append(pred)
                batch_labels.append(label)

                if len(batch_preds) % 16 == 0:
                    _step(batch_preds, batch_labels)
                    batch_preds, batch_labels = [], []

        # Note: This flushes any leftover slides (< 16) so the tail of the epoch still contributes a gradient update instead of being ignored
        if training and batch_preds:
            _step(batch_preds, batch_labels)

    preds  = torch.stack(all_preds).numpy()
    labels = torch.stack(all_labels).numpy()
    mean_pcc = compute_gene_pccs(preds, labels).mean()

    return total_loss / len(loader), mean_pcc, preds, labels


# Main training script
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    df, gene_cols, clinical_cols, _, _ = load_metadata(args.metadata)

    # Build gene symbol -> index mapping for panel evaluation.
    # Strips _fpkm_uq suffix and leaves TMB, APM, and TIS unaltered.
    gene_symbol_to_idx = {}
    for i, g in enumerate(gene_cols):
        symbol = g.replace("_fpkm_uq", "")   # e.g. HLA-A_fpkm_uq -> HLA-A
        gene_symbol_to_idx[symbol] = i        # TMB, APM, TIS kept unchanged
    log.info(f"Predicting {len(gene_cols)} targets: "
             f"{len([g for g in gene_cols if g.endswith('_fpkm_uq')])} genes "
             f"+ TMB + APM + TIS")

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

    # 5-fold cross validation
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=98)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(df_dev)):
        log.info(f"\n{'='*60}")
        log.info(f"FOLD {fold} / {args.n_folds - 1}")
        log.info(f"{'='*60}")

        df_train = df_dev.iloc[train_idx].reset_index(drop=True)
        df_val   = df_dev.iloc[val_idx].reset_index(drop=True)

        # Datasets
        train_ds = FiLMDataset(df_train, feature_dirs, gene_cols, clinical_cols,
                               n_tiles=None, deterministic=False)
        val_ds   = FiLMDataset(df_val,   feature_dirs, gene_cols, clinical_cols,
                               n_tiles=None, deterministic=True)
        test_ds  = FiLMDataset(df_test,  feature_dirs, gene_cols, clinical_cols,
                               n_tiles=None, deterministic=True)

        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,  num_workers=4)
        val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, num_workers=4)
        test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False, num_workers=4)

        # Model
        model    = FiLMMILModel(feat_dim=1536, n_genes=len(gene_cols)).to(device)
        loss_fn  = CompositeLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5, verbose=True
        )

        # Training loop with early stopping
        best_val_pcc  = -np.inf
        best_weights  = None
        patience_ctr  = 0

        for epoch in range(1, args.max_epochs + 1):
            train_loss, train_pcc, _, _ = run_epoch(
                model, train_loader, loss_fn, optimizer, epoch, device, training=True
            )
            val_loss, val_pcc, _, _ = run_epoch(
                model, val_loader, loss_fn, optimizer, epoch, device, training=False
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
        _, _, test_preds, test_labels = run_epoch(
            model, test_loader, loss_fn, optimizer, 999, device, training=False
        )

        # Gene-level PCCs
        gene_pccs = compute_gene_pccs(test_preds, test_labels)

        # Panel-level metrics (LUAD + LUSC combined, then subtype-split below)
        apm_pcc  = compute_panel_pcc(test_preds, test_labels, gene_cols, APM_GENES, gene_symbol_to_idx)
        tis_pcc  = compute_panel_pcc(test_preds, test_labels, gene_cols, TIS_GENES, gene_symbol_to_idx)
        apm_auc  = compute_auc(test_preds, test_labels, gene_cols, APM_GENES, gene_symbol_to_idx)
        tis_auc  = compute_auc(test_preds, test_labels, gene_cols, TIS_GENES, gene_symbol_to_idx)

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
            subtype_results[name] = {
                "APM_PCC": compute_panel_pcc(p, l, gene_cols, APM_GENES, gene_symbol_to_idx),
                "TIS_PCC": compute_panel_pcc(p, l, gene_cols, TIS_GENES, gene_symbol_to_idx),
                "APM_AUC": compute_auc(p, l, gene_cols, APM_GENES, gene_symbol_to_idx),
                "TIS_AUC": compute_auc(p, l, gene_cols, TIS_GENES, gene_symbol_to_idx),
                "n":       int(mask.sum()),
            }

        fold_result = {
            "fold":           fold,
            "best_val_pcc":   float(best_val_pcc),
            "APM_PCC":        float(apm_pcc),
            "TIS_PCC":        float(tis_pcc),
            "APM_AUC":        float(apm_auc),
            "TIS_AUC":        float(tis_auc),
            "gene_pccs":      {
                g.replace("_fpkm_uq", ""): float(gene_pccs[i])
                for i, g in enumerate(gene_cols)
            },
            "subtype":        subtype_results,
        }
        fold_results.append(fold_result)

        log.info(f"\nFold {fold} Test Results:")
        log.info(f"  APM  PCC={apm_pcc:.4f}  AUC={apm_auc:.4f}")
        log.info(f"  TIS  PCC={tis_pcc:.4f}  AUC={tis_auc:.4f}")
        for name, res in subtype_results.items():
            log.info(f"  {name} (n={res['n']}): APM PCC={res['APM_PCC']:.4f}, TIS PCC={res['TIS_PCC']:.4f}")

        # Save model weights for this fold
        torch.save(
            best_weights,
            output_dir / f"fold{fold}_best_model.pt"
        )

    # Summary across folds
    log.info(f"\n{'='*60}")
    log.info("CROSS-VALIDATION SUMMARY")
    log.info(f"{'='*60}")
    for metric in ["APM_PCC", "TIS_PCC", "APM_AUC", "TIS_AUC"]:
        vals = [r[metric] for r in fold_results]
        log.info(f"  {metric}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    for subtype in ["LUAD", "LUSC"]:
        for metric in ["APM_PCC", "TIS_PCC", "APM_AUC", "TIS_AUC"]:
            vals = [r["subtype"][subtype][metric]
                    for r in fold_results if subtype in r["subtype"]]
            if vals:
                log.info(f"  {subtype} {metric}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # Save all results
    with open(output_dir / "results.json", "w") as f:
        json.dump(fold_results, f, indent=2)
    log.info(f"\nResults saved to {output_dir / 'results.json'}")


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
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
#!/usr/bin/env python3
"""
build_cptac_metadata.py

Builds a CPTAC external-validation metadata CSV matching the format
FiLMDataset/load_metadata() expects from TCGA training, EXCEPT that gene
and age z-scoring reuse the saved TCGA statistics (not recomputed from
CPTAC) so the model sees inputs on the same scale it was trained on.

Inputs:
  --cptac-download-dir   Folder of UUID-named subfolders, each containing
                          one *.rna_seq.augmented_star_gene_counts.tsv
  --metadata-json        GDC metadata JSON (list of file records) mapping
                          file_id (= subfolder name) -> case_id
  --clinical-tsv         GDC clinical.tsv with cases.case_id,
                          cases.submitter_id, demographic.days_to_birth,
                          demographic.sex_at_birth, [diagnoses.ajcc_pathologic_stage]
  --model-config          model_config.json from TCGA training (gene_cols,
                          y_means, y_stds, APM_GENES, TIS_GENES)
  --tcga-age-mean/--tcga-age-std
                          REQUIRED. Mean/std of age_years from the ORIGINAL
                          TCGA training metadata CSV. Do not compute these
                          from CPTAC - that reintroduces exactly the cohort-
                          shift bias external validation is supposed to
                          catch. If you don't have these, recompute them
                          from your saved TCGA training CSV:
                              df["age_years"].mean(), df["age_years"].std()
  --out-csv               Output path

Optional, for a slide-count sanity check only:
  --luad-features-dir / --lusc-features-dir
                          Local folders of .h5 WSI features, to report how
                          many slides exist per matched case.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def load_star_gene_counts(path: Path) -> pd.DataFrame:
    """Robustly parse a GDC STAR - Counts gene quantification TSV.

    These files have a variable number of leading comment lines, then a
    header row, then 4 N_* summary rows before the actual gene rows.
    """
    with open(path) as f:
        lines = f.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("gene_id"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"Could not find header row (starting 'gene_id') in {path}")

    df = pd.read_csv(path, sep="\t", skiprows=header_idx)
    df = df[~df["gene_id"].astype(str).str.startswith("N_")].copy()

    # Resolve duplicate gene_name entries (e.g. PAR_Y regions) by keeping the max fpkm_uq_unstranded per symbol
    dup_names = df["gene_name"][df["gene_name"].duplicated(keep=False)].unique()
    if len(dup_names) > 0:
        print(f"    note: {len(dup_names)} duplicate gene_name(s) in {path.name}, "
              f"keeping max fpkm_uq_unstranded per symbol: {list(dup_names)[:5]}"
              f"{'...' if len(dup_names) > 5 else ''}")
    df = df.sort_values("fpkm_uq_unstranded", ascending=False).drop_duplicates("gene_name", keep="first")
    return df.set_index("gene_name")["fpkm_uq_unstranded"]


def load_file_to_case_map(metadata_json_path: Path) -> dict:
    with open(metadata_json_path) as f:
        records = json.load(f)
    if isinstance(records, dict):
        records = [records]

    file_to_case = {}
    for rec in records:
        file_id = rec.get("file_id")
        entities = rec.get("associated_entities", [])
        case_ids = {e["case_id"] for e in entities if "case_id" in e}
        if file_id is None or len(case_ids) != 1:
            print(f"    skipping metadata record with file_id={file_id}: "
                  f"expected exactly 1 associated case_id, got {case_ids}")
            continue
        file_to_case[file_id] = next(iter(case_ids))
    return file_to_case


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--cptac-download-dir", type=Path, required=True)
ap.add_argument("--metadata-json", type=Path, required=True)
ap.add_argument("--clinical-tsv", type=Path, required=True)
ap.add_argument("--model-config", type=Path, required=True)
ap.add_argument("--tcga-age-mean", type=float, required=True,
                    help="age_years mean from the ORIGINAL TCGA training CSV - do not recompute from CPTAC")
ap.add_argument("--tcga-age-std", type=float, required=True,
                    help="age_years std from the ORIGINAL TCGA training CSV - do not recompute from CPTAC")
ap.add_argument("--luad-features-dir", type=Path, default=None)
ap.add_argument("--lusc-features-dir", type=Path, default=None)
ap.add_argument("--out-csv", type=Path, required=True)
args = ap.parse_args()

with open(args.model_config) as f:
    config = json.load(f)
y_means = config["y_means"]
y_stds = config["y_stds"]
apm_genes = config["APM_GENES"]
tis_genes = config["TIS_GENES"]
# gene_cols minus TMB/APM/TIS = the 35 raw fpkm_uq columns to extract
raw_gene_cols = [c for c in config["gene_cols"] if c.endswith("_fpkm_uq")]
gene_symbols = [c.replace("_fpkm_uq", "") for c in raw_gene_cols]

# 1. file_id -> case_id (GDC internal UUID)
print("Loading metadata JSON ...")
file_to_case = load_file_to_case_map(args.metadata_json)
print(f"  {len(file_to_case)} RNA-seq files mapped to cases")

# 2. clinical.tsv: case_id -> submitter_id, age, sex, stage
print("Loading clinical data ...")
clinical = pd.read_csv(args.clinical_tsv, sep="\t")
n_rows = len(clinical)
clinical = clinical.drop_duplicates(subset="cases.case_id", keep="first")
if len(clinical) != n_rows:
    print(f"clinical.tsv had {n_rows} rows -> {len(clinical)} unique cases "
            f"(deduped extra diagnosis/treatment/exposure rows)")
clinical = clinical.set_index("cases.case_id")

#Parse each RNA-seq file, extract target genes
print("Parsing RNA-seq gene counts files ...")
rows = []
skipped_no_case = 0
skipped_no_clinical = 0
skipped_missing_genes = 0

for subdir in sorted(args.cptac_download_dir.iterdir()):
    if not subdir.is_dir():
        continue
    file_id = subdir.name
    tsvs = list(subdir.glob("*.tsv"))
    if not tsvs:
        continue
    if len(tsvs) > 1:
            print(f"{file_id}: multiple .tsv files found, using {tsvs[0].name}")
    tsv_path = tsvs[0]

    case_id = file_to_case.get(file_id)
    if case_id is None:
        skipped_no_case += 1
        continue
    if case_id not in clinical.index:
        skipped_no_clinical += 1
        continue

    gene_values = load_star_gene_counts(tsv_path)
    missing = [g for g in gene_symbols if g not in gene_values.index]
    if missing:
        skipped_missing_genes += 1
        print(f"{file_id} (case {case_id}): missing genes {missing}, skipping")
        continue

    clin_row = clinical.loc[case_id]
    row = {
        "submitter_id": str(clin_row["cases.submitter_id"]).upper().strip(),
        "file_id": file_id,
    }
    for g, col in zip(gene_symbols, raw_gene_cols):
        row[col] = gene_values[g]
    try:
        days_to_birth = str(clin_row["demographic.days_to_birth"]).strip().strip("'\"")
        row["age_years"] = abs(float(days_to_birth)) / 365.25
    except (ValueError, TypeError):
        row["age_years"] = args.tcga_age_mean
    row["demographic.gender"] = str(clin_row["demographic.sex_at_birth"]).lower()
    if "diagnoses.ajcc_pathologic_stage" in clinical.columns:
        row["ajcc_pathologic_stage"] = clin_row.get("diagnoses.ajcc_pathologic_stage")
    rows.append(row)

print(f"\nParsed {len(rows)} cases "
        f"(skipped: {skipped_no_case} no case_id, {skipped_no_clinical} no clinical match, "
        f"{skipped_missing_genes} missing target genes)")

df = pd.DataFrame(rows)
if df.empty:
    print("\nNo cases survived parsing - nothing to write. "
            "Check the warnings above (missing genes / unmatched IDs / clinical join).")
df = df.drop_duplicates(subset="submitter_id", keep="first")
if len(df) < len(rows):
    print(f"  dropped {len(rows) - len(df)} duplicate submitter_id rows "
            f"(case had multiple RNA-seq aliquots - kept first)")

# Apply TCGA-derived preprocessing: log1p, then z-score with SAVED training stats (not recomputed from CPTAC)
print("\nApplying log1p + TCGA-trained z-score normalization ...")
for col in raw_gene_cols:
    df[col + "_raw"] = df[col].copy()
    df[col] = np.log1p(df[col])
    df[col] = (df[col] - y_means[col]) / y_stds[col]

# APM/TIS composite scores: mean of the z-scored panel genes, matching the panel_score() convention used throughout evaluation/figures
def panel_mean(genes):
    cols = [f"{g}_fpkm_uq" for g in genes]
    return df[cols].mean(axis=1)

df["APM"] = panel_mean(apm_genes)
df["TIS"] = panel_mean(tis_genes)

# TMB cannot be derived from RNA-seq (needs somatic mutation data),leave as NaN
df["TMB"] = np.nan
print("  note: TMB set to NaN - not derivable from RNA-seq. Exclude from "
        "any TMB-specific evaluation, or source it separately from CPTAC WES data.")

# Age: z-score using the TRAINING mean/std (critical for external validation, therefore NOT recomputed from CPTAC)
df["age_years_z"] = (df["age_years"] - args.tcga_age_mean) / args.tcga_age_std
print(f"  age_years_z computed using TCGA training stats "
        f"(mean={args.tcga_age_mean:.2f}, std={args.tcga_age_std:.2f}), NOT CPTAC's own distribution")

# Gender: match load_metadata()'s expected encoding downstream
df["gender_encoded"] = df["demographic.gender"].map({"male": 0, "female": 1}).fillna(0).astype(np.float32)

# 5. Optional: cross-check slide counts per matched case
if args.luad_features_dir or args.lusc_features_dir:
    print("\nCross-checking matched cases against local WSI feature files ...")
    case_id_re = re.compile(r"(C3[LN]-\d{5})", re.IGNORECASE)
    slide_counts = {}
    for feat_dir in filter(None, [args.luad_features_dir, args.lusc_features_dir]):
        for h5 in Path(feat_dir).glob("*.h5"):
            m = case_id_re.search(h5.stem)
            if m:
                cid = m.group(1).upper()
                slide_counts.setdefault(cid, []).append(h5.name)
    matched_with_slides = [sid for sid in df["submitter_id"] if sid in slide_counts]
    matched_without_slides = [sid for sid in df["submitter_id"] if sid not in slide_counts]
    multi_slide = {sid: len(v) for sid, v in slide_counts.items() if sid in df["submitter_id"].values and len(v) > 1}
    print(f"  {len(matched_with_slides)}/{len(df)} cases with RNA-seq also have >=1 local WSI slide")
    if matched_without_slides:
        print(f"{len(matched_without_slides)} cases have RNA-seq but NO matching local .h5 file "
                f"- these will be unusable, e.g. {matched_without_slides[:5]}")
    if multi_slide:
        print(f"{len(multi_slide)} cases have multiple slides"
                f"e.g. {dict(list(multi_slide.items())[:5])}")

# 6. Save
args.out_csv.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(args.out_csv, index=False)
print(f"\Saved {len(df)} cases to {args.out_csv}")
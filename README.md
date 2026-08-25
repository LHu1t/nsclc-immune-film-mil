# NSCLC Immune Profiling — FiLM-MIL

Predicting immune gene-expression signatures, antigen presentation machinery (APM) and T-cell inflammation signature (TIS), directly from H&E whole-slide images of non-small-cell lung cancer (NSCLC), using an attention-based multiple-instance-learning (MIL) model optionally conditioned on tumor subtype (LUAD/LUSC) via FiLM.

Trained on TCGA (n≈987) and externally validated on an independent CPTAC cohort (n=330) with a same pipeline no-FiLM ablation to isolate whether explicit subtype conditioning is doing any of the work.

> Companion workshop paper: *"Antigen Presentation Is Subtype-Intrinsic, T-Cell Inflammation Is Not: A Cross-Cohort Dissociation Discovered via Attention-MIL on NSCLC Histology"* — in preparation.

## Overview

```
WSI tiles (256x256px, 20x)
        │  UNI2-h (frozen, 1536-dim)
        ▼
Attention-MIL pooling  →  512-dim slide embedding
        │  FiLM (subtype-conditioned, optional)
        ▼
+ clinical covariates (age, sex)
        ▼
Regression head  ->  35 gene targets + APM/TIS panel scores
```

The `use_film` flag switches the conditioning mechanism on/off so the same pipeline can be trained and evaluated both ways — this is the basis of the FiLM vs. no-FiLM comparison in `src/compare_film_vs_nofilm.py`.

## Repository structure

```
├── data/
│   ├── tcga-nsclc-metadata.csv      # TCGA training/discovery metadata
│   └── cptac-nsclc-metadata.csv     # CPTAC external-validation metadata
├── documents/
│   ├── data_access.md               # where the WSIs, gene expression, and
│   │                                 # UNI2-h features come from
│   ├── dataset_structure.md         # expected on-disk layout for features
│   └── kaggle_setup.md              # Kaggle dataset/GPU setup for training
├── notebooks/
│   ├── kaggle-film-mil.ipynb            # TCGA training (FiLM / no-FiLM)
│   └── kaggle-external-validation.ipynb # CPTAC external validation
├── results/
│   ├── FiLM/{training,external_validation}/
│   └── No-FiLM/{training,external_validation}/
│       # per-fold + ensemble predictions (.npz), metrics (.csv/.json)
│       # for both conditioning settings
├── src/
│   ├── build_metadata.py            # builds CPTAC metadata matching the
│   │                                 # TCGA training format (reuses TCGA's
│   │                                 # saved z-scoring statistics)
│   ├── train_film_mil.py            # Attention-MIL + FiLM training/CV
│   ├── compare_film_vs_nofilm.py    # FiLM vs. no-FiLM stats + figures
│   └── make_paper_figures.py        # figure set for the paper
├── requirements.txt
└── CITATION.cff
```

## Data

WSIs, clinical metadata, and RNA-seq for TCGA-LUAD/LUSC and CPTAC-3 are public via the [NCI Genomic Data Commons](https://portal.gdc.cancer.gov/).

Tile-level UNI2-h feature embeddings (1536-dim, extracted at 256×256px / 20×) are not redistributed in this repository due to license terms. To access them:

1. Request access to the [UNI2-h model](https://huggingface.co/MahmoodLab/UNI2-h) (Mahmood Lab).
2. Download features following the layout in [`documents/dataset_structure.md`](documents/dataset_structure.md)
   — one `.h5` file per slide (`submitter_id.h5`, keys `features`/`coords`,
   shape `[N_tiles, 1536]`) from pre-extracted TCGA features from the
   [Mahmood Lab UNI2-h-features dataset](https://huggingface.co/datasets/MahmoodLab/UNI2-h-features/tree/main/TCGA). Same can be done for CPTAC.

More access details are in [`documents/data_access.md`](documents/data_access.md).

## Setup

```bash
git clone https://github.com/LHu1t/nsclc-immune-film-mil.git
cd nsclc-immune-film-mil
pip install -r requirements.txt
```

Training was run on Kaggle (T4×2 GPU + internet access) rather than locally, given the WSI feature volume — see [`documents/kaggle_setup.md`](documents/kaggle_setup.md) for uploading
features/metadata as Kaggle datasets and configuring the notebook environment.

## Reproducing the results

**1. Train (TCGA), 5-fold CV — [`notebooks/kaggle-film-mil.ipynb`](notebooks/kaggle-film-mil.ipynb) or equivalently:**
```bash
python src/train_film_mil.py \
    --luad_features /path/to/UNI2/Features/TCGA-LUAD \
    --lusc_features /path/to/UNI2/Features/TCGA-LUSC \
    --metadata      /path/to/tcga-nsclc-metadata.csv \
    --output_dir    ./results/FiLM/training \   # or non-film
    --n_folds       5 \
    --max_epochs    50 \
    --patience      3 \
    --use_film      true   # false for the no-FiLM ablation
```

**2. Build CPTAC external-validation metadata** (reuses TCGA's saved
gene/age z-scoring statistics rather than recomputing on CPTAC, to avoid
reintroducing cohort-shift bias):
```bash
python src/build_metadata.py \
    --cptac-download-dir <GDC download folder> \
    --metadata-json      <GDC metadata JSON> \
    --clinical-tsv       <GDC clinical.tsv> \
    --model-config       results/FiLM/training/model_config.json \
    --tcga-age-mean <mean> --tcga-age-std <std> \
    --out-csv data/cptac-nsclc-metadata.csv
```

**3. External validation (CPTAC)** — [`notebooks/kaggle-external-validation.ipynb`](notebooks/kaggle-external-validation.ipynb),
run once per trained model (FiLM and no-FiLM), producing
`ensemble_cptac_predictions.npz`, per-fold prediction files, and
`model_config_used.json` in `results/{FiLM,No-FiLM}/external_validation/`.

**4. FiLM vs. no-FiLM comparison, stats, and figures:**
```bash
python src/compare_film_vs_nofilm.py   # edit FILM_DIR / NOFILM_DIR paths at the top
```
Produces paired-bootstrap PCC differences, DeLong AUC tests, Levene/Bartlett fold-variance tests, LUAD-vs-LUSC subtype comparisons (Fisher r-to-z and paired fold-level t-test), gene-rank concordance, and the paper's forest / subtype-bar / gene-scatter figures.

**5. Paper figures:**
```bash
python src/make_paper_figures.py --artifacts-dir results/paper_artifacts --out-dir figures
```

## Gene panels

- **APM** (18 genes): `PSMB5/6/7/8/9/10, TAP1/2, ERAP1/2, TAPBP, CANX, CALR, PDIA3, B2M, HLA-A/B/C`
- **TIS** (18 genes): `PSMB10, HLA-DQA1, HLA-DRB1, CMKLR1, HLA-E, NKG7, CD8A, CCL5, CXCL9, CD27, CXCR6, IDO1, STAT1, CD274, CD276, LAG3, PDCD1LG2, TIGIT`

(`PSMB10` is shared between panels.) Headline metrics use the mean of individually z-scored panel genes, not the model's separate auxiliary panel-score output head.

## Citation

See [`CITATION.cff`](CITATION.cff). If you use this code, please cite the associated paper (details to be finalized on publication).

## License / data-use note

Code is provided as-is for research reproducibility. Underlying WSI, RNA-seq, and clinical data remain subject to the data-use terms of TCGA/CPTAC and the GDC; UNI2-h feature embeddings are subject to the Mahmood Lab's license terms and are not redistributed here.
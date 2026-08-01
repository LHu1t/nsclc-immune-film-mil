## Data Structure

data/
│
├── tcga-nsclc-metadata.csv
│
├── features/
│   ├── LUAD/
│   │   ├── TCGA-XX-XXXX.pt
│   │
│   └── LUSC/
│       ├── TCGA-YY-YYYY.pt

Each feature file contains:

Tensor:
[N_tiles, feature_dimension] e.g.: TCGA - [7936, 1536]
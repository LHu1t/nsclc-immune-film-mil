## Data Structure

```text
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
```

Each feature file contains:

**Tensor shape:**

```text
[N_tiles, feature_dimension]
```

For example:

```text
TCGA -> [7936, 1536]
```
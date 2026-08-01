## Kaggle Setup

# Add datasets

1. Login/create Kaggle account (https://www.kaggle.com)
2. In the terminal, ensure kaggle is installed (pip install kaggle)
3. Generate a Kaggle API Token: https://www.kaggle.com > Account > Create new API Token (Legacy)
4. Move the file to the appropriate location (e.g. Linux: ~/.kaggle/kaggle.json)

# Upload and run training

1. In the terminal, navigate into the folder you want to upload to kaggle (e.g. the TCGA LUAD feature folder)
2. Initialise the dataset: kaggle datasets init -p /path/to/dataset
3. Edit the dataset-metadata.json file created with an approriate name and slug
4. Create the dataset: kaggle datasets create -p /path/to/dataset
5. Repeat to upload all of the following: TCGA LUSC features, TCGA LUAD features, and tcga-nsclc-metadata.csv
6. Enable GPU accelerator (T4 x2) and internet access
7. Add:
   - metadata dataset
   - feature embedding datasets
8. Run all
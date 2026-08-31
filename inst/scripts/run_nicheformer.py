import os
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import anndata as ad
from typing import Optional, Dict, Any
from tqdm import tqdm

from nicheformer.models import Nicheformer
from nicheformer.data import NicheformerDataset

# download nicheformer.ckpt from 
# https://data.mendeley.com/preview/87gm9hrgm8?a=d95a6dde-e054-4245-a7eb-0522d6ea7dff.

config = {
    'data_path': '../extdata/spe_converted_subset_200_iter_1.h5ad', #'path/to/your/data.h5ad',  # Path to your AnnData file
    'technology_mean_path': 'data/xenium_mean_script.npy',
    'checkpoint_path': 'data/nicheformer.ckpt',  # Path to model checkpoint
    'output_path': 'data_with_embeddings.h5ad',  # Where to save the result, it is a new h5ad
    'output_dir': '.',  # Directory for any intermediate outputs
    'batch_size': 32,
    'max_seq_len': 1500, 
    'aux_tokens': 30, 
    'chunk_size': 1000, # to prevent OOM
    'num_workers': 4,
    'precision': 32,
    'embedding_layer': -1,  # Which layer to extract embeddings from (-1 for last layer)
    'embedding_name': 'embeddings'  # Name suffix for the embedding key in adata.obsm
}

model = ad.read_h5ad('data/model.h5ad')

# Set random seed for reproducibility
pl.seed_everything(42)

# Load data
adata = ad.read_h5ad(config['data_path'])
adata.X = adata.layers["counts"]
technology_mean = np.load(config['technology_mean_path'])

# format data properly with the model
adata = ad.concat([model, adata], join='outer', axis=0)
# dropping the first observation
adata = adata[1:, model.var_names].copy()

# Change accordingly

adata.obs['modality'] = 4 # spatial
adata.obs['specie'] = 5 # human
adata.obs['assay'] = 9 # xenium
adata.obs['nicheformer_split'] = 'train'

# Create dataset
dataset = NicheformerDataset(
    adata=adata,
    technology_mean=technology_mean,
    split='train',
    max_seq_len=1500,
    aux_tokens=config.get('aux_tokens', 30),
    chunk_size=config.get('chunk_size', 1000),
    metadata_fields={'obs': ['modality', 'specie', 'assay']}
)

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=config['batch_size'],
    shuffle=False,
    num_workers=0,
    pin_memory=False
)

# Load pre-trained model
model = Nicheformer.load_from_checkpoint(checkpoint_path=config['checkpoint_path'], strict=False, weights_only=False)
model.eval()  # Set to evaluation mode
# model.to("mps") # run this if you need to define a device

print("Extracting embeddings...")
embeddings = []
device = model.embeddings.weight.device

with torch.no_grad():
    for batch in tqdm(dataloader):
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

        # Get embeddings from the model
        emb = model.get_embeddings(
            batch=batch,
            layer=config.get('embedding_layer', -1)  # Default to last layer
        )
        embeddings.append(emb.cpu().numpy())


# Concatenate all embeddings
embeddings = np.concatenate(embeddings, axis=0)
print(embeddings)
print(embeddings.shape)

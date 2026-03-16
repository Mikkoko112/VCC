#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

# Reduce OpenMP/Numba threading conflicts that can crash on macOS.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

import anndata as ad
from anndata.experimental.pytorch import AnnLoader
import numpy as np
import scanpy as sc
import pandas as pd
import torch
from torch import nn, optim 
import matplotlib.pyplot as plt
from pathlib import Path

# Configure device for PyTorch. For this workflow, CPU is the stable default.
use_mps = False
if use_mps and torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print(x)
else:
    mps_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {mps_device}")

# TODO: Read the .h5ad file using Scanpy
file_path = Path(__file__).parent.parent.parent / "data" / "mesenchymal_subset.h5ad"
adata = sc.read_h5ad(file_path)

# Make sure to reset the core matrix to raw counts
# If you want to keep only cells 1000: and keep full AnnData consistency
adata = adata[:8030, :].copy()
adata.X = adata.raw.X.copy()   # optional: reset X to raw counts after subsetting

# TODO: Normalize raw count matrix to median total counts using Scanpy
sc.pp.normalize_total(adata, target_sum=1e6, inplace=True)

# TODO: Logarithmize the normalized count matrix using Scanpy
sc.pp.log1p(adata)

# TODO: Subset the Anndata object to the top 3000 highly variable genes
sc.pp.highly_variable_genes(adata, n_top_genes=3000, subset=True, flavor="seurat_v3", inplace=True)

# TODO: Define the AnnLoader object using the entire Anndata object as training data with a batch_size of 128 and sample shuffling enabled 
dataloader = AnnLoader(adata, batch_size=128, shuffle=True)

# TODO: Define the full single-cell autoencoder model (see exercise sheet for layer definitions)
cdims = adata.shape[-1]

class SCAutoEncoder(nn.Module):
    def __init__(self):
        super(SCAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(cdims, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 2))
        self.decoder = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, cdims),
            nn.Softplus()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def layer_size_bounds(self):
        encoder_out = [m.out_features for m in self.encoder if isinstance(m, nn.Linear)]
        decoder_out = [m.out_features for m in self.decoder if isinstance(m, nn.Linear)]
        # Exclude the final decoder output size (cdims) to focus on AE architecture widths.
        ae_sizes = encoder_out + decoder_out[:-1]
        return min(ae_sizes), max(ae_sizes)

# TODO: Initialize the SCAutoEncoder model
model = SCAutoEncoder().to(mps_device)
min_layer_size, max_layer_size = model.layer_size_bounds()
layer_tag = f"_min{min_layer_size}_max{max_layer_size}"

# TODO: Configure MSE loss
loss_function = nn.MSELoss()

# TODO: Configure Adam optimizer with learning rate 1e-3 and weight decay of 1e-8
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)

# Train the autoencoder for 20 epochs using the above defined AnnLoader
epochs = 20
outputs = []
losses = []
model.to(mps_device)
plot_dir = Path(__file__).parent / "figures" / f"{layer_tag}"
plot_dir.mkdir(parents=True, exist_ok=True)

for epoch in range(epochs):
    for batch in dataloader:
        pbatch = batch.X
        if pbatch.is_sparse:
            pbatch = pbatch.to_dense()
        pbatch = pbatch.to(mps_device, dtype=torch.float32)
        reconstructed = model(pbatch)
        # TODO: Compare autoencoder reconstruction against input by computing the loss
        loss = loss_function(reconstructed, pbatch)
        # Compute gradient, perform backpropagation and take a step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Store loss
        losses.append(loss.item())
    
    outputs.append((epoch, batch, reconstructed))
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

# Plot the global training loss
plt.style.use('fivethirtyeight')
plt.figure(figsize=(8, 5))
plt.plot(losses, label='Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig(plot_dir / f"training_loss{layer_tag}.png", dpi=150, bbox_inches="tight")
plt.close()

# TODO: Define a utility AnnLoader with batch size equal to the total number of cells and without sample shuffling
fulldataloader = AnnLoader(adata, batch_size=adata.n_obs, shuffle=False)

# Process full sample batch
fulldataiter = iter(fulldataloader)
fullbatch = next(fulldataiter).X.to(mps_device)

# TODO: Compute the embedding representation for the full training data set
embedding = model.encoder(fullbatch.float())

# TODO: Compute the reconstruction for the full training data set
reconstruction = model.decoder(embedding)

# Convert to Pandas DataFrame and add it back to the original AnnData object
embedding_df = pd.DataFrame(embedding.cpu().detach().numpy())
adata.obsm["autoencoder"] = embedding_df.to_numpy()
#embedding_np = embedding.cpu().detach().numpy()
#assert embedding_np.shape == (adata.n_obs, 2)

# convert to DataFrame with matching index and column names
#embedding_df = pd.DataFrame(embedding_np, index=adata.obs_names, columns=["AE1", "AE2"])
#adata.obsm["autoencoder"] = embedding_df.to_numpy()

# Compute dimensionality reductions for plotting.
sc.tl.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)

# TODO: Plot the computed autoencoder embedding using Scanpy and color the cells once for each covariate 1) "donor_id"; 2) "batch"; 3) "cell_type"; 4) "sex"; 5) "tissue"
for cov in ["donor_id", "batch", "cell_type", "sex", "tissue"]:
    fig = sc.pl.embedding(adata, basis="autoencoder", color=cov, title=f"Autoencoder: {cov}", show=False, return_fig=True)
    fig.savefig(plot_dir / f"autoencoder_{cov}{layer_tag}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

# TODO: Now plot the first two PCs using Scanpy and color the cells once for each covariate 1) "donor_id"; 2) "batch"; 3) "cell_type"; 4) "sex"; 5) "tissue"
for cov in ["donor_id", "batch", "cell_type", "sex", "tissue"]:
    fig = sc.pl.pca(adata, color=cov, title=f"PCA: {cov}", show=False, return_fig=True)
    fig.savefig(plot_dir / f"pca_{cov}{layer_tag}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

# TODO: Finally plot the first two UMAP dimensions using Scanpy and color the cells once for each covariate 1) "donor_id"; 2) "batch"; 3) "cell_type"; 4) "sex"; 5) "tissue"
for cov in ["donor_id", "batch", "cell_type", "sex", "tissue"]:
    fig = sc.pl.umap(adata, color=cov, title=f"UMAP: {cov}", show=False, return_fig=True)
    fig.savefig(plot_dir / f"umap_{cov}{layer_tag}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

# TODO: Compute the row-wise cosine similarity between original and reconstructed cell profiles 
orig = fullbatch.float()
recon = reconstruction.float()
orig_norm = orig / (orig.norm(dim=1, keepdim=True) + 1e-8)
recon_norm = recon / (recon.norm(dim=1, keepdim=True) + 1e-8)
cos = (orig_norm * recon_norm).sum(dim=1)  # per-cell cosine similarity
similarities = cos

# Plot the global distribution of cosine correlation values
pd.DataFrame(similarities.cpu().detach().numpy()).plot(kind="box", xlabel="Cosine similarity", ylabel="Training data vs. Reconstruction")
plt.tight_layout()
plt.savefig(plot_dir / f"cosine_similarity_box{layer_tag}.png", dpi=150, bbox_inches="tight")
plt.close()

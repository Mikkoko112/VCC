#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
autoencoder.py

Implements a single-cell autoencoder with negative binomial likelihood for 
raw count data.
This script loads preprocessed AnnData with raw counts and then do a HVG selection, 
defines the SCAutoEncoder model, trains it on the training split, 
evaluates on validation and test splits, and saves all relevant metadata, 
model weights, training curves, and evaluation metrics in a reproducible 
experiment folder.

Needed changes to the code:
- Adjust file paths and dataset loading as needed.
- Modify model architecture, hyperparameters, or training loop for experimentation.

=> maybe change it to a function that takes the file path as input and then runs 
the whole workflow, so that it can be easily called with different datasets 
without changing the code.
"""

import os
import json
import random
import uuid
import datetime
import hashlib
import time
import platform
import sys
import shutil

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
import torch.nn.functional as F
from torch import nn, optim 
import matplotlib.pyplot as plt
from pathlib import Path
try:
    import pkg_resources
except ImportError:
    pkg_resources = None

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
file_path = "/Users/nike/Documents/Uni/Master/dieVirtuelleZelle/VCC/test/Nike/pbmc_data/pbmc68k/adata_hvg.h5ad"
adata = sc.read_h5ad(file_path)

# Make sure to reset the core matrix to raw counts
# If you want to keep only cells 1000: and keep full AnnData consistency
#adata = adata[:8030, :].copy()
#adata = adata.copy()  # ensure we have a copy to modify without affecting original
#adata.X = adata.raw.X.copy()   # keep raw counts in X for NB likelihood training

# make an AnnData that contains raw counts and then select HVGs on it
adata_full = adata.raw.to_adata().copy()   # full counts AnnData
# TODO: Subset the Anndata object to the top 3000 highly variable genes
sc.pp.highly_variable_genes(adata_full, n_top_genes=3000, flavor="seurat_v3", subset=True, inplace=True)
adata = adata_full  # now adata.X contains counts for the HVGs

# Keep raw counts in X for negative binomial training and avoid log1p transformations
# sc.pp.normalize_total(adata, target_sum=1e6, inplace=True)
# sc.pp.log1p(adata)

## reproducibility and experiment tracking setup
def compute_sha256(file_path):
    try:
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as reader:
            for chunk in iter(lambda: reader.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    except FileNotFoundError:
        return None

def save_json(path, data):
    path = Path(path)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(data, handle, indent=2)

def save_rng_state(path):
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state().cpu(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(state, path)

# Create a reproducible experiment folder.
experiment_date = datetime.date.today().isoformat()
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
experiment_id = uuid.uuid4().hex[:8]
experiment_name = f"experiment_{experiment_date}_{timestamp}_{experiment_id}"
# TODO: put here the directory where you want to save the experiment results, e.g. "/path/to/experiments"
experiment_dir = Path(__file__).parent / "experiments" / experiment_name
experiment_dir.mkdir(parents=True, exist_ok=False)
plot_dir = experiment_dir / "figures"
plot_dir.mkdir(parents=True, exist_ok=True)

# Save raw file checksum and provenance.
raw_file_checksum = compute_sha256(file_path)
raw_info = {
    'raw_file_path': file_path,
    'raw_file_checksum': raw_file_checksum,
    'adata_obs_shape': adata.obs.shape,
    'adata_var_shape': adata.var.shape,
}
save_json(experiment_dir / 'raw_provenance.json', raw_info)

# Configure reproducible seeds.
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Split data into train / validation / test for reproducibility.
n_cells = adata.n_obs
indices = np.arange(n_cells)
np.random.shuffle(indices)
train_cut = int(n_cells * 0.8)
val_cut = int(n_cells * 0.9)
train_idx = indices[:train_cut]
val_idx = indices[train_cut:val_cut]
test_idx = indices[val_cut:]

split_labels = np.empty(n_cells, dtype=object)
split_labels[train_idx] = 'train'
split_labels[val_idx] = 'validation'
split_labels[test_idx] = 'test'
adata.obs['experiment_split'] = split_labels

# Save split indices and metadata.
np.save(experiment_dir / 'cell_indices_train.npy', train_idx)
np.save(experiment_dir / 'cell_indices_validation.npy', val_idx)
np.save(experiment_dir / 'cell_indices_test.npy', test_idx)
pd.DataFrame({'cell_id': adata.obs_names, 'split': split_labels}).to_csv(
    experiment_dir / 'cell_split_table.csv', index=False
)

# Save cell metadata and gene list.
cell_metadata = adata.obs.copy()
if 'total_counts' not in cell_metadata.columns:
    total_counts = np.asarray(adata.X.sum(axis=1)).reshape(-1)
    cell_metadata['total_counts'] = total_counts
cell_metadata.to_csv(experiment_dir / 'cell_metadata.csv', index=True)

pd.DataFrame({'gene': adata.var_names}).to_csv(experiment_dir / 'gene_list.csv', index=False)

# Train/validation/test AnnLoaders.
train_adata = adata[train_idx].copy()
val_adata = adata[val_idx].copy()
test_adata = adata[test_idx].copy()

batch_size = 128
test_batch_size = 256
train_loader = AnnLoader(train_adata, batch_size=batch_size, shuffle=True)
val_loader = AnnLoader(val_adata, batch_size=test_batch_size, shuffle=False)
test_loader = AnnLoader(test_adata, batch_size=test_batch_size, shuffle=False)

# Define the full single-cell autoencoder model.
cdims = adata.shape[-1]
latent_dim = 32

# TODO: Define the SCAutoEncoder model architecture with negative binomial likelihood and global dispersion parameter.  
class SCAutoEncoder(nn.Module):
    def __init__(self):
        super(SCAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(cdims, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1), #
            nn.Linear(256, 128), #
            nn.LayerNorm(128), #
            nn.LeakyReLU(0.2),#
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, cdims),
            nn.Softplus()
        )
        self.dispersion_param = nn.Parameter(torch.full((cdims,), 0.5))

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def decode(self, z):
        return self.decoder(z)

    def get_dispersion(self):
        return F.softplus(self.dispersion_param) + 1e-4

    def layer_size_bounds(self):
        encoder_out = [m.out_features for m in self.encoder if isinstance(m, nn.Linear)]
        decoder_out = [m.out_features for m in self.decoder if isinstance(m, nn.Linear)]
        # Exclude the final decoder output size (cdims) to focus on AE architecture widths.
        ae_sizes = encoder_out + decoder_out[:-1]
        return min(ae_sizes), max(ae_sizes)

# Initialize the SCAutoEncoder model
model = SCAutoEncoder().to(mps_device)
min_layer_size, max_layer_size = model.layer_size_bounds()
layer_tag = f"_min{min_layer_size}_max{max_layer_size}"

# Define the negative binomial loss function and evaluation metrics for the autoencoder.
def negative_binomial_log_prob(counts, mu, dispersion, eps=1e-8):
    mu = mu.clamp(min=eps)
    theta = dispersion.clamp(min=eps)
    if theta.ndim == 1:
        theta = theta.unsqueeze(0)  # (1, G) then broadcast
    log_theta_mu = torch.log(theta + mu + eps)
    log_mu = torch.log(mu + eps)
    log_theta = torch.log(theta + eps)

    t1 = torch.lgamma(theta + counts) - torch.lgamma(theta) - torch.lgamma(counts + 1.0)
    t2 = theta * (log_theta - log_theta_mu)
    t3 = counts * (log_mu - log_theta_mu)
    return t1 + t2 + t3

def negative_binomial_loss(counts, mu, dispersion, eps=1e-8):
    log_prob = negative_binomial_log_prob(counts, mu, dispersion, eps=eps)
    return -(log_prob.sum(dim=1)).mean()

def negative_binomial_deviance(counts, mu, dispersion, eps=1e-8):
    mu = mu.clamp(min=eps)
    mu_sat = counts.clamp(min=eps)
    log_prob_model = negative_binomial_log_prob(counts, mu, dispersion, eps=eps)
    log_prob_sat = negative_binomial_log_prob(counts, mu_sat, dispersion, eps=eps)
    return 2.0 * (log_prob_sat - log_prob_model)

# Evaluation function for validation and test sets. Computes average negative log-likelihood loss across the dataset.
def evaluate_loader(loader):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            pbatch = batch.X
            if pbatch.is_sparse:
                pbatch = pbatch.to_dense()
            pbatch = pbatch.to(mps_device, dtype=torch.float32)
            reconstructed = model(pbatch)
            dispersion = model.get_dispersion()
            loss = negative_binomial_loss(pbatch, reconstructed, dispersion)
            losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0


optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)
scheduler = None

training_metadata = {
    'experiment_id': experiment_id,
    'timestamp': datetime.datetime.now().isoformat(),
    'model_architecture': {
        'encoder_layers': [cdims, 512, 256, latent_dim],
        'decoder_layers': [latent_dim, 256, 512, cdims],
        'activation': 'LeakyReLU(0.2)',
        'dropout': 0.1,
        'output_activation': 'Softplus',
        'latent_dim': latent_dim,
        'global_dispersion': True,
    },
    'hyperparameters': {
        'learning_rate': 1e-3,
        'weight_decay': 1e-8,
        'batch_size': batch_size,
        'epochs': 20,
        'loss_type': 'negative_binomial',
        'dispersion_scheme': 'gene-wise global dispersion',
    },
    'preprocessing': {
        'counts_mode': 'raw counts',
        'log1p': False,
        'normalization': False,
        'highly_variable_genes': True,
        'n_genes': int(adata.n_vars),
    },
    'split_strategy': {
        'train_fraction': 0.8,
        'validation_fraction': 0.1,
        'test_fraction': 0.1,
    },
    'seeds': {
        'python': seed,
        'numpy': seed,
        'torch': seed,
        'cuda': seed if torch.cuda.is_available() else None,
    },
    'raw_data_source': {
        'file_path': file_path,
        'file_checksum': raw_file_checksum,
    },
}

# Save package environment if available.
if pkg_resources is not None:
    try:
        packages = {dist.project_name: dist.version for dist in pkg_resources.working_set}
        training_metadata['package_versions'] = {
            'python': platform.python_version(),
            'torch': torch.__version__,
            'scanpy': sc.__version__,
            'anndata': ad.__version__,
            'numpy': np.__version__,
            'pandas': pd.__version__,
            'packages': {k: packages[k] for k in ['torch', 'scanpy', 'anndata', 'numpy', 'pandas'] if k in packages}
        }
    except Exception:
        training_metadata['package_versions'] = {
            'python': platform.python_version(),
            'torch': torch.__version__,
            'scanpy': sc.__version__,
            'anndata': ad.__version__,
            'numpy': np.__version__,
            'pandas': pd.__version__,
        }
else:
    training_metadata['package_versions'] = {
        'python': platform.python_version(),
        'torch': torch.__version__,
        'scanpy': sc.__version__,
        'anndata': ad.__version__,
        'numpy': np.__version__,
        'pandas': pd.__version__,
    }

training_metadata['runtime'] = {
    'device': str(mps_device),
    'cuda_available': torch.cuda.is_available(),
    'cuda_version': torch.version.cuda,
    'cudnn_version': torch.backends.cudnn.version(),
    'python_executable': sys.executable,
    'platform': platform.platform(),
}

save_json(experiment_dir / 'metadata.json', training_metadata)
save_rng_state(experiment_dir / 'rng_state_initial.pt')

# Save a small readme for reproducibility.
with open(experiment_dir / 'README.txt', 'w', encoding='utf-8') as handle:
    handle.write('This experiment folder contains model weights, optimizer state, data splits, and evaluation metrics.\n')
    handle.write('Load gene order from gene_list.csv and cell split data from cell_split_table.csv.\n')
    handle.write('Example: model.load_state_dict(torch.load(\"model_epoch20.pt\"))\n')
    handle.write('Reproducible command: python autoencoder.py\n')

# Train the autoencoder.
# TODO: Implement the training loop for the SCAutoEncoder model, including saving 
# model checkpoints, training curves, and final evaluation metrics on the test set.
epochs = 50
train_history = []
val_history = []
epoch_times = []

for epoch in range(epochs):
    epoch_start = time.time()
    model.train()
    batch_losses = []
    for batch in train_loader:
        pbatch = batch.X
        if pbatch.is_sparse:
            pbatch = pbatch.to_dense()
        pbatch = pbatch.to(mps_device, dtype=torch.float32)
        reconstructed = model(pbatch)
        dispersion = model.get_dispersion()
        loss = negative_binomial_loss(pbatch, reconstructed, dispersion)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
    train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
    val_loss = evaluate_loader(val_loader)
    epoch_end = time.time()
    epoch_times.append(epoch_end - epoch_start)
    train_history.append(train_loss)
    val_history.append(val_loss)

    torch.save(model.state_dict(), experiment_dir / f'model_epoch{epoch+1}.pt')
    torch.save(optimizer.state_dict(), experiment_dir / f'optim_epoch{epoch+1}.pt')
    save_rng_state(experiment_dir / f'rng_state_epoch{epoch+1}.pt')
    if scheduler is not None:
        torch.save(scheduler.state_dict(), experiment_dir / f'scheduler_epoch{epoch+1}.pt')

    print(f'Epoch {epoch+1}/{epochs} | train_loss: {train_loss:.6f} | val_loss: {val_loss:.6f} | time: {epoch_times[-1]:.2f}s')

# Save training curves.
train_curve = pd.DataFrame({'epoch': np.arange(1, epochs + 1), 'train_loss': train_history, 'val_loss': val_history, 'epoch_time_s': epoch_times})
train_curve.to_csv(experiment_dir / 'training_history.csv', index=False)

plt.figure(figsize=(8, 5))
plt.plot(train_curve['epoch'], train_curve['train_loss'], marker='o', label='train_loss')
plt.plot(train_curve['epoch'], train_curve['val_loss'], marker='o', label='val_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.tight_layout()
plt.savefig(plot_dir / f'training_loss{layer_tag}.png', dpi=150, bbox_inches='tight')
plt.close()

# Final evaluation on full train/validation/test sets.
final_test_loss = evaluate_loader(test_loader)
final_train_loss = evaluate_loader(train_loader)
final_val_loss = evaluate_loader(val_loader)

# Compute full dataset embeddings and metrics.
full_loader = AnnLoader(adata, batch_size=adata.n_obs, shuffle=False)
full_counts = None
full_reconstruction = None
full_latents = None
with torch.no_grad():
    for batch in full_loader:
        pbatch = batch.X
        if pbatch.is_sparse:
            pbatch = pbatch.to_dense()
        full_counts = pbatch.to(mps_device, dtype=torch.float32)
        full_latents = model.encoder(full_counts)
        full_reconstruction = model.decoder(full_latents)
        break

dispersion = model.get_dispersion()
full_log_prob = negative_binomial_log_prob(full_counts, full_reconstruction, dispersion)
full_nll = -full_log_prob.sum(dim=1)
final_avg_nll = float(full_nll.mean().item())

mse = torch.mean((full_reconstruction - full_counts) ** 2).item()
mae = torch.mean(torch.abs(full_reconstruction - full_counts)).item()
rmse = float(torch.sqrt(torch.mean((full_reconstruction - full_counts) ** 2)).item())
full_deviance = negative_binomial_deviance(full_counts, full_reconstruction, dispersion)
nb_deviance = float(full_deviance.mean().item())

per_gene_mse = torch.mean((full_reconstruction - full_counts) ** 2, dim=0).cpu().numpy()
per_gene_mae = torch.mean(torch.abs(full_reconstruction - full_counts), dim=0).cpu().numpy()
per_gene_nb_deviance = full_deviance.sum(dim=0).detach().cpu().numpy()

pd.DataFrame({
    'gene': adata.var_names,
    'mse': per_gene_mse,
    'mae': per_gene_mae,
    'nb_deviance': per_gene_nb_deviance,
}).to_csv(experiment_dir / 'per_gene_metrics.csv', index=False)

# Save latent embeddings for train/validation/test splits.
np.savez(
    experiment_dir / 'latent_embeddings.npz',
    train=full_latents[train_idx].cpu().numpy(),
    validation=full_latents[val_idx].cpu().numpy(),
    test=full_latents[test_idx].cpu().numpy(),
    cell_ids=np.array(adata.obs_names),
)

metrics = {
    'final_train_loss': final_train_loss,
    'final_validation_loss': final_val_loss,
    'final_test_loss': final_test_loss,
    'final_avg_negative_log_likelihood': final_avg_nll,
    'mse': mse,
    'mae': mae,
    'rmse': rmse,
    'nb_deviance': nb_deviance,
    'num_cells': int(n_cells),
    'num_genes': int(cdims),
}
save_json(experiment_dir / 'final_metrics.json', metrics)

system_info = {
    'python_version': platform.python_version(),
    'sys_executable': sys.executable,
    'platform': platform.platform(),
    'cpu': platform.processor(),
    'device': str(mps_device),
    'cuda_available': torch.cuda.is_available(),
    'cuda_version': torch.version.cuda,
    'cudnn_version': torch.backends.cudnn.version(),
}
save_json(experiment_dir / 'system_info.json', system_info)

archive_path = shutil.make_archive(str(experiment_dir), 'gztar', root_dir=experiment_dir.parent, base_dir=experiment_dir.name)
archive_checksum = compute_sha256(archive_path)
save_json(experiment_dir / 'archive_checksum.json', {'archive_path': str(archive_path), 'sha256': archive_checksum})

print(f'Experiment saved to {experiment_dir}')
print(f'Archive created at {archive_path}')

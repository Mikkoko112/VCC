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

Needed changes to the code: see TODO
- Adjust file paths and dataset loading as needed.
- Modify model architecture, hyperparameters, or training loop for experimentation.

=> maybe change it to a function that takes the file path as input and then runs 
the whole workflow, so that it can be easily called with different datasets 
without changing the code.
"""

# --------------------------------imports---------------------------------
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
try:
    import psutil
except Exception:
    psutil = None


# --------------------------------settings--------------------------------- 
# Configure device for PyTorch. For this workflow, CPU is the stable default.
use_mps = False
if use_mps and torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print(x)
else:
    mps_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {mps_device}")

# Fraction of available RAM considered safe to use (e.g. 0.75 = 75%)
MEMORY_SAFETY_FRACTION = 0.75

# Create a reproducible experiment folder.
experiment_date = datetime.date.today().isoformat()
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
experiment_id = uuid.uuid4().hex[:8]
experiment_name = f"experiment_{experiment_date}_{timestamp}_{experiment_id}"

# TODO: put here the directory where you want to save the experiment results, 
# e.g. "/path/to/experiments"
experiment_dir = f"/beegfs/$USER/autoencoder_runs/run1/{experiment_name}"
#experiment_dir = Path(__file__).parent / "experiments" / experiment_name

experiment_dir.mkdir(parents=True, exist_ok=False)
plot_dir = experiment_dir / "figures"
plot_dir.mkdir(parents=True, exist_ok=True)

# TODO: Read the .h5ad file using Scanpy
file_path = "/beegfs/g/kern/bbb7169/vcc_data/PBMC-X-HealthyAging-X-scRNAseq-X-2023-X-Terekhova-X-10.1016_j.immuni.2023.10.013.h5ad"

# Read the AnnData file. For very large files prefer backed mode 
# `sc.read_h5ad(..., backed='r')`.
adata = sc.read_h5ad(file_path, backed='r')

# make an AnnData that contains raw counts and then select HVGs on it
if adata.raw is not None:
    adata = adata.raw.to_adata() # full counts AnnData
else:
    print("No raw data stored in this AnnData object")
    # Use the current adata instead
    adata = adata.copy()  
# Subset the Anndata object to the top 5000 highly variable genes
# now adata.X contains counts for the HVGs
sc.pp.highly_variable_genes(adata, n_top_genes=5000, flavor="seurat_v3", subset=True, inplace=True)

# maybe delete later ---------
# Make sure to reset the core matrix to raw counts
# If you want to keep only cells 1000: and keep full AnnData consistency
#adata = adata[:8030, :].copy()
#adata = adata.copy()  # ensure we have a copy to modify without affecting original
#adata.X = adata.raw.X.copy()   # keep raw counts in X for NB likelihood training
# Keep raw counts in X for negative binomial training and avoid log1p transformations
# sc.pp.normalize_total(adata, target_sum=1e6, inplace=True)
# sc.pp.log1p(adata)

# --------------------------------functions--------------------------------
# minimal safety checks before heavy model construction and dense conversions
def _available_memory_bytes():
    if psutil is None:
        return None
    return psutil.virtual_memory().available

def _human_bytes(n):
    for unit in ['B','KB','MB','GB','TB']:
        if abs(n) < 1024.0:
            return f"{n:3.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}PB"

def _estimate_and_check_memory(
        adata, 
        file_path, 
        model, 
        safety_fraction=MEMORY_SAFETY_FRACTION):
    """
    Estimate memory footprint based on actual model architecture.
    """
    avail = _available_memory_bytes()
    n_cells, n_genes = int(adata.n_obs), int(adata.n_vars)
    is_backed = getattr(adata.X, 'filename', None) is not None

    # Estimate dense data bytes (float32)
    try:
        if getattr(adata.X, 'nnz', None) is not None:
            nnz = int(adata.X.nnz)
            dense_data_bytes = n_cells * n_genes * 4
            sparse_data_bytes = nnz * 8
            data_repr = 'sparse'
        else:
            dense_data_bytes = n_cells * n_genes * 4
            sparse_data_bytes = None
            data_repr = 'dense'
    except Exception:
        dense_data_bytes = n_cells * n_genes * 4
        sparse_data_bytes = None
        data_repr = 'unknown'

    # Count actual model parameters
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_bytes = param_count * 4

    # If backed, only estimate batch-size memory, not full dataset
    if is_backed:
        # Assume largest batch size (1024) will be in memory
        batch_data_bytes = 1024 * n_genes * 4
        estimate_total = batch_data_bytes + param_bytes * 3
    else:
        # Full dataset in memory
        estimate_total = dense_data_bytes + param_bytes * 3

    report = {
        'file_size_bytes': os.path.getsize(file_path) if os.path.exists(file_path) else None,
        'n_cells': n_cells,
        'n_genes': n_genes,
        'is_backed': is_backed,
        'data_representation': data_repr,
        'dense_data_bytes': dense_data_bytes,
        'sparse_data_bytes': sparse_data_bytes,
        'param_count': int(param_count),
        'param_bytes': int(param_bytes),
        'estimated_total_bytes': int(estimate_total),
        'available_memory_bytes': avail,
    }

    # Save small diagnostics next to the script (if writable)
    try:
        diag_path = Path(__file__).parent / 'memory_diagnostics.json'
        with diag_path.open('w', encoding='utf-8') as fh:
            json.dump(report, fh, indent=2, default=str)
    except Exception:
        pass

    # Check memory
    if avail is not None:
        if estimate_total > avail * safety_fraction:
            msg = (
                f"Estimated memory need { _human_bytes(estimate_total) } exceeds "
                f"{int(safety_fraction*100)}% of available RAM {_human_bytes(avail)}.\n"
                "Aborting to avoid OOM. Options:\n"
                " - Use a smaller gene set (HVG selection) before training.\n"
                " - Use `backed='r'` mode (you are using backed={is_backed}).\n"
                " - Reduce model size (fewer hidden units / smaller latent_dim).\n"
                " - Train on a machine with more RAM.\n"
            )
            raise MemoryError(msg)

    return report

# ------functions: Reproducibility and experiment tracking setup---------
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

# ------------functions: Training of the autoencoder----------------------
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

def negative_binomial_loss(counts, mu, dispersion, eps=1e-8, reduce='sum_genes_mean_batch'):
    """
    reduce: 'mean' (default, across genes and batch) or 'sum_genes_mean_batch'
    """
    log_prob = negative_binomial_log_prob(counts, mu, dispersion, eps=eps)
    if reduce == 'mean':
        return -(log_prob.mean())  # Average across ALL elements
    elif reduce == 'sum_genes_mean_batch':
        return -(log_prob.sum(dim=1)).mean()  # Sum genes per sample, avg across batch
    return -log_prob

def negative_binomial_deviance(counts, mu, dispersion, eps=1e-8):
    mu = mu.clamp(min=eps)
    mu_sat = counts.clamp(min=eps)
    log_prob_model = negative_binomial_log_prob(counts, mu, dispersion, eps=eps)
    log_prob_sat = negative_binomial_log_prob(counts, mu_sat, dispersion, eps=eps)
    return 2.0 * (log_prob_sat - log_prob_model)

# Evaluation function for validation and test sets. Computes average negative log-likelihood loss across the dataset.
def evaluate_loader(loader, compute_deviance=False):
    """Returns loss and optionally deviance and reconstruction error."""
    model.eval()
    losses, deviances, recon_errors = [], [], []
    with torch.no_grad():
        for batch in loader:
            pbatch = batch.X
            if getattr(pbatch, 'is_sparse', False):
                try:
                    rows, cols = pbatch.shape
                    dense_bytes = int(rows) * int(cols) * 4
                    avail = _available_memory_bytes()
                    if avail is not None and dense_bytes > avail * MEMORY_SAFETY_FRACTION:
                        raise MemoryError(
                            f"Batch dense conversion would need {_human_bytes(dense_bytes)}, "
                            f">{MEMORY_SAFETY_FRACTION}% of available RAM {_human_bytes(avail)}. Aborting to avoid OOM."
                        )
                except Exception:
                    pass
                pbatch = pbatch.to_dense()
            pbatch = pbatch.to(mps_device, dtype=torch.float32)
            reconstructed = model(pbatch)
            dispersion = model.get_dispersion()
            
            loss = negative_binomial_loss(pbatch, reconstructed, dispersion)
            losses.append(loss.item())
            
            # L2 reconstruction error (optional, for sanity)
            recon_error = torch.mean((pbatch - reconstructed) ** 2).item()
            recon_errors.append(recon_error)
            
            if compute_deviance:
                dev = negative_binomial_deviance(pbatch, reconstructed, dispersion).mean()
                deviances.append(dev.item())
    
    result = {
        'loss': float(np.mean(losses)) if losses else 0.0,
        'recon_mse': float(np.mean(recon_errors)) if recon_errors else 0.0,
    }
    if compute_deviance:
        result['deviance'] = float(np.mean(deviances)) if deviances else 0.0
    return result

# --------------------functions: Final evaluation------------------------
def process_full_dataset(adata, model, mps_device, MEMORY_SAFETY_FRACTION=0.8):
    """
    Process full dataset in batches to compute latents and reconstructions.
    Returns tensors on CPU to avoid GPU overflow.
    """
    full_latents_list = []
    full_reconstruction_list = []
    full_counts_list = []
    
    full_loader = AnnLoader(adata, batch_size=adata.n_obs, shuffle=False)
    
    model.eval()
    with torch.no_grad():
        for batch in full_loader:
            pbatch = batch.X
            
            # Memory check
            if getattr(pbatch, 'is_sparse', False):
                try:
                    rows, cols = pbatch.shape
                    dense_bytes = int(rows) * int(cols) * 4
                    avail = _available_memory_bytes()
                    if avail is not None and dense_bytes > avail * MEMORY_SAFETY_FRACTION:
                        raise MemoryError(
                            f"Full-dataset dense conversion would need {_human_bytes(dense_bytes)}, "
                            f">{MEMORY_SAFETY_FRACTION*100:.0f}% of available RAM {_human_bytes(avail)}. Aborting to avoid OOM."
                        )
                except MemoryError as e:
                    print(f"Error: {e}")
                    raise
                except Exception as e:
                    print(f"Unexpected error during memory check: {e}")
                    raise
                pbatch = pbatch.to_dense()
            
            pbatch = pbatch.to(mps_device, dtype=torch.float32)
            latents = model.encoder(pbatch)
            recon = model.decoder(latents)
            
            # Move to CPU immediately to free GPU memory
            full_latents_list.append(latents.detach().cpu())
            full_reconstruction_list.append(recon.detach().cpu())
            full_counts_list.append(pbatch.detach().cpu())
    
    # Concatenate all batches (though typically just one for full dataset)
    full_latents = torch.cat(full_latents_list, dim=0)
    full_reconstruction = torch.cat(full_reconstruction_list, dim=0)
    full_counts = torch.cat(full_counts_list, dim=0)
    
    print(f"Full dataset processed: latents shape {full_latents.shape}, recon shape {full_reconstruction.shape}")
    
    return full_latents, full_reconstruction, full_counts

def save_latent_embeddings(full_latents, train_idx, val_idx, test_idx, cell_ids, experiment_dir):
    """Save latent embeddings with train/val/test split indices."""
    np.savez(
        experiment_dir / 'latent_embeddings.npz',
        train_latents=full_latents[train_idx].numpy() if isinstance(full_latents, torch.Tensor) else full_latents[train_idx],
        val_latents=full_latents[val_idx].numpy() if isinstance(full_latents, torch.Tensor) else full_latents[val_idx],
        test_latents=full_latents[test_idx].numpy() if isinstance(full_latents, torch.Tensor) else full_latents[test_idx],
        cell_ids=np.array(cell_ids),
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
    )
    print(f"Latent embeddings saved: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

def compute_per_gene_metrics(full_counts, full_reconstruction, dispersion, adata, experiment_dir):
    """
    Compute per-gene reconstruction and NB metrics.
    All inputs should be on CPU or numpy arrays.
    """
    # Ensure tensors are on CPU
    if isinstance(full_counts, torch.Tensor):
        full_counts = full_counts.cpu()
    if isinstance(full_reconstruction, torch.Tensor):
        full_reconstruction = full_reconstruction.cpu()
    
    # Reconstruction metrics
    mse = torch.mean((full_reconstruction - full_counts) ** 2, dim=0).numpy()
    mae = torch.mean(torch.abs(full_reconstruction - full_counts), dim=0).numpy()
    
    # NB deviance per gene
    dispersion_np = dispersion.detach().cpu().numpy() if isinstance(dispersion, torch.Tensor) else dispersion
    full_deviance = negative_binomial_deviance(full_counts, full_reconstruction, dispersion)
    deviance_per_gene = full_deviance.sum(dim=0).detach().cpu().numpy() if isinstance(full_deviance, torch.Tensor) else full_deviance.sum(axis=0)
    
    # Save to CSV
    df_metrics = pd.DataFrame({
        'gene': adata.var_names,
        'mse': mse,
        'mae': mae,
        'nb_deviance': deviance_per_gene,
    })
    df_metrics.to_csv(experiment_dir / 'per_gene_metrics.csv', index=False)
    print(f"Per-gene metrics saved to {experiment_dir / 'per_gene_metrics.csv'}")
    
    return df_metrics

#######################################################################
######## Reproducibility and experiment tracking setup ################
#######################################################################
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

#######################################################################
################# Initialize the SCAutoEncoder model ###################
#######################################################################
train_adata = adata[train_idx]
val_adata = adata[val_idx]
test_adata = adata[test_idx]

# Recommended batch sizes for scRNA-seq (adjust if you have more/less RAM)
# Use smaller batches to reduce memory pressure on large datasets.
batch_size = 512
test_batch_size = 1024
# Save model checkpoints every `checkpoint_freq` epochs
checkpoint_freq = 10
train_loader = AnnLoader(train_adata, batch_size=batch_size, shuffle=True)
val_loader = AnnLoader(val_adata, batch_size=test_batch_size, shuffle=False)
test_loader = AnnLoader(test_adata, batch_size=test_batch_size, shuffle=False)

# Define the full single-cell autoencoder model.
cdims = adata.shape[-1]
# latent dimensionality (tunable)
latent_dim = 256

# TODO: Define the SCAutoEncoder model architecture with negative binomial 
# likelihood and global dispersion parameter.  
class SCAutoEncoder(nn.Module):
    def __init__(self):
        super(SCAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(cdims, 4096),
            nn.LayerNorm(4096),
            nn.LeakyReLU(0.2),
            nn.Linear(4096, 2048),
            nn.LayerNorm(2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 2048),
            nn.LayerNorm(2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 4096),
            nn.LayerNorm(4096),
            nn.LeakyReLU(0.2),
            nn.Linear(4096, cdims),
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

# Run memory checks and possibly abort early with a helpful message
try:
    _estimate_and_check_memory(adata, file_path, model)
except MemoryError as e:
    print("Memory check failed:\n", e)
    # Write a short failure note to the current directory and exit cleanly
    try:
        fail_note = Path('.') / 'memory_check_failed.txt'  # Current working directory
        fail_note.write_text(str(e))
    except Exception:
        pass
    sys.exit(1)

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)
scheduler = None

#######################################################################
################# Reproducibility for training setup ##################
#######################################################################
training_metadata = {
    'experiment_id': experiment_id,
    'timestamp': datetime.datetime.now().isoformat(),
    'model_architecture': {
        'encoder_layers': [cdims] + [m.out_features for m in model.encoder if isinstance(m, nn.Linear)],
        'decoder_layers': [latent_dim] + [m.out_features for m in model.decoder if isinstance(m, nn.Linear)],
        'activation': 'LeakyReLU(0.2)',
        'dropout': None,
        'output_activation': 'Softplus',
        'latent_dim': int(latent_dim),
        'global_dispersion': True,
    },
    'hyperparameters': {
        'learning_rate': 1e-3,
        'weight_decay': 1e-8,
        'batch_size': int(batch_size),
        'test_batch_size': int(test_batch_size),
        'epochs': 200,
        'checkpoint_frequency': int(checkpoint_freq),
        'early_stopping': {
            'enabled': True,
            'patience': 15,
            'min_delta': 1e-4,
        },
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
    'loss_details': {
    'reduction': 'sum_genes_mean_batch',  # Per-cell sum, batch mean
    'numerical_stability': 'log-space throughout',
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
    'scheduler': {
    'type': None,
    'enabled': False,
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
    handle.write('gene_list.csv \t — ordered list of approx 5k HVGs used\n')
    handle.write('cell_split_table.csv \t — train/val/test cell indices\n')
    handle.write('model_epoch_*.pt \t — checkpoint weights\n')
    handle.write('metadata.json \t — complete hyperparameter + environment record\n')

#######################################################################
################# Training of the autoencoder #########################
#######################################################################

# TODO: Implement the training loop for the SCAutoEncoder model, including saving 
# model checkpoints, training curves, and final evaluation metrics on the test set.
epochs = 1
train_history = []
val_history = []
epoch_times = []
# Early stopping configuration: stop if validation loss doesn't improve
# for `early_stopping_patience` epochs by at least `early_stopping_min_delta`.
early_stopping_enabled = True
early_stopping_patience = 15
early_stopping_min_delta = 1e-4
best_val = float('inf')
epochs_no_improve = 0
best_val_epoch = 0 

for epoch in range(epochs):
    epoch_start = time.time()
    model.train()
    batch_losses = []
    for batch in train_loader:
        pbatch = batch.X
        if getattr(pbatch, 'is_sparse', False):
            try:
                rows, cols = pbatch.shape
                dense_bytes = int(rows) * int(cols) * 4
                avail = _available_memory_bytes()
                if avail is not None and dense_bytes > avail * MEMORY_SAFETY_FRACTION:
                    raise MemoryError(
                        f"Batch dense conversion would need {_human_bytes(dense_bytes)}, "
                        f">{MEMORY_SAFETY_FRACTION}% of available RAM {_human_bytes(avail)}. Aborting to avoid OOM."
                    )
            except Exception:
                pass
            pbatch = pbatch.to_dense()
        pbatch = pbatch.to(mps_device, dtype=torch.float32)
        reconstructed = model(pbatch)
        dispersion = model.get_dispersion()
        loss = negative_binomial_loss(pbatch, reconstructed, dispersion)
        optimizer.zero_grad()
        loss.backward()
        # NB loss can produce unstable gradients early in training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        batch_losses.append(loss.item())
    train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
    val_metrics = evaluate_loader(val_loader, compute_deviance=False)
    val_loss = val_metrics['loss']
    val_recon_mse = val_metrics['recon_mse']
    epoch_end = time.time()
    epoch_times.append(epoch_end - epoch_start)
    train_history.append(train_loss)
    val_history.append(val_loss)

    # Early stopping: save best model and stop if no improvement for patience epochs
    if early_stopping_enabled:
        if val_loss < best_val - early_stopping_min_delta:
            best_val = val_loss
            best_val_epoch = epoch + 1
            epochs_no_improve = 0
            # save best model snapshot
            torch.save(model.state_dict(), experiment_dir / 'model_best.pt')
            torch.save(optimizer.state_dict(), experiment_dir / 'optim_best.pt')
            save_rng_state(experiment_dir / 'rng_state_best.pt')
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping: validation loss did not improve for {epochs_no_improve} epochs. Stopping training.")
            break

    # Save checkpoints only at the configured frequency to reduce I/O and disk usage.
    if (epoch + 1) % checkpoint_freq == 0:
        torch.save(model.state_dict(), experiment_dir / f'model_epoch{epoch+1}.pt')
        torch.save(optimizer.state_dict(), experiment_dir / f'optim_epoch{epoch+1}.pt')
        save_rng_state(experiment_dir / f'rng_state_epoch{epoch+1}.pt')
        if scheduler is not None:
            torch.save(scheduler.state_dict(), experiment_dir / f'scheduler_epoch{epoch+1}.pt')
        
        # Save training curves so far
        training_curves = {
            'epochs_completed': epoch + 1,
            'train_loss': train_history,
            'val_loss': val_history,
            'epoch_times': epoch_times,
            'best_epoch': best_val_epoch
        }
        save_json(experiment_dir / 'training_curves.json', training_curves)

    print(f'Epoch {epoch+1}/{epochs} | train_loss: {train_loss:.6f} | val_loss: {val_loss:.6f} | val_mse: {val_recon_mse:.6f} | time: {epoch_times[-1]:.2f}s')

# Save training curves using the actual number of epochs run.
num_epochs_run = len(train_history)
train_curve = pd.DataFrame({'epoch': np.arange(1, num_epochs_run + 1), 'train_loss': train_history, 'val_loss': val_history, 'epoch_time_s': epoch_times})
train_curve.to_csv(experiment_dir / 'training_history.csv', index=False)

# Load best model for final evaluation
model.load_state_dict(torch.load(experiment_dir / 'model_best.pt'))
test_metrics = evaluate_loader(test_loader, compute_deviance=True)

final_results = {
    'best_epoch': best_val_epoch,
    'best_val_loss': best_val,
    'test_loss': test_metrics['loss'],
    'test_mse': test_metrics['recon_mse'],
    'test_deviance': test_metrics.get('deviance', None),
    'total_epochs_trained': len(train_history),
    'total_training_time_seconds': sum(epoch_times),
    'avg_epoch_time_seconds': float(np.mean(epoch_times)),
}
save_json(experiment_dir / 'final_results.json', final_results)
print(f"\n{'='*60}")
print(f"Training complete!")
print(f"Best validation loss: {best_val:.6f}")
print(f"Test loss: {test_metrics['loss']:.6f}")
print(f"Total training time: {sum(epoch_times)/60:.1f} minutes")
print(f"{'='*60}\n")

plt.figure(figsize=(8, 5))
plt.plot(train_curve['epoch'], train_curve['train_loss'], marker='o', label='train_loss', linewidth=1.5)
plt.plot(train_curve['epoch'], train_curve['val_loss'], marker='o', label='val_loss', linewidth=1.5)
plt.axvline(x=best_val_epoch, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Best Val (epoch {best_val_epoch})')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(plot_dir / f'training_loss{layer_tag}.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Training curves saved to {plot_dir / f'training_loss{layer_tag}.png'}")


#######################################################################
######################### Final evaluation ############################
#######################################################################
model.load_state_dict(torch.load(experiment_dir / 'model_best.pt'))
model.eval()

final_train_loss = train_history[-1]
final_val_loss = best_val
final_test_loss = test_metrics['loss']

# Compute full-dataset embeddings (batched, safe)
full_latents, full_reconstruction, full_counts = process_full_dataset(
    adata, model, mps_device, MEMORY_SAFETY_FRACTION
)

# Save latent embeddings immediately
save_latent_embeddings(
    full_latents, train_idx, val_idx, test_idx, 
    adata.obs_names, experiment_dir
)

# Compute & save per-gene metrics
dispersion = model.get_dispersion()
compute_per_gene_metrics(full_counts, full_reconstruction, dispersion, adata, experiment_dir)

# Compute global metrics
mse = torch.mean((full_reconstruction - full_counts) ** 2).item()
mae = torch.mean(torch.abs(full_reconstruction - full_counts)).item()
rmse = torch.sqrt(torch.mean((full_reconstruction - full_counts) ** 2)).item()

full_counts_gpu = full_counts.to(mps_device, dtype=torch.float32)
full_recon_gpu = full_reconstruction.to(mps_device, dtype=torch.float32)
full_log_prob = negative_binomial_log_prob(full_counts_gpu, full_recon_gpu, dispersion)
full_nll = -full_log_prob.sum(dim=1)
final_avg_nll = float(full_nll.mean().item())

full_deviance = negative_binomial_deviance(full_counts_gpu, full_recon_gpu, dispersion)
nb_deviance = float(full_deviance.mean().item())

metrics = {
    'final_train_loss': float(final_train_loss),
    'final_validation_loss': float(final_val_loss),
    'final_test_loss': float(final_test_loss),
    'final_avg_negative_log_likelihood': final_avg_nll,
    'mse': mse,
    'mae': mae,
    'rmse': rmse,
    'nb_deviance': nb_deviance,
    'num_cells': int(adata.n_obs),
    'num_genes': int(adata.n_vars),
    'latent_dim': int(model.latent_dim),
}

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

save_json(experiment_dir / 'final_metrics.json', metrics)
save_json(experiment_dir / 'system_info.json', system_info)

# Archive
archive_path = shutil.make_archive(str(experiment_dir), 'gztar', root_dir=experiment_dir.parent, base_dir=experiment_dir.name)
archive_checksum = compute_sha256(archive_path)
save_json(experiment_dir / 'archive_checksum.json', {'archive_path': str(archive_path), 'sha256': archive_checksum})

print(f'\nExperiment saved to {experiment_dir}')
print(f'Archive created at {archive_path}')

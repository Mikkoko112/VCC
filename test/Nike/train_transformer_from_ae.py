#!/usr/bin/env python3

"""
train_transformer_from_ae.py
Example script to load a pretrained AE, encode a new perturbed dataset, and train a simple Transformer model on the AE latent space embeddings.
This is a starting point for experimentation. Adjust model architecture, hyperparameters, and training loop as needed.
Make sure to adjust file paths and dataset loading as needed for your specific use case.

Needed changes to the code:
- Adjust file paths and dataset loading as needed.
- Modify model architecture, hyperparameters, or training loop for experimentation.

=> maybe change it to a function that takes the file path as input and then runs 
the whole workflow, so that it can be easily called with different ae models and 
datasets without changing the code. 
"""

import json
import random
import uuid
import datetime
import hashlib
import time
import platform
import sys
import shutil
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from anndata.experimental.pytorch import AnnLoader
from anndata import AnnData

# --- Configuration: edit paths ---
experiment_dir = Path("experiments/68k_50epochs_onemorelayer_min32_experiment_2026-03-31_20260331_171738_c2dbbcd0")    # dir where AE model_epochN.pt exist
ae_weights = experiment_dir / "model_epoch22.pt"
ae_code = experiment_dir / "metadata.json"              # optional metadata
perturbed_h5ad = Path("/Users/nike/Documents/Uni/Master/dieVirtuelleZelle/vcc_data/adata_Training.h5ad")  # your new dataset to train Transformer on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model definitions (must match AE architecture used when training) ---
cdims = 3000      # number of HVGs used for AE
latent_dim = 32

class SCAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
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
        z = self.encoder(x)
        return self.decoder(z)
    def encode(self, x):
        return self.encoder(x)

# --- Load AE and encoder weights ---
ae = SCAutoEncoder().to(device)
state = torch.load(ae_weights, map_location=device)
ae.load_state_dict(state)
ae.eval()

# Decide whether to freeze encoder
freeze_encoder = True
if freeze_encoder:
    for p in ae.encoder.parameters():
        p.requires_grad = False

# --- Transformer model using AE encoder outputs as token embeddings ---
# Here we treat each cell as a single "token sequence" of length 1 with embedding = encoder(cell).
# For a sequence task across cells, adapt sequence construction (below uses per-cell classification/regression).
# TODO: Transformer architecture and hyperparameters can be adjusted as needed. This is a simple example to get started!!
emb_dim = latent_dim
num_heads = 4
num_layers = 2
transformer_ff = 128
dropout = 0.1
out_dim = cdims  # example: reconstruct counts (or change to number of classes if classification)

class SmallTransformer(nn.Module):
    def __init__(self, emb_dim, num_heads, num_layers, ff_dim, out_dim, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # simple head for per-token output (here just maps embedding back to gene space)
        self.head = nn.Sequential(
            nn.Linear(emb_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, out_dim),
            nn.Softplus()
        )
    def forward(self, x):
        # x shape: (B, S, emb_dim)
        h = self.transformer(x)    # (B, S, emb_dim)
        out = self.head(h)         # (B, S, out_dim)
        return out

transformer = SmallTransformer(emb_dim, num_heads, num_layers, transformer_ff, out_dim, dropout).to(device)

# reproducible experiment output directory for transformer training
experiment_date = datetime.date.today().isoformat()
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
transformer_experiment_id = uuid.uuid4().hex[:8]
results_dir = Path(__file__).parent / 'experiments' / f'transformer_experiment_{experiment_date}_{timestamp}_{transformer_experiment_id}'
results_dir.mkdir(parents=True, exist_ok=False)
plot_dir = results_dir / 'figures'
plot_dir.mkdir(parents=True, exist_ok=True)


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

# --- Data loading: encode perturbed dataset with AE encoder to get embeddings ---
adata_p = sc.read_h5ad(perturbed_h5ad)
# smaller subset to test!
#adata_p = adata_p[:2000].copy()
adata_p = adata_p.copy()
# Ensure adata_p has same genes/HVG order as AE training (gene_list.csv). If not, subset/reorder:
# genes_expected = list(pd.read_csv(experiment_dir / "gene_list.csv")["gene"])
# adata_p = adata_p[:, genes_expected].copy()
# If adata_p has raw counts in adata_p.raw, use adata_p.raw.X similarly.

# Place raw counts into X if needed:
if adata_p.raw is not None:
    try:
        # pick only the AE HVGs if present in raw
        if set(adata_p.var_names).issuperset(set(adata_p.var_names)):
            pass
    except Exception:
        pass

# Make loader that yields raw count batches
batch_size = 128
# genes_expected: list of 3000 AE genes (ordered)
genes_expected = pd.read_csv(experiment_dir / "gene_list.csv")["gene"].tolist()

# adata_p is your perturbed AnnData (2168 genes)
present = [g for g in genes_expected if g in adata_p.var_names]
missing = [g for g in genes_expected if g not in adata_p.var_names]

# subset present genes in the expected order
adata_present = adata_p[:, present].copy()
n_cells = adata_present.n_obs
n_missing = len(missing)

# create zero matrix for missing genes (sparse)
if n_missing > 0:
    zero_mat = sparse.csr_matrix((n_cells, n_missing), dtype=adata_present.X.dtype)
    aligned_X = sparse.hstack([adata_present.X, zero_mat], format="csr")
    aligned_var = pd.Index(present + missing)
else:
    aligned_X = adata_present.X
    aligned_var = pd.Index(present)

# build new AnnData with same obs and new var
adata_aligned = AnnData(X=aligned_X, obs=adata_present.obs.copy(), var=pd.DataFrame(index=aligned_var))

# copy any needed var annotations from original where present
common = adata_p.var.index.intersection(adata_aligned.var.index)
adata_aligned.var.loc[common] = adata_p.var.loc[common]

# verify shapes
assert adata_aligned.n_vars == len(genes_expected)
adata_aligned.var_names = aligned_var  # ensure ordering

# save provenance and gene list for the aligned dataset
raw_file_checksum = compute_sha256(perturbed_h5ad)
raw_info = {
    'perturbed_h5ad_path': str(perturbed_h5ad),
    'perturbed_h5ad_checksum': raw_file_checksum,
    'n_cells': int(adata_aligned.n_obs),
    'n_genes': int(adata_aligned.n_vars),
}
save_json(results_dir / 'raw_provenance.json', raw_info)
pd.DataFrame({'gene': adata_aligned.var_names}).to_csv(results_dir / 'gene_list.csv', index=False)

# create reproducible splits for train / validation / test
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

n_cells = adata_aligned.n_obs
indices = np.arange(n_cells)
np.random.shuffle(indices)
train_cut = int(0.8 * n_cells)
val_cut = int(0.9 * n_cells)
train_idx = indices[:train_cut]
val_idx = indices[train_cut:val_cut]
test_idx = indices[val_cut:]

split_labels = np.empty(n_cells, dtype=object)
split_labels[train_idx] = 'train'
split_labels[val_idx] = 'validation'
split_labels[test_idx] = 'test'
adata_aligned.obs['experiment_split'] = split_labels

np.save(results_dir / 'cell_indices_train.npy', train_idx)
pd.DataFrame({'cell_id': adata_aligned.obs_names, 'split': split_labels}).to_csv(results_dir / 'cell_split_table.csv', index=False)
np.save(results_dir / 'cell_indices_validation.npy', val_idx)
np.save(results_dir / 'cell_indices_test.npy', test_idx)

# save cell metadata
cell_metadata = adata_aligned.obs.copy()
if 'total_counts' not in cell_metadata.columns:
    total_counts = np.asarray(adata_aligned.X.sum(axis=1)).reshape(-1)
    cell_metadata['total_counts'] = total_counts
cell_metadata.to_csv(results_dir / 'cell_metadata.csv', index=True)

train_adata = adata_aligned[train_idx].copy()
val_adata = adata_aligned[val_idx].copy()
test_adata = adata_aligned[test_idx].copy()

train_loader = AnnLoader(train_adata, batch_size=batch_size, shuffle=True)
val_loader = AnnLoader(val_adata, batch_size=batch_size, shuffle=False)
test_loader = AnnLoader(test_adata, batch_size=batch_size, shuffle=False)

# now use adata_aligned for AnnLoader / encoding
loader = AnnLoader(adata_aligned, batch_size=batch_size, shuffle=True)

# Helper: produce encoder embeddings for a batch
def batch_to_embeddings(batch):
    x = batch.X
    if getattr(x, "is_sparse", False):
        x = x.to_dense()
    x = x.to(device, dtype=torch.float32)
    with torch.no_grad():
        z = ae.encode(x)
    return z


def evaluate_loader(loader):
    transformer.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in loader:
            x = batch.X
            if getattr(x, "is_sparse", False):
                x = x.to_dense()
            x = x.to(device, dtype=torch.float32)
            z = ae.encode(x)
            seq = z.unsqueeze(1)
            pred = transformer(seq).squeeze(1)
            loss = criterion(pred, x)
            total_loss += loss.item() * x.shape[0]
            total_count += x.shape[0]
    return float(total_loss / total_count) if total_count else 0.0

# --- Training loop: use train/validation/test splits and save per-epoch outputs ---
optimizer = optim.Adam(transformer.parameters(), lr=1e-4, weight_decay=1e-6)
criterion = nn.MSELoss()   # objective: reconstruct raw counts
scheduler = None
n_epochs = 50 # TODO: adjust number of epochs, learning rate, and other hyperparameters as needed for convergence. Monitor training/validation loss to decide when to stop or if adjustments are needed.

# TODO: if using other ae change model architecture, update metadata accordingly, 
# and ensure batch_to_embeddings matches the AE encoding process (e.g. log1p, normalization, etc.) used during AE training. 
training_metadata = {
    'experiment_id': transformer_experiment_id,
    'timestamp': datetime.datetime.now().isoformat(),
    'model_architecture': {
        'ae_encoder_layers': [cdims, 512, 256, 128, latent_dim],
        'ae_decoder_layers': [latent_dim, 128, 256, 512, cdims],
        'transformer_num_heads': num_heads,
        'transformer_num_layers': num_layers,
        'transformer_ff_dim': transformer_ff,
        'activation': 'LeakyReLU(0.2)',
        'dropout': dropout,
        'output_activation': 'Softplus',
        'latent_dim': latent_dim,
        'global_dispersion': False,
    },
    'hyperparameters': {
        'learning_rate': 1e-4,
        'weight_decay': 1e-6,
        'batch_size': batch_size,
        'epochs': n_epochs,
        'loss_type': 'mse',
        'dispersion_scheme': 'None',
    },
    'preprocessing': {
        'counts_mode': 'raw counts',
        'log1p': False,
        'normalization': False,
        'highly_variable_genes': False,
        'hvg_gene_count': len(genes_expected),
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
        'perturbed_h5ad_path': str(perturbed_h5ad),
        'perturbed_h5ad_checksum': raw_file_checksum,
    },
}

save_json(results_dir / 'metadata.json', training_metadata)
save_rng_state(results_dir / 'rng_state_initial.pt')

train_history = []
val_history = []
test_history = []
epoch_times = []

for epoch in range(n_epochs):
    epoch_start = time.time()
    transformer.train()
    total_loss = 0.0
    total_count = 0
    for batch in train_loader:
        x = batch.X
        if getattr(x, "is_sparse", False):
            x = x.to_dense()
        x = x.to(device, dtype=torch.float32)
        with torch.no_grad():
            z = ae.encode(x)
        seq = z.unsqueeze(1)
        pred = transformer(seq).squeeze(1)
        loss = criterion(pred, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.shape[0]
        total_count += x.shape[0]

    train_loss = float(total_loss / total_count) if total_count else 0.0
    val_loss = evaluate_loader(val_loader)
    test_loss = evaluate_loader(test_loader)
    epoch_end = time.time()
    epoch_times.append(epoch_end - epoch_start)
    train_history.append(train_loss)
    val_history.append(val_loss)
    test_history.append(test_loss)

    torch.save(transformer.state_dict(), results_dir / f'model_epoch{epoch+1}.pt')
    torch.save(optimizer.state_dict(), results_dir / f'optim_epoch{epoch+1}.pt')
    save_rng_state(results_dir / f'rng_state_epoch{epoch+1}.pt')
    if scheduler is not None:
        torch.save(scheduler.state_dict(), results_dir / f'scheduler_epoch{epoch+1}.pt')

    print(f"Epoch {epoch+1}/{n_epochs} | train_loss: {train_loss:.6f} | val_loss: {val_loss:.6f} | test_loss: {test_loss:.6f} | time: {epoch_times[-1]:.2f}s")

# Save training curves.
train_curve = pd.DataFrame({
    'epoch': np.arange(1, n_epochs + 1),
    'train_loss': train_history,
    'val_loss': val_history,
    'test_loss': test_history,
    'epoch_time_s': epoch_times,
})
train_curve.to_csv(results_dir / 'training_history.csv', index=False)

plt.figure(figsize=(8, 5))
plt.plot(train_curve['epoch'], train_curve['train_loss'], marker='o', label='train_loss')
plt.plot(train_curve['epoch'], train_curve['val_loss'], marker='o', label='validation_loss')
plt.plot(train_curve['epoch'], train_curve['test_loss'], marker='o', label='test_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.tight_layout()
plt.savefig(plot_dir / 'training_loss.png', dpi=150, bbox_inches='tight')
plt.close()

# Final evaluation on full dataset predictions.
all_counts = []
all_preds = []
all_latents = []
with torch.no_grad():
    full_loader = AnnLoader(adata_aligned, batch_size=batch_size, shuffle=False)
    for batch in full_loader:
        x = batch.X
        if getattr(x, "is_sparse", False):
            x = x.to_dense()
        x = x.to(device, dtype=torch.float32)
        z = ae.encode(x)
        seq = z.unsqueeze(1)
        pred = transformer(seq).squeeze(1)
        all_latents.append(z.cpu().numpy())
        all_counts.append(x.cpu())
        all_preds.append(pred.cpu())

all_counts = torch.cat(all_counts, dim=0)
all_preds = torch.cat(all_preds, dim=0)
all_latents = np.vstack(all_latents)

mse = torch.mean((all_preds - all_counts) ** 2).item()
mae = torch.mean(torch.abs(all_preds - all_counts)).item()
rmse = float(torch.sqrt(torch.mean((all_preds - all_counts) ** 2)).item())

per_gene_mse = torch.mean((all_preds - all_counts) ** 2, dim=0).cpu().numpy()
per_gene_mae = torch.mean(torch.abs(all_preds - all_counts), dim=0).cpu().numpy()
pd.DataFrame({
    'gene': adata_aligned.var_names,
    'mse': per_gene_mse,
    'mae': per_gene_mae,
}).to_csv(results_dir / 'per_gene_metrics.csv', index=False)

np.savez(
    results_dir / 'latent_embeddings.npz',
    train=all_latents[train_idx],
    validation=all_latents[val_idx],
    test=all_latents[test_idx],
    cell_ids=np.array(adata_aligned.obs_names),
)

metrics = {
    'final_train_loss': train_history[-1] if train_history else None,
    'final_validation_loss': val_history[-1] if val_history else None,
    'final_test_loss': test_history[-1] if test_history else None,
    'mse': mse,
    'mae': mae,
    'rmse': rmse,
    'num_cells': int(n_cells),
    'num_genes': int(adata_aligned.n_vars),
}
save_json(results_dir / 'final_metrics.json', metrics)

system_info = {
    'python_version': platform.python_version(),
    'sys_executable': sys.executable,
    'platform': platform.platform(),
    'device': str(device),
    'cuda_available': torch.cuda.is_available(),
    'cuda_version': torch.version.cuda,
    'cudnn_version': torch.backends.cudnn.version(),
}
save_json(results_dir / 'system_info.json', system_info)

with open(results_dir / 'README.txt', 'w', encoding='utf-8') as handle:
    handle.write('This transformer experiment folder contains saved model checkpoints, optimizer states, data splits, and evaluation metrics.\n')
    handle.write('Load gene order from gene_list.csv and cell splits from cell_split_table.csv.\n')
    handle.write('Example: torch.load(\'model_epoch20.pt\') into SmallTransformer state_dict.\n')
    handle.write('Reproducible command: python train_transformer_from_ae.py\n')

print(f'Transformer experiment saved to {results_dir}')

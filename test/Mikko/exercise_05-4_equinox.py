#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax


# ----------------------------
# Load and preprocess data
# ----------------------------
file_path = Path(__file__).parent.parent.parent / "data" / "mesenchymal_subset.h5ad"
adata = sc.read_h5ad(file_path)

adata = adata[:8030, :].copy()
adata.X = adata.raw.X.copy()

sc.pp.normalize_total(adata, target_sum=1e6, inplace=True)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(
    adata,
    n_top_genes=3000,
    subset=True,
    flavor="seurat_v3",
    inplace=True,
)

# Convert matrix to dense float32 NumPy/JAX array
X = adata.X
if hasattr(X, "toarray"):
    X = X.toarray()
X = np.asarray(X, dtype=np.float32)

n_cells, cdims = X.shape


# ----------------------------
# Mini-batching helper
# ----------------------------
def batch_iterator(X, batch_size, rng):
    n = X.shape[0]
    perm = rng.permutation(n)
    for start in range(0, n, batch_size):
        idx = perm[start:start + batch_size]
        yield X[idx]


# ----------------------------
# Model definition
# ----------------------------
class SCAutoEncoder(eqx.Module):
    enc1: eqx.nn.Linear
    enc2: eqx.nn.Linear
    enc3: eqx.nn.Linear
    enc4: eqx.nn.Linear
    enc5: eqx.nn.Linear
    enc6: eqx.nn.Linear

    dec1: eqx.nn.Linear
    dec2: eqx.nn.Linear
    dec3: eqx.nn.Linear
    dec4: eqx.nn.Linear
    dec5: eqx.nn.Linear
    dec6: eqx.nn.Linear

    def __init__(self, input_dim: int, *, key):
        keys = jr.split(key, 12)

        self.enc1 = eqx.nn.Linear(input_dim, 512, key=keys[0])
        self.enc2 = eqx.nn.Linear(512, 256, key=keys[1])
        self.enc3 = eqx.nn.Linear(256, 128, key=keys[2])
        self.enc4 = eqx.nn.Linear(128, 64, key=keys[3])
        self.enc5 = eqx.nn.Linear(64, 16, key=keys[4])
        self.enc6 = eqx.nn.Linear(16, 2, key=keys[5])

        self.dec1 = eqx.nn.Linear(2, 16, key=keys[6])
        self.dec2 = eqx.nn.Linear(16, 64, key=keys[7])
        self.dec3 = eqx.nn.Linear(64, 128, key=keys[8])
        self.dec4 = eqx.nn.Linear(128, 256, key=keys[9])
        self.dec5 = eqx.nn.Linear(256, 512, key=keys[10])
        self.dec6 = eqx.nn.Linear(512, input_dim, key=keys[11])

    def encode_one(self, x):
        x = jax.nn.relu(self.enc1(x))
        x = jax.nn.relu(self.enc2(x))
        x = jax.nn.relu(self.enc3(x))
        x = jax.nn.relu(self.enc4(x))
        x = jax.nn.relu(self.enc5(x))
        x = self.enc6(x)
        return x

    def decode_one(self, z):
        z = jax.nn.relu(self.dec1(z))
        z = jax.nn.relu(self.dec2(z))
        z = jax.nn.relu(self.dec3(z))
        z = jax.nn.relu(self.dec4(z))
        z = jax.nn.relu(self.dec5(z))
        z = jax.nn.softplus(self.dec6(z))
        return z

    def __call__(self, x):
        z = self.encode_one(x)
        y = self.decode_one(z)
        return y

    def encode_batch(self, x):
        return jax.vmap(self.encode_one)(x)

    def decode_batch(self, z):
        return jax.vmap(self.decode_one)(z)

    def reconstruct_batch(self, x):
        return jax.vmap(self)(x)

    def layer_size_bounds(self):
        ae_sizes = [512, 256, 128, 64, 16, 2, 16, 64, 128, 256, 512]
        return min(ae_sizes), max(ae_sizes)


# ----------------------------
# Loss and optimizer
# ----------------------------
def mse_loss(model, batch):
    recon = model.reconstruct_batch(batch)
    return jnp.mean((recon - batch) ** 2)


@eqx.filter_jit
def train_step(model, opt_state, batch, optimizer):
    loss, grads = eqx.filter_value_and_grad(mse_loss)(model, batch)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


# ----------------------------
# Initialize
# ----------------------------
key = jr.PRNGKey(0)
model = SCAutoEncoder(cdims, key=key)
min_layer_size, max_layer_size = model.layer_size_bounds()
layer_tag = f"_min{min_layer_size}_max{max_layer_size}"

# AdamW is the closest common replacement for Adam + weight decay
optimizer = optax.adamw(learning_rate=1e-3, weight_decay=1e-8)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

epochs = 20
batch_size = 128
losses = []

plot_dir = Path(__file__).parent / "figures" / f"{layer_tag}"
plot_dir.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(0)

print("start training")
import time
start = time.time()
for epoch in range(epochs):
    epoch_losses = []
    for batch_np in batch_iterator(X, batch_size=batch_size, rng=rng):
        batch = jnp.asarray(batch_np)
        model, opt_state, loss = train_step(model, opt_state, batch, optimizer)
        loss_value = float(loss)
        losses.append(loss_value)
        epoch_losses.append(loss_value)

    print(f"Epoch {epoch+1}/{epochs}, Loss: {np.mean(epoch_losses):.6f}")
end = time.time()
print(f"Training took {end-start}s")

# ----------------------------
# Plot training loss
# ----------------------------
plt.style.use("fivethirtyeight")
plt.figure(figsize=(8, 5))
plt.plot(losses, label="Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(plot_dir / f"training_loss{layer_tag}.png", dpi=150, bbox_inches="tight")
plt.close()


# ----------------------------
# Full-data embedding/reconstruction
# ----------------------------
X_jax = jnp.asarray(X)
embedding = model.encode_batch(X_jax)
reconstruction = model.decode_batch(embedding)

embedding_np = np.asarray(embedding)
reconstruction_np = np.asarray(reconstruction)

adata.obsm["autoencoder"] = embedding_np


# ----------------------------
# Scanpy plots
# ----------------------------
sc.tl.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)

for cov in ["donor_id", "batch", "cell_type", "sex", "tissue"]:
    fig = sc.pl.embedding(
        adata,
        basis="autoencoder",
        color=cov,
        title=f"Autoencoder: {cov}",
        show=False,
        return_fig=True,
    )
    fig.savefig(plot_dir / f"autoencoder_{cov}{layer_tag}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

for cov in ["donor_id", "batch", "cell_type", "sex", "tissue"]:
    fig = sc.pl.pca(
        adata,
        color=cov,
        title=f"PCA: {cov}",
        show=False,
        return_fig=True,
    )
    fig.savefig(plot_dir / f"pca_{cov}{layer_tag}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

for cov in ["donor_id", "batch", "cell_type", "sex", "tissue"]:
    fig = sc.pl.umap(
        adata,
        color=cov,
        title=f"UMAP: {cov}",
        show=False,
        return_fig=True,
    )
    fig.savefig(plot_dir / f"umap_{cov}{layer_tag}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Cosine similarity
# ----------------------------
orig = X_jax
recon = jnp.asarray(reconstruction_np)

orig_norm = orig / (jnp.linalg.norm(orig, axis=1, keepdims=True) + 1e-8)
recon_norm = recon / (jnp.linalg.norm(recon, axis=1, keepdims=True) + 1e-8)
similarities = jnp.sum(orig_norm * recon_norm, axis=1)

pd.DataFrame(np.asarray(similarities)).plot(
    kind="box",
    xlabel="Cosine similarity",
    ylabel="Training data vs. Reconstruction",
)
plt.tight_layout()
plt.savefig(plot_dir / f"cosine_similarity_box{layer_tag}.png", dpi=150, bbox_inches="tight")
plt.close()
#!/usr/bin/env python3
"""
prepare_rawpbmc.py
Loads local 10x PBMC raw count matrices from `raw_mtx/` directories,
selects HVGs, computes size factors, creates train/val/test splits, and saves outputs.

Run:
  python prepare_rawpbmc.py --dataset pbmc3k --indir ./pbmc_data --outdir ./pbmc_data

Options:
  --dataset  one of: pbmc3k, pbmc8k, pbmc68k
  --indir    root directory containing <dataset>/raw_mtx/
  --outdir   output directory to store prepared AnnData
"""

import argparse
from pathlib import Path
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd

def load_10x_filtered_mtx_dir(mtx_dir):
    print("Loading Matrix Market directory at:", mtx_dir)
    # sc.read_10x_mtx expects path to directory containing barcodes.tsv, genes.tsv or features.tsv, matrix.mtx
    adata = sc.read_10x_mtx(str(mtx_dir), var_names='gene_symbols', make_unique=True)
    return adata

def prepare_adatas(adata, outdir, n_hvg=3000, seed=42):
    outdir = Path(outdir)
    # Ensure raw contains counts: set raw if not present
    if adata.raw is None:
        # store a copy of raw counts into .raw
        adata.raw = adata.copy()
    # Put raw counts into X for processing (we want integer counts)
    adata.X = adata.raw.X.copy()

    # compute size factors (library sizes)
    cell_totals = np.asarray(adata.X.sum(axis=1)).ravel()
    adata.obs['size_factors'] = cell_totals

    # HVG selection on raw counts
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor='seurat_v3', inplace=True)
    adata = adata[:, adata.var['highly_variable']].copy()

    # train/val/test split
    n_cells = adata.n_obs
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_cells)
    train_cut = int(0.8 * n_cells)
    val_cut = int(0.9 * n_cells)
    train_idx = perm[:train_cut]
    val_idx = perm[train_cut:val_cut]
    test_idx = perm[val_cut:]

    adata.obs['split'] = ''
    adata.obs['split'].iloc[train_idx] = 'train'
    adata.obs['split'].iloc[val_idx] = 'validation'
    adata.obs['split'].iloc[test_idx] = 'test'

    # save AnnData objects and metadata
    outdir.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(outdir / 'adata_hvg.h5ad')
    # Save gene list and size factors and indices
    pd.DataFrame({'gene': adata.var_names}).to_csv(outdir / 'gene_list.csv', index=False)
    np.save(outdir / 'size_factors.npy', adata.obs['size_factors'].values)
    np.save(outdir / 'train_idx.npy', train_idx)
    np.save(outdir / 'val_idx.npy', val_idx)
    np.save(outdir / 'test_idx.npy', test_idx)
    print("Prepared and saved AnnData and metadata to:", outdir)
    return adata

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['pbmc3k', 'pbmc8k', 'pbmc68k'], default='pbmc3k')
    parser.add_argument('--indir', default='./pbmc_data')
    parser.add_argument('--outdir', default='./pbmc_data')
    parser.add_argument('--n_hvg', type=int, default=3000)
    args = parser.parse_args()

    key = args.dataset
    in_dir = Path(args.indir) / key / 'raw_mtx'
    if not in_dir.exists():
        raise FileNotFoundError(
            f"Local raw mtx directory not found: {in_dir}. "
            "Please place barcodes.tsv, genes.tsv, matrix.mtx under <indir>/<dataset>/raw_mtx/"
        )

    adata = load_10x_filtered_mtx_dir(in_dir)
    outdir = Path(args.outdir) / key
    outdir.mkdir(parents=True, exist_ok=True)

    # basic checks
    print("adata.X dtype:", getattr(adata.X, 'dtype', type(adata.X)))
    print("adata shape:", adata.shape)
    prepared = prepare_adatas(adata, outdir, n_hvg=args.n_hvg)
    print("Done. Files in:", outdir.resolve())

if __name__ == '__main__':
    main()
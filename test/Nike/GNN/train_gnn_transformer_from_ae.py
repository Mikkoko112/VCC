#!/usr/bin/env python3
"""Train a graph transformer to predict expression changes after gene knockout."""

import argparse
import datetime
import json
import random
import uuid
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torch_geometric.nn import TransformerConv

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
AE_EXPERIMENT = (
    PROJECT_DIR
    / "experiments"
    / "hummel_runs"
    / "run1_experiment_2026-08-04_20260804_193738_df5e45a8_continue5"
)
PERTURBATION_DATA = SCRIPT_DIR / "vcc_data" / "adata_Training.h5ad"
VALIDATION_CONTROL_DIR = SCRIPT_DIR / "vcc_2026_controls"
VALIDATION_TARGETS = SCRIPT_DIR / "vcc_data" / "pert_counts_Validation.csv"
PATHWAY_GRAPH = PROJECT_DIR / "pathway_network_gnn.npz"
CONTROL_LABEL = "non-targeting"
AE_INPUT_DIM = 5000
AE_LATENT_DIM = 256


class PretrainedAEEncoder(nn.Module):
    """Encoder matching the saved 5,000-to-256 AE checkpoint."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(AE_INPUT_DIM, 4096),
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
            nn.Linear(512, AE_LATENT_DIM),
        )

    def forward(self, counts):
        return self.encoder(counts)


class PerturbationGraphTransformer(nn.Module):
    """Predict a gene-expression delta on a shared, directed regulatory graph."""

    def __init__(self, n_genes, hidden_dim=64, num_heads=4, num_layers=2, edge_dim=5):
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.gene_embedding = nn.Embedding(n_genes, hidden_dim)
        self.baseline_projection = nn.Linear(1, hidden_dim)
        self.ae_projection = nn.Linear(AE_LATENT_DIM, hidden_dim)
        self.target_flag_projection = nn.Linear(1, hidden_dim, bias=False)
        self.target_context_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.convolutions = nn.ModuleList(
            TransformerConv(
                hidden_dim,
                hidden_dim // num_heads,
                heads=num_heads,
                concat=True,
                edge_dim=edge_dim,
                dropout=0.1,
                beta=True,
            )
            for _ in range(num_layers)
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    @staticmethod
    def _repeat_graph_edges(edge_index, edge_attr, batch_size, n_genes):
        offsets = torch.arange(batch_size, device=edge_index.device) * n_genes
        repeated_index = (
            edge_index.unsqueeze(0) + offsets[:, None, None]
        ).permute(1, 0, 2).reshape(2, -1)
        repeated_attr = edge_attr.repeat(batch_size, 1)
        return repeated_index, repeated_attr

    def forward(self, baseline, ae_latent, target_indices, edge_index, edge_attr):
        batch_size, n_genes = baseline.shape
        gene_ids = torch.arange(n_genes, device=baseline.device)
        target_mask = gene_ids.unsqueeze(0) == target_indices.unsqueeze(1)
        target_context = self.gene_embedding(target_indices)

        hidden = self.gene_embedding(gene_ids).unsqueeze(0)
        hidden = hidden + self.baseline_projection(baseline.unsqueeze(-1))
        hidden = hidden + self.ae_projection(ae_latent).unsqueeze(1)
        hidden = hidden + self.target_context_projection(target_context).unsqueeze(1)
        hidden = hidden + self.target_flag_projection(target_mask.float().unsqueeze(-1))
        hidden = hidden.expand(batch_size, -1, -1).reshape(
            batch_size * n_genes, -1
        )

        batched_edges, batched_attributes = self._repeat_graph_edges(
            edge_index, edge_attr, batch_size, n_genes
        )
        for convolution in self.convolutions:
            hidden = nn.functional.gelu(
                convolution(hidden, batched_edges, batched_attributes)
            )
        return self.output(hidden).reshape(batch_size, n_genes)


def load_pretrained_encoder(experiment_dir, device):
    weights_path = experiment_dir / "model_best.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"AE checkpoint not found: {weights_path}")

    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    encoder_state = {
        key: value for key, value in checkpoint.items() if key.startswith("encoder.")
    }
    encoder = PretrainedAEEncoder()
    encoder.load_state_dict(encoder_state, strict=True)
    encoder.to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder


def make_gene_graph(gene_order, graph_path=PATHWAY_GRAPH):
    """Load the exported pathway graph and align its edges to the gene order."""
    if not graph_path.is_file():
        raise FileNotFoundError(f"Pathway graph export not found: {graph_path}")

    gene_to_index = {gene: index for index, gene in enumerate(gene_order)}
    with np.load(graph_path, allow_pickle=False) as data:
        required_arrays = {
            "node_genes",
            "node_labels",
            "edge_index",
            "edge_attr",
            "edge_effects",
        }
        missing = required_arrays - set(data.files)
        if missing:
            raise ValueError(
                f"Pathway graph {graph_path} is missing arrays: {sorted(missing)}"
            )
        node_genes = data["node_genes"].astype(str)
        node_labels = data["node_labels"].astype(str)
        edge_index = data["edge_index"]
        edge_attr = data["edge_attr"]
        edge_effects = data["edge_effects"].astype(str)

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            f"Expected edge_index with shape (2, E), got {edge_index.shape}"
        )
    if edge_attr.ndim != 2 or edge_attr.shape[0] != edge_index.shape[1]:
        raise ValueError(
            f"edge_attr shape {edge_attr.shape} does not match "
            f"{edge_index.shape[1]} edges"
        )
    if len(edge_effects) != edge_index.shape[1]:
        raise ValueError("edge_effects length does not match edge_index")
    if len(node_genes) != len(node_labels):
        raise ValueError("node_genes and node_labels have different lengths")

    graph_node_to_gene_index = {}
    for graph_node_index, names in enumerate(node_genes):
        candidates = set(str(names).split(","))
        candidates.update(str(node_labels[graph_node_index]).split(","))
        for candidate in candidates:
            gene = candidate.strip()
            if gene in gene_to_index:
                graph_node_to_gene_index[graph_node_index] = gene_to_index[gene]
                break

    graph = {}
    for edge_number, (source, target) in enumerate(edge_index.T):
        source_gene_index = graph_node_to_gene_index.get(int(source))
        target_gene_index = graph_node_to_gene_index.get(int(target))
        if (
            source_gene_index is None
            or target_gene_index is None
            or source_gene_index == target_gene_index
        ):
            continue
        edge_key = (source_gene_index, target_gene_index)
        effect = str(edge_effects[edge_number])
        weight = float(edge_attr[edge_number, 0])
        if edge_key not in graph:
            graph[edge_key] = {"weight": weight, "effects": {effect}}
        else:
            previous = graph[edge_key]
            previous["weight"] = (previous["weight"] + weight) / 2
            previous["effects"].add(effect)

    edge_index = []
    edge_attributes = []
    effect_types = ("activation", "inhibition", "expression", "other")
    for (source_index, target_index), attributes in graph.items():
        edge_index.append((source_index, target_index))
        effects = attributes["effects"]
        effect = next(iter(effects)) if len(effects) == 1 else "other"
        if effect not in effect_types:
            effect = "other"
        effect_vector = [float(effect == effect_type) for effect_type in effect_types]
        edge_attributes.append([attributes["weight"], *effect_vector])

    if edge_index:
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr_tensor = torch.tensor(edge_attributes, dtype=torch.float32)
    else:
        edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
        edge_attr_tensor = torch.empty((0, 5), dtype=torch.float32)
    return edge_index_tensor, edge_attr_tensor, len(graph)


def aggregate_conditions(
    data_path,
    ae_genes,
    encoder,
    device,
    min_cells_per_condition,
    chunk_size=512,
    additional_genes=(),
):
    """Aggregate perturbed cells against non-targeting controls from the same batch."""
    adata = ad.read_h5ad(data_path, backed="r")
    required_obs = {"target_gene", "batch"}
    missing_obs = required_obs - set(adata.obs.columns)
    if missing_obs:
        raise ValueError(f"Perturbation data is missing obs columns: {sorted(missing_obs)}")

    var_index = pd.Index(adata.var_names.astype(str))
    targets = adata.obs["target_gene"].astype(str).to_numpy()
    batches = adata.obs["batch"].astype(str).to_numpy()
    if CONTROL_LABEL not in set(targets):
        raise ValueError(f"No control rows labelled {CONTROL_LABEL!r} in target_gene")

    perturbation_targets = sorted(set(targets) - {CONTROL_LABEL})
    ae_genes = list(ae_genes)
    measured_ae_genes = [gene for gene in ae_genes if gene in var_index]
    included_genes = set(measured_ae_genes)
    gene_order = measured_ae_genes + [
        gene for gene in perturbation_targets
        if gene in var_index and gene not in included_genes
    ]
    included_genes.update(gene_order)
    for gene in additional_genes:
        if gene in var_index and gene not in included_genes:
            gene_order.append(gene)
            included_genes.add(gene)
    if not gene_order:
        raise ValueError("No AE genes or perturbation targets overlap the H5AD var_names")

    gene_to_index = {gene: index for index, gene in enumerate(gene_order)}
    data_columns = var_index.get_indexer(gene_order)
    ae_source_indices = np.array(
        [gene_to_index[gene] for gene in measured_ae_genes], dtype=np.int64
    )
    ae_target_indices = np.array(
        [ae_genes.index(gene) for gene in measured_ae_genes], dtype=np.int64
    )

    batch_codes, batch_levels = pd.factorize(batches, sort=True)
    batch_levels = list(batch_levels.astype(str))
    batch_to_index = {batch: index for index, batch in enumerate(batch_levels)}
    condition_index = pd.MultiIndex.from_arrays([targets, batches])
    condition_codes, conditions = pd.factorize(condition_index, sort=True)
    condition_targets = conditions.get_level_values(0).astype(str).to_numpy()
    condition_batches = conditions.get_level_values(1).astype(str).to_numpy()
    n_conditions = len(conditions)
    n_genes = len(gene_order)

    expression_sums = np.zeros((n_conditions, n_genes), dtype=np.float32)
    expression_counts = np.bincount(condition_codes, minlength=n_conditions)
    control_raw_sums = np.zeros((len(batch_levels), len(ae_genes)), dtype=np.float64)
    control_counts = np.zeros(len(batch_levels), dtype=np.int64)

    for start in range(0, adata.n_obs, chunk_size):
        stop = min(start + chunk_size, adata.n_obs)
        full_counts = adata.X[start:stop, :]
        total_counts = np.asarray(full_counts.sum(axis=1)).reshape(-1).astype(np.float32)
        gene_counts = full_counts[:, data_columns].toarray().astype(np.float32, copy=False)
        normalized = gene_counts / np.maximum(total_counts[:, None], 1.0) * 10000.0
        log_expression = np.log1p(normalized).astype(np.float32, copy=False)

        chunk_conditions = condition_codes[start:stop]
        np.add.at(expression_sums, chunk_conditions, log_expression)

        is_control = targets[start:stop] == CONTROL_LABEL
        if np.any(is_control):
            chunk_batches = batch_codes[start:stop]
            ae_counts = np.zeros((stop - start, len(ae_genes)), dtype=np.float32)
            ae_counts[:, ae_target_indices] = gene_counts[:, ae_source_indices]
            control_batch_codes = chunk_batches[is_control]
            np.add.at(control_raw_sums, control_batch_codes, ae_counts[is_control])
            control_counts += np.bincount(
                control_batch_codes, minlength=len(batch_levels)
            )

    expression_sums /= np.maximum(expression_counts[:, None], 1)
    control_expression = {}
    for condition_index, target in enumerate(condition_targets):
        if target == CONTROL_LABEL:
            batch_index = batch_to_index[condition_batches[condition_index]]
            control_expression[batch_index] = expression_sums[condition_index].copy()

    valid_batches = sorted(set(control_expression) & set(np.flatnonzero(control_counts)))
    if not valid_batches:
        raise ValueError("No batches contain both non-targeting controls and perturbations")

    baseline_expression = np.zeros((len(batch_levels), n_genes), dtype=np.float32)
    for batch_index, profile in control_expression.items():
        baseline_expression[batch_index] = profile

    ae_control_means = control_raw_sums / np.maximum(control_counts[:, None], 1)
    examples = []
    for index, target in enumerate(condition_targets):
        if target == CONTROL_LABEL or expression_counts[index] < min_cells_per_condition:
            continue
        batch_index = batch_to_index[condition_batches[index]]
        if batch_index not in control_expression or control_counts[batch_index] == 0:
            continue
        expression_sums[index] -= control_expression[batch_index]
        examples.append((index, target, batch_index, int(expression_counts[index])))

    if not examples:
        raise ValueError(
            "No target/batch conditions meet min_cells_per_condition and have controls"
        )

    with torch.no_grad():
        latent_batches = []
        for start in range(0, len(batch_levels), 64):
            control_batch = torch.as_tensor(
                ae_control_means[start:start + 64],
                dtype=torch.float32,
                device=device,
            )
            latent_batches.append(encoder(control_batch).cpu().numpy())
        ae_latents = np.concatenate(latent_batches, axis=0)
    adata.file.close()

    example_groups = np.array([item[0] for item in examples], dtype=np.int64)
    example_targets = np.array([item[1] for item in examples], dtype=object)
    example_target_indices = np.array(
        [gene_to_index[item[1]] for item in examples], dtype=np.int64
    )
    example_batch_indices = np.array([item[2] for item in examples], dtype=np.int64)
    example_cell_counts = np.array([item[3] for item in examples], dtype=np.int64)

    return {
        "gene_order": gene_order,
        "batch_levels": batch_levels,
        "deltas": expression_sums,
        "baseline_expression": baseline_expression,
        "ae_latents": ae_latents,
        "group_indices": example_groups,
        "target_names": example_targets,
        "target_indices": example_target_indices,
        "batch_indices": example_batch_indices,
        "cell_counts": example_cell_counts,
        "n_controls": int(control_counts.sum()),
    }


def split_by_target(target_names, seed):
    """Hold out entire knockout targets for validation and testing."""
    unique_targets = sorted(set(target_names))
    if len(unique_targets) < 3:
        raise ValueError("At least three knockout targets are needed for target-held-out splits")
    random.Random(seed).shuffle(unique_targets)
    n_test = max(1, round(0.15 * len(unique_targets)))
    n_validation = max(1, round(0.15 * len(unique_targets)))
    test_targets = set(unique_targets[:n_test])
    validation_targets = set(unique_targets[n_test:n_test + n_validation])
    train_targets = set(unique_targets[n_test + n_validation:])
    masks = {
        "train": np.array([target in train_targets for target in target_names]),
        "validation": np.array([target in validation_targets for target in target_names]),
        "test": np.array([target in test_targets for target in target_names]),
    }
    if not all(mask.any() for mask in masks.values()):
        raise ValueError("A split has no conditions; lower min_cells_per_condition")
    return masks


def load_validation_control_baselines(
    control_dir,
    gene_order,
    ae_genes,
    encoder,
    device,
    chunk_size=512,
):
    """Aggregate control-only contexts into model input baselines and AE latents."""
    control_paths = sorted(control_dir.glob("context_*.h5ad"))
    if not control_paths:
        raise FileNotFoundError(
            f"No context_*.h5ad files found in validation control folder: {control_dir}"
        )

    gene_to_index = {gene: index for index, gene in enumerate(gene_order)}
    ae_gene_positions = [
        (ae_index, gene_to_index[gene])
        for ae_index, gene in enumerate(ae_genes)
        if gene in gene_to_index
    ]
    contexts = []
    baseline_profiles = []
    ae_control_means = []
    total_control_cells = 0

    for control_path in control_paths:
        adata = ad.read_h5ad(control_path, backed="r")
        try:
            required_obs = {"target_gene", "context"}
            missing_obs = required_obs - set(adata.obs.columns)
            if missing_obs:
                raise ValueError(
                    f"{control_path} is missing observation columns: "
                    f"{sorted(missing_obs)}"
                )

            targets = adata.obs["target_gene"].astype(str)
            if set(targets.unique()) != {CONTROL_LABEL}:
                raise ValueError(
                    f"{control_path} must contain only {CONTROL_LABEL!r} cells"
                )
            context_values = adata.obs["context"].astype(str).unique()
            if len(context_values) != 1:
                raise ValueError(
                    f"Expected one context in {control_path}, found "
                    f"{context_values.tolist()}"
                )
            context = context_values[0]
            var_index = pd.Index(adata.var_names.astype(str))
            selected_source_columns = var_index.get_indexer(gene_order)
            measured_model_positions = np.flatnonzero(selected_source_columns >= 0)
            measured_source_columns = selected_source_columns[measured_model_positions]
            gene_expression_sum = np.zeros(len(gene_order), dtype=np.float64)
            ae_count_sum = np.zeros(len(ae_genes), dtype=np.float64)
            ae_model_positions = [
                (ae_index, model_index)
                for ae_index, model_index in ae_gene_positions
                if selected_source_columns[model_index] >= 0
            ]

            for start in range(0, adata.n_obs, chunk_size):
                stop = min(start + chunk_size, adata.n_obs)
                full_counts = adata.X[start:stop, :]
                total_counts = np.asarray(full_counts.sum(axis=1)).reshape(-1)
                selected_counts = full_counts[:, measured_source_columns].toarray()
                normalized = (
                    selected_counts
                    / np.maximum(total_counts[:, None], 1.0)
                    * 10000.0
                )
                gene_expression_sum[measured_model_positions] += np.log1p(
                    normalized
                ).sum(axis=0)

                for ae_index, model_index in ae_model_positions:
                    column_index = np.searchsorted(
                        measured_model_positions, model_index
                    )
                    ae_count_sum[ae_index] += selected_counts[:, column_index].sum()

            cell_count = int(adata.n_obs)
            if cell_count == 0:
                raise ValueError(f"No control cells found in {control_path}")
            contexts.append(context)
            baseline_profiles.append(gene_expression_sum / cell_count)
            ae_control_means.append(ae_count_sum / cell_count)
            total_control_cells += cell_count
        finally:
            adata.file.close()

    control_ae_tensor = torch.as_tensor(
        np.asarray(ae_control_means, dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        ae_latents = encoder(control_ae_tensor).cpu().numpy()

    return {
        "contexts": contexts,
        "baseline_expression": np.asarray(baseline_profiles, dtype=np.float32),
        "ae_latents": ae_latents,
        "n_controls": total_control_cells,
    }


def predict_validation_controls(
    model,
    target_path,
    gene_order,
    edge_index,
    edge_attr,
    control_dir,
    ae_genes,
    encoder,
    device,
    batch_size,
    top_genes,
    output_path,
):
    """Write exploratory model predictions using validation controls as baselines."""
    targets_frame = pd.read_csv(target_path)
    if "target_gene" not in targets_frame.columns:
        raise ValueError(f"Validation target file has no 'target_gene' column: {target_path}")

    validation_targets = sorted(
        set(targets_frame["target_gene"].dropna().astype(str))
        - {CONTROL_LABEL}
    )
    gene_to_index = {gene: index for index, gene in enumerate(gene_order)}
    matched_targets = [
        target for target in validation_targets if target in gene_to_index
    ]
    skipped_targets = sorted(set(validation_targets) - set(matched_targets))
    if not matched_targets:
        raise ValueError(
            "No validation perturbation targets occur in the model gene order"
        )

    controls = load_validation_control_baselines(
        control_dir,
        gene_order,
        ae_genes,
        encoder,
        device,
        chunk_size=512,
    )
    records = []
    model.eval()
    with torch.no_grad():
        for context_index, context in enumerate(controls["contexts"]):
            baseline = controls["baseline_expression"][context_index]
            latent = controls["ae_latents"][context_index]
            for start in range(0, len(matched_targets), batch_size):
                batch_targets = matched_targets[start:start + batch_size]
                batch_length = len(batch_targets)
                prediction = model(
                    torch.as_tensor(
                        np.repeat(baseline[None, :], batch_length, axis=0),
                        dtype=torch.float32,
                        device=device,
                    ),
                    torch.as_tensor(
                        np.repeat(latent[None, :], batch_length, axis=0),
                        dtype=torch.float32,
                        device=device,
                    ),
                    torch.as_tensor(
                        [gene_to_index[target] for target in batch_targets],
                        dtype=torch.long,
                        device=device,
                    ),
                    edge_index,
                    edge_attr,
                ).cpu().numpy()
                for target, row in zip(batch_targets, prediction):
                    top_indices = np.argsort(np.abs(row))[-top_genes:][::-1]
                    records.extend(
                        {
                            "context": context,
                            "target_gene": target,
                            "gene": gene_order[gene_index],
                            "predicted_log1p_delta": float(row[gene_index]),
                        }
                        for gene_index in top_indices
                    )

    pd.DataFrame(records).to_csv(output_path, index=False)
    return {
        "contexts": controls["contexts"],
        "n_control_cells": controls["n_controls"],
        "n_validation_targets": len(validation_targets),
        "n_targets_predicted": len(matched_targets),
        "unmatched_targets": skipped_targets,
        "predictions_file": str(output_path),
    }


def save_performance_plots(history, observed, predicted, output_dir, seed):
    """Save learning curves and a sampled held-out prediction-vs-observation plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    test_mse = float(np.mean(np.square(predicted - observed)))
    zero_baseline_mse = float(np.mean(np.square(observed)))
    total_sum_squares = float(np.square(observed - observed.mean()).sum())
    test_r2 = (
        1.0 - float(np.square(predicted - observed).sum()) / total_sum_squares
        if total_sum_squares > 0
        else float("nan")
    )

    rng = np.random.default_rng(seed)
    sample_size = min(len(observed), 25000)
    sample_indices = rng.choice(len(observed), size=sample_size, replace=False)
    plot_observed = observed[sample_indices]
    plot_predicted = predicted[sample_indices]
    axis_limits = (
        min(float(plot_observed.min()), float(plot_predicted.min())),
        max(float(plot_observed.max()), float(plot_predicted.max())),
    )

    figure, axis = plt.subplots(figsize=(8, 8))
    axis.scatter(plot_observed, plot_predicted, s=5, alpha=0.25, rasterized=True)
    axis.plot(axis_limits, axis_limits, color="black", linestyle="--", linewidth=1)
    axis.set(
        xlim=axis_limits,
        ylim=axis_limits,
        xlabel="Observed log1p expression delta",
        ylabel="Predicted log1p expression delta",
        title=(
            "Held-out target predictions\n"
            f"MSE={test_mse:.4g} | zero-baseline MSE={zero_baseline_mse:.4g} "
            f"| R²={test_r2:.4g}"
        ),
    )
    figure.tight_layout()
    figure.savefig(output_dir / "test_prediction_scatter.png", dpi=180)
    plt.close(figure)

    history_frame = pd.DataFrame(history)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(history_frame["epoch"], history_frame["train_mse"], label="Train")
    axis.plot(
        history_frame["epoch"],
        history_frame["validation_mse"],
        label="Validation",
    )
    axis.set(
        xlabel="Epoch",
        ylabel="Mean squared error",
        title="Training history",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "training_history.png", dpi=180)
    plt.close(figure)
    return test_mse, zero_baseline_mse, test_r2


def run_smoke_test():
    """Check a forward/backward pass without opening the large dataset or KEGG."""
    torch.manual_seed(1)
    model = PerturbationGraphTransformer(n_genes=5)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge_attr = torch.zeros((3, 5), dtype=torch.float32)
    baseline = torch.rand((2, 5))
    ae_latent = torch.rand((2, AE_LATENT_DIM))
    targets = torch.tensor([0, 3], dtype=torch.long)
    prediction = model(baseline, ae_latent, targets, edge_index, edge_attr)
    loss = prediction.square().mean()
    loss.backward()
    assert prediction.shape == (2, 5)
    assert torch.isfinite(loss)
    print("GNN Transformer forward/backward smoke test passed")


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ae_gene_path = args.ae_experiment / "gene_list.csv"
    if not ae_gene_path.exists():
        raise FileNotFoundError(f"AE gene order not found: {ae_gene_path}")
    ae_genes = pd.read_csv(ae_gene_path)["gene"].astype(str).tolist()
    if len(ae_genes) != AE_INPUT_DIM:
        raise ValueError(f"Expected {AE_INPUT_DIM} AE genes, found {len(ae_genes)}")

    validation_targets_frame = pd.read_csv(args.validation_targets)
    if "target_gene" not in validation_targets_frame.columns:
        raise ValueError(
            "Validation target file has no 'target_gene' column: "
            f"{args.validation_targets}"
        )
    validation_targets = sorted(
        set(validation_targets_frame["target_gene"].dropna().astype(str))
        - {CONTROL_LABEL}
    )

    encoder = load_pretrained_encoder(args.ae_experiment, device)
    data = aggregate_conditions(
        args.perturbation_data,
        ae_genes,
        encoder,
        device,
        min_cells_per_condition=args.min_cells_per_condition,
        chunk_size=args.chunk_size,
        additional_genes=validation_targets,
    )
    gene_order = data["gene_order"]
    edge_index, edge_attr, n_edges = make_gene_graph(
        gene_order, args.pathway_graph
    )
    print(
        f"Prepared {len(data['group_indices'])} target/batch conditions; "
        f"{len(gene_order)} genes, {n_edges} regulatory edges, "
        f"{data['n_controls']} control cells."
    )

    masks = split_by_target(data["target_names"], args.seed)
    dataset = TensorDataset(
        torch.from_numpy(data["group_indices"]),
        torch.from_numpy(data["target_indices"]),
        torch.from_numpy(data["batch_indices"]),
    )
    train_loader = DataLoader(
        Subset(dataset, np.flatnonzero(masks["train"]).tolist()),
        batch_size=args.batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        Subset(dataset, np.flatnonzero(masks["validation"]).tolist()),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        Subset(dataset, np.flatnonzero(masks["test"]).tolist()),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = PerturbationGraphTransformer(
        n_genes=len(gene_order),
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    ).to(device)
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
    deltas = torch.from_numpy(data["deltas"]).to(device)
    baseline_expression = torch.from_numpy(data["baseline_expression"]).to(device)
    ae_latents = torch.from_numpy(data["ae_latents"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    def evaluate(loader):
        model.eval()
        losses = []
        with torch.no_grad():
            for group_indices, target_indices, batch_indices in loader:
                group_indices = group_indices.to(device)
                target_indices = target_indices.to(device)
                batch_indices = batch_indices.to(device)
                predicted = model(
                    baseline_expression[batch_indices],
                    ae_latents[batch_indices],
                    target_indices,
                    edge_index,
                    edge_attr,
                )
                losses.append(
                    torch.mean((predicted - deltas[group_indices]) ** 2, dim=1)
                )
        return torch.cat(losses).mean().item()

    results_dir = SCRIPT_DIR / "experiments" / (
        f"gnn_transformer_{datetime.date.today().isoformat()}_"
        f"{datetime.datetime.now():%H%M%S}_{uuid.uuid4().hex[:8]}"
    )
    results_dir.mkdir(parents=True, exist_ok=False)
    best_validation = float("inf")
    best_state = None
    stale_epochs = 0
    history = []

    for epoch in range(args.epochs):
        model.train()
        batch_losses = []
        for group_indices, target_indices, batch_indices in train_loader:
            group_indices = group_indices.to(device)
            target_indices = target_indices.to(device)
            batch_indices = batch_indices.to(device)
            prediction = model(
                baseline_expression[batch_indices],
                ae_latents[batch_indices],
                target_indices,
                edge_index,
                edge_attr,
            )
            loss = torch.mean((prediction - deltas[group_indices]) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_losses.append(loss.item())

        train_loss = float(np.mean(batch_losses))
        validation_loss = evaluate(validation_loader)
        history.append(
            {"epoch": epoch + 1, "train_mse": train_loss, "validation_mse": validation_loss}
        )
        print(
            f"Epoch {epoch + 1}/{args.epochs}: train_mse={train_loss:.6f}, "
            f"validation_mse={validation_loss:.6f}"
        )

        if validation_loss < best_validation:
            best_validation = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print("Early stopping: validation loss did not improve.")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a model checkpoint")
    model.load_state_dict(best_state)

    predictions = []
    test_observed = []
    test_predicted = []
    model.eval()
    with torch.no_grad():
        for group_indices, target_indices, batch_indices in test_loader:
            group_indices = group_indices.to(device)
            target_indices = target_indices.to(device)
            batch_indices = batch_indices.to(device)
            predicted = model(
                baseline_expression[batch_indices],
                ae_latents[batch_indices],
                target_indices,
                edge_index,
                edge_attr,
            ).cpu().numpy()
            group_ids = group_indices.cpu().numpy()
            batch_ids = batch_indices.cpu().numpy()
            target_ids = target_indices.cpu().numpy()
            test_observed.append(data["deltas"][group_ids])
            test_predicted.append(predicted)
            for row, group_id, batch_id, target_id in zip(
                predicted, group_ids, batch_ids, target_ids
            ):
                delta = data["deltas"][group_id]
                top_indices = np.argsort(np.abs(row))[-args.top_genes:][::-1]
                for gene_index in top_indices:
                    predictions.append(
                        {
                            "target_gene": gene_order[target_id],
                            "batch": data["batch_levels"][batch_id],
                            "gene": gene_order[gene_index],
                            "predicted_log1p_delta": float(row[gene_index]),
                            "observed_log1p_delta": float(delta[gene_index]),
                        }
                    )

    test_observed_array = np.concatenate(test_observed, axis=0).reshape(-1)
    test_predicted_array = np.concatenate(test_predicted, axis=0).reshape(-1)
    test_mse, zero_baseline_mse, test_r2 = save_performance_plots(
        history,
        test_observed_array,
        test_predicted_array,
        results_dir,
        args.seed,
    )

    checkpoint = {
        "model_state_dict": best_state,
        "gene_order": gene_order,
        "pathway_graph": str(args.pathway_graph),
        "edge_index": edge_index.cpu(),
        "edge_attr": edge_attr.cpu(),
        "ae_experiment": str(args.ae_experiment),
        "model_config": {
            "hidden_dim": args.hidden_dim,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "ae_latent_dim": AE_LATENT_DIM,
        },
    }
    torch.save(checkpoint, results_dir / "model_best.pt")
    pd.DataFrame(history).to_csv(results_dir / "training_history.csv", index=False)
    pd.DataFrame(predictions).to_csv(results_dir / "top_test_effects.csv", index=False)

    validation_prediction_summary = predict_validation_controls(
        model,
        args.validation_targets,
        gene_order,
        edge_index,
        edge_attr,
        args.validation_control_dir,
        ae_genes,
        encoder,
        device,
        batch_size=args.batch_size,
        top_genes=args.top_genes,
        output_path=results_dir / "validation_control_predictions.csv",
    )
    (results_dir / "validation_control_prediction_summary.json").write_text(
        json.dumps(validation_prediction_summary, indent=2),
        encoding="utf-8",
    )

    metrics = {
        "test_mse": test_mse,
        "zero_baseline_test_mse": zero_baseline_mse,
        "test_r2": test_r2,
        "best_validation_mse": best_validation,
        "evaluation_scope": (
            "Exploratory target-held-out split within vcc_data/adata_Training.h5ad; "
            "not an official vcc_2026 test score."
        ),
        "n_genes": len(gene_order),
        "n_graph_edges": n_edges,
        "n_conditions": len(data["group_indices"]),
        "n_targets": len(set(data["target_names"])),
        "n_controls": data["n_controls"],
        "validation_control_prediction_summary": validation_prediction_summary,
        "split_target_counts": {
            split: len(set(data["target_names"][mask])) for split, mask in masks.items()
        },
        "target_split": {
            split: sorted(set(data["target_names"][mask]))
            for split, mask in masks.items()
        },
    }
    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Test MSE: {test_mse:.6f}")
    print(f"Zero-change baseline test MSE: {zero_baseline_mse:.6f}")
    print(f"Test R²: {test_r2:.6f}")
    print(f"Saved model, metrics, and top effects to {results_dir}")
    print(f"Test prediction plot: {results_dir / 'test_prediction_scatter.png'}")
    print(
        "2026 control-baseline predictions (not scored): "
        f"{results_dir / 'validation_control_predictions.csv'}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ae-experiment", type=Path, default=AE_EXPERIMENT)
    parser.add_argument("--perturbation-data", type=Path, default=PERTURBATION_DATA)
    parser.add_argument("--pathway-graph", type=Path, default=PATHWAY_GRAPH)
    parser.add_argument(
        "--validation-control-dir", type=Path, default=VALIDATION_CONTROL_DIR
    )
    parser.add_argument(
        "--validation-targets", type=Path, default=VALIDATION_TARGETS
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--min-cells-per-condition", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-genes", type=int, default=20)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.smoke_test:
        run_smoke_test()
    else:
        train(arguments)
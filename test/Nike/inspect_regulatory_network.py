#!/usr/bin/env python3
"""Inspect the node and edge data exported by regulatoryNetwork_3DPlot.py."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_GRAPH_PATH = Path(__file__).with_name("pathway_network_gnn.npz")


def load_graph_tables(graph_path):
    """Load the GNN bundle as node and edge DataFrames."""
    with np.load(graph_path, allow_pickle=False) as data:
        required = {
            "node_ids",
            "node_labels",
            "node_genes",
            "node_types",
            "node_pathways",
            "edge_index",
            "edge_attr",
            "edge_effects",
            "edge_relations",
            "edge_pathways",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(
                f"{graph_path} is missing required arrays: {sorted(missing)}"
            )

        node_ids = data["node_ids"].astype(str)
        node_labels = data["node_labels"].astype(str)
        node_genes = data["node_genes"].astype(str)
        node_types = data["node_types"].astype(str)
        node_pathways = data["node_pathways"].astype(str)
        edge_index = data["edge_index"]
        edge_attr = data["edge_attr"]
        edge_effects = data["edge_effects"].astype(str)
        edge_relations = data["edge_relations"].astype(str)
        edge_pathways = data["edge_pathways"].astype(str)

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            f"edge_index must have shape (2, number_of_edges), got {edge_index.shape}"
        )
    if edge_attr.ndim != 2 or edge_attr.shape[0] != edge_index.shape[1]:
        raise ValueError(
            f"edge_attr shape {edge_attr.shape} does not match "
            f"{edge_index.shape[1]} edges"
        )

    node_columns = {
        "node_index": np.arange(len(node_ids), dtype=np.int64),
        "node_id": node_ids,
        "label": node_labels,
        "gene": node_genes,
        "type": node_types,
        "pathways": [_format_pathways(value) for value in node_pathways],
    }
    for column, values in node_columns.items():
        if len(values) != len(node_ids):
            raise ValueError(f"Node metadata column {column!r} has the wrong length")
    nodes = pd.DataFrame(node_columns)

    source_indices = edge_index[0].astype(np.int64, copy=False)
    target_indices = edge_index[1].astype(np.int64, copy=False)
    if edge_index.shape[1] and (
        source_indices.min() < 0
        or target_indices.min() < 0
        or source_indices.max() >= len(nodes)
        or target_indices.max() >= len(nodes)
    ):
        raise ValueError("edge_index refers to a node index outside the node table")
    if any(
        len(values) != edge_index.shape[1]
        for values in (edge_effects, edge_relations, edge_pathways)
    ):
        raise ValueError("Edge metadata lengths do not match edge_index")

    edges = pd.DataFrame(
        {
            "edge_index": np.arange(edge_index.shape[1], dtype=np.int64),
            "source_index": source_indices,
            "target_index": target_indices,
            "source_id": node_ids[source_indices],
            "source_gene": node_genes[source_indices],
            "source_label": node_labels[source_indices],
            "target_id": node_ids[target_indices],
            "target_gene": node_genes[target_indices],
            "target_label": node_labels[target_indices],
            "weight": edge_attr[:, 0].astype(np.float64, copy=False),
            "effect": edge_effects,
            "relation": edge_relations,
            "pathways": [_format_pathways(value) for value in edge_pathways],
        }
    )
    return nodes, edges


def _format_pathways(value):
    """Format the JSON-encoded pathway lists stored in the graph bundle."""
    try:
        pathways = json.loads(value)
    except json.JSONDecodeError:
        return value
    if isinstance(pathways, list):
        return ", ".join(str(pathway) for pathway in pathways)
    return str(pathways)


def inspect_graph(
    graph_path=DEFAULT_GRAPH_PATH,
    gene=None,
    effect=None,
    top_nodes=20,
    top_edges=20,
    edges_csv=None,
    nodes_csv=None,
):
    """Print graph statistics and filtered node/edge tables."""
    nodes, edges = load_graph_tables(graph_path)

    if gene:
        query = gene.casefold()
        matching_nodes = nodes[
            nodes[["node_id", "label", "gene"]]
            .fillna("")
            .astype(str)
            .apply(lambda column: column.str.casefold().str.contains(query, regex=False))
            .any(axis=1)
        ]
        matching_indices = set(matching_nodes["node_index"])
        edges = edges[
            edges["source_index"].isin(matching_indices)
            | edges["target_index"].isin(matching_indices)
        ]
        nodes = matching_nodes

    if effect:
        edges = edges[edges["effect"].str.casefold() == effect.casefold()]

    print(f"Graph file: {graph_path}")
    print(f"Nodes: {len(nodes):,} shown")
    print(f"Edges: {len(edges):,} shown")
    if len(edges):
        print(
            "Weight range: "
            f"{edges['weight'].min():g} to {edges['weight'].max():g}; "
            f"mean {edges['weight'].mean():g}"
        )
        print("\nEdge counts by effect:")
        print(edges["effect"].value_counts().to_string())

    if nodes_csv:
        nodes.to_csv(nodes_csv, index=False)
        print(f"\nNode table saved to {nodes_csv}")
    if edges_csv:
        edges.to_csv(edges_csv, index=False)
        print(f"Edge table saved to {edges_csv}")

    print("\nNodes:")
    if nodes.empty:
        print("No matching nodes.")
    else:
        display_nodes = nodes if top_nodes == 0 else nodes.head(top_nodes)
        print(display_nodes.to_string(index=False))
        if top_nodes > 0 and len(nodes) > top_nodes:
            print(f"... {len(nodes) - top_nodes:,} additional nodes not shown")

    print("\nEdges:")
    if edges.empty:
        print("No matching edges.")
    else:
        display_edges = edges.sort_values("weight")
        if top_edges > 0:
            display_edges = display_edges.head(top_edges)
        print(display_edges.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Inspect exported regulatory graph nodes, edges, and weights."
    )
    parser.add_argument(
        "--graph",
        type=Path,
        default=DEFAULT_GRAPH_PATH,
        help=f"Path to the graph .npz file (default: {DEFAULT_GRAPH_PATH.name})",
    )
    parser.add_argument(
        "--gene",
        help="Show nodes matching this gene/ID and edges connected to them.",
    )
    parser.add_argument(
        "--effect",
        choices=("activation", "inhibition", "expression", "other", "mixed"),
        help="Show only edges with this effect.",
    )
    parser.add_argument(
        "--top-nodes",
        type=int,
        default=20,
        help="Maximum nodes to print (0 prints all; default: 20).",
    )
    parser.add_argument(
        "--top-edges",
        type=int,
        default=20,
        help="Maximum edges to print, sorted by weight (0 prints all; default: 20).",
    )
    parser.add_argument("--edges-csv", type=Path, help="Optionally save matching edges.")
    parser.add_argument("--nodes-csv", type=Path, help="Optionally save matching nodes.")
    args = parser.parse_args()

    if args.top_nodes < 0 or args.top_edges < 0:
        parser.error("--top-nodes and --top-edges must be zero or greater")
    if not args.graph.is_file():
        parser.error(
            f"Graph file not found: {args.graph}. "
            "Generate it first with regulatoryNetwork_3DPlot.py --output graph."
        )

    try:
        inspect_graph(
            graph_path=args.graph,
            gene=args.gene,
            effect=args.effect,
            top_nodes=args.top_nodes,
            top_edges=args.top_edges,
            edges_csv=args.edges_csv,
            nodes_csv=args.nodes_csv,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

# python inspect_regulatory_network.py --gene TP53
# python inspect_regulatory_network.py --effect inhibition
# python inspect_regulatory_network.py --gene TP53 --edges-csv tp53_edges.csv
'''
Create a regulatory network as a graph object. The nodes are the genes and the 
edges are the regulatory interactions between them. The graph is directed and weighted, 
where the weight represents the strength of the interaction.

Adding pathway information from KEGG in KGML format and add the node and edge 
informations in the graph object. KGML: Gene/Entry-IDs; Relation types; sometimes reactions
    
    Need unique name for each gene node, as the same gene can be present in 
    multiple pathways. Need second mapping from KEGG-ID to gene symbol, as the 
    KGML file contains KEGG-IDs and not gene symbols. 
        => DONE def _kegg_gene_symbols(species) and def kegg_gene_to_symbol(kegg_id, species="hsa")
    
    And to compare with scRNA-seq data -> filter
        First: edge weights are +1.0 for activation/expression, -1.0 for
        inhibition/repression, and 0.0 when the subtype does not specify direction.
        Across pathways, the
        edge weight is the sum of the weights from each pathway.
        Second: Adjust nodes and edges to only include genes present in the scRNA-seq data.
        => DONE def filter_graph_to_gene_list(graph, allowed_genes)

    Combine different Pathways into one graph, and add pathway information to the nodes and edges. 
        => DONE def build_combined_graph(pathway_ids, scrna_gene_set=None, species="hsa")

# relation dissocation weight -1?

'''

import argparse
import json
from typing import Callable

import numpy as np
import requests
import xml.etree.ElementTree as ET
import networkx as nx
import csv
from functools import lru_cache
from pathlib import Path


POSITIVE_RELATION_SUBTYPES = {"activation", "expression"}
NEGATIVE_RELATION_SUBTYPES = {"inhibition", "repression"}


def _relation_effect_and_weight(subtype_names):
    """Assign signed weights only to KEGG subtypes with explicit direction."""
    subtype_names = set(subtype_names)
    has_positive_effect = bool(subtype_names & POSITIVE_RELATION_SUBTYPES)
    has_negative_effect = bool(subtype_names & NEGATIVE_RELATION_SUBTYPES)

    if has_positive_effect and has_negative_effect:
        return "mixed", 0.0
    if has_negative_effect:
        return "inhibition", -1.0
    if has_positive_effect:
        effect = "activation" if "activation" in subtype_names else "expression"
        return effect, 1.0
    return "other", 0.0


def _effect_for_weight(weight, effects=()):
    """Describe the sign of an aggregated edge weight."""
    if weight < 0:
        return "inhibition"
    if weight > 0:
        return "activation"
    effects = set(effects)
    if (
        effects & POSITIVE_RELATION_SUBTYPES
        and effects & NEGATIVE_RELATION_SUBTYPES
    ):
        return "mixed"
    return "other"


@lru_cache(maxsize=None)
def _kegg_gene_symbols(species):
    """Load KEGG's gene-name table once per organism."""
    response = requests.get(f"https://rest.kegg.jp/list/{species}", timeout=20)
    response.raise_for_status()
    symbols = {}
    for line in response.text.splitlines():
        kegg_id, separator, names = line.partition("\t")
        if separator:
            symbols[kegg_id] = names.split("\t")[-1].strip().split(";", 1)[0].strip()
    # print(list(symbols.items())[:5])  # Print first 5 items
    # [('hsa:124905564', 'CDS\t1:complement(4611..>32450)\tneuroblastoma breakpoint family member 1-like'), ('hsa:102466751', 'ncRNA\t1:complement(17369..17391)\tMIR6859-1'), ('hsa:124905321', 'ncRNA\t1:complement(21861..22024)\tU1 spliceosomal RNA'), ('hsa:317705', 'CDS\t1:22515..23588\tVN1R5'), ('hsa:107987367', 'tRNA\t1:complement(28755..28828)\tTRN-GTT26-1; tRNA-Asn')]
    return symbols


def kegg_gene_to_symbol(kegg_id, species="hsa"):
    '''
    Convert KEGG gene ID to gene symbol using KEGG REST API.
    '''
    if not kegg_id:
        return None
    if ":" not in kegg_id:
        kegg_id = f"{species}:{kegg_id}"
    try:
        return _kegg_gene_symbols(species).get(kegg_id)
    except requests.RequestException:
        return None


def build_graph(pathway_id="hsa04151", scrna_gene_set=None, species="hsa"):
    '''
    Build a directed graph from KEGG pathway KGML data. Node attributes include 
    gene symbol, KEGG ID, and type (gene, pathway, etc.).
    Edge attributes include relation type and weight.
    '''
    url = f"https://rest.kegg.jp/get/{pathway_id}/kgml"
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    root = ET.fromstring(response.text)

    graph = nx.DiGraph()
    entry_to_nodes = {}
    allowed_genes = set(scrna_gene_set) if scrna_gene_set is not None else None

    for entry in root.findall("entry"):
        entry_id = entry.get("id")
        entry_name = entry.get("name", "")
        entry_type = entry.get("type", "unknown")
        if not entry_id:
            continue

        node_ids = []
        if entry_type == "gene":
            for kegg_id in entry_name.split():
                if ":" not in kegg_id:
                    continue
                symbol = kegg_gene_to_symbol(kegg_id, species=species)
                if allowed_genes is not None and symbol not in allowed_genes:
                    continue

                node_id = f"{pathway_id}:{entry_id}:{kegg_id}"
                graph.add_node(
                    node_id,
                    label=symbol or kegg_id,
                    gene=symbol,
                    kegg_id=kegg_id,
                    entry_id=entry_id,
                    type=entry_type,
                    pathways=[pathway_id],
                )
                node_ids.append(node_id)
        else:
            node_id = f"{pathway_id}:{entry_id}"
            graph.add_node(
                node_id,
                label=entry_name,
                entry_id=entry_id,
                type=entry_type,
                pathways=[pathway_id],
            )
            node_ids.append(node_id)

        entry_to_nodes[entry_id] = node_ids

    for relation in root.findall("relation"):
        source_nodes = entry_to_nodes.get(relation.get("entry1"), [])
        target_nodes = entry_to_nodes.get(relation.get("entry2"), [])
        subtypes = [subtype.get("name") for subtype in relation.findall("subtype")]
        subtype_names = [name for name in subtypes if name]
        relation_type = relation.get("type", "other")
        relation_name = ";".join(subtype_names) or relation_type
        effect, weight = _relation_effect_and_weight(subtype_names)

        for source in source_nodes:
            for target in target_nodes:
                graph.add_edge(
                    source,
                    target,
                    relation=relation_name,
                    relation_type=relation_type,
                    effect=effect,
                    pathway=pathway_id,
                    weight=weight,
                )

    return graph


def load_gene_list(csv_path, symbol_column="gene"):
    """Read allowed gene symbols from a CSV column."""
    with open(csv_path, newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if symbol_column not in (reader.fieldnames or []):
            raise ValueError(
                f"Column {symbol_column!r} not found in {csv_path}; "
                f"available columns: {reader.fieldnames}"
            )
        genes = set()
        for row in reader:
            values = [row.get(symbol_column) or ""]
            values.extend(row.get(None) or [])
            genes.update(
                variant.strip()
                for value in values
                for variant in value.split(",")
                if variant.strip()
            )
        return genes


def _gene_name_variants(gene_name):
    """Return individual symbols from a comma-separated gene-name field."""
    if not gene_name:
        return set()
    return {variant.strip() for variant in str(gene_name).split(",") if variant.strip()}


def filter_graph_to_gene_list(graph, allowed_genes, *, copy=True):
    """Remove non-listed nodes and connect their predecessors to successors."""
    filtered_graph = graph.copy() if copy else graph
    allowed_genes = {gene.strip() for gene in allowed_genes if gene and gene.strip()}
    removed_nodes = [
        node
        for node, attributes in filtered_graph.nodes(data=True)
        if attributes.get("type") != "gene"
        or not (_gene_name_variants(attributes.get("gene")) & allowed_genes)
    ]

    for removed_node in removed_nodes:
        if removed_node not in filtered_graph:
            continue

        incoming_edges = [
            (source, dict(attributes))
            for source, _, attributes in filtered_graph.in_edges(removed_node, data=True)
            if source != removed_node
        ]
        outgoing_edges = [
            (target, dict(attributes))
            for _, target, attributes in filtered_graph.out_edges(removed_node, data=True)
            if target != removed_node
        ]

        for predecessor, incoming in incoming_edges:
            for successor, outgoing in outgoing_edges:
                if predecessor == successor:
                    continue

                weight = (
                    incoming.get("weight", 0.0) + outgoing.get("weight", 0.0)
                ) / 2
                relation = f"{incoming.get('relation', 'relation')} -> {outgoing.get('relation', 'relation')}"
                pathways = sorted(
                    set(incoming.get("pathways", []))
                    | set(outgoing.get("pathways", []))
                )
                through = filtered_graph.nodes[removed_node].get(
                    "gene", filtered_graph.nodes[removed_node].get("label", removed_node)
                )
                shortcut = {
                    "relation": relation,
                    "relation_type": f"{incoming.get('relation_type', 'other')} -> {outgoing.get('relation_type', 'other')}",
                    "effect": _effect_for_weight(
                        weight,
                        (incoming.get("effect", "other"), outgoing.get("effect", "other")),
                    ),
                    "weight": weight,
                    "pathways": pathways,
                    "adjusted_through": [through],
                }

                if not filtered_graph.has_edge(predecessor, successor):
                    filtered_graph.add_edge(predecessor, successor, **shortcut)
                    continue

                existing = filtered_graph[predecessor][successor]
                existing["weight"] = (existing.get("weight", 0.0) + weight) / 2
                existing["pathways"] = sorted(
                    set(existing.get("pathways", [])) | set(pathways)
                )
                existing["adjusted_through"] = sorted(
                    set(existing.get("adjusted_through", [])) | {through}
                )
                relations = set(existing.get("relations", [existing.get("relation", "")]))
                relations.add(relation)
                existing["relations"] = sorted(relations - {""})
                existing["relation"] = "; ".join(existing["relations"])
                existing["effect"] = _effect_for_weight(
                    existing["weight"],
                    (
                        *existing.get(
                            "effects", [existing.get("effect", "other")]
                        ),
                        incoming.get("effect", "other"),
                        outgoing.get("effect", "other"),
                    ),
                )

        filtered_graph.remove_node(removed_node)

    return filtered_graph


def build_combined_graph(
    pathway_ids,
    scrna_gene_set=None,
    species="hsa",
    progress_callback: Callable[[str], None] | None = None,
):
    """Filter each pathway before merging repeated genes by KEGG gene ID."""
    combined_graph = nx.DiGraph()

    pathway_ids = list(pathway_ids)
    allowed_genes = set(scrna_gene_set) if scrna_gene_set is not None else None
    for index, pathway_id in enumerate(pathway_ids, start=1):
        if progress_callback is not None:
            progress_callback(
                f"Loading pathway {index}/{len(pathway_ids)}: {pathway_id}"
            )
        pathway_graph = build_graph(
            pathway_id,
            scrna_gene_set=None,
            species=species,
        )
        if allowed_genes is not None:
            raw_nodes = pathway_graph.number_of_nodes()
            raw_edges = pathway_graph.number_of_edges()
            filter_graph_to_gene_list(
                pathway_graph, allowed_genes, copy=False
            )
            if progress_callback is not None:
                progress_callback(
                    f"Filtered {pathway_id}: {raw_nodes} nodes/{raw_edges} edges "
                    f"to {pathway_graph.number_of_nodes()} nodes/"
                    f"{pathway_graph.number_of_edges()} edges"
                )

        node_mapping = {}

        for node_id, attributes in pathway_graph.nodes(data=True):
            if attributes.get("type") == "gene" and attributes.get("kegg_id"):
                combined_node_id = attributes["kegg_id"]
            else:
                combined_node_id = node_id
            node_mapping[node_id] = combined_node_id

            if combined_node_id not in combined_graph:
                combined_graph.add_node(combined_node_id, **attributes)
                combined_graph.nodes[combined_node_id]["pathways"] = [pathway_id]
                combined_graph.nodes[combined_node_id]["entry_ids"] = [
                    attributes.get("entry_id")
                ]
            else:
                node_attributes = combined_graph.nodes[combined_node_id]
                if pathway_id not in node_attributes["pathways"]:
                    node_attributes["pathways"].append(pathway_id)
                entry_id = attributes.get("entry_id")
                if entry_id not in node_attributes["entry_ids"]:
                    node_attributes["entry_ids"].append(entry_id)

        for source, target, attributes in pathway_graph.edges(data=True):
            combined_source = node_mapping[source]
            combined_target = node_mapping[target]
            if combined_source == combined_target:
                continue

            edge_attributes = dict(attributes)
            edge_attributes["pathways"] = [pathway_id]
            edge_attributes["pathway_weights"] = {
                pathway_id: edge_attributes.get("weight", 0.0)
            }
            if not combined_graph.has_edge(combined_source, combined_target):
                combined_graph.add_edge(
                    combined_source, combined_target, **edge_attributes
                )
                continue

            existing = combined_graph[combined_source][combined_target]
            if pathway_id not in existing["pathways"]:
                existing["pathways"].append(pathway_id)
            effects = set(existing.get("effects", [existing.get("effect", "other")]))
            effects.add(edge_attributes.get("effect", "other"))
            existing["effects"] = sorted(effects)
            existing["effect"] = next(iter(effects)) if len(effects) == 1 else "mixed"
            relations = set(existing.get("relations", [existing.get("relation", "")]))
            relations.add(edge_attributes.get("relation", ""))
            existing["relations"] = sorted(relations - {""})
            existing["relation"] = ";".join(existing["relations"])
            existing.setdefault(
                "pathway_weights",
                {existing["pathways"][0]: existing.get("weight", 0.0)},
            )
            existing["pathway_weights"][pathway_id] = edge_attributes.get("weight", 0.0)
            existing["weight"] = sum(existing["pathway_weights"].values()) / len(
                existing["pathway_weights"]
            )
            existing["effect"] = _effect_for_weight(
                existing["weight"], existing["effects"]
            )

        if progress_callback is not None:
            progress_callback(
                f"Merged {pathway_id}: {combined_graph.number_of_nodes()} nodes, "
                f"{combined_graph.number_of_edges()} edges"
            )

    return combined_graph


def plot_graph_3d(graph, html_output_path, png_output_path):
    """Save an interactive Plotly view and a static PNG of the 3D network."""
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    if not graph:
        raise ValueError("Cannot plot an empty regulatory network.")

    if len(graph) >= 500:
        positions = nx.spectral_layout(graph, dim=3, weight=None)
    else:
        positions = nx.spring_layout(
            graph, dim=3, seed=42, weight=None, iterations=40
        )
    nodes = list(graph.nodes)
    coordinates = np.array([positions[node] for node in nodes])
    node_colors = [
        graph.nodes[node].get("color", "lightgray") for node in nodes
    ]
    node_labels = [str(graph.nodes[node].get("label", node)) for node in nodes]
    node_text = [
        f"{node_labels[index]}"
        f"<br>Pathways: {', '.join(graph.nodes[node].get('pathways', []))}"
        for index, node in enumerate(nodes)
    ]

    interactive_figure = go.Figure()
    edge_colors = {
        "activation": "red",
        "expression": "red",
        "inhibition": "blue",
        "other": "gray",
        "mixed": "gray",
    }
    max_direction_arrows = 1500
    max_edge_hover_points = 3000
    edges = list(graph.edges(data=True))
    arrow_stride = max(1, (len(edges) + max_direction_arrows - 1) // max_direction_arrows)
    hover_stride = max(1, (len(edges) + max_edge_hover_points - 1) // max_edge_hover_points)
    edges_by_effect = {}
    for edge_index, (source, target, attributes) in enumerate(edges):
        effect = attributes.get("effect", "other")
        group = edges_by_effect.setdefault(
            effect,
            {
                "x": [],
                "y": [],
                "z": [],
                "arrows": [],
                "hover_x": [],
                "hover_y": [],
                "hover_z": [],
                "hover_text": [],
            },
        )
        start = positions[source]
        end = positions[target]
        group["x"].extend([start[0], end[0], None])
        group["y"].extend([start[1], end[1], None])
        group["z"].extend([start[2], end[2], None])
        if edge_index % arrow_stride == 0:
            group["arrows"].append((source, target))
        if edge_index % hover_stride == 0:
            group["hover_x"].append((start[0] + end[0]) / 2)
            group["hover_y"].append((start[1] + end[1]) / 2)
            group["hover_z"].append((start[2] + end[2]) / 2)
            group["hover_text"].append(
                f"{graph.nodes[source].get('label', source)} → "
                f"{graph.nodes[target].get('label', target)}"
                f"<br>Effect: {effect}"
                f"<br>Relation: {attributes.get('relation', '')}"
                f"<br>Weight: {attributes.get('weight', 1.0):g}"
            )

    for effect, group in edges_by_effect.items():
        interactive_figure.add_trace(
            go.Scatter3d(
                x=group["x"],
                y=group["y"],
                z=group["z"],
                mode="lines",
                line={"color": edge_colors.get(effect, "gray"), "width": 2},
                name=f"{effect.title()} edges",
                hoverinfo="skip",
            )
        )
        arrows = group["arrows"]
        if arrows:
            interactive_figure.add_trace(
                go.Cone(
                    x=[positions[target][0] for _, target in arrows],
                    y=[positions[target][1] for _, target in arrows],
                    z=[positions[target][2] for _, target in arrows],
                    u=[
                        positions[target][0] - positions[source][0]
                        for source, target in arrows
                    ],
                    v=[
                        positions[target][1] - positions[source][1]
                        for source, target in arrows
                    ],
                    w=[
                        positions[target][2] - positions[source][2]
                        for source, target in arrows
                    ],
                    anchor="tip",
                    sizemode="absolute",
                    sizeref=0.12,
                    colorscale=[
                        [0, edge_colors.get(effect, "gray")],
                        [1, edge_colors.get(effect, "gray")],
                    ],
                    showscale=False,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        if group["hover_text"]:
            interactive_figure.add_trace(
                go.Scatter3d(
                    x=group["hover_x"],
                    y=group["hover_y"],
                    z=group["hover_z"],
                    mode="markers",
                    marker={"size": 5, "opacity": 0},
                    text=group["hover_text"],
                    hoverinfo="text",
                    showlegend=False,
                )
            )

    interactive_figure.add_trace(
        go.Scatter3d(
            x=coordinates[:, 0],
            y=coordinates[:, 1],
            z=coordinates[:, 2],
            mode="markers+text" if len(nodes) <= 100 else "markers",
            text=node_labels if len(nodes) <= 100 else None,
            textposition="top center",
            hovertext=node_text,
            hoverinfo="text",
            marker={
                "size": 5,
                "color": node_colors,
                "opacity": 0.9,
                "line": {"width": 0.5, "color": "white"},
            },
            name="Genes",
        )
    )
    interactive_figure.update_layout(
        title="3D Regulatory Network",
        scene={
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "zaxis": {"visible": False},
            "aspectmode": "data",
        },
        legend={"title": "Regulatory effects"},
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
    )
    interactive_figure.write_html(html_output_path, include_plotlyjs=True)

    figure = plt.figure(figsize=(12, 9))
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        coordinates[:, 2],
        c=node_colors,
        s=24,
        alpha=0.9,
        depthshade=True,
    )

    if edges:
        edge_segments = [
            [positions[source], positions[target]]
            for source, target, _ in edges
        ]
        edge_line_colors = [
            {
                "activation": "red",
                "expression": "red",
                "inhibition": "blue",
            }.get(attributes.get("effect"), "gray")
            for _, _, attributes in edges
        ]
        edge_collection = Line3DCollection(
            edge_segments,
            colors=edge_line_colors,
            linewidths=0.5,
            alpha=0.35,
        )
        axis.add_collection3d(edge_collection)

    if len(nodes) <= 100:
        for node, (x, y, z) in zip(nodes, coordinates):
            axis.text(
                x, y, z, str(graph.nodes[node].get("label", node)), fontsize=7
            )

    spans = np.ptp(coordinates, axis=0)
    axis.set_box_aspect(np.where(spans == 0, 1, spans))
    axis.set_axis_off()
    axis.set_title("3D Regulatory Network")
    axis.legend(
        handles=[
            Line2D([0], [0], color="red", label="Activation/expression (+)"),
            Line2D([0], [0], color="blue", label="Inhibition/repression (-)"),
            Line2D([0], [0], color="gray", label="Unspecified direction (0)"),
        ],
        loc="upper left",
    )
    figure.tight_layout()
    figure.savefig(png_output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_graph_for_gnn(graph, output_path):
    """Save a directed graph as a compressed NumPy bundle for GNN workflows."""
    nodes = list(graph.nodes)
    node_to_index = {node: index for index, node in enumerate(nodes)}
    edges = list(graph.edges(data=True))
    effect_classes = ("activation", "inhibition", "expression", "other")

    edge_index = np.asarray(
        [
            [node_to_index[source] for source, _, _ in edges],
            [node_to_index[target] for _, target, _ in edges],
        ],
        dtype=np.int64,
    ).reshape(2, -1)
    edge_attr = np.zeros((len(edges), 1 + len(effect_classes)), dtype=np.float32)
    edge_effects = []
    edge_relations = []
    edge_pathways = []
    for index, (_, _, attributes) in enumerate(edges):
        effect = attributes.get("effect", "other")
        edge_attr[index, 0] = float(attributes.get("weight", 1.0))
        if effect in effect_classes:
            edge_attr[index, 1 + effect_classes.index(effect)] = 1.0
        else:
            edge_attr[index, 1 + effect_classes.index("other")] = 1.0
        edge_effects.append(str(effect))
        edge_relations.append(str(attributes.get("relation", "")))
        edge_pathways.append(
            json.dumps(attributes.get("pathways", []), ensure_ascii=True)
        )

    np.savez_compressed(
        output_path,
        node_ids=np.asarray([str(node) for node in nodes], dtype=np.str_),
        node_labels=np.asarray(
            [str(graph.nodes[node].get("label", node)) for node in nodes],
            dtype=np.str_,
        ),
        node_genes=np.asarray(
            [str(graph.nodes[node].get("gene") or "") for node in nodes],
            dtype=np.str_,
        ),
        node_types=np.asarray(
            [str(graph.nodes[node].get("type", "")) for node in nodes],
            dtype=np.str_,
        ),
        node_pathways=np.asarray(
            [
                json.dumps(graph.nodes[node].get("pathways", []), ensure_ascii=True)
                for node in nodes
            ],
            dtype=np.str_,
        ),
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_effects=np.asarray(edge_effects, dtype=np.str_),
        edge_relations=np.asarray(edge_relations, dtype=np.str_),
        edge_pathways=np.asarray(edge_pathways, dtype=np.str_),
        effect_classes=np.asarray(effect_classes, dtype=np.str_),
    )


def main(output_mode="both"):
    if output_mode not in {"both", "plot", "graph"}:
        raise ValueError("output_mode must be 'both', 'plot', or 'graph'.")

    #pathway_ids = ["hsa04110", "hsa04210", "hsa04151"]
    pathway_ids = [
    "hsa04110",
    "hsa04114",
    "hsa04115",
    "hsa03410",
    "hsa03420",
    "hsa04010",
    "hsa04014",
    "hsa04015",
    "hsa04012",
    "hsa04020",
    "hsa04024",
    "hsa04022",
    "hsa04151",
    "hsa04152",
    "hsa04150",
    "hsa04310",
    "hsa04330",
    "hsa04350",
    "hsa04370",
    "hsa04630",
    "hsa04668",
    "hsa04657",
    "hsa04620",
    "hsa04621",
    "hsa04660",
    "hsa04662",
    "hsa04722",
    "hsa04710",
    "hsa04210"
    ]
    
    gene_csv = Path(__file__).with_name("gene_list.csv")
    gene_set = load_gene_list(gene_csv)
    def report_progress(message):
        print(message, flush=True)

    graph = build_combined_graph(
        pathway_ids,
        scrna_gene_set=gene_set,
        progress_callback=report_progress,
    )

    for node in graph:
        if graph.nodes[node].get("pathways"):
            graph.nodes[node]["color"] = {
                "hsa04110": "lightgreen",
                "hsa04210": "orange",
                "hsa04151": "lightblue",
                "hsa04110, hsa04210": "yellow",
                "hsa04110, hsa04151": "lightyellow",
                "hsa04210, hsa04151": "lightcoral",
                "hsa04110, hsa04210, hsa04151": "lightgray",
            }.get(", ".join(sorted(graph.nodes[node]["pathways"])), "lightgray")
            graph.nodes[node]["title"] = (
                f"In pathways {', '.join(graph.nodes[node]['pathways'])}"
            )
        else:
            graph.nodes[node]["color"] = "lightgray"

    output_directory = Path(__file__).parent
    if output_mode in {"both", "plot"}:
        html_output_path = output_directory / "pathway_network_3d.html"
        png_output_path = output_directory / "pathway_network_3d.png"
        plot_graph_3d(graph, html_output_path, png_output_path)
        print(f"Interaktiver 3D-Plot gespeichert: {html_output_path}")
        print(f"PNG-Export gespeichert: {png_output_path}")

    if output_mode in {"both", "graph"}:
        graph_output_path = output_directory / "pathway_network_gnn.npz"
        save_graph_for_gnn(graph, graph_output_path)
        print(f"GNN-Graph gespeichert: {graph_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a regulatory network and export a plot, graph data, or both."
    )
    parser.add_argument(
        "--output",
        choices=("both", "plot", "graph"),
        default="both",
        help="Export both visualizations and graph data, only the plots, or only the GNN graph file.",
    )
    main(output_mode=parser.parse_args().output)

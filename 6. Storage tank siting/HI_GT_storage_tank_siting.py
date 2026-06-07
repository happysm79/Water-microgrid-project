"""
Hydraulically Informed Graph-Theoretic Framework for Storage Tank Siting in WDNs

Author: Amirmahdi Ghanaatikashani
Institution: Southern Illinois University Carbondale

Description:
This script analyzes water distribution networks using hydraulic simulation,
graph-theoretic centrality metrics, entropy-based weighting, sectorization,
and IGT-based source configuration evaluation to identify candidate storage
tank locations.
"""
# ============================================================
# HI-GT FRAMEWORK FOR STORAGE TANK SITING IN WDNs
# ============================================================

# -----------------------------
# Install required packages
# -----------------------------
# !pip -q install wntr openpyxl scikit-learn

# -----------------------------
# Imports
# -----------------------------
import os
import re
import math
import warnings
import zipfile
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import wntr

from google.colab import files
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    # Output folder
    "output_dir": "/content/higt_outputs",

    # Hydraulic simulation
    "steady_state_duration_sec": 0,

    # Ranking
    "top_n_full": 10,
    "top_n_per_cluster": 3,

    # Clustering options
    "cluster_k_range": list(range(2, 9)),
    "user_defined_k": 3,
    "spectral_random_state": 42,
    "n_init": 20,

    # IGT
    "k_shortest_paths": 30,
    "igt_trim_fraction": 0.10,

    # Hazen-Williams resistance
    "hw_constant": 10.67,
    "hw_exp_c": 1.852,
    "hw_exp_d": 4.871,

    # Numerical safety
    "epsilon": 1e-12,

    # Plot settings
    "plot_node_size": 22,
    "plot_source_size": 90,
    "plot_top_node_size": 130,
    "dpi": 300
}


# ============================================================
# FILE UPLOAD AND OUTPUT FOLDERS
# ============================================================

def upload_inp_file_colab() -> str:
    uploaded = files.upload()
    if not uploaded:
        raise FileNotFoundError("No file was uploaded.")

    inp_candidates = [name for name in uploaded.keys() if name.lower().endswith(".inp")]
    if not inp_candidates:
        raise FileNotFoundError("No .inp file found among uploaded files.")

    inp_file = inp_candidates[0]
    print(f"Uploaded INP file: {inp_file}")
    return inp_file


def ensure_output_dirs(base_dir: str) -> Dict[str, str]:
    subdirs = {
        "base": base_dir,
        "tables": os.path.join(base_dir, "tables"),
        "figures": os.path.join(base_dir, "figures"),
        "clusters": os.path.join(base_dir, "figures", "clusters"),
        "scores": os.path.join(base_dir, "figures", "scores"),
        "igt": os.path.join(base_dir, "figures", "igt"),
    }

    for path in subdirs.values():
        os.makedirs(path, exist_ok=True)

    return subdirs


def save_table(df: pd.DataFrame, filepath: str):
    df.to_csv(filepath, index=False)


# ============================================================
# BASIC UTILITIES
# ============================================================

def minmax_normalize(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()

    for c in cols:
        vals = out[c].astype(float)
        vmin = vals.min()
        vmax = vals.max()

        if pd.isna(vmin) or pd.isna(vmax):
            out[f"norm_{c}"] = 0.0
        elif abs(vmax - vmin) <= CONFIG["epsilon"]:
            out[f"norm_{c}"] = 1.0
        else:
            out[f"norm_{c}"] = (vals - vmin) / (vmax - vmin)

    return out


def entropy_weights(norm_df: pd.DataFrame,
                    norm_cols: List[str],
                    epsilon: float = 1e-12) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Entropy weighting:
      p_ij = (xhat_ij + epsilon) / sum_i(xhat_ij + epsilon)
      E_j = -1/ln(n) * sum_i p_ij ln(p_ij)
      d_j = 1 - E_j
      alpha_j = d_j / sum_j d_j
    """

    X = norm_df[norm_cols].fillna(0.0).to_numpy(dtype=float)
    n = X.shape[0]

    if n == 0:
        return pd.DataFrame(), {}

    X_eps = X + epsilon
    col_sums = X_eps.sum(axis=0)
    P = X_eps / col_sums

    if n <= 1:
        alpha = np.ones(len(norm_cols)) / len(norm_cols)
        E = np.zeros(len(norm_cols))
        d = np.ones(len(norm_cols))
    else:
        E = -(1.0 / np.log(n)) * np.sum(P * np.log(P + epsilon), axis=0)
        d = 1.0 - E

        if abs(d.sum()) <= epsilon:
            alpha = np.ones(len(norm_cols)) / len(norm_cols)
        else:
            alpha = d / d.sum()

    entropy_tbl = pd.DataFrame({
        "metric": norm_cols,
        "entropy": E,
        "diversification": d,
        "alpha_weight": alpha
    })

    weights = dict(zip(norm_cols, alpha))

    return entropy_tbl, weights


def weighted_score(df: pd.DataFrame,
                   norm_cols: List[str],
                   weights: Dict[str, float],
                   score_col: str = "CS") -> pd.DataFrame:
    out = df.copy()
    out[score_col] = 0.0

    for c in norm_cols:
        out[score_col] += out[c] * weights[c]

    out["rank"] = out[score_col].rank(ascending=False, method="dense")
    out = out.sort_values(["rank", score_col], ascending=[True, False]).reset_index(drop=True)

    return out


def trimmed_mean_series(s: pd.Series, trim_fraction: float = 0.10) -> float:
    vals = np.sort(s.dropna().astype(float).to_numpy())
    n = len(vals)

    if n == 0:
        return np.nan

    g = int(np.floor(trim_fraction * n))

    if 2 * g >= n:
        return float(np.mean(vals))

    trimmed = vals[g:n - g]

    if len(trimmed) == 0:
        return float(np.mean(vals))

    return float(np.mean(trimmed))


# ============================================================
# LOAD NETWORK AND EXTRACT TABLES
# ============================================================

def load_network(inp_file: str):
    wn = wntr.network.WaterNetworkModel(inp_file)
    return wn


def extract_node_table(wn) -> pd.DataFrame:
    records = []

    coords = wn.query_node_attribute("coordinates")
    demands = wn.query_node_attribute("base_demand")
    elevations = wn.query_node_attribute("elevation")

    for name, node in wn.nodes():
        x, y = coords.get(name, (np.nan, np.nan)) if hasattr(coords, "get") else (np.nan, np.nan)

        records.append({
            "node_id": str(name),
            "node_type": node.node_type,
            "x": x,
            "y": y,
            "elevation": elevations.get(name, np.nan) if hasattr(elevations, "get") else np.nan,
            "base_demand": demands.get(name, 0.0) if hasattr(demands, "get") else 0.0,
        })

    return pd.DataFrame(records)


def extract_link_table(wn) -> pd.DataFrame:
    records = []

    for name, link in wn.links():
        records.append({
            "link_id": str(name),
            "link_type": link.link_type,
            "start_node": str(link.start_node_name),
            "end_node": str(link.end_node_name),
            "length": getattr(link, "length", np.nan),
            "diameter": getattr(link, "diameter", np.nan),
            "roughness": getattr(link, "roughness", np.nan),
            "minor_loss": getattr(link, "minor_loss", np.nan),
            "status": str(getattr(link, "initial_status", "Open")),
        })

    return pd.DataFrame(records)


def get_source_nodes(wn) -> Dict[str, List[str]]:
    tanks = [str(name) for name, node in wn.nodes() if node.node_type == "Tank"]
    reservoirs = [str(name) for name, node in wn.nodes() if node.node_type == "Reservoir"]

    return {
        "tanks": sorted(tanks),
        "reservoirs": sorted(reservoirs),
        "existing_storage_sources": sorted(tanks + reservoirs)
    }


def build_network_summary(wn, link_df: pd.DataFrame, sources: Dict[str, List[str]]) -> pd.DataFrame:
    records = [
        {"component": "junctions", "count": sum(1 for _, n in wn.nodes() if n.node_type == "Junction")},
        {"component": "reservoirs", "count": len(sources["reservoirs"])},
        {"component": "tanks", "count": len(sources["tanks"])},
        {"component": "pipes", "count": int((link_df["link_type"] == "Pipe").sum())},
        {"component": "pumps", "count": int((link_df["link_type"] == "Pump").sum())},
        {"component": "valves", "count": int((link_df["link_type"] == "Valve").sum())},
        {"component": "total_nodes", "count": len(list(wn.nodes()))},
        {"component": "total_links", "count": link_df.shape[0]},
    ]

    return pd.DataFrame(records)


def check_units_for_hw(link_df: pd.DataFrame):
    pipes = link_df[link_df["link_type"] == "Pipe"].copy()

    if pipes.empty:
        print("No pipes found for unit check.")
        return

    median_d = pipes["diameter"].dropna().median()
    median_l = pipes["length"].dropna().median()

    print("\n--- Unit sanity check for Hazen-Williams resistance ---")
    print(f"Median pipe length: {median_l}")
    print(f"Median pipe diameter: {median_d}")

    if median_d > 5:
        print("WARNING: Median pipe diameter is larger than expected for SI units.")
        print("WNTR usually converts EPANET input values to SI, but please verify units.")
    else:
        print("Diameter values appear reasonable for SI-based Hazen-Williams resistance.")


# ============================================================
# PREDEFINED DMA PARSING
# ============================================================

def parse_dma_labels_from_inp(inp_file: str) -> Dict[str, str]:
    """
    Parse predefined DMA labels from the [JUNCTIONS] section.

    Expected C-Town format:
        node_id elevation base_demand DMA1_pat

    Returns:
        {node_id: "DMA1", node_id: "DMA2", ...}
    """

    dma_labels = {}
    in_junctions = False

    with open(inp_file, "r", encoding="latin-1", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith("["):
                in_junctions = line.upper().startswith("[JUNCTIONS]")
                continue

            if not in_junctions:
                continue

            if line.startswith(";"):
                continue

            line_no_comment = line.split(";")[0].strip()

            if not line_no_comment:
                continue

            parts = line_no_comment.split()

            if len(parts) >= 4:
                node_id = str(parts[0])
                pattern = parts[3]

                match = re.match(r"(DMA\d+)_pat", pattern, re.IGNORECASE)
                if match:
                    dma_label = match.group(1).upper()
                    dma_labels[node_id] = dma_label

    return dma_labels


def add_dma_labels_to_node_df(node_df: pd.DataFrame,
                              dma_labels: Dict[str, str]) -> pd.DataFrame:
    out = node_df.copy()
    out["dma_label"] = out["node_id"].map(dma_labels)
    return out


# ============================================================
# STEADY-STATE HYDRAULIC SIMULATION
# ============================================================

def run_steady_state_simulation(wn, duration_sec: int = 0):
    wn.options.time.duration = duration_sec
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()
    return results


def extract_hydraulic_results(wn, results) -> Tuple[pd.DataFrame, pd.DataFrame]:
    t0 = results.node["head"].index[0]

    try:
        headloss = results.link["headloss"].loc[t0]
    except Exception:
        headloss = pd.Series(dtype=float)

    try:
        status = results.link["status"].loc[t0]
    except Exception:
        status = pd.Series(dtype=float)

    link_records = []

    for name, link in wn.links():
        name = str(name)
        hl = headloss.get(name, np.nan) if len(headloss) else np.nan

        link_records.append({
            "link_id": name,
            "link_type": link.link_type,
            "headloss_result": float(hl) if pd.notna(hl) else np.nan,
            "abs_headloss_result": abs(float(hl)) if pd.notna(hl) else np.nan,
            "sim_status": float(status.get(name, np.nan)) if len(status) else np.nan,
        })

    head = results.node["head"].loc[t0]
    pressure = results.node["pressure"].loc[t0]

    node_records = []

    for name, node in wn.nodes():
        name = str(name)
        node_records.append({
            "node_id": name,
            "head": float(head.get(name, np.nan)),
            "pressure": float(pressure.get(name, np.nan)),
        })

    return pd.DataFrame(link_records), pd.DataFrame(node_records)


# ============================================================
# HYDRAULIC EDGE WEIGHTS
# ============================================================

def compute_hazen_williams_resistance(length, roughness, diameter, cfg=CONFIG):
    """
    Flow-independent Hazen-Williams pipe resistance coefficient:
    r = 10.67 * L / (C^1.852 * D^4.871)
    """

    if any(pd.isna(v) for v in [length, roughness, diameter]):
        return np.nan

    if length <= 0 or roughness <= 0 or diameter <= 0:
        return np.nan

    r = cfg["hw_constant"] * length / (
        (roughness ** cfg["hw_exp_c"]) *
        (diameter ** cfg["hw_exp_d"]) +
        cfg["epsilon"]
    )

    return r


def build_link_hydraulic_table(link_df: pd.DataFrame,
                               hydraulic_link_df: pd.DataFrame,
                               cfg=CONFIG) -> pd.DataFrame:
    merged = link_df.merge(hydraulic_link_df, on=["link_id", "link_type"], how="left")

    def edge_weight(row):
        link_type = row["link_type"]

        if link_type == "Pipe":
            r = compute_hazen_williams_resistance(
                row["length"],
                row["roughness"],
                row["diameter"],
                cfg=cfg
            )
            return max(r, cfg["epsilon"]) if pd.notna(r) else np.nan

        elif link_type in ["Pump", "Valve"]:
            hl = row["headloss_result"]
            return max(abs(hl), cfg["epsilon"]) if pd.notna(hl) else np.nan

        else:
            return np.nan

    merged["edge_weight"] = merged.apply(edge_weight, axis=1)

    merged["edge_weight_was_missing"] = merged["edge_weight"].isna()
    merged["edge_weight"] = merged["edge_weight"].fillna(1.0)

    return merged


# ============================================================
# GRAPH BUILDING
# ============================================================

def build_graphs(node_df: pd.DataFrame,
                 link_hyd_df: pd.DataFrame,
                 cfg=CONFIG) -> Tuple[nx.Graph, nx.Graph]:
    """
    G_topo: topological graph for clustering and degree centrality.
    Gw: hydraulically weighted graph for closeness, betweenness, and IGT.
    """

    G_topo = nx.Graph()
    Gw = nx.Graph()

    node_attrs = node_df.set_index("node_id").to_dict("index")

    for node_id, attrs in node_attrs.items():
        G_topo.add_node(str(node_id), **attrs)
        Gw.add_node(str(node_id), **attrs)

    for _, row in link_hyd_df.iterrows():
        u = str(row["start_node"])
        v = str(row["end_node"])

        if pd.isna(u) or pd.isna(v):
            continue

        hydraulic_w = float(max(row["edge_weight"], cfg["epsilon"]))

        common_attrs = {
            "link_id": row["link_id"],
            "link_type": row["link_type"],
            "length": row.get("length", np.nan),
            "diameter": row.get("diameter", np.nan),
            "roughness": row.get("roughness", np.nan),
            "headloss_result": row.get("headloss_result", np.nan),
            "edge_weight": hydraulic_w,
        }

        if Gw.has_edge(u, v):
            if hydraulic_w < Gw[u][v].get("weight", np.inf):
                G_topo[u][v].update({**common_attrs, "weight": 1.0})
                Gw[u][v].update({**common_attrs, "weight": hydraulic_w})
        else:
            G_topo.add_edge(u, v, **common_attrs, weight=1.0)
            Gw.add_edge(u, v, **common_attrs, weight=hydraulic_w)

    return G_topo, Gw


def get_pos(G: nx.Graph) -> Dict[str, Tuple[float, float]]:
    pos = {}

    for n, data in G.nodes(data=True):
        x = data.get("x", 0.0)
        y = data.get("y", 0.0)

        if pd.isna(x) or pd.isna(y):
            x, y = 0.0, 0.0

        pos[n] = (x, y)

    return pos


# ============================================================
# CENTRALITY METRICS AND RANKING
# ============================================================

def compute_centrality_metrics(G_topo: nx.Graph,
                               G_weighted: nx.Graph,
                               node_subset: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Degree centrality is topological.
    Closeness and betweenness are hydraulically weighted.
    """

    if node_subset is not None:
        node_subset = [str(n) for n in node_subset]
        Gt = G_topo.subgraph(node_subset).copy()
        Gw = G_weighted.subgraph(node_subset).copy()
    else:
        Gt = G_topo.copy()
        Gw = G_weighted.copy()

    nodes = list(Gw.nodes())

    degree = nx.degree_centrality(Gt)
    closeness = nx.closeness_centrality(Gw, distance="weight")
    betweenness = nx.betweenness_centrality(Gw, weight="weight", normalized=True)

    return pd.DataFrame({
        "node_id": nodes,
        "degree_centrality": [degree.get(n, 0.0) for n in nodes],
        "closeness_centrality": [closeness.get(n, 0.0) for n in nodes],
        "betweenness_centrality": [betweenness.get(n, 0.0) for n in nodes],
    })


def filter_candidate_nodes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Candidate nodes exclude existing tanks and reservoirs.
    """

    out = df.copy()
    out = out[~out["node_type"].isin(["Tank", "Reservoir"])]
    return out.reset_index(drop=True)


def compute_ranking(metrics_df: pd.DataFrame,
                    score_col: str,
                    scenario_name: str,
                    cluster_id: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    use_cols = [
        "degree_centrality",
        "closeness_centrality",
        "betweenness_centrality"
    ]

    candidate_df = filter_candidate_nodes(metrics_df)

    if candidate_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    norm_df = minmax_normalize(candidate_df, use_cols)
    norm_cols = [f"norm_{c}" for c in use_cols]

    entropy_tbl, weights = entropy_weights(norm_df, norm_cols, epsilon=CONFIG["epsilon"])
    ranked = weighted_score(norm_df, norm_cols, weights, score_col=score_col)

    ranked["scenario"] = scenario_name
    entropy_tbl["scenario"] = scenario_name

    if cluster_id is not None:
        ranked["cluster_id"] = cluster_id
        entropy_tbl["cluster_id"] = cluster_id

    return ranked, entropy_tbl


def compute_full_network_ranking(node_df: pd.DataFrame,
                                 node_hyd_df: pd.DataFrame,
                                 G_topo: nx.Graph,
                                 Gw: nx.Graph,
                                 score_col: str = "CS_full") -> Tuple[pd.DataFrame, pd.DataFrame]:
    metrics = compute_centrality_metrics(G_topo, Gw)
    merged = node_df.merge(node_hyd_df, on="node_id", how="left").merge(metrics, on="node_id", how="left")
    ranked, entropy_tbl = compute_ranking(merged, score_col=score_col, scenario_name="full_network")
    return ranked, entropy_tbl


# ============================================================
# SPECTRAL CLUSTERING
# ============================================================

def build_topological_affinity(G: nx.Graph) -> Tuple[np.ndarray, List[str]]:
    nodes = list(G.nodes())
    A = nx.to_numpy_array(G, nodelist=nodes, weight=None)
    A = (A > 0).astype(float)
    np.fill_diagonal(A, 0.0)
    return A, nodes


def relabel_compact(labels: np.ndarray) -> np.ndarray:
    unique = sorted(pd.unique(labels))
    mapping = {old: new for new, old in enumerate(unique)}
    return np.array([mapping[x] for x in labels], dtype=int)


def count_cluster_fragments(G: nx.Graph, nodes: List[str], labels: np.ndarray) -> int:
    df = pd.DataFrame({"node_id": nodes, "cluster_id": labels})
    extra_fragments = 0

    for _, sub in df.groupby("cluster_id"):
        sub_nodes = sub["node_id"].tolist()
        comps = list(nx.connected_components(G.subgraph(sub_nodes)))
        extra_fragments += max(0, len(comps) - 1)

    return extra_fragments


def enforce_connected_clusters(G: nx.Graph,
                               nodes: List[str],
                               labels: np.ndarray,
                               max_iter: int = 20) -> np.ndarray:
    """
    Reassign disconnected fragments to neighboring clusters.
    """

    node_to_label = dict(zip(nodes, labels))

    for _ in range(max_iter):
        changed = False

        for cid in sorted(set(node_to_label.values())):
            cluster_nodes = [n for n, lab in node_to_label.items() if lab == cid]

            if len(cluster_nodes) <= 1:
                continue

            comps = sorted(
                nx.connected_components(G.subgraph(cluster_nodes)),
                key=len,
                reverse=True
            )

            if len(comps) <= 1:
                continue

            for comp in comps[1:]:
                neighbor_scores = {}

                for u in comp:
                    for v in G.neighbors(u):
                        if v in comp:
                            continue

                        other_cid = node_to_label[v]

                        if other_cid == cid:
                            continue

                        neighbor_scores[other_cid] = neighbor_scores.get(other_cid, 0.0) + 1.0

                if neighbor_scores:
                    new_cid = max(neighbor_scores, key=neighbor_scores.get)

                    for u in comp:
                        node_to_label[u] = new_cid

                    changed = True

        if not changed:
            break

    fixed_labels = np.array([node_to_label[n] for n in nodes], dtype=int)
    return relabel_compact(fixed_labels)


def run_spectral_clustering(G_topo: nx.Graph,
                            k: int,
                            scenario_name: str,
                            cfg=CONFIG) -> Tuple[pd.DataFrame, pd.DataFrame]:
    A, nodes = build_topological_affinity(G_topo)

    sc = SpectralClustering(
        n_clusters=int(k),
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=cfg["spectral_random_state"],
        n_init=cfg["n_init"]
    )

    raw_labels = sc.fit_predict(A)
    labels = enforce_connected_clusters(G_topo, nodes, raw_labels)

    n_unique = len(np.unique(labels))

    try:
        sil = silhouette_score(A, labels, metric="euclidean") if n_unique > 1 else np.nan
    except Exception:
        sil = np.nan

    size_series = pd.Series(labels).value_counts().sort_index()
    mean_size = size_series.mean()
    std_size = size_series.std(ddof=0) if len(size_series) > 1 else 0.0
    size_cv = std_size / mean_size if mean_size > 0 else np.nan
    fragments = count_cluster_fragments(G_topo, nodes, labels)

    eval_df = pd.DataFrame([{
        "scenario": scenario_name,
        "k_requested": int(k),
        "k_realized": n_unique,
        "silhouette_score": sil,
        "min_cluster_size": int(size_series.min()),
        "max_cluster_size": int(size_series.max()),
        "mean_cluster_size": float(mean_size),
        "std_cluster_size": float(std_size),
        "cluster_size_cv": float(size_cv),
        "n_extra_fragments": int(fragments)
    }])

    membership_df = pd.DataFrame({
        "node_id": nodes,
        "cluster_id": labels,
        "scenario": scenario_name
    })

    return membership_df, eval_df


def evaluate_automatic_k(G_topo: nx.Graph,
                         k_values: List[int],
                         cfg=CONFIG) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    eval_parts = []
    membership_by_k = {}

    for k in k_values:
        if k < 2 or k >= G_topo.number_of_nodes():
            continue

        membership, eval_df = run_spectral_clustering(
            G_topo,
            k=k,
            scenario_name=f"automatic_k_candidate_{k}",
            cfg=cfg
        )

        row = eval_df.iloc[0].to_dict()

        sil = row["silhouette_score"] if pd.notna(row["silhouette_score"]) else -999.0
        size_cv = row["cluster_size_cv"] if pd.notna(row["cluster_size_cv"]) else 999.0
        fragments = row["n_extra_fragments"]

        selection_score = sil - 0.10 * size_cv - 0.25 * fragments

        row["selection_score"] = selection_score
        eval_parts.append(row)
        membership_by_k[k] = membership

    eval_all = pd.DataFrame(eval_parts).sort_values("k_requested").reset_index(drop=True)

    best_row = eval_all.sort_values(
        ["selection_score", "silhouette_score"],
        ascending=[False, False]
    ).iloc[0]

    best_k = int(best_row["k_requested"])
    best_membership = membership_by_k[best_k].copy()
    best_membership["scenario"] = "automatic_k"

    eval_all["selected_as_best"] = eval_all["k_requested"] == best_k

    return best_membership, eval_all, best_k


def infer_missing_dma_labels_by_nearest_labeled_node(
    G_topo: nx.Graph,
    node_df: pd.DataFrame,
    dma_col: str = "dma_label"
) -> Dict[str, str]:
    """
    Assign unlabeled nodes to the nearest labeled DMA node using unweighted graph distance.
    """

    label_map = (
        node_df
        .dropna(subset=[dma_col])
        .set_index("node_id")[dma_col]
        .astype(str)
        .to_dict()
    )

    if len(label_map) == 0:
        return {}

    inferred = dict(label_map)
    labeled_nodes = set(label_map.keys())

    for n in G_topo.nodes:
        if n in inferred:
            continue

        try:
            lengths = nx.single_source_shortest_path_length(G_topo, n)
        except Exception:
            continue

        nearest = []

        for labeled_node in labeled_nodes:
            if labeled_node in lengths:
                nearest.append((lengths[labeled_node], label_map[labeled_node]))

        if not nearest:
            continue

        min_dist = min(d for d, lab in nearest)
        nearest_labels = [lab for d, lab in nearest if d == min_dist]

        chosen_label = sorted(nearest_labels)[0]
        inferred[n] = chosen_label

    return inferred


def build_dma_membership(
    G_topo: nx.Graph,
    node_df: pd.DataFrame,
    scenario_name: str = "predefined_dma"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build membership table for predefined DMAs.
    """

    if "dma_label" not in node_df.columns:
        return pd.DataFrame(), pd.DataFrame()

    if node_df["dma_label"].dropna().empty:
        return pd.DataFrame(), pd.DataFrame()

    inferred_labels = infer_missing_dma_labels_by_nearest_labeled_node(
        G_topo=G_topo,
        node_df=node_df,
        dma_col="dma_label"
    )

    if len(inferred_labels) == 0:
        return pd.DataFrame(), pd.DataFrame()

    unique_dmas = sorted(set(inferred_labels.values()))
    dma_to_cluster = {dma: idx for idx, dma in enumerate(unique_dmas)}

    records = []

    for n in G_topo.nodes:
        dma_label = inferred_labels.get(n, np.nan)

        if pd.isna(dma_label):
            continue

        records.append({
            "node_id": n,
            "dma_label": dma_label,
            "cluster_id": dma_to_cluster[dma_label],
            "scenario": scenario_name
        })

    membership_df = pd.DataFrame(records)

    if membership_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    labels = membership_df.set_index("node_id")["cluster_id"].to_dict()
    nodes = membership_df["node_id"].tolist()
    label_array = np.array([labels[n] for n in nodes])

    size_series = membership_df["cluster_id"].value_counts().sort_index()
    fragments = count_cluster_fragments(G_topo, nodes, label_array)

    eval_df = pd.DataFrame([{
        "scenario": scenario_name,
        "k_requested": len(unique_dmas),
        "k_realized": len(unique_dmas),
        "silhouette_score": np.nan,
        "min_cluster_size": int(size_series.min()),
        "max_cluster_size": int(size_series.max()),
        "mean_cluster_size": float(size_series.mean()),
        "std_cluster_size": float(size_series.std(ddof=0)),
        "cluster_size_cv": float(size_series.std(ddof=0) / size_series.mean()) if size_series.mean() > 0 else np.nan,
        "n_extra_fragments": int(fragments),
        "dma_labels": ";".join(unique_dmas)
    }])

    return membership_df, eval_df


def assign_cluster_attributes(node_df: pd.DataFrame,
                              membership_df: pd.DataFrame) -> pd.DataFrame:
    return node_df.merge(
        membership_df[["node_id", "cluster_id", "scenario"]],
        on="node_id",
        how="left"
    )


def compute_cluster_rankings(G_topo: nx.Graph,
                             Gw: nx.Graph,
                             node_df: pd.DataFrame,
                             membership_df: pd.DataFrame,
                             scenario_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    clustered_node_df = assign_cluster_attributes(node_df, membership_df)

    ranking_parts = []
    entropy_parts = []

    for cluster_id, sub in clustered_node_df.groupby("cluster_id"):
        cluster_nodes = sub["node_id"].tolist()

        if len(cluster_nodes) < 2:
            continue

        metrics = compute_centrality_metrics(G_topo, Gw, node_subset=cluster_nodes)
        merged = sub.drop(columns=["scenario"], errors="ignore").merge(metrics, on="node_id", how="left")

        ranked, entropy_tbl = compute_ranking(
            merged,
            score_col=f"CS_{scenario_name}",
            scenario_name=scenario_name,
            cluster_id=int(cluster_id)
        )

        if not ranked.empty:
            ranked["cluster_rank"] = ranked["rank"]
            ranking_parts.append(ranked)

        if not entropy_tbl.empty:
            entropy_parts.append(entropy_tbl)

    rank_df = pd.concat(ranking_parts, ignore_index=True) if ranking_parts else pd.DataFrame()
    entropy_df = pd.concat(entropy_parts, ignore_index=True) if entropy_parts else pd.DataFrame()

    return rank_df, entropy_df


# ============================================================
# IGT EVALUATION
# ============================================================

def path_resistance(G: nx.Graph,
                    path: List[str],
                    weight_attr: str = "weight") -> float:
    total = 0.0

    for u, v in zip(path[:-1], path[1:]):
        total += G[u][v].get(weight_attr, 1.0)

    return total


def k_shortest_paths_between_nodes(G: nx.Graph,
                                   source: str,
                                   target: str,
                                   k: int,
                                   weight_attr: str = "weight") -> List[List[str]]:
    try:
        gen = nx.shortest_simple_paths(G, source=source, target=target, weight=weight_attr)
        paths = []

        for idx, p in enumerate(gen):
            if idx >= k:
                break
            paths.append(p)

        return paths

    except Exception:
        return []


def get_demand_nodes(node_df: pd.DataFrame,
                     graph_nodes: List[str]) -> List[str]:
    sub = node_df[
        (node_df["node_id"].isin(graph_nodes)) &
        (node_df["node_type"] == "Junction") &
        (node_df["base_demand"] > 0)
    ].copy()

    return sub["node_id"].astype(str).tolist()


def compute_node_igt_to_source_set(G: nx.Graph,
                                   demand_node: str,
                                   source_set: List[str],
                                   k: int,
                                   cfg=CONFIG) -> Tuple[float, int]:
    total_igt = 0.0
    total_paths_found = 0

    for s in source_set:
        if s not in G.nodes:
            continue

        if demand_node == s:
            continue

        paths = k_shortest_paths_between_nodes(
            G,
            source=s,
            target=demand_node,
            k=k,
            weight_attr="weight"
        )

        inv_vals = []

        for p in paths:
            r = path_resistance(G, p, weight_attr="weight")
            inv_vals.append(1.0 / (r + cfg["epsilon"]))

        if inv_vals:
            total_igt += float(np.mean(inv_vals))
            total_paths_found += len(inv_vals)

    return total_igt, total_paths_found


def summarize_igt_values(node_igt_df: pd.DataFrame,
                         cfg=CONFIG) -> Dict[str, float]:
    vals = node_igt_df["igt_node_value"].dropna().astype(float)

    if len(vals) == 0:
        return {
            "trimmed_mean_igt": np.nan,
            "min_igt": np.nan,
            "mean_igt": np.nan,
            "normalized_variance_igt": np.nan,
            "n_demand_nodes_evaluated": 0,
            "n_reachable_demand_nodes": 0,
            "reachability_ratio": np.nan
        }

    mean_igt = float(vals.mean())
    var_igt = float(vals.var(ddof=0))
    min_igt = float(vals.min())
    trimmed_mean_igt = trimmed_mean_series(vals, trim_fraction=cfg["igt_trim_fraction"])
    normalized_variance_igt = var_igt / (mean_igt**2 + cfg["epsilon"])

    n_eval = len(node_igt_df)
    n_reachable = int((node_igt_df["n_paths_found"] > 0).sum())

    return {
        "trimmed_mean_igt": trimmed_mean_igt,
        "min_igt": min_igt,
        "mean_igt": mean_igt,
        "normalized_variance_igt": normalized_variance_igt,
        "n_demand_nodes_evaluated": n_eval,
        "n_reachable_demand_nodes": n_reachable,
        "reachability_ratio": n_reachable / n_eval if n_eval > 0 else np.nan
    }


def evaluate_source_configuration_igt(G: nx.Graph,
                                      node_df: pd.DataFrame,
                                      source_set: List[str],
                                      scenario_name: str,
                                      configuration_name: str,
                                      candidate_node: Optional[str] = None,
                                      cluster_id: Optional[int] = None,
                                      cfg=CONFIG) -> Tuple[pd.DataFrame, pd.DataFrame]:
    graph_nodes = list(G.nodes())
    demand_nodes = get_demand_nodes(node_df, graph_nodes)

    node_records = []

    for dn in demand_nodes:
        igt_val, n_paths = compute_node_igt_to_source_set(
            G,
            demand_node=dn,
            source_set=source_set,
            k=cfg["k_shortest_paths"],
            cfg=cfg
        )

        node_records.append({
            "scenario": scenario_name,
            "configuration": configuration_name,
            "cluster_id": cluster_id,
            "candidate_node": candidate_node,
            "demand_node": dn,
            "igt_node_value": igt_val,
            "n_paths_found": n_paths
        })

    node_igt_df = pd.DataFrame(node_records)

    summary = summarize_igt_values(node_igt_df, cfg=cfg)
    summary.update({
        "scenario": scenario_name,
        "configuration": configuration_name,
        "cluster_id": cluster_id,
        "candidate_node": candidate_node,
        "active_source_set": ";".join(source_set),
        "n_active_sources": len(source_set)
    })

    summary_df = pd.DataFrame([summary])

    return node_igt_df, summary_df


def select_top_full_candidates(full_rank_df: pd.DataFrame,
                               top_n: int) -> pd.DataFrame:
    if full_rank_df.empty:
        return pd.DataFrame()

    return (
        full_rank_df
        .sort_values(["rank"])
        .head(top_n)
        .copy()
        .reset_index(drop=True)
    )


def select_top_cluster_candidates(cluster_rank_df: pd.DataFrame,
                                  top_n_per_cluster: int) -> pd.DataFrame:
    if cluster_rank_df.empty:
        return pd.DataFrame()

    return (
        cluster_rank_df
        .sort_values(["cluster_id", "cluster_rank"])
        .groupby("cluster_id", group_keys=False)
        .head(top_n_per_cluster)
        .copy()
        .reset_index(drop=True)
    )


def run_full_network_igt(Gw: nx.Graph,
                         node_df: pd.DataFrame,
                         sources: Dict[str, List[str]],
                         full_candidates: pd.DataFrame,
                         cfg=CONFIG) -> Tuple[pd.DataFrame, pd.DataFrame]:
    node_parts = []
    summary_parts = []

    existing_sources = sources["reservoirs"] + sources["tanks"]

    node_df_existing, summary_existing = evaluate_source_configuration_igt(
        G=Gw,
        node_df=node_df,
        source_set=existing_sources,
        scenario_name="full_network",
        configuration_name="existing_tanks_plus_reservoirs",
        candidate_node="EXISTING_TANKS",
        cluster_id=np.nan,
        cfg=cfg
    )

    node_parts.append(node_df_existing)
    summary_parts.append(summary_existing)

    reservoirs = sources["reservoirs"]

    for _, row in full_candidates.iterrows():
        candidate = str(row["node_id"])
        proposed_sources = reservoirs + [candidate]

        node_df_c, summary_c = evaluate_source_configuration_igt(
            G=Gw,
            node_df=node_df,
            source_set=proposed_sources,
            scenario_name="full_network",
            configuration_name="proposed_candidate_plus_reservoirs",
            candidate_node=candidate,
            cluster_id=np.nan,
            cfg=cfg
        )

        node_parts.append(node_df_c)
        summary_parts.append(summary_c)

    all_nodes = pd.concat(node_parts, ignore_index=True) if node_parts else pd.DataFrame()
    all_summary = pd.concat(summary_parts, ignore_index=True) if summary_parts else pd.DataFrame()

    return all_nodes, all_summary


def run_cluster_igt(Gw: nx.Graph,
                    node_df: pd.DataFrame,
                    sources: Dict[str, List[str]],
                    membership_df: pd.DataFrame,
                    cluster_candidates: pd.DataFrame,
                    scenario_name: str,
                    cfg=CONFIG) -> Tuple[pd.DataFrame, pd.DataFrame]:
    node_parts = []
    summary_parts = []

    clustered_nodes = membership_df[["node_id", "cluster_id"]].copy()

    source_tanks = sources["tanks"]
    reservoirs = sources["reservoirs"]

    # Existing configuration for each sector
    for cluster_id, sub in clustered_nodes.groupby("cluster_id"):
        cluster_nodes = sub["node_id"].tolist()
        subG = Gw.subgraph(cluster_nodes).copy()

        existing_sources_local = [
            s for s in (reservoirs + source_tanks)
            if s in subG.nodes
        ]

        node_df_e, summary_e = evaluate_source_configuration_igt(
            G=subG,
            node_df=node_df,
            source_set=existing_sources_local,
            scenario_name=scenario_name,
            configuration_name="existing_tanks_plus_reservoirs",
            candidate_node="EXISTING_TANKS",
            cluster_id=int(cluster_id),
            cfg=cfg
        )

        node_parts.append(node_df_e)
        summary_parts.append(summary_e)

    # Proposed candidate configuration for each selected local candidate
    for _, row in cluster_candidates.iterrows():
        candidate = str(row["node_id"])
        cluster_id = int(row["cluster_id"])

        cluster_nodes = clustered_nodes.loc[
            clustered_nodes["cluster_id"] == cluster_id,
            "node_id"
        ].tolist()

        subG = Gw.subgraph(cluster_nodes).copy()

        reservoirs_local = [r for r in reservoirs if r in subG.nodes]
        proposed_sources = reservoirs_local + [candidate]

        node_df_c, summary_c = evaluate_source_configuration_igt(
            G=subG,
            node_df=node_df,
            source_set=proposed_sources,
            scenario_name=scenario_name,
            configuration_name="proposed_candidate_plus_reservoirs",
            candidate_node=candidate,
            cluster_id=cluster_id,
            cfg=cfg
        )

        node_parts.append(node_df_c)
        summary_parts.append(summary_c)

    all_nodes = pd.concat(node_parts, ignore_index=True) if node_parts else pd.DataFrame()
    all_summary = pd.concat(summary_parts, ignore_index=True) if summary_parts else pd.DataFrame()

    return all_nodes, all_summary


# ============================================================
# PLOTTING
# ============================================================

def plot_edge_weight_map(Gw: nx.Graph,
                         figpath: str,
                         title: str = "Hydraulic Edge Weights"):
    pos = get_pos(Gw)

    weights = np.array([Gw[u][v].get("weight", 1.0) for u, v in Gw.edges()], dtype=float)
    log_weights = np.log10(weights + CONFIG["epsilon"])

    fig, ax = plt.subplots(figsize=(10, 8))

    nx.draw_networkx_edges(
        Gw,
        pos,
        edge_color=log_weights,
        edge_cmap=plt.cm.viridis,
        width=1.5,
        alpha=0.85,
        ax=ax
    )

    node_types = nx.get_node_attributes(Gw, "node_type")
    sources = [n for n, t in node_types.items() if t in ["Tank", "Reservoir"]]
    others = [n for n in Gw.nodes if n not in sources]

    nx.draw_networkx_nodes(
        Gw,
        pos,
        nodelist=others,
        node_size=18,
        node_color="lightgray",
        edgecolors="0.4",
        linewidths=0.3,
        ax=ax
    )

    nx.draw_networkx_nodes(
        Gw,
        pos,
        nodelist=sources,
        node_size=70,
        node_color="black",
        node_shape="s",
        ax=ax
    )

    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
    sm.set_array(log_weights)
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("log10(edge weight)")

    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(figpath, dpi=CONFIG["dpi"], bbox_inches="tight")
    plt.close()


def plot_node_score_map(G: nx.Graph,
                        score_df: pd.DataFrame,
                        score_col: str,
                        figpath: str,
                        title: str):
    pos = get_pos(G)
    score_map = score_df.set_index("node_id")[score_col].to_dict()

    scored_nodes = [n for n in G.nodes if n in score_map]
    scored_vals = [score_map[n] for n in scored_nodes]
    unscored_nodes = [n for n in G.nodes if n not in score_map]

    fig, ax = plt.subplots(figsize=(10, 8))

    nx.draw_networkx_edges(
        G,
        pos,
        edge_color="0.75",
        width=0.8,
        alpha=0.8,
        ax=ax
    )

    if unscored_nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=unscored_nodes,
            node_color="lightgray",
            node_size=15,
            ax=ax
        )

    if scored_nodes:
        nodes = nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=scored_nodes,
            node_color=scored_vals,
            cmap="viridis",
            vmin=0,
            vmax=1,
            node_size=28,
            edgecolors="0.25",
            linewidths=0.3,
            ax=ax
        )

        cbar = plt.colorbar(nodes, ax=ax)
        cbar.set_label(score_col)

    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(figpath, dpi=CONFIG["dpi"], bbox_inches="tight")
    plt.close()


def plot_cluster_map(G: nx.Graph,
                     membership_df: pd.DataFrame,
                     figpath: str,
                     title: str):
    pos = get_pos(G)
    cluster_map = membership_df.set_index("node_id")["cluster_id"].to_dict()

    unique_clusters = sorted(membership_df["cluster_id"].dropna().unique().astype(int).tolist())
    n_clusters = len(unique_clusters)

    cmap = plt.cm.get_cmap("tab10", max(n_clusters, 1))
    cluster_index = {cid: i for i, cid in enumerate(unique_clusters)}

    node_colors = []
    source_nodes = []
    normal_nodes = []

    for n, data in G.nodes(data=True):
        if data.get("node_type", "") in ["Tank", "Reservoir"]:
            source_nodes.append(n)
        else:
            normal_nodes.append(n)
            cid = cluster_map.get(n, np.nan)
            node_colors.append(cluster_index.get(int(cid), -1) if pd.notna(cid) else -1)

    fig, ax = plt.subplots(figsize=(10, 8))

    nx.draw_networkx_edges(
        G,
        pos,
        edge_color="0.70",
        width=0.8,
        alpha=0.85,
        ax=ax
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=normal_nodes,
        node_color=node_colors,
        cmap=cmap,
        node_size=28,
        edgecolors="0.25",
        linewidths=0.3,
        ax=ax
    )

    if source_nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=source_nodes,
            node_color="black",
            node_shape="s",
            node_size=70,
            ax=ax
        )

    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array(np.arange(n_clusters))
    cbar = plt.colorbar(sm, ax=ax, ticks=np.arange(n_clusters))
    cbar.ax.set_yticklabels([str(cid) for cid in unique_clusters])
    cbar.set_label("Sector ID")

    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(figpath, dpi=CONFIG["dpi"], bbox_inches="tight")
    plt.close()


def plot_cluster_score_map(G: nx.Graph,
                           membership_df: pd.DataFrame,
                           cluster_rank_df: pd.DataFrame,
                           score_col: str,
                           figpath: str,
                           title: str):
    pos = get_pos(G)

    score_map = cluster_rank_df.set_index("node_id")[score_col].to_dict()
    scored_nodes = [n for n in G.nodes if n in score_map]
    scored_vals = [score_map[n] for n in scored_nodes]
    unscored_nodes = [n for n in G.nodes if n not in score_map]

    fig, ax = plt.subplots(figsize=(10, 8))

    nx.draw_networkx_edges(
        G,
        pos,
        edge_color="0.75",
        width=0.8,
        alpha=0.75,
        ax=ax
    )

    if unscored_nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=unscored_nodes,
            node_color="lightgray",
            node_size=15,
            ax=ax
        )

    if scored_nodes:
        nodes = nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=scored_nodes,
            node_color=scored_vals,
            cmap="viridis",
            vmin=0,
            vmax=1,
            node_size=30,
            edgecolors="0.25",
            linewidths=0.3,
            ax=ax
        )

        cbar = plt.colorbar(nodes, ax=ax)
        cbar.set_label(score_col)

    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(figpath, dpi=CONFIG["dpi"], bbox_inches="tight")
    plt.close()


def plot_dma_score_map(G: nx.Graph,
                       membership_df: pd.DataFrame,
                       dma_rank_df: pd.DataFrame,
                       score_col: str,
                       figpath: str,
                       title: str = "DMA-Level Composite Centrality Score"):
    """
    Plot composite centrality score distribution within predefined DMAs.
    """

    pos = get_pos(G)

    score_map = dma_rank_df.set_index("node_id")[score_col].to_dict()

    scored_nodes = [n for n in G.nodes if n in score_map]
    scored_vals = [score_map[n] for n in scored_nodes]
    unscored_nodes = [n for n in G.nodes if n not in score_map]

    fig, ax = plt.subplots(figsize=(10, 8))

    nx.draw_networkx_edges(
        G,
        pos,
        edge_color="0.78",
        width=0.8,
        alpha=0.75,
        ax=ax
    )

    if unscored_nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=unscored_nodes,
            node_color="lightgray",
            node_size=15,
            edgecolors="0.4",
            linewidths=0.2,
            ax=ax
        )

    if scored_nodes:
        nodes = nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=scored_nodes,
            node_color=scored_vals,
            cmap="viridis",
            vmin=0,
            vmax=1,
            node_size=30,
            edgecolors="0.25",
            linewidths=0.3,
            ax=ax
        )

        cbar = plt.colorbar(nodes, ax=ax)
        cbar.set_label(r"DMA-level composite score, $CS_i$")

    node_types = nx.get_node_attributes(G, "node_type")
    source_nodes = [
        n for n, t in node_types.items()
        if t in ["Tank", "Reservoir"]
    ]

    if source_nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=source_nodes,
            node_color="black",
            node_shape="s",
            node_size=80,
            ax=ax
        )

    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(figpath, dpi=CONFIG["dpi"], bbox_inches="tight")
    plt.close()


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    inp_file = upload_inp_file_colab()
    outdirs = ensure_output_dirs(CONFIG["output_dir"])

    print("\nLoading network...")
    wn = load_network(inp_file)

    node_df = extract_node_table(wn)
    link_df = extract_link_table(wn)
    sources = get_source_nodes(wn)

    # ------------------------------------------------------------
    # Optional predefined DMA labels from INP file
    # ------------------------------------------------------------
    dma_labels = parse_dma_labels_from_inp(inp_file)
    node_df = add_dma_labels_to_node_df(node_df, dma_labels)

    if len(dma_labels) > 0:
        print(f"\nPredefined DMA labels detected: {sorted(set(dma_labels.values()))}")
    else:
        print("\nNo predefined DMA labels detected in the INP file.")

    print("\nExisting reservoirs:", sources["reservoirs"])
    print("Existing tanks:", sources["tanks"])

    network_summary = build_network_summary(wn, link_df, sources)
    save_table(network_summary, os.path.join(outdirs["tables"], "network_component_summary.csv"))
    save_table(node_df, os.path.join(outdirs["tables"], "node_attributes_with_dma_labels.csv"))

    check_units_for_hw(link_df)

    print("\nRunning steady-state simulation...")
    results = run_steady_state_simulation(wn, CONFIG["steady_state_duration_sec"])
    hydraulic_link_df, node_hyd_df = extract_hydraulic_results(wn, results)

    print("\nBuilding hydraulic edge weights...")
    link_hyd_df = build_link_hydraulic_table(link_df, hydraulic_link_df, CONFIG)
    save_table(link_hyd_df, os.path.join(outdirs["tables"], "hydraulic_edge_weights.csv"))

    print("\nBuilding graphs...")
    G_topo, Gw = build_graphs(node_df, link_hyd_df, CONFIG)

    print("Graph nodes:", Gw.number_of_nodes())
    print("Graph edges:", Gw.number_of_edges())

    # ------------------------------------------------------------
    # Plot: edge weights
    # ------------------------------------------------------------
    plot_edge_weight_map(
        Gw,
        figpath=os.path.join(outdirs["figures"], "edge_weight_map.png"),
        title="Hydraulic Edge Weights of the WDN"
    )

    # ------------------------------------------------------------
    # Full-network ranking
    # ------------------------------------------------------------
    print("\nComputing full-network ranking...")
    full_rank_df, full_entropy_df = compute_full_network_ranking(
        node_df=node_df,
        node_hyd_df=node_hyd_df,
        G_topo=G_topo,
        Gw=Gw,
        score_col="CS_full"
    )

    save_table(full_rank_df, os.path.join(outdirs["tables"], "full_network_ranking.csv"))
    save_table(full_entropy_df, os.path.join(outdirs["tables"], "full_network_entropy_weights.csv"))

    plot_node_score_map(
        Gw,
        score_df=full_rank_df,
        score_col="CS_full",
        figpath=os.path.join(outdirs["scores"], "full_network_centrality_score_map.png"),
        title="Full-Network Composite Centrality Score"
    )

    full_candidates = select_top_full_candidates(full_rank_df, CONFIG["top_n_full"])
    save_table(full_candidates, os.path.join(outdirs["tables"], "full_network_selected_candidates.csv"))

    # ------------------------------------------------------------
    # Full-network IGT
    # ------------------------------------------------------------
    print("\nRunning full-network IGT evaluation...")
    full_igt_nodes, full_igt_summary = run_full_network_igt(
        Gw=Gw,
        node_df=node_df,
        sources=sources,
        full_candidates=full_candidates,
        cfg=CONFIG
    )

    save_table(full_igt_nodes, os.path.join(outdirs["tables"], "full_network_igt_node_values.csv"))
    save_table(full_igt_summary, os.path.join(outdirs["tables"], "full_network_igt_summary.csv"))

    # ------------------------------------------------------------
    # Sector scenarios
    # ------------------------------------------------------------
    print("\nRunning sector scenarios...")

    cluster_scenarios = {}

    # 1. Tank-based k
    tank_based_k = max(2, len(sources["tanks"]))
    membership_tank, eval_tank = run_spectral_clustering(
        G_topo,
        k=tank_based_k,
        scenario_name="tank_based_k",
        cfg=CONFIG
    )
    cluster_scenarios["tank_based_k"] = (membership_tank, eval_tank)

    # 2. Automatic k
    membership_auto, eval_auto_all, best_k_auto = evaluate_automatic_k(
        G_topo,
        k_values=CONFIG["cluster_k_range"],
        cfg=CONFIG
    )
    cluster_scenarios["automatic_k"] = (
        membership_auto,
        eval_auto_all[eval_auto_all["selected_as_best"] == True].copy()
    )

    save_table(eval_auto_all, os.path.join(outdirs["tables"], "automatic_k_cluster_diagnostics_all.csv"))

    print(f"Automatic selected k: {best_k_auto}")

    # 3. User-defined k
    user_k = int(CONFIG["user_defined_k"])
    membership_user, eval_user = run_spectral_clustering(
        G_topo,
        k=user_k,
        scenario_name="user_defined_k",
        cfg=CONFIG
    )
    cluster_scenarios["user_defined_k"] = (membership_user, eval_user)

    # 4. Optional predefined DMA scenario
    membership_dma, eval_dma = build_dma_membership(
        G_topo=G_topo,
        node_df=node_df,
        scenario_name="predefined_dma"
    )

    if not membership_dma.empty:
        print("Predefined DMA scenario added.")
        cluster_scenarios["predefined_dma"] = (membership_dma, eval_dma)

        save_table(
            membership_dma,
            os.path.join(outdirs["tables"], "predefined_dma_membership.csv")
        )

        save_table(
            eval_dma,
            os.path.join(outdirs["tables"], "predefined_dma_diagnostics.csv")
        )
    else:
        print("Predefined DMA scenario skipped because no DMA labels were found.")

    # ------------------------------------------------------------
    # Process each sector scenario
    # ------------------------------------------------------------
    all_cluster_eval = []
    all_cluster_rank = []
    all_cluster_entropy = []
    all_cluster_igt_nodes = []
    all_cluster_igt_summary = []
    all_cluster_candidates = []

    for scenario_name, (membership_df, eval_df) in cluster_scenarios.items():
        print(f"\nProcessing sector scenario: {scenario_name}")

        save_table(
            membership_df,
            os.path.join(outdirs["tables"], f"{scenario_name}_cluster_membership.csv")
        )

        eval_df = eval_df.copy()
        eval_df["scenario"] = scenario_name
        all_cluster_eval.append(eval_df)

        plot_cluster_map(
            Gw,
            membership_df=membership_df,
            figpath=os.path.join(outdirs["clusters"], f"{scenario_name}_cluster_map.png"),
            title=f"Sector-Based Decentralized Configuration: {scenario_name}"
        )

        cluster_rank_df, cluster_entropy_df = compute_cluster_rankings(
            G_topo=G_topo,
            Gw=Gw,
            node_df=node_df,
            membership_df=membership_df,
            scenario_name=scenario_name
        )

        save_table(
            cluster_rank_df,
            os.path.join(outdirs["tables"], f"{scenario_name}_cluster_ranking.csv")
        )

        save_table(
            cluster_entropy_df,
            os.path.join(outdirs["tables"], f"{scenario_name}_cluster_entropy_weights.csv")
        )

        all_cluster_rank.append(cluster_rank_df)
        all_cluster_entropy.append(cluster_entropy_df)

        score_col = f"CS_{scenario_name}"

        if not cluster_rank_df.empty:
            if scenario_name == "predefined_dma":
                plot_dma_score_map(
                    G=Gw,
                    membership_df=membership_df,
                    dma_rank_df=cluster_rank_df,
                    score_col=score_col,
                    figpath=os.path.join(outdirs["scores"], f"{scenario_name}_dma_score_map.png"),
                    title="DMA-Level Composite Centrality Score: Predefined DMAs"
                )
            else:
                plot_cluster_score_map(
                    G=Gw,
                    membership_df=membership_df,
                    cluster_rank_df=cluster_rank_df,
                    score_col=score_col,
                    figpath=os.path.join(outdirs["scores"], f"{scenario_name}_cluster_score_map.png"),
                    title=f"Cluster-Level Composite Centrality Score: {scenario_name}"
                )

        cluster_candidates = select_top_cluster_candidates(
            cluster_rank_df,
            top_n_per_cluster=CONFIG["top_n_per_cluster"]
        )

        save_table(
            cluster_candidates,
            os.path.join(outdirs["tables"], f"{scenario_name}_selected_cluster_candidates.csv")
        )

        all_cluster_candidates.append(cluster_candidates)

        print(f"Running IGT for sector scenario: {scenario_name}")

        cluster_igt_nodes, cluster_igt_summary = run_cluster_igt(
            Gw=Gw,
            node_df=node_df,
            sources=sources,
            membership_df=membership_df,
            cluster_candidates=cluster_candidates,
            scenario_name=scenario_name,
            cfg=CONFIG
        )

        save_table(
            cluster_igt_nodes,
            os.path.join(outdirs["tables"], f"{scenario_name}_igt_node_values.csv")
        )

        save_table(
            cluster_igt_summary,
            os.path.join(outdirs["tables"], f"{scenario_name}_igt_summary.csv")
        )

        all_cluster_igt_nodes.append(cluster_igt_nodes)
        all_cluster_igt_summary.append(cluster_igt_summary)

    # ------------------------------------------------------------
    # Combined tables
    # ------------------------------------------------------------
    combined_cluster_eval = pd.concat(all_cluster_eval, ignore_index=True) if all_cluster_eval else pd.DataFrame()
    combined_cluster_rank = pd.concat(all_cluster_rank, ignore_index=True) if all_cluster_rank else pd.DataFrame()
    combined_cluster_entropy = pd.concat(all_cluster_entropy, ignore_index=True) if all_cluster_entropy else pd.DataFrame()
    combined_cluster_candidates = pd.concat(all_cluster_candidates, ignore_index=True) if all_cluster_candidates else pd.DataFrame()
    combined_cluster_igt_summary = pd.concat(all_cluster_igt_summary, ignore_index=True) if all_cluster_igt_summary else pd.DataFrame()

    save_table(combined_cluster_eval, os.path.join(outdirs["tables"], "combined_sector_diagnostics.csv"))
    save_table(combined_cluster_rank, os.path.join(outdirs["tables"], "combined_sector_rankings.csv"))
    save_table(combined_cluster_entropy, os.path.join(outdirs["tables"], "combined_sector_entropy_weights.csv"))
    save_table(combined_cluster_candidates, os.path.join(outdirs["tables"], "combined_sector_selected_candidates.csv"))
    save_table(combined_cluster_igt_summary, os.path.join(outdirs["tables"], "combined_sector_igt_summary.csv"))

    # ------------------------------------------------------------
    # Sort IGT summaries for decision support
    # ------------------------------------------------------------
    if not full_igt_summary.empty:
        full_sorted = full_igt_summary.sort_values(
            ["trimmed_mean_igt", "min_igt", "normalized_variance_igt"],
            ascending=[False, False, True]
        )
        save_table(full_sorted, os.path.join(outdirs["tables"], "full_network_igt_summary_sorted.csv"))

    if not combined_cluster_igt_summary.empty:
        cluster_sorted = combined_cluster_igt_summary.sort_values(
            ["scenario", "cluster_id", "trimmed_mean_igt", "min_igt", "normalized_variance_igt"],
            ascending=[True, True, False, False, True]
        )
        save_table(cluster_sorted, os.path.join(outdirs["tables"], "combined_sector_igt_summary_sorted.csv"))

    # ------------------------------------------------------------
    # Zip outputs
    # ------------------------------------------------------------
    zip_path = "/content/higt_outputs.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, file_names in os.walk(CONFIG["output_dir"]):
            for file_name in file_names:
                full_path = os.path.join(root, file_name)
                rel_path = os.path.relpath(full_path, CONFIG["output_dir"])
                zipf.write(full_path, rel_path)

    print("\n======================================================")
    print("HI-GT analysis completed.")
    print(f"Output folder: {CONFIG['output_dir']}")
    print(f"ZIP file: {zip_path}")
    print("======================================================")

    files.download(zip_path)

    return {
        "node_df": node_df,
        "link_hyd_df": link_hyd_df,
        "full_rank_df": full_rank_df,
        "full_entropy_df": full_entropy_df,
        "full_igt_summary": full_igt_summary,
        "combined_sector_eval": combined_cluster_eval,
        "combined_sector_rank": combined_cluster_rank,
        "combined_sector_entropy": combined_cluster_entropy,
        "combined_sector_igt_summary": combined_cluster_igt_summary,
    }


# ============================================================
# RUN
# ============================================================

outputs = main()

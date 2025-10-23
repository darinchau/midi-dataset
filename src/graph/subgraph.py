from ..extract import MusicXMLNote
from .make_graph import graph_to_pyg_data
from .make_graph import NoteGraph
from .filter import is_good_midi
from .make_graph import construct_music_graph
from ..extract import musicxml_to_notes
from ..utils import get_path
from ..constants import XML_ROOT
from ..test import BACH_CHORALE, LE_MOLDEAU, BACH_C_MAJOR_PRELUDE
import numpy as np
import random
from collections import deque, defaultdict
from math import exp, log


def get_edge_weight(graph: NoteGraph, src: int, dst: int, time_weight_beta: float, pitch_weight_beta: float) -> float:
    """Gets the weight associated to each graph edge."""
    # Inverse of 1 + onset time difference + pitch difference
    # Make sure src and dst are connected properly
    start_idx = graph.feature_info.feature_names.index('start')
    pitch_idx = graph.feature_info.feature_names.index('pitch')

    if graph.node_features[dst, start_idx] < graph.node_features[src, start_idx]:
        raise ValueError("Destination note occurs before source note")

    time_diff = abs(graph.node_features[dst, start_idx] - graph.node_features[src, start_idx])
    pitch_diff = abs(graph.node_features[dst, pitch_idx] - graph.node_features[src, pitch_idx])
    # Think of this as log probability weights
    # = log(exp(-time_diff/time_weight_beta) * exp(-pitch_diff/pitch_weight_beta))
    return -time_weight_beta * time_diff - pitch_weight_beta * pitch_diff


def sample_from_logprobs(log_probs_dict: dict[int, float]) -> int:
    # Use log-sum-exp trick to sample from log weights
    if not log_probs_dict:
        raise ValueError("log_probs_dict is empty")
    log_p_max = max(log_probs_dict.values())
    weights = {k: exp(v - log_p_max) for k, v in log_probs_dict.items()}
    total_weight = sum(weights.values())
    threshold = random.random() * total_weight
    cumulative = 0.0
    for k, w in weights.items():
        cumulative += w
        if cumulative >= threshold:
            return k
    return list(log_probs_dict.keys())[-1]


def find_path_inner(
    graph: NoteGraph,
    adj_list: dict[int, list[int]],
    current_path: list[int],
    n: int,
    time_weight_beta: float,
    pitch_weight_beta: float,
) -> list[int] | None:
    """Finds a path of length n starting from start node using DFS."""
    if len(current_path) == n:
        return current_path
    node = current_path[-1]
    neighbor_pick_weight = {i: get_edge_weight(graph, node, i, time_weight_beta, pitch_weight_beta) for i in adj_list[node] if i not in current_path}
    tried = {}
    while neighbor_pick_weight:
        chosen_neighbor = sample_from_logprobs(neighbor_pick_weight)
        current_path.append(chosen_neighbor)
        result = find_path_inner(graph, adj_list, current_path, n, time_weight_beta, pitch_weight_beta)
        if result is not None:
            return result
        current_path.pop()
        tried[chosen_neighbor] = neighbor_pick_weight[chosen_neighbor]
        del neighbor_pick_weight[chosen_neighbor]
    return None


def extract_subgraph(
    notes: list[MusicXMLNote],
    n: int,
    max_seconds_apart: float = 1,
    time_weight_beta: float = 1,
    pitch_weight_beta: float = 1,
) -> list[int]:
    """
    Extracts a connected subgraph of size n from the given graph.
    Returns the list of selected node indices and the corresponding subgraph.

    Args:
        notes (list[MusicXMLNote]): The original music notes.
        n (int): The number of nodes in the desired subgraph.
        time_weight_beta (float): Weighting factor for time difference in edge weights. The higher the beta, the more likely you get note groups with small time differences.
        pitch_weight_beta (float): Weighting factor for pitch difference in edge weights. The higher the beta, the more likely you get note groups with small pitch differences.

    Returns:
        list[int]: The list of selected node indices forming the subgraph.
    """
    graph = construct_music_graph(notes, max_seconds_apart=max_seconds_apart)

    if n > graph.num_nodes:
        raise ValueError(f"Requested subgraph size {n} is larger than total nodes {graph.num_nodes}")

    adj_list: dict[int, list[int]] = {i: [] for i in range(graph.num_nodes)}
    edge_attr_map: dict[tuple[int, int], int] = {}  # Map (src, dst) to edge attribute index
    for i in range(graph.edge_index.shape[1]):
        src, dst = graph.edge_index[0, i].item(), graph.edge_index[1, i].item()
        adj_list[src].append(dst)
        edge_attr_map[(src, dst)] = i

    # Attempt to find a connected subgraph of size N
    attempted_starts = set()
    for _ in range(graph.num_nodes):
        available_starts = list(set(range(graph.num_nodes)) - attempted_starts)
        if not available_starts:
            break
        start_node = random.choice(available_starts)
        attempted_starts.add(start_node)
        current_path = [start_node]
        result = find_path_inner(graph, adj_list, current_path, n, time_weight_beta, pitch_weight_beta)
        if result is not None:
            return result
    raise ValueError(f"Could not find a subgraph of size {n} after trying all possible starting nodes.")


def test():
    from scripts.graph.validate import visualize_music_graph
    path = get_path(XML_ROOT, BACH_C_MAJOR_PRELUDE)
    notes = musicxml_to_notes(path)
    is_good_midi(path)

    sampled_notes = extract_subgraph(notes, 10, max_seconds_apart=1)

    subgraph = construct_music_graph([n for i, n in enumerate(notes) if i in sampled_notes])
    visualize_music_graph(subgraph)


if __name__ == "__main__":
    test()

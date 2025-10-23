import numpy as np
from typing import List, Dict, Any, Optional, Set

from ..extract.utils import get_inv_time_signature_map
from ..extract import MusicXMLNote

import torch
from torch_geometric.data import Data

from dataclasses import dataclass


@dataclass
class FeatureInfo:
    """Metadata about graph features."""
    feature_names: List[str]
    categorical_features: List[str]
    continuous_features: List[str]
    binary_features: List[str]

    @classmethod
    def empty(cls) -> "FeatureInfo":
        return cls(
            feature_names=[],
            categorical_features=[],
            continuous_features=[],
            binary_features=[]
        )

    def __post_init__(self):
        # Ensure all feature names are accounted for
        all_features = set(self.feature_names)
        accounted_features = set(self.categorical_features) | set(self.continuous_features) | set(self.binary_features)
        assert all_features == accounted_features, f"Feature names do not match accounted features: {all_features} vs {accounted_features}"
        assert len(self.feature_names) == len(self.categorical_features) + len(self.continuous_features) + len(self.binary_features), "Feature names mismatch"


@dataclass
class NoteGraph:
    """Data structure for a music note graph."""
    node_features: np.ndarray  # Shape: [num_nodes, num_features]
    edge_index: np.ndarray     # Shape: [2, num_edges]
    edge_attr: np.ndarray      # Shape: [num_edges, num_edge_features]
    feature_info: FeatureInfo  # Metadata about features

    def __post_init__(self):
        assert self.num_edges == self.edge_attr.shape[0], f"Edge index and edge attr size mismatch: {self.edge_index.shape[1]} != {self.edge_attr.shape[0]}"
        assert self.edge_index.shape[0] == 2, f"Edge index first dimension must be 2, got {self.edge_index.shape[0]}"
        assert self.node_features.shape[1] == len(self.feature_info.feature_names), f"Node features second dimension does not match feature info: {self.node_features.shape[1]} != {len(self.feature_info.feature_names)}"
        assert self.node_features.ndim == 2, f"Node features must be 2D, got {self.node_features.shape}"
        assert self.edge_index.ndim == 2, f"Edge index must be 2D, got {self.edge_index.shape}"
        assert self.edge_attr.ndim == 2, f"Edge attr must be 2D, got {self.edge_attr.shape}"

    @property
    def num_nodes(self) -> int:
        return self.node_features.shape[0]

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]

    def show_stats(self):
        print("Node features shape:", self.node_features.shape)
        print("Edge index shape:", self.edge_index.shape)
        print("Edge attributes shape:", self.edge_attr.shape)


def construct_music_graph(
    notes: List[MusicXMLNote],
    remove_barlines: bool = True,
    max_seconds_apart: float = 10.0,
) -> NoteGraph:
    """
    Constructs a graph with RAW features. One-hot encoding should be done later.
    Args:
        notes: List of MusicXMLNote objects.
        remove_barlines: Whether to remove barline notes from the graph.
        max_seconds_apart: Maximum time difference to connect notes with edges.
    """
    if not notes:
        return NoteGraph(
            node_features=np.array([]),
            edge_index=np.array([[], []]),
            edge_attr=np.array([]),
            feature_info=FeatureInfo.empty()
        )

    # Remove all barlines since we don't care about those for now?
    if remove_barlines:
        notes = [note for note in notes if not note.barline]

    # Extract RAW node features
    node_features = []
    for note in notes:
        features: list[int | float] = [
            note.instrument,        # Categorical
            note.pitch,             # Categorical
            note.start,             # Continuous
            note.duration,          # Continuous
            note.start_ql,          # Continuous
            note.duration_ql,       # Continuous
            note.index,             # Categorical
            note.octave,            # Categorical
            get_inv_time_signature_map()[note.timesig]  # Categorical
        ]
        if not remove_barlines:
            features.append(1.0 if note.barline else 0.0)  # Binary, only add if barlines are kept
        node_features.append(features)

    node_features = np.array(node_features, dtype=np.float32)

    onsets = np.array([note.start for note in notes])
    num_nodes = len(notes)

    # Create edges
    i_indices, j_indices = np.meshgrid(np.arange(num_nodes), np.arange(num_nodes), indexing='ij')
    onset_diffs = onsets[j_indices] - onsets[i_indices]

    # Create mask for valid edges:
    # 1. onset_a <= onset_b (onset_diff >= 0)
    # 2. No self-loops (i != j)
    # 3. Within window (onset_diff <= max_seconds_apart)
    valid_mask = (onset_diffs >= 0) & (i_indices != j_indices) & (onset_diffs <= max_seconds_apart)

    edge_sources = i_indices[valid_mask]
    edge_targets = j_indices[valid_mask]
    valid_diffs = onset_diffs[valid_mask]

    # Compute weights (unused for now: all ones)
    edge_weights = np.ones_like(valid_diffs, dtype=np.float32)  # Shape: [num_edges]

    # Final filter for positive weights (should be redundant but just in case)
    positive_mask = edge_weights > 0
    edge_sources = edge_sources[positive_mask]
    edge_targets = edge_targets[positive_mask]
    edge_weights = edge_weights[positive_mask]

    # Convert to proper format
    edge_index = np.stack([edge_sources, edge_targets], axis=0).astype(np.int64)
    edge_attr = edge_weights.astype(np.float32)[..., None]

    # Store metadata about features
    feature_info = FeatureInfo(
        feature_names=[
            'instrument', 'pitch', 'start', 'duration',
            'start_ql', 'duration_ql', 'index', 'octave',
            'timesig'
        ] + (['barline'] if not remove_barlines else []),
        categorical_features=['instrument', 'octave', 'pitch', 'index', 'timesig'],
        continuous_features=['start', 'duration', 'start_ql', 'duration_ql'],
        binary_features=['barline'] if not remove_barlines else []
    )

    return NoteGraph(
        node_features=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        feature_info=feature_info
    )


def graph_to_pyg_data(graph: NoteGraph):
    """Convert dictionary format to PyTorch Geometric Data object"""
    node_features = torch.tensor(graph.node_features, dtype=torch.float32)
    edge_index = torch.tensor(graph.edge_index, dtype=torch.long)
    edge_attr = torch.tensor(graph.edge_attr, dtype=torch.float32)

    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        feature_info=graph.feature_info
    )

    return data

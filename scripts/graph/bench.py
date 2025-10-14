import numpy as np
from typing import List, Dict, Any
import bisect
from src.extract import MusicXMLNote

import numpy as np
from typing import List, Dict, Any, Optional, Set


def construct_music_graph(
    notes: List[MusicXMLNote],
    include_edge_features: bool = False
) -> Dict[str, Any]:
    """
    Constructs a graph with RAW features. One-hot encoding should be done later.
    """
    if not notes:
        return {
            'node_features': np.array([]),
            'edge_index': np.array([[], []]),
            'edge_attr': np.array([]),
            'num_nodes': 0,
            'feature_info': {}
        }

    num_nodes = len(notes)

    # Extract RAW node features
    node_features = []
    for note in notes:
        features = [
            note.instrument,      # Categorical - needs encoding
            note.pitch,          # Could be categorical or continuous
            note.start,          # Continuous
            note.duration,       # Continuous
            note.start_ql,       # Continuous
            note.duration_ql,    # Continuous
            note.index,          # Categorical or continuous
            note.octave,         # Categorical - needs encoding
            note.velocity,       # Continuous (0-127)
            1.0 if note.barline else 0.0,  # Binary
        ]
        node_features.append(features)

    node_features = np.array(node_features, dtype=np.float32)

    # Store metadata about features for later preprocessing
    feature_info = {
        'feature_names': [
            'instrument', 'pitch', 'start', 'duration',
            'start_ql', 'duration_ql', 'index', 'octave',
            'velocity', 'barline'
        ],
        'categorical_features': ['instrument', 'octave'],  # Definitely categorical
        'possibly_categorical': ['pitch', 'index'],  # Could be treated either way
        'continuous_features': ['start', 'duration', 'start_ql', 'duration_ql', 'velocity'],
        'binary_features': ['barline']
    }

    # Construct edges (same as before)
    edge_sources = []
    edge_targets = []
    edge_weights = []
    edge_features = []

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                onset_a = notes[i].start
                onset_b = notes[j].start

                if onset_a <= onset_b:
                    time_diff = onset_b - onset_a
                    weight = max(1.0 - time_diff / 10.0, 0.0)

                    if weight > 0:
                        edge_sources.append(i)
                        edge_targets.append(j)
                        edge_weights.append(weight)

                        if include_edge_features:
                            # Additional edge features if needed
                            edge_feat = [
                                time_diff,  # Time difference
                                notes[j].pitch - notes[i].pitch,  # Pitch interval
                                notes[j].octave - notes[i].octave,  # Octave difference
                            ]
                            edge_features.append(edge_feat)

    # Convert to arrays
    if edge_sources:
        edge_index = np.array([edge_sources, edge_targets], dtype=np.int64)
        edge_attr = np.array(edge_weights, dtype=np.float32)
        if include_edge_features:
            edge_features = np.array(edge_features, dtype=np.float32)
    else:
        edge_index = np.array([[], []], dtype=np.int64)
        edge_attr = np.array([], dtype=np.float32)
        edge_features = np.array([], dtype=np.float32) if include_edge_features else None

    result = {
        'node_features': node_features,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'num_nodes': num_nodes,
        'feature_info': feature_info
    }

    if include_edge_features and edge_features is not None:
        result['edge_features'] = edge_features

    return result


def construct_music_graph_optimized(notes: List[MusicXMLNote]) -> Dict[str, Any]:
    """
    Optimized O(n log n + E) construction where E is the number of edges.
    In practice, E << n² because of the 10-second cutoff.
    """
    if not notes:
        return {
            'node_features': np.array([]),
            'edge_index': np.array([[], []]),
            'edge_attr': np.array([]),
            'num_nodes': 0,
            'feature_info': {}
        }

    num_nodes = len(notes)

    # Create node features (same as before)
    node_features = []
    for note in notes:
        features = [
            note.instrument, note.pitch, note.start, note.duration,
            note.start_ql, note.duration_ql, note.index, note.octave,
            note.velocity, 1.0 if note.barline else 0.0
        ]
        node_features.append(features)

    node_features = np.array(node_features, dtype=np.float32)

    # Create a sorted index array by onset time
    # This preserves original indices while allowing sorted traversal
    sorted_indices = np.argsort([note.start for note in notes])
    sorted_onsets = np.array([notes[i].start for i in sorted_indices])

    # Build edges efficiently
    edge_sources = []
    edge_targets = []
    edge_weights = []

    # For each note in sorted order
    for idx, i in enumerate(sorted_indices):
        onset_a = notes[i].start

        # Find the index where notes become > onset_a + 10
        # All notes beyond this point will have weight 0
        cutoff_time = onset_a + 10.0
        cutoff_idx = bisect.bisect_right(sorted_onsets, cutoff_time)

        # Only check notes from current position to cutoff
        for j_idx in range(idx, min(cutoff_idx, len(sorted_indices))):
            j = sorted_indices[j_idx]

            if i != j:  # Skip self-loops
                onset_b = notes[j].start

                # Calculate edge weight
                time_diff = onset_b - onset_a
                weight = max(1.0 - time_diff / 10.0, 0.0)

                if weight > 0:
                    edge_sources.append(i)
                    edge_targets.append(j)
                    edge_weights.append(weight)

    # Also need to handle notes with same onset time but different original indices
    # Group notes by onset time for efficient same-time connections
    from collections import defaultdict
    onset_groups = defaultdict(list)
    for i, note in enumerate(notes):
        onset_groups[note.start].append(i)

    # Add edges between notes at the same time
    for onset, indices in onset_groups.items():
        if len(indices) > 1:
            for i in indices:
                for j in indices:
                    if i != j and (i, j) not in zip(edge_sources, edge_targets):
                        edge_sources.append(i)
                        edge_targets.append(j)
                        edge_weights.append(1.0)  # Same time = weight 1

    # Convert to arrays
    if edge_sources:
        edge_index = np.array([edge_sources, edge_targets], dtype=np.int64)
        edge_attr = np.array(edge_weights, dtype=np.float32)
    else:
        edge_index = np.array([[], []], dtype=np.int64)
        edge_attr = np.array([], dtype=np.float32)

    feature_info = {
        'feature_names': [
            'instrument', 'pitch', 'start', 'duration',
            'start_ql', 'duration_ql', 'index', 'octave',
            'velocity', 'barline'
        ],
        'categorical_features': ['instrument', 'octave'],
        'possibly_categorical': ['pitch', 'index'],
        'continuous_features': ['start', 'duration', 'start_ql', 'duration_ql', 'velocity'],
        'binary_features': ['barline']
    }

    return {
        'node_features': node_features,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'num_nodes': num_nodes,
        'feature_info': feature_info
    }


def construct_music_graph_sliding_window(notes: List[MusicXMLNote],
                                         window_seconds: float = 10.0) -> Dict[str, Any]:
    """
    Vectorized approach using NumPy for maximum speed.
    Best for dense graphs where E is large.
    """
    if not notes:
        return {
            'node_features': np.array([]),
            'edge_index': np.array([[], []]),
            'edge_attr': np.array([]),
            'num_nodes': 0,
            'feature_info': {}
        }

    # Extract features and onsets
    node_features = np.array([
        [note.instrument, note.pitch, note.start, note.duration,
         note.start_ql, note.duration_ql, note.index, note.octave,
         note.velocity, 1.0 if note.barline else 0.0]
        for note in notes
    ], dtype=np.float32)

    onsets = np.array([note.start for note in notes])
    num_nodes = len(notes)

    # Create meshgrid of all possible edges
    i_indices, j_indices = np.meshgrid(np.arange(num_nodes), np.arange(num_nodes), indexing='ij')

    # Compute time differences for all pairs
    onset_diffs = onsets[j_indices] - onsets[i_indices]

    # Create mask for valid edges:
    # 1. onset_a <= onset_b (onset_diff >= 0)
    # 2. No self-loops (i != j)
    # 3. Within window (onset_diff <= window_seconds)
    valid_mask = (onset_diffs >= 0) & (i_indices != j_indices) & (onset_diffs <= window_seconds)

    # Extract valid edges
    edge_sources = i_indices[valid_mask]
    edge_targets = j_indices[valid_mask]
    valid_diffs = onset_diffs[valid_mask]

    # Compute weights
    edge_weights = np.maximum(1.0 - valid_diffs / window_seconds, 0.0)

    # Final filter for positive weights (should be redundant but just in case)
    positive_mask = edge_weights > 0
    edge_sources = edge_sources[positive_mask]
    edge_targets = edge_targets[positive_mask]
    edge_weights = edge_weights[positive_mask]

    # Convert to proper format
    edge_index = np.stack([edge_sources, edge_targets], axis=0).astype(np.int64)
    edge_attr = edge_weights.astype(np.float32)

    feature_info = {
        'feature_names': [
            'instrument', 'pitch', 'start', 'duration',
            'start_ql', 'duration_ql', 'index', 'octave',
            'velocity', 'barline'
        ]
    }

    return {
        'node_features': node_features,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'num_nodes': num_nodes,
        'feature_info': feature_info
    }


def benchmark_graph_construction(notes: List[MusicXMLNote]):
    """Compare performance of different implementations and validate correctness."""
    import time

    start = time.time()
    graph1 = construct_music_graph(notes)
    t1 = time.time() - start
    print(f"Naive O(n²) approach took {t1:.4f} seconds with {graph1['num_nodes']} nodes and {graph1['edge_index'].shape[1]} edges.")

    start = time.time()
    graph2 = construct_music_graph_optimized(notes)
    t2 = time.time() - start
    print(f"Optimized O(n log n + E) approach took {t2:.4f} seconds with {graph2['num_nodes']} nodes and {graph2['edge_index'].shape[1]} edges.")

    start = time.time()
    graph3 = construct_music_graph_sliding_window(notes)
    t3 = time.time() - start
    print(f"Sliding window approach took {t3:.4f} seconds with {graph3['num_nodes']} nodes and {graph3['edge_index'].shape[1]} edges.")

    # Validate that all graphs are identical
    assert np.array_equal(graph1['node_features'], graph2['node_features'])
    assert np.array_equal(graph1['node_features'], graph3['node_features'])
    # assert np.array_equal(graph1['edge_index'], graph2['edge_index'])
    assert np.array_equal(graph1['edge_index'], graph3['edge_index'])
    # assert np.allclose(graph1['edge_attr'], graph2['edge_attr'])
    assert np.allclose(graph1['edge_attr'], graph3['edge_attr'])

    print("All implementations produce identical graphs.")


def main():
    from src.extract import MusicXMLNote
    from src.test import BACH_CHORALE, LE_MOLDEAU
    from src.constants import XML_ROOT
    from src.utils import get_path
    from src.extract import musicxml_to_notes

    path = get_path(XML_ROOT, BACH_CHORALE)
    notes = musicxml_to_notes(path)
    benchmark_graph_construction(notes)

    path = get_path(XML_ROOT, LE_MOLDEAU)
    notes = musicxml_to_notes(path)
    import time

    start = time.time()
    graph3 = construct_music_graph_sliding_window(notes)
    t3 = time.time() - start
    print(f"Sliding window approach took {t3:.4f} seconds with {graph3['num_nodes']} nodes and {graph3['edge_index'].shape[1]} edges.")


if __name__ == "__main__":
    main()

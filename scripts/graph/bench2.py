from src.extract import musicxml_to_notes
from src.utils import get_path, iterate_xmls
from src.constants import XML_ROOT
from src.test import BACH_CHORALE, LE_MOLDEAU
import time
import numpy as np
from typing import List, Dict, Any
import bisect

from tqdm import tqdm
from scripts.bench import construct_music_graph_sliding_window
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


def benchmark_graph_construction(notes: List[MusicXMLNote]):
    """Compare performance of different implementations and validate correctness."""
    from src.graph.make_graph import construct_music_graph as construct_music_graph_sliding_window

    start = time.time()
    graph1 = construct_music_graph(notes)
    t1 = time.time() - start
    print(f"Naive O(n²) approach took {t1:.4f} seconds with {graph1['num_nodes']} nodes and {graph1['edge_index'].shape[1]} edges.")

    start = time.time()
    graph3 = construct_music_graph_sliding_window(notes)
    t3 = time.time() - start
    print(f"Vectorized approach took {t3:.4f} seconds with {graph3['num_nodes']} nodes and {graph3['edge_index'].shape[1]} edges.")

    # Validate that all graphs are identical
    assert np.array_equal(graph1['node_features'], graph3['node_features'])
    assert np.array_equal(graph1['edge_index'], graph3['edge_index'])
    assert np.allclose(graph1['edge_attr'], graph3['edge_attr'])

    print("All implementations produce identical graphs.")


def main():
    test_set_count = 100
    bar = tqdm(iterate_xmls(), total=test_set_count, desc="Benchmarking graph construction")
    for i, path in enumerate(bar):
        if i >= test_set_count:
            break
        tqdm.write(f"====== Processing {path} (set {i+1}/{test_set_count}) ======")
        notes = musicxml_to_notes(path)
        benchmark_graph_construction(notes)

    path = get_path(XML_ROOT, LE_MOLDEAU)
    notes = musicxml_to_notes(path)

    start = time.time()
    graph3 = construct_music_graph_sliding_window(notes)
    t3 = time.time() - start
    print(f"Vectorized approach took {t3:.4f} seconds with {graph3['num_nodes']} nodes and {graph3['edge_index'].shape[1]} edges.")


if __name__ == "__main__":
    main()

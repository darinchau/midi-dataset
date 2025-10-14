import numpy as np
from typing import List, Dict, Any, Optional, Set
from ..extract import MusicXMLNote
from line_profiler import profile


class MusicGraphPreprocessor:
    """Handles preprocessing of music graph features."""

    def __init__(self):
        self.instrument_mapping = {}
        self.octave_mapping = {}
        self.pitch_mapping = {}
        self.feature_indices = {}
        self.is_fitted = False

    @profile
    def fit(self, notes: List[MusicXMLNote]):
        """Learn the categorical mappings from the data."""
        # Collect unique values
        # Can hardcode but use dynamic for potentially smaller model sizes later?
        instruments = set(note.instrument for note in notes)
        octaves = set(note.octave for note in notes)
        pitches = set(note.pitch for note in notes)

        # Create mappings
        self.instrument_mapping = {inst: i for i, inst in enumerate(sorted(instruments))}
        self.octave_mapping = {oct: i for i, oct in enumerate(sorted(octaves))}
        self.pitch_mapping = {pitch: i for i, pitch in enumerate(sorted(pitches))}

        self.is_fitted = True
        return self

    @profile
    def transform_features(
        self,
        node_features: np.ndarray,
        feature_info: Dict[str, Any],
        one_hot_categoricals: bool = True,
        normalize_continuous: bool = True
    ) -> np.ndarray:
        """Transform raw features into processed features.

        Args:
            node_features: Raw node features array.
            feature_info: Metadata about features.
            one_hot_categoricals: Whether to one-hot encode categorical features.
            normalize_continuous: Whether to normalize continuous features.
        Returns:
            Transformed node features array.
        """

        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        feature_names = feature_info['feature_names']
        transformed_features = []

        for i, feat_name in enumerate(feature_names):
            feat_col = node_features[:, i]

            if feat_name == 'instrument' and one_hot_categoricals:
                one_hot = np.zeros((len(feat_col), len(self.instrument_mapping)))
                for j, val in enumerate(feat_col):
                    if int(val) in self.instrument_mapping:
                        one_hot[j, self.instrument_mapping[int(val)]] = 1
                transformed_features.append(one_hot)

            elif feat_name == 'octave' and one_hot_categoricals:
                one_hot = np.zeros((len(feat_col), len(self.octave_mapping)))
                for j, val in enumerate(feat_col):
                    if int(val) in self.octave_mapping:
                        one_hot[j, self.octave_mapping[int(val)]] = 1
                transformed_features.append(one_hot)

            elif feat_name == 'pitch':
                if one_hot_categoricals:
                    # Option 1: One-hot encode (88 piano keys or 128 MIDI values)
                    one_hot = np.zeros((len(feat_col), 128))
                    for j, val in enumerate(feat_col):
                        one_hot[j, int(val)] = 1
                    transformed_features.append(one_hot)
                else:
                    # Option 2: Keep as continuous and normalize
                    normalized = feat_col / 127.0
                    transformed_features.append(normalized.reshape(-1, 1))

            elif feat_name in ['velocity']:
                # Normalize to [0, 1]
                normalized = feat_col / 127.0
                transformed_features.append(normalized.reshape(-1, 1))

            elif feat_name in ['start', 'duration', 'start_ql', 'duration_ql']:
                if normalize_continuous:
                    # Apply log transformation for duration-like features
                    if feat_name in ['duration', 'duration_ql']:
                        transformed = np.log1p(feat_col)
                    else:
                        transformed = feat_col
                    # Standardize
                    mean = transformed.mean()
                    std = transformed.std() + 1e-8
                    normalized = (transformed - mean) / std
                    transformed_features.append(normalized.reshape(-1, 1))
                else:
                    transformed_features.append(feat_col.reshape(-1, 1))

            else:
                # Keep as is (binary features, etc.)
                transformed_features.append(feat_col.reshape(-1, 1))

        # Concatenate all features
        return np.hstack(transformed_features).astype(np.float32)


@profile
def construct_music_graph(
    notes: List[MusicXMLNote],
    note_velocity_threshold: int = 20,
    remove_barlines: bool = True,
    max_seconds_apart: float = 10.0,
) -> Dict[str, Any]:
    """
    Constructs a graph with RAW features. One-hot encoding should be done later.
    Args:
        notes: List of MusicXMLNote objects.
        note_velocity_threshold: Velocity threshold to consider a note as "on".
        remove_barlines: Whether to remove barline notes from the graph.
        max_seconds_apart: Maximum time difference to connect notes with edges.
    """
    if not notes:
        return {
            'node_features': np.array([]),
            'edge_index': np.array([[], []]),
            'edge_attr': np.array([]),
            'num_nodes': 0,
            'feature_info': {}
        }

    # Remove all barlines since we don't care about those for now?
    if remove_barlines:
        notes = [note for note in notes if not note.barline]

    # Extract RAW node features
    node_features = []
    for note in notes:
        features = [
            note.instrument,        # Categorical
            note.pitch,             # Categorical
            note.start,             # Continuous
            note.duration,          # Continuous
            note.start_ql,          # Continuous
            note.duration_ql,       # Continuous
            note.index,             # Categorical
            note.octave,            # Categorical
            1.0 if note.velocity > note_velocity_threshold else 0.0,  # Binary
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

    # Compute weights
    edge_weights = np.maximum(1.0 - valid_diffs / max_seconds_apart, 0.0)

    # Final filter for positive weights (should be redundant but just in case)
    positive_mask = edge_weights > 0
    edge_sources = edge_sources[positive_mask]
    edge_targets = edge_targets[positive_mask]
    edge_weights = edge_weights[positive_mask]

    # Convert to proper format
    edge_index = np.stack([edge_sources, edge_targets], axis=0).astype(np.int64)
    edge_attr = edge_weights.astype(np.float32)

    # Store metadata about features
    feature_info = {
        'feature_names': [
            'instrument', 'pitch', 'start', 'duration',
            'start_ql', 'duration_ql', 'index', 'octave',
            'velocity'
        ],
        'categorical_features': ['instrument', 'octave', 'pitch', 'index'],
        'continuous_features': ['start', 'duration', 'start_ql', 'duration_ql'],
        'binary_features': ['velocity']
    }
    if not remove_barlines:
        feature_info['feature_names'].append('barline')
        feature_info['binary_features'].append('barline')

    return {
        'node_features': node_features,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'num_nodes': num_nodes,
        'feature_info': feature_info
    }


@profile
def create_preprocessed_graph(
    notes: List[MusicXMLNote],
    preprocessor: Optional[MusicGraphPreprocessor] = None,
    *,
    note_velocity_threshold: int = 20,
    remove_barlines: bool = True,
    max_seconds_apart: float = 10.0,
) -> Dict[str, Any]:
    """Create graph with preprocessed features."""

    graph = construct_music_graph(
        notes,
        note_velocity_threshold=note_velocity_threshold,
        remove_barlines=remove_barlines,
        max_seconds_apart=max_seconds_apart
    )

    if preprocessor is None:
        preprocessor = MusicGraphPreprocessor()
        preprocessor.fit(notes)

    graph['node_features'] = preprocessor.transform_features(
        graph['node_features'],
        graph['feature_info'],
        one_hot_categoricals=True,
        normalize_continuous=True
    )

    return graph

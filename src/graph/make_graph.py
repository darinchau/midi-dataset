import numpy as np
from typing import List, Dict, Any, Optional, Set
from ..extract import MusicXMLNote  # Converts the graph to PyG Data format

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


class MusicGraphPreprocessor:
    """Handles preprocessing of music graph features."""

    def __init__(self):
        self.instrument_mapping = {}
        self.octave_mapping = {}
        self.pitch_mapping = {}
        self.feature_indices = {}
        self.is_fitted = False

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

    def transform_features(
        self,
        node_features: np.ndarray,
        feature_info: FeatureInfo,
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

        transformed_features = []

        for i, feat_name in enumerate(feature_info.feature_names):
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
        features = [
            note.instrument,        # Categorical
            note.pitch,             # Categorical
            note.start,             # Continuous
            note.duration,          # Continuous
            note.start_ql,          # Continuous
            note.duration_ql,       # Continuous
            note.index,             # Categorical
            note.octave,            # Categorical
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
        ] + (['barline'] if not remove_barlines else []),
        categorical_features=['instrument', 'octave', 'pitch', 'index'],
        continuous_features=['start', 'duration', 'start_ql', 'duration_ql'],
        binary_features=['barline'] if not remove_barlines else []
    )

    return NoteGraph(
        node_features=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        feature_info=feature_info
    )


def create_preprocessed_graph(
    notes: List[MusicXMLNote],
    preprocessor: Optional[MusicGraphPreprocessor] = None,
    *,
    max_seconds_apart: float = 10.0,
) -> NoteGraph:
    """Create graph with preprocessed features."""

    graph = construct_music_graph(
        notes,
        remove_barlines=True,
        max_seconds_apart=max_seconds_apart
    )

    if preprocessor is None:
        preprocessor = MusicGraphPreprocessor()
        preprocessor.fit(notes)

    processed_features = preprocessor.transform_features(
        graph.node_features,
        graph.feature_info,
        one_hot_categoricals=True,
        normalize_continuous=True
    )

    return NoteGraph(
        node_features=processed_features,
        edge_index=graph.edge_index,
        edge_attr=graph.edge_attr,
        feature_info=graph.feature_info
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
    )

    return data


def pyg_data_to_graph(data: Data, feature_info: FeatureInfo) -> NoteGraph:
    """Convert PyTorch Geometric Data object back to NoteGraph format."""
    # Convert tensors back to numpy arrays
    node_features = data.x.cpu().numpy() if data.x is not None else np.array([])
    edge_index = data.edge_index.cpu().numpy() if data.edge_index is not None else np.array([[], []])
    edge_attr = data.edge_attr.cpu().numpy() if data.edge_attr is not None else np.array([])

    return NoteGraph(
        node_features=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        feature_info=feature_info
    )


def graph_to_xml_notes(
    graph: NoteGraph,
    preprocessor: Optional[MusicGraphPreprocessor] = None,
    default_velocity: int = 64,
    default_timesig: Optional[str] = None,
    preprocessed: bool = False
) -> List[MusicXMLNote]:
    """
    Convert NoteGraph back to list of MusicXMLNote objects.

    Args:
        graph: NoteGraph object to convert
        preprocessor: If features were preprocessed, provide the preprocessor to reverse transformations
        default_velocity: Default MIDI velocity for notes (not stored in graph)
        default_timesig: Default time signature (not stored in graph)
        preprocessed: Whether the graph features are preprocessed (one-hot encoded, normalized)

    Returns:
        List of MusicXMLNote objects
    """
    if graph.num_nodes == 0:
        return []

    # First, we need to reverse any preprocessing if applicable
    if preprocessed and preprocessor is not None:
        features = reverse_preprocessing(graph.node_features, graph.feature_info, preprocessor)
    else:
        features = graph.node_features

    # Extract features based on feature_info
    feature_indices = {name: i for i, name in enumerate(graph.feature_info.feature_names)}

    notes: list[MusicXMLNote] = []
    for i in range(graph.num_nodes):
        if 'instrument' not in feature_indices:
            raise ValueError("Feature 'instrument' missing from graph features")
        instrument = int(features[i, feature_indices['instrument']])

        if 'start' not in feature_indices:
            raise ValueError("Feature 'start' missing from graph features")
        start = float(features[i, feature_indices['start']])

        if 'duration' not in feature_indices:
            raise ValueError("Feature 'duration' missing from graph features")
        duration = float(features[i, feature_indices['duration']])

        if 'start_ql' not in feature_indices:
            raise ValueError("Feature 'start_ql' missing from graph features")
        start_ql = float(features[i, feature_indices['start_ql']])

        if 'duration_ql' not in feature_indices:
            raise ValueError("Feature 'duration_ql' missing from graph features")
        duration_ql = float(features[i, feature_indices['duration_ql']])

        if 'index' not in feature_indices:
            raise ValueError("Feature 'index' missing from graph features")
        index = int(features[i, feature_indices['index']])

        if 'octave' not in feature_indices:
            raise ValueError("Feature 'octave' missing from graph features")
        octave = int(features[i, feature_indices['octave']])

        if 'barline' in feature_indices:
            barline = bool(features[i, feature_indices['barline']] > 0.5)
        else:
            barline = False

        velocity = default_velocity
        timesig = default_timesig

        note = MusicXMLNote(
            instrument=instrument,
            start=start,
            duration=duration,
            start_ql=start_ql,
            duration_ql=duration_ql,
            index=index,
            octave=octave,
            barline=barline,
            velocity=velocity,
            timesig=timesig
        )

        notes.append(note)

    notes.sort(key=lambda x: (x.start, x.pitch))
    return notes


def reverse_preprocessing(
    processed_features: np.ndarray,
    feature_info: FeatureInfo,
    preprocessor: MusicGraphPreprocessor
) -> np.ndarray:
    """
    Reverse the preprocessing transformations to get back raw features.

    This function handles:
    - Reversing one-hot encoding
    - Denormalizing continuous features
    """
    if not preprocessor.is_fitted:
        raise ValueError("Preprocessor must be fitted to reverse transformations")

    raw_features = []
    current_idx = 0

    # Create reverse mappings
    reverse_instrument = {v: k for k, v in preprocessor.instrument_mapping.items()}
    reverse_octave = {v: k for k, v in preprocessor.octave_mapping.items()}
    reverse_pitch = {v: k for k, v in preprocessor.pitch_mapping.items()}

    for feat_name in feature_info.feature_names:
        if feat_name == 'instrument':
            # Reverse one-hot encoding
            num_instruments = len(preprocessor.instrument_mapping)
            one_hot = processed_features[:, current_idx:current_idx + num_instruments]
            raw_values = []
            for row in one_hot:
                idx = np.argmax(row).item()
                raw_values.append(reverse_instrument.get(idx, 0))
            raw_features.append(np.array(raw_values))
            current_idx += num_instruments

        elif feat_name == 'octave':
            # Reverse one-hot encoding
            num_octaves = len(preprocessor.octave_mapping)
            one_hot = processed_features[:, current_idx:current_idx + num_octaves]
            raw_values = []
            for row in one_hot:
                idx = np.argmax(row).item()
                raw_values.append(reverse_octave.get(idx, 0))
            raw_features.append(np.array(raw_values))
            current_idx += num_octaves

        elif feat_name == 'pitch':
            # Assuming one-hot encoding with 128 MIDI values
            one_hot = processed_features[:, current_idx:current_idx + 128]
            raw_values = []
            for row in one_hot:
                idx = np.argmax(row)
                raw_values.append(idx)
            raw_features.append(np.array(raw_values))
            current_idx += 128

        elif feat_name == 'velocity':
            # Denormalize from [0, 1] to [0, 127]
            normalized = processed_features[:, current_idx]
            raw_values = normalized * 127.0
            raw_features.append(raw_values)
            current_idx += 1

        elif feat_name in ['start', 'duration', 'start_ql', 'duration_ql']:
            # For now, just keep as is - proper denormalization would require
            # storing mean/std from original data
            raw_features.append(processed_features[:, current_idx])
            current_idx += 1

        elif feat_name == 'index':
            # Keep as is
            raw_features.append(processed_features[:, current_idx])
            current_idx += 1

        elif feat_name == 'barline':
            # Binary feature
            raw_features.append(processed_features[:, current_idx])
            current_idx += 1

        else:
            # Unknown feature, keep as is
            raw_features.append(processed_features[:, current_idx])
            current_idx += 1

    return np.column_stack(raw_features)


# Example usage:
def convert_pyg_to_xml(
    data: Data,
    feature_info: FeatureInfo,
    preprocessor: Optional[MusicGraphPreprocessor] = None,
    preprocessed: bool = False
) -> List[MusicXMLNote]:
    """
    Convenience function to convert PyTorch Geometric Data directly to XML notes.

    Args:
        data: PyTorch Geometric Data object
        feature_info: Metadata about features in the graph
        preprocessor: Preprocessor used for transforming features (if applicable)
        preprocessed: Whether the features in data are preprocessed

    Returns:
        List of MusicXMLNote objects
    """
    graph = pyg_data_to_graph(data, feature_info)
    return graph_to_xml_notes(graph, preprocessor, preprocessed=preprocessed)

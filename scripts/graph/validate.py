import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
from matplotlib.axes import Axes
from typing import List, Dict, Any, Tuple

from src.constants import XML_ROOT
from src.extract.analyze import musicxml_to_notes
from src.test import BACH_CHORALE, BACH_C_MAJOR_PRELUDE
from src.extract import MusicXMLNote
from src.graph import construct_music_graph, NoteGraph
from src.utils import get_path


def visualize_music_graph(notes: List[MusicXMLNote],
                          graph_data: NoteGraph,
                          figsize: Tuple[int, int] = (15, 10),
                          show_edges: bool = True,
                          edge_alpha: float = 0.3,
                          max_edges_to_show: int = 1000) -> None:
    """
    Visualize the music graph with notes as nodes and temporal connections as edges.

    Args:
        notes: List of MusicXMLNote objects
        graph_data: Output from construct_music_graph function
        figsize: Figure size
        show_edges: Whether to show edges (can be slow for dense graphs)
        edge_alpha: Transparency of edges
        max_edges_to_show: Maximum number of edges to display
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'Music Graph Visualization - {graph_data.num_nodes} nodes, {graph_data.edge_index.shape[1]} edges')

    # 1. Piano roll visualization with graph edges
    ax1 = axes[0, 0]
    plot_piano_roll_with_edges(graph_data, ax1, show_edges, edge_alpha, max_edges_to_show)

    # 2. Node degree distribution
    ax2 = axes[0, 1]
    plot_degree_distribution(graph_data, ax2)

    # 3. Edge weight distribution
    ax3 = axes[1, 0]
    plot_edge_weight_distribution(graph_data, ax3)

    # 4. Temporal connectivity pattern
    ax4 = axes[1, 1]
    plot_temporal_connectivity(graph_data, ax4)

    plt.tight_layout()
    plt.show()

    # Print validation statistics
    print_validation_stats(graph_data)


def plot_piano_roll_with_edges(
    graph_data: NoteGraph,
    ax: Axes,
    show_edges: bool = True,
    edge_alpha: float = 0.3,
    max_edges_to_show: int = 1000
) -> None:
    """Plot piano roll visualization with edges."""

    start_idx = graph_data.feature_info.feature_names.index('start')
    duration_idx = graph_data.feature_info.feature_names.index('duration')
    pitch_idx = graph_data.feature_info.feature_names.index('pitch')
    num_nodes = graph_data.num_nodes
    node_features = graph_data.node_features

    # Plot notes as rectangles
    for i in range(num_nodes):
        rect = Rectangle(
            (node_features[i, start_idx], node_features[i, pitch_idx] - 0.4),
            node_features[i, duration_idx], 0.8,
            facecolor='skyblue',
            edgecolor='darkblue',
            linewidth=1
        )
        ax.add_patch(rect)
        ax.text(
            node_features[i, start_idx] + node_features[i, duration_idx]/2, node_features[i, pitch_idx],
            str(i),
            ha='center',
            va='center',
            fontsize=6
        )

    # Plot edges if requested
    if show_edges and graph_data.edge_index.shape[1] > 0:
        edge_index = graph_data.edge_index
        edge_weights = graph_data.edge_attr

        num_edges = edge_index.shape[1]
        if num_edges > max_edges_to_show:
            indices = np.random.choice(num_edges, max_edges_to_show, replace=False)
            edge_index = edge_index[:, indices]
            edge_weights = edge_weights[indices]

        segments = []
        colors = []

        for idx in range(edge_index.shape[1]):
            src, tgt = edge_index[0, idx], edge_index[1, idx]

            x1 = node_features[src, start_idx] + node_features[src, duration_idx]/2
            y1 = node_features[src, pitch_idx]
            x2 = node_features[tgt, start_idx] + node_features[tgt, duration_idx]/2
            y2 = node_features[tgt, pitch_idx]

            segments.append([(x1, y1), (x2, y2)])
            colors.append(edge_weights[idx])

        # Create line collection
        lc = LineCollection(segments, cmap='viridis', alpha=edge_alpha)
        lc.set_array(np.array(colors))
        lc.set_linewidth(0.5)
        ax.add_collection(lc)

        # Add colorbar
        cbar = plt.colorbar(lc, ax=ax, label='Edge Weight')

    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('MIDI Pitch')
    ax.set_title('Piano Roll with Graph Edges')
    ax.grid(True, alpha=0.3)

    offsets = node_features[:, start_idx] + node_features[:, duration_idx]

    # Set reasonable limits
    ax.set_xlim(-0.5, max(offsets) + 0.5)
    ax.set_ylim(node_features[:, pitch_idx].min() - 2,
                node_features[:, pitch_idx].max() + 2)


def plot_degree_distribution(graph_data: NoteGraph, ax: Axes) -> None:
    """Plot in-degree and out-degree distributions."""

    edge_index = graph_data.edge_index
    num_nodes = graph_data.num_nodes

    if edge_index.shape[1] == 0:
        ax.text(0.5, 0.5, 'No edges in graph', ha='center', va='center')
        ax.set_title('Degree Distribution')
        return

    # Calculate degrees
    in_degrees = np.bincount(edge_index[1], minlength=num_nodes)
    out_degrees = np.bincount(edge_index[0], minlength=num_nodes)

    # Plot histogram
    bins = np.arange(0, max(in_degrees.max(), out_degrees.max()) + 2) - 0.5
    ax.hist([out_degrees, in_degrees], bins=bins, label=['Out-degree', 'In-degree'],
            alpha=0.7, color=['blue', 'red'])

    ax.set_xlabel('Degree')
    ax.set_ylabel('Number of Nodes')
    ax.set_title('Node Degree Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add statistics
    stats_text = f'Avg out-degree: {out_degrees.mean():.1f}\n'
    stats_text += f'Avg in-degree: {in_degrees.mean():.1f}\n'
    stats_text += f'Max out-degree: {out_degrees.max()}\n'
    stats_text += f'Max in-degree: {in_degrees.max()}'
    ax.text(0.7, 0.7, stats_text, transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def plot_edge_weight_distribution(graph_data: NoteGraph, ax: Axes) -> None:
    """Plot distribution of edge weights."""

    edge_weights = graph_data.edge_attr

    if len(edge_weights) == 0:
        ax.text(0.5, 0.5, 'No edges in graph', ha='center', va='center')
        ax.set_title('Edge Weight Distribution')
        return

    # Plot histogram
    ax.hist(edge_weights, bins=50, alpha=0.7, color='green', edgecolor='darkgreen')
    ax.set_xlabel('Edge Weight')
    ax.set_ylabel('Number of Edges')
    ax.set_title('Edge Weight Distribution')
    ax.grid(True, alpha=0.3)

    # Add statistics
    stats_text = f'Mean weight: {edge_weights.mean():.3f}\n'
    stats_text += f'Std weight: {edge_weights.std():.3f}\n'
    stats_text += f'Min weight: {edge_weights.min():.3f}\n'
    stats_text += f'Max weight: {edge_weights.max():.3f}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def plot_temporal_connectivity(graph_data: NoteGraph,
                               ax: Axes) -> None:
    """Plot how connectivity changes over time."""

    edge_index = graph_data.edge_index
    edge_weights = graph_data.edge_attr

    if edge_index.shape[1] == 0:
        ax.text(0.5, 0.5, 'No edges in graph', ha='center', va='center')
        ax.set_title('Temporal Connectivity')
        return

    note_starts = graph_data.node_features[:, graph_data.feature_info.feature_names.index('start')]

    # Group edges by source note time
    time_bins = np.linspace(0, note_starts.max(), 50)
    connectivity = np.zeros(len(time_bins) - 1)
    weights_sum = np.zeros(len(time_bins) - 1)

    for idx in range(edge_index.shape[1]):
        src = edge_index[0, idx]
        src_time = note_starts[src]
        bin_idx = np.searchsorted(time_bins, src_time) - 1
        if 0 <= bin_idx < len(connectivity):
            connectivity[bin_idx] += 1
            weights_sum[bin_idx] += edge_weights[idx]

    # Plot
    bin_centers = (time_bins[:-1] + time_bins[1:]) / 2
    ax.bar(bin_centers, connectivity, width=time_bins[1] - time_bins[0],
           alpha=0.7, label='Edge count')

    # Plot average weight on secondary axis
    ax2 = ax.twinx()
    mask = connectivity > 0
    avg_weights = np.zeros_like(weights_sum)
    avg_weights[mask] = weights_sum[mask] / connectivity[mask]
    ax2.plot(bin_centers, avg_weights, 'r-', linewidth=2, label='Avg weight')
    ax2.set_ylabel('Average Edge Weight', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Number of Outgoing Edges')
    ax.set_title('Temporal Connectivity Pattern')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')


def print_validation_stats(graph_data: NoteGraph) -> None:
    """Print validation statistics to verify graph construction."""

    print("\n=== Graph Validation Statistics ===")
    print(f"Number of nodes: {graph_data.num_nodes}")
    print(f"Number of edges: {graph_data.edge_index.shape[1]}")

    note_starts = graph_data.node_features[:, graph_data.feature_info.feature_names.index('start')]

    if graph_data.edge_index.shape[1] > 0:
        edge_index = graph_data.edge_index
        edge_weights = graph_data.edge_attr

        # Check edge directions
        forward_edges = 0
        same_time_edges = 0

        for idx in range(edge_index.shape[1]):
            src, tgt = edge_index[0, idx], edge_index[1, idx]
            if note_starts[src] < note_starts[tgt]:
                forward_edges += 1
            elif note_starts[src] == note_starts[tgt]:
                same_time_edges += 1

        print(f"Forward edges (src.start < tgt.start): {forward_edges}")
        print(f"Same-time edges (src.start == tgt.start): {same_time_edges}")
        print(f"Edge density: {graph_data.edge_index.shape[1] / (graph_data.num_nodes ** 2):.4f}")

        # Verify edge weight calculation
        print("\nSample edge verification (first 5 edges):")
        for idx in range(min(5, edge_index.shape[1])):
            src, tgt = edge_index[0, idx], edge_index[1, idx]
            time_diff = note_starts[tgt] - note_starts[src]
            expected_weight = max(1.0 - time_diff / 1, 0.0)
            actual_weight = edge_weights[idx]
            print(f"  Edge {src}->{tgt}: time_diff={time_diff:.3f}, "
                  f"weight={actual_weight:.3f} (expected: {expected_weight:.3f})")


# Example usage
if __name__ == "__main__":
    path = get_path(XML_ROOT, BACH_C_MAJOR_PRELUDE)
    test_notes = musicxml_to_notes(path)

    # Construct graph
    graph = construct_music_graph(test_notes, max_seconds_apart=1)

    # Visualize
    visualize_music_graph(test_notes, graph, show_edges=True, max_edges_to_show=500)

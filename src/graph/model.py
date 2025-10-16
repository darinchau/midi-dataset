import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool, GlobalAttention
from torch_geometric.nn import AttentionalAggregation
from torch_geometric.data import Data, Batch
import numpy as np
from torch import Tensor
from typing import Optional, Tuple, Dict, List


class AttentionalReadout(nn.Module):
    """Attention-based graph readout layer"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = AttentionalAggregation(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1)
            )
        )

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        return self.attention(x, batch)


class EdgeEncoder(nn.Module):
    """Encode edge attributes"""

    def __init__(self, edge_attr_dim: int, edge_emb_dim: int):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_attr_dim, edge_emb_dim),
            nn.ReLU(),
            nn.Linear(edge_emb_dim, edge_emb_dim)
        )

    def forward(self, edge_attr: Tensor) -> Tensor:
        return self.edge_mlp(edge_attr)


class SymMusicMotifGNN(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        output_dim: int = 64,
        num_instruments: int = 16,
        num_octaves: int = 12,
        num_pitches: int = 128,
        num_indices: int = 16,
        instrument_emb_dim: int = 8,
        octave_emb_dim: int = 4,
        pitch_emb_dim: int = 16,
        index_emb_dim: int = 8,
        time_emb_dim: int = 64,
        edge_attr_dim: int = 1,
        edge_emb_dim: int = 32,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Embedding layers for categorical features
        self.instrument_embedding = nn.Embedding(num_instruments, instrument_emb_dim)
        self.octave_embedding = nn.Embedding(num_octaves, octave_emb_dim)
        self.pitch_embedding = nn.Embedding(num_pitches, pitch_emb_dim)
        self.index_embedding = nn.Embedding(num_indices, index_emb_dim)

        # MLP for continuous features
        self.continuous_encoder = nn.Sequential(
            nn.Linear(4, time_emb_dim * 2),  # 4 continuous features
            nn.ReLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim)
        )

        # Calculate total embedding dimension
        self.embedded_feature_dim = (
            instrument_emb_dim + octave_emb_dim + pitch_emb_dim +
            index_emb_dim + time_emb_dim
        )

        # Initial feature transformation
        self.feature_transform = nn.Sequential(
            nn.Linear(self.embedded_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Edge encoder
        self.edge_encoder = EdgeEncoder(edge_attr_dim, edge_emb_dim)

        # GATv2 layers
        self.conv1 = GATv2Conv(
            hidden_dim, hidden_dim,
            heads=num_heads,
            edge_dim=edge_emb_dim,
            dropout=dropout,
            concat=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim * num_heads)

        self.conv2 = GATv2Conv(
            hidden_dim * num_heads, hidden_dim,
            heads=num_heads,
            edge_dim=edge_emb_dim,
            dropout=dropout,
            concat=False
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.attention_readout = AttentionalReadout(hidden_dim)

        # Final projection with non-negative constraint
        self.final_projection = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # 3x for attention + mean + max
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Softplus()  # Ensures non-negative output for monotonicity
        )

        if output_dim > hidden_dim * 3:
            warnings.warn(
                "Output dimension is greater than hidden_dim * 3; "
                "this may limit the expressiveness of the final layer."
            )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None
    ) -> Tensor:
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        instrument = x[:, 0].long()
        pitch = x[:, 1].long()
        timeinfo = x[:, 2:6].float()
        index = x[:, 6].long()
        octave = x[:, 7].long()

        instrument_emb = self.instrument_embedding(instrument)
        pitch_emb = self.pitch_embedding(pitch)
        index_emb = self.index_embedding(index)
        octave_emb = self.octave_embedding(octave)
        time_emb = self.continuous_encoder(timeinfo)

        x = torch.cat([
            instrument_emb, pitch_emb, index_emb, octave_emb, time_emb
        ], dim=1)

        x = self.feature_transform(x)

        # Encode edges if provided
        if edge_attr is not None:
            edge_emb = self.edge_encoder(edge_attr)
        else:
            edge_emb = None

        # GATv2 layers with skip connections
        identity = x
        x = self.conv1(x, edge_index, edge_attr=edge_emb)
        x = self.norm1(x)
        x = F.relu(x)  # Skip connection
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_attr=edge_emb)
        x = self.norm2(x)
        x = F.relu(x + identity)  # Skip connection

        # Multiple readout strategies
        x_att = self.attention_readout(x, batch)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)

        # Combine readouts
        x = torch.cat([x_att, x_mean, x_max], dim=1)
        x = self.final_projection(x)
        return x


if __name__ == "__main__":

    from src.extract import MusicXMLNote
    from src.test import BACH_CHORALE, LE_MOLDEAU, BACH_C_MAJOR_PRELUDE
    from src.constants import XML_ROOT
    from src.utils import get_path
    from src.extract import musicxml_to_notes
    from src.graph.make_graph import construct_music_graph, graph_to_pyg_data
    from src.graph.filter import is_good_midi

    path = get_path(XML_ROOT, BACH_C_MAJOR_PRELUDE)
    notes = musicxml_to_notes(path)

    is_good_midi(path)

    graph = construct_music_graph(notes)

    print(graph)
    print("Node features shape:", graph.node_features.shape)
    print("Edge index shape:", graph.edge_index.shape)
    print("Edge attributes shape:", graph.edge_attr.shape)

    data = graph_to_pyg_data(graph)

    # Initialize the model
    model = SymMusicMotifGNN(
        hidden_dim=64,
        output_dim=192,
        num_instruments=8,
        num_octaves=12,
        num_pitches=128,
        num_indices=16,
        instrument_emb_dim=8,
        octave_emb_dim=4,
        pitch_emb_dim=16,
        index_emb_dim=8,
        time_emb_dim=64,
    )

    print(sum(p.numel() for p in model.parameters() if p.requires_grad), "trainable parameters")

    output = model(data.x, data.edge_index)

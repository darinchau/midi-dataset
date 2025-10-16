import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool, GlobalAttention
from torch_geometric.data import Data, Batch
import numpy as np
from torch import Tensor
from typing import Optional, Tuple, Dict, List


class AttentionalReadout(nn.Module):
    """Attention-based graph readout layer"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = GlobalAttention(
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
        continuous_emb_dim: int = 64,
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
            nn.Linear(4, continuous_emb_dim * 2),  # 4 continuous features
            nn.ReLU(),
            nn.Linear(continuous_emb_dim * 2, continuous_emb_dim)
        )

        # Calculate total embedding dimension
        self.embedded_feature_dim = (
            instrument_emb_dim + octave_emb_dim + pitch_emb_dim +
            index_emb_dim + continuous_emb_dim
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

        # GATv2 layers with edge conditioning
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
            concat=False  # Average instead of concat for last layer
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Readout layers
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

        # Extract features (assuming x has shape [N, 9] with velocity)
        instrument = x[:, 0].long()
        pitch = x[:, 1].long()
        start = x[:, 2:3]
        duration = x[:, 3:4]
        start_ql = x[:, 4:5]
        duration_ql = x[:, 5:6]
        index = x[:, 6].long()
        octave = x[:, 7].long()

        # Get embeddings for categorical features
        instrument_emb = self.instrument_embedding(instrument)
        pitch_emb = self.pitch_embedding(pitch)
        index_emb = self.index_embedding(index)
        octave_emb = self.octave_embedding(octave)

        # Process continuous features together
        continuous_features = torch.cat([start, duration, start_ql, duration_ql], dim=1)
        continuous_emb = self.continuous_encoder(continuous_features)

        # Concatenate all features
        x = torch.cat([
            instrument_emb, pitch_emb, index_emb, octave_emb, continuous_emb
        ], dim=1)

        # Initial transformation
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
        x = F.relu(x)
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

        # Final projection with non-negative constraint
        x = self.final_projection(x)

        return x


class MotifLoss(nn.Module):
    """Combined loss function for motif learning with constraints"""

    def __init__(
        self,
        lambda_mono: float = 1.0,
        lambda_edit: float = 1.0,
        lambda_diff: float = 0.5,
        lambda_origin: float = 0.1,
        edit_threshold: float = 0.5,
        diff_margin: float = 2.0
    ):
        super().__init__()
        self.lambda_mono = lambda_mono
        self.lambda_edit = lambda_edit
        self.lambda_diff = lambda_diff
        self.lambda_origin = lambda_origin
        self.edit_threshold = edit_threshold
        self.diff_margin = diff_margin

    def monotonicity_loss(self, embeddings_sub: Tensor, embeddings_super: Tensor) -> Tensor:
        """
        Enforces P(M1) <= P(M2) elementwise when M1 is subgraph of M2
        """
        # ReLU-based hinge penalty for violations
        violations = F.relu(embeddings_sub - embeddings_super)
        return violations.mean()

    def edit_distance_loss(self, embeddings1: Tensor, embeddings2: Tensor) -> Tensor:
        """
        Enforces d(P(M1), P(M2)) <= threshold for 1-edit neighbors
        """
        distances = torch.norm(embeddings1 - embeddings2, p=2, dim=1)
        return F.relu(distances - self.edit_threshold).mean()

    def difference_loss(self, embeddings1: Tensor, embeddings2: Tensor) -> Tensor:
        """
        Enforces minimum distance between unrelated motifs
        """
        distances = torch.norm(embeddings1 - embeddings2, p=2, dim=1)
        return F.relu(self.diff_margin - distances).mean()

    def origin_loss(self, empty_embedding: Tensor) -> Tensor:
        """
        Regularizes empty motif to map near origin
        """
        return torch.norm(empty_embedding, p=2)

    def forward(
        self,
        subgraph_pairs: Optional[Tuple[Tensor, Tensor]] = None,
        edit_pairs: Optional[Tuple[Tensor, Tensor]] = None,
        negative_pairs: Optional[Tuple[Tensor, Tensor]] = None,
        empty_embedding: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        Compute combined loss

        Args:
            subgraph_pairs: (embeddings_sub, embeddings_super) for monotonicity
            edit_pairs: (embeddings1, embeddings2) for edit distance constraint
            negative_pairs: (embeddings1, embeddings2) for dissimilar motifs
            empty_embedding: embedding of empty motif

        Returns:
            Dictionary with individual losses and total loss
        """
        losses = {}
        total_loss = 0.0

        if subgraph_pairs is not None:
            loss_mono = self.monotonicity_loss(*subgraph_pairs)
            losses['monotonicity'] = loss_mono
            total_loss += self.lambda_mono * loss_mono

        if edit_pairs is not None:
            loss_edit = self.edit_distance_loss(*edit_pairs)
            losses['edit_distance'] = loss_edit
            total_loss += self.lambda_edit * loss_edit

        if negative_pairs is not None:
            loss_diff = self.difference_loss(*negative_pairs)
            losses['difference'] = loss_diff
            total_loss += self.lambda_diff * loss_diff

        if empty_embedding is not None:
            loss_origin = self.origin_loss(empty_embedding)
            losses['origin'] = loss_origin
            total_loss += self.lambda_origin * loss_origin

        losses['total'] = total_loss
        return losses


# Helper function to create batches of motif pairs
def create_motif_pairs(
    graphs: List[Data],
    model: nn.Module,
    pair_type: str = 'subgraph'
) -> Tuple[Tensor, Tensor]:
    """
    Create pairs of motif embeddings for training

    Args:
        graphs: List of graph data objects
        model: The GNN model
        pair_type: 'subgraph', 'edit', or 'negative'

    Returns:
        Tuple of (embeddings1, embeddings2)
    """
    # This is a placeholder - you'll need to implement the actual pairing logic
    # based on your subgraph/edit distance computation
    batch = Batch.from_data_list(graphs)
    embeddings = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

    # For now, return dummy pairs
    n = len(graphs)
    if n > 1:
        return embeddings[:-1], embeddings[1:]
    else:
        return embeddings, embeddings


# Example usage
if __name__ == "__main__":
    # Initialize model
    model = SymMusicMotifGNN(
        hidden_dim=128,
        output_dim=64,
        num_heads=4,
        edge_attr_dim=1,  # Adjust based on your edge attributes
        edge_emb_dim=32
    )

    # Initialize loss function
    criterion = MotifLoss(
        lambda_mono=1.0,
        lambda_edit=1.0,
        lambda_diff=0.5,
        lambda_origin=0.1,
        edit_threshold=0.5,
        diff_margin=2.0
    )

    # Example forward pass (assuming you have a data object)
    # output = model(data.x, data.edge_index, data.edge_attr)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List


@dataclass(frozen=True)
class VQVAEConfig:
    """Configuration for VQ-VAE model and training"""

    # Data parameters
    input_hz: int = 48  # Input temporal resolution in Hz
    target_hz: int = 24  # Target temporal resolution in Hz (for compression)
    num_instruments: int = 128  # Number of MIDI instruments
    lowest_note_number: int = 21  # Lowest MIDI note number (A0)
    highest_note_number: int = 108  # Highest MIDI note number (C8)
    num_notes: int = field(init=False)  # Number of MIDI notes (computed)
    velocity_threshold: float = 0.3  # Velocity threshold for active notes

    # Model architecture
    hidden_dim: int = 512  # Hidden dimension for transformer
    num_layers: int = 4  # Number of transformer layers in encoder/decoder
    num_heads: int = 8  # Number of attention heads
    ff_dim: int = -1  # Feed-forward dimension (-1 = 4 * hidden_dim)
    dropout: float = 0.1  # Dropout rate

    # Vector Quantization
    num_embeddings: int = 512  # Size of VQ codebook
    embedding_dim: int = 512  # Dimension of VQ embeddings
    commitment_cost: float = 0.25  # VQ commitment loss weight

    # Training parameters
    batch_size: int = 8
    num_epochs: int = 100
    learning_rate: float = 1e-4
    recon_loss_weight: float = 1.0
    vq_loss_weight: float = 0.25
    gradient_clip: Optional[float] = 1.0  # Gradient clipping value

    # Optimization
    adam_betas: Tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    weight_decay: float = 0.0

    # Learning rate schedule
    use_lr_scheduler: bool = True
    lr_scheduler_type: str = "cosine"  # "cosine", "step", "exponential"
    lr_warmup_steps: int = 1000
    lr_min: float = 1e-6

    # Logging and checkpointing
    log_interval: int = 10  # Log every N batches
    save_interval: int = 5  # Save checkpoint every N epochs
    checkpoint_dir: str = "./checkpoints"

    # Device settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True

    # Derived parameters (computed post-init)
    downsample_factor: int = field(init=False)
    input_channels: int = field(init=False)

    def __post_init__(self):
        """Compute derived parameters"""
        assert self.input_hz % self.target_hz == 0, "Input Hz must be divisible by target Hz"
        object.__setattr__(self, 'num_notes', self.highest_note_number - self.lowest_note_number + 1)
        object.__setattr__(self, 'downsample_factor', self.input_hz // self.target_hz)
        object.__setattr__(self, 'input_channels', self.num_instruments * self.num_notes)

        if self.ff_dim == -1:
            object.__setattr__(self, 'ff_dim', 4 * self.hidden_dim)


class BidirectionalAttention(nn.Module):
    def __init__(self, config: VQVAEConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.dim = config.hidden_dim
        self.head_dim = self.dim // self.num_heads
        assert self.head_dim * self.num_heads == self.dim

        self.q_proj = nn.Linear(self.dim, self.dim)
        self.k_proj = nn.Linear(self.dim, self.dim)
        self.v_proj = nn.Linear(self.dim, self.dim)
        self.out_proj = nn.Linear(self.dim, self.dim)

        self.dropout = nn.Dropout(config.dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x):
        B, T, D = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(B, T, D)

        return self.out_proj(attn_output)


class TransformerBlock(nn.Module):
    def __init__(self, config: VQVAEConfig):
        super().__init__()

        self.attention = BidirectionalAttention(config)
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)

        self.ff = nn.Sequential(
            nn.Linear(config.hidden_dim, config.ff_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ff_dim, config.hidden_dim),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        # Self-attention with residual
        x = x + self.attention(self.norm1(x))
        # Feed-forward with residual
        x = x + self.ff(self.norm2(x))
        return x


class VectorQuantizer(nn.Module):
    def __init__(self, config: VQVAEConfig):
        super().__init__()
        self.num_embeddings = config.num_embeddings
        self.embedding_dim = config.embedding_dim
        self.commitment_cost = config.commitment_cost

        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, z):
        # z shape: (B, D, T)
        z = z.permute(0, 2, 1).contiguous()  # (B, T, D)
        z_flattened = z.view(-1, self.embedding_dim)

        # Calculate distances to embeddings
        distances = (z_flattened.pow(2).sum(1, keepdim=True)
                     + self.embeddings.weight.pow(2).sum(1)
                     - 2 * z_flattened @ self.embeddings.weight.t())

        # Get closest embeddings
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=z.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize
        quantized = encodings @ self.embeddings.weight
        quantized = quantized.view(z.shape)

        # Compute losses
        e_latent_loss = F.mse_loss(quantized.detach(), z)
        q_latent_loss = F.mse_loss(quantized, z.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized = z + (quantized - z).detach()

        # Return to original shape
        quantized = quantized.permute(0, 2, 1).contiguous()

        # Also return indices for token representation
        encoding_indices = encoding_indices.view(z.shape[0], -1)

        return quantized, loss, encoding_indices

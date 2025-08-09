from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import ReformerConfig, ReformerModel
from einops import rearrange, repeat


@dataclass
class ModelConfig:
    """Configuration for the Reformer-RVQ compression model"""
    # Input dimensions
    total_input_dim: int = 276
    discrete_dim: int = 274
    continuous_dim: int = 2

    # Model architecture
    hidden_dim: int = 512
    compressed_dim: int = 128

    # Reformer configuration
    num_layers: int = 6
    num_heads: int = 8
    feed_forward_size: int = 2048
    max_position_embeddings: int = 4096
    axial_pos_shape: Tuple[int, int] = (64, 64)
    num_buckets: int = 64
    num_hashes: int = 4
    lsh_attn_chunk_length: int = 256
    local_attn_chunk_length: int = 128

    # RVQ configuration
    num_quantizers: int = 4
    codebook_size: int = 512
    commitment_weight: float = 0.25

    # Mixed feature encoder configuration
    vocab_size: int = 1000
    num_embeddings: int = 8

    def __post_init__(self) -> None:
        assert self.total_input_dim == self.discrete_dim + self.continuous_dim, \
            "total_input_dim must equal discrete_dim + continuous_dim"


class ResidualVectorQuantizer(nn.Module):
    """Residual Vector Quantization module for compression"""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_quantizers: int = config.num_quantizers
        self.codebook_size: int = config.codebook_size
        self.dim: int = config.compressed_dim
        self.commitment_weight: float = config.commitment_weight

        # Initialize codebooks for each quantizer
        self.codebooks: nn.ParameterList = nn.ParameterList([
            nn.Parameter(torch.randn(self.codebook_size, self.dim) * 0.1)
            for _ in range(self.num_quantizers)
        ])

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: input tensor of shape (batch, seq_len, compressed_dim)

        Returns:
            quantized: quantized tensor of shape (batch, seq_len, compressed_dim)
            indices: codebook indices of shape (batch, seq_len, num_quantizers)
            commitment_loss: scalar tensor
        """
        batch, seq_len, dim = x.shape

        residual: torch.Tensor = x
        quantized: torch.Tensor = torch.zeros_like(x)
        all_indices: List[torch.Tensor] = []
        commitment_loss: torch.Tensor = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        for i, codebook in enumerate(self.codebooks):
            distances: torch.Tensor = torch.cdist(residual, codebook)
            indices: torch.Tensor = distances.argmin(dim=-1)
            all_indices.append(indices)
            quantized_step: torch.Tensor = F.embedding(indices, codebook)
            quantized = quantized + quantized_step
            commitment_loss = commitment_loss + F.mse_loss(
                quantized_step.detach(), residual
            )

            # Update residual
            residual = residual - quantized_step.detach()

        all_indices_stacked: torch.Tensor = torch.stack(all_indices, dim=-1)

        # Straight-through estimator
        quantized = x + (quantized - x).detach()

        return quantized, all_indices_stacked, commitment_loss * self.commitment_weight


class MixedFeatureEncoder(nn.Module):
    """Handles mixed discrete and continuous features"""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.discrete_dim: int = config.discrete_dim
        self.continuous_dim: int = config.continuous_dim
        self.hidden_dim: int = config.hidden_dim
        self.vocab_size: int = config.vocab_size
        self.num_embeddings: int = config.num_embeddings

        # Multiple embedding tables for handling sparse discrete features
        self.embeddings: nn.ModuleList = nn.ModuleList([
            nn.Embedding(self.vocab_size, self.hidden_dim // self.num_embeddings)
            for _ in range(self.num_embeddings)
        ])

        # Project discrete features to embedding indices
        chunk_size: int = self.discrete_dim // self.num_embeddings
        self.discrete_projections: nn.ModuleList = nn.ModuleList([
            nn.Linear(
                chunk_size if i < self.num_embeddings - 1
                else self.discrete_dim - chunk_size * (self.num_embeddings - 1),
                1
            )
            for i in range(self.num_embeddings)
        ])

        # Process continuous features
        self.continuous_projection: nn.Sequential = nn.Sequential(
            nn.Linear(self.continuous_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim)
        )

        # Combine discrete and continuous features
        self.fusion: nn.Sequential = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (batch, seq_len, total_input_dim)

        Returns:
            output: encoded features of shape (batch, seq_len, hidden_dim)
        """
        batch, seq_len, _ = x.shape

        # Split discrete and continuous features
        discrete_features: torch.Tensor = x[..., :self.discrete_dim]
        continuous_features: torch.Tensor = x[..., self.discrete_dim:]

        # Process discrete features through multiple embeddings
        discrete_embeds: List[torch.Tensor] = []
        chunk_size: int = self.discrete_dim // self.num_embeddings

        for i, (embed, proj) in enumerate(zip(self.embeddings, self.discrete_projections)):  # type: ignore
            embed: torch.nn.Embedding
            proj: nn.Linear
            # Get chunk of discrete features
            start_idx: int = i * chunk_size
            end_idx: int = (i + 1) * chunk_size if i < self.num_embeddings - 1 else self.discrete_dim
            chunk: torch.Tensor = discrete_features[..., start_idx:end_idx]

            # Project to indices (handling sparsity)
            indices: torch.Tensor = proj(chunk).squeeze(-1)
            indices = torch.clamp(indices, 0, embed.num_embeddings - 1).long()

            # Get embeddings
            embed_chunk: torch.Tensor = embed(indices)
            discrete_embeds.append(embed_chunk)

        # Concatenate all discrete embeddings
        discrete_encoded: torch.Tensor = torch.cat(discrete_embeds, dim=-1)

        # Process continuous features
        continuous_encoded: torch.Tensor = self.continuous_projection(continuous_features)

        # Combine features
        combined: torch.Tensor = torch.cat([discrete_encoded, continuous_encoded], dim=-1)
        output: torch.Tensor = self.fusion(combined)

        return output


class ReformerCompressor(nn.Module):
    """Main model combining Reformer and RVQ for sequence compression"""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config: ModelConfig = config

        # Feature encoder for mixed input
        self.feature_encoder: MixedFeatureEncoder = MixedFeatureEncoder(config)

        # Reformer encoder configuration
        self.encoder_config: ReformerConfig = ReformerConfig(
            hidden_size=config.hidden_dim,
            num_attention_heads=config.num_heads,
            num_hidden_layers=config.num_layers,
            feed_forward_size=config.feed_forward_size,
            max_position_embeddings=config.max_position_embeddings,
            axial_pos_shape=list(config.axial_pos_shape),
            num_buckets=config.num_buckets,
            num_hashes=config.num_hashes,
            lsh_attn_chunk_length=config.lsh_attn_chunk_length,
            local_attn_chunk_length=config.local_attn_chunk_length,
            lsh_num_chunks_before=1,
            lsh_num_chunks_after=1,
            is_decoder=False
        )

        # Reformer encoder
        self.encoder: ReformerModel = ReformerModel(self.encoder_config)

        # Compression projection
        self.compress_projection: nn.Linear = nn.Linear(
            config.hidden_dim, config.compressed_dim
        )

        # RVQ module
        self.rvq: ResidualVectorQuantizer = ResidualVectorQuantizer(config)

        # Decoder projection
        self.decompress_projection: nn.Linear = nn.Linear(
            config.compressed_dim, config.hidden_dim
        )

        # Reformer decoder configuration
        self.decoder_config: ReformerConfig = ReformerConfig(
            hidden_size=config.hidden_dim,
            num_attention_heads=config.num_heads,
            num_hidden_layers=config.num_layers,
            feed_forward_size=config.feed_forward_size,
            max_position_embeddings=config.max_position_embeddings,
            axial_pos_shape=list(config.axial_pos_shape),
            num_buckets=config.num_buckets,
            num_hashes=config.num_hashes,
            lsh_attn_chunk_length=config.lsh_attn_chunk_length,
            local_attn_chunk_length=config.local_attn_chunk_length,
            is_decoder=True
        )

        # Reformer decoder
        self.decoder: ReformerModel = ReformerModel(self.decoder_config)

        # Output reconstruction layers
        self.output_projection: nn.Linear = nn.Linear(
            config.hidden_dim, config.hidden_dim
        )

        # Reconstruct discrete features
        self.discrete_reconstruction: nn.Linear = nn.Linear(
            config.hidden_dim, config.discrete_dim
        )

        # Reconstruct continuous features
        self.continuous_reconstruction: nn.Sequential = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, config.continuous_dim)
        )

    def encode(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode input sequences

        Args:
            x: input tensor of shape (batch, seq_len, total_input_dim)

        Returns:
            quantized: quantized representation (batch, seq_len, compressed_dim)
            indices: quantization indices (batch, seq_len, num_quantizers)
            vq_loss: vector quantization loss
            encoded: encoder output before quantization (batch, seq_len, hidden_dim)
        """
        # Process mixed features
        features: torch.Tensor = self.feature_encoder(x)

        # Encode with Reformer
        encoder_output = self.encoder(inputs_embeds=features)
        encoded: torch.Tensor = encoder_output.last_hidden_state

        # Project to compression dimension
        compressed: torch.Tensor = self.compress_projection(encoded)

        # Apply RVQ
        quantized, indices, vq_loss = self.rvq(compressed)

        return quantized, indices, vq_loss, encoded

    def decode(self, quantized: torch.Tensor) -> torch.Tensor:
        """
        Decode from quantized representation

        Args:
            quantized: quantized tensor of shape (batch, seq_len, compressed_dim)

        Returns:
            reconstructed: reconstructed input (batch, seq_len, total_input_dim)
        """
        # Project back to hidden dimension
        decompressed: torch.Tensor = self.decompress_projection(quantized)

        # Decode with Reformer
        decoder_output = self.decoder(inputs_embeds=decompressed)
        decoded: torch.Tensor = decoder_output.last_hidden_state

        # Project to output
        output_features: torch.Tensor = self.output_projection(decoded)

        # Reconstruct discrete and continuous parts
        discrete_recon: torch.Tensor = self.discrete_reconstruction(output_features)
        continuous_recon: torch.Tensor = self.continuous_reconstruction(output_features)

        # Combine reconstructions
        reconstructed: torch.Tensor = torch.cat([discrete_recon, continuous_recon], dim=-1)

        return reconstructed

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass with encoding and decoding

        Args:
            x: input tensor of shape (batch, seq_len, total_input_dim)

        Returns:
            reconstructed: reconstructed input (batch, seq_len, total_input_dim)
            indices: quantization indices (batch, seq_len, num_quantizers)
            vq_loss: vector quantization loss
        """
        # Encode
        quantized, indices, vq_loss, _ = self.encode(x)

        # Decode
        reconstructed = self.decode(quantized)

        return reconstructed, indices, vq_loss

    def compress_ratio(self) -> float:
        """Calculate theoretical compression ratio"""
        # Original: total_input_dim * 32 bits (assuming float32)
        # Compressed: num_quantizers * log2(codebook_size) bits
        original_bits: float = self.config.total_input_dim * 32
        compressed_bits: float = self.config.num_quantizers * np.log2(self.config.codebook_size)
        return original_bits / compressed_bits

    def get_compressed_representation(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get only the compressed representation without decoding

        Args:
            x: input tensor of shape (batch, seq_len, total_input_dim)

        Returns:
            quantized: quantized representation (batch, seq_len, compressed_dim)
            indices: quantization indices (batch, seq_len, num_quantizers)
        """
        quantized, indices, _, _ = self.encode(x)
        return quantized, indices

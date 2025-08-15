# Tokenizes MIDI compound words (cp) into discrete tokens

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import Dataset
import numpy as np
from dataclasses import dataclass
import logging
from functools import cache
from tqdm import tqdm
from ..extract import musicxml_to_notes
from ..extract.tokenize import notes_to_tokens
from ..constants import XML_ROOT
from ..util import get_path


@dataclass(frozen=True)
class CpConfig:
    """
    Configuration for CP tokenizer training.
    """
    # The latent dimension of the model
    hidden_dims: int = 512

    # The number of codebook entries
    num_embeddings: int = 8192

    # Number of attention heads
    n_heads: int = 8

    # Number of encoder blocks
    n_encoder_blocks: int = 8

    # Number of decoder blocks
    n_decoder_blocks: int = 8

    # Whether to use DCAE https://arxiv.org/abs/2504.00496
    use_dcae: bool = False

    # Dropout rate for attention and feed-forward layers
    dropout: float = 0.1

    # Temperature for softmax in DCAE
    temperature: float = 1.0

    # Batch size for training
    batch_size: int = 32


class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer that maps continuous representations to discrete codes.
    Supports both L2 distance and DCAE.
    """

    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25,
                 use_dcae=False, temperature=1.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.use_dcae = use_dcae
        self.temperature = temperature

        # Initialize codebook
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

        if use_dcae:
            # Additional projections for dcae based assignment
            self.query_proj = nn.Linear(embedding_dim, embedding_dim)
            self.key_proj = nn.Linear(embedding_dim, embedding_dim)
            # Learnable temperature parameter
            self.temperature_param = nn.Parameter(torch.ones(1) * temperature)

    def forward(self, inputs):
        # inputs shape: (batch, sequence_length, embedding_dim)
        batch_size, seq_len, _ = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)

        if self.use_dcae:
            # dcae based assignment (similarity-based)
            # Project inputs and codebook to query and key spaces
            queries = self.query_proj(flat_input)  # (batch*seq, embedding_dim)
            keys = self.key_proj(self.embedding.weight)  # (num_embeddings, embedding_dim)

            # Compute similarity scores (dot product)
            similarity_scores = torch.matmul(queries, keys.t()) / self.temperature_param

            # Get assignments using softmax (soft assignment during training)
            attention_weights = F.softmax(similarity_scores, dim=-1)

            # For hard assignment (inference), get argmax
            encoding_indices = torch.argmax(similarity_scores, dim=-1, keepdim=True)

            # Create one-hot encodings for straight-through estimator
            encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
            encodings.scatter_(1, encoding_indices, 1)

            # Soft quantization during training (for better gradients)
            quantized_soft = torch.matmul(attention_weights, self.embedding.weight)
            quantized_hard = torch.matmul(encodings, self.embedding.weight)

            # Reshape back
            quantized_soft = quantized_soft.view(batch_size, seq_len, self.embedding_dim)
            quantized_hard = quantized_hard.view(batch_size, seq_len, self.embedding_dim)

            # Compute losses
            # Commitment loss: encourage input to be close to selected codebook entry
            commitment_loss = F.mse_loss(quantized_hard.detach(), inputs)

            # Codebook loss: encourage codebook to be close to input
            codebook_loss = F.mse_loss(quantized_hard, inputs.detach())

            # Entropy regularization (encourage diverse codebook usage)
            avg_probs = torch.mean(attention_weights, dim=0)
            entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
            entropy_reg = -0.01 * entropy  # negative because we want to maximize entropy

            loss = codebook_loss + self.commitment_cost * commitment_loss + entropy_reg

            # Straight-through estimator with soft-to-hard transition
            quantized = inputs + (quantized_hard - inputs).detach()

            # Use attention weights for perplexity calculation
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        else:
            # Original L2 distance based assignment
            distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                         + torch.sum(self.embedding.weight**2, dim=1)
                         - 2 * torch.matmul(flat_input, self.embedding.weight.t()))

            # Get nearest codebook entries
            encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
            encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
            encodings.scatter_(1, encoding_indices, 1)

            # Quantize using codebook
            quantized = torch.matmul(encodings, self.embedding.weight).view(batch_size, seq_len, self.embedding_dim)

            # Compute losses
            e_latent_loss = F.mse_loss(quantized.detach(), inputs)
            q_latent_loss = F.mse_loss(quantized, inputs.detach())
            loss = q_latent_loss + self.commitment_cost * e_latent_loss

            # Straight-through estimator
            quantized = inputs + (quantized - inputs).detach()

            # Perplexity for monitoring
            avg_probs = torch.mean(encodings, dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        encoding_indices = encoding_indices.view(batch_size, seq_len)
        return quantized, loss, perplexity, encoding_indices


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention module.
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Linear transformations and split into heads
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = torch.matmul(attn_weights, v)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Final linear transformation
        output = self.w_o(context)

        return output


class TransformerBlock(nn.Module):
    """
    Transformer block with self-attention and feed-forward network.
    """

    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model

        self.attention = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Self-attention with residual connection
        attn_output = self.attention(x)
        x = self.norm1(x + attn_output)

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)

        return x


class VQVAEEncoder(nn.Module):
    """
    Encoder that projects input, applies self-attention, and quantizes.
    """

    def __init__(self, d1, d2, num_embeddings, n_attention_blocks=3, n_heads=8,
                 dropout=0.1, use_dcae=False, temperature=1.0):
        super().__init__()

        # Initial projection
        self.input_projection = nn.Linear(d1, d2)

        # Self-attention blocks
        self.attention_blocks = nn.ModuleList([
            TransformerBlock(d2, n_heads, dropout=dropout)
            for _ in range(n_attention_blocks)
        ])

        # Pre-quantization normalization
        self.pre_quant_norm = nn.LayerNorm(d2)

        # Vector quantization with dcae option
        self.quantizer = VectorQuantizer(
            num_embeddings, d2,
            use_dcae=use_dcae,
            temperature=temperature
        )

    def forward(self, x):
        # Project input from d1 to d2
        x = self.input_projection(x)

        # Apply self-attention blocks
        for block in self.attention_blocks:
            x = block(x)

        # Normalize before quantization
        x = self.pre_quant_norm(x)

        # Quantize
        quantized, vq_loss, perplexity, encoding_indices = self.quantizer(x)

        return quantized, vq_loss, perplexity, encoding_indices


class VQVAEDecoder(nn.Module):
    """
    Decoder that reconstructs the original input from quantized representations.
    """

    def __init__(self, d2, d1, n_attention_blocks=3, n_heads=8, dropout=0.1):
        super().__init__()

        # Self-attention blocks
        self.attention_blocks = nn.ModuleList([
            TransformerBlock(d2, n_heads, dropout=dropout)
            for _ in range(n_attention_blocks)
        ])

        # Output projection
        self.output_projection = nn.Sequential(
            nn.LayerNorm(d2),
            nn.Linear(d2, d1)
        )

    def forward(self, x):
        # Apply self-attention blocks
        for block in self.attention_blocks:
            x = block(x)

        # Project back to original dimension
        x = self.output_projection(x)

        return x


class VQVAE(nn.Module):
    """
    Complete VQ-VAE model with encoder and decoder.
    Supports both L2 distance and dcae based vector quantization.
    """

    def __init__(self, d1, d2, num_embeddings, n_encoder_blocks=3, n_decoder_blocks=3,
                 n_heads=8, dropout=0.1, commitment_cost=0.25,
                 use_dcae=False, temperature=1.0):
        super().__init__()

        self.encoder = VQVAEEncoder(
            d1, d2, num_embeddings,
            n_attention_blocks=n_encoder_blocks,
            n_heads=n_heads,
            dropout=dropout,
            use_dcae=use_dcae,
            temperature=temperature
        )

        self.decoder = VQVAEDecoder(
            d2, d1,
            n_attention_blocks=n_decoder_blocks,
            n_heads=n_heads,
            dropout=dropout
        )

        # Store commitment cost for loss calculation
        self.commitment_cost = commitment_cost
        self.use_dcae = use_dcae

    def forward(self, x):
        # Encode and quantize
        quantized, vq_loss, perplexity, encoding_indices = self.encoder(x)

        # Decode
        reconstructed = self.decoder(quantized)

        return reconstructed, quantized, vq_loss, perplexity, encoding_indices

    def compute_loss(self, x, reconstructed, vq_loss, beta=1.0):
        """
        Compute the total loss for VQ-VAE.

        Args:
            x: Original input tensor (batch_size, seq_len, d1)
            reconstructed: Reconstructed tensor (batch_size, seq_len, d1)
            vq_loss: Vector quantization loss from encoder
            beta: Weight for reconstruction loss

        Returns:
            Dictionary containing individual losses and total loss
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstructed, x)

        # Total loss
        total_loss = beta * recon_loss + vq_loss

        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'vq_loss': vq_loss
        }

    def encode(self, x):
        """
        Encode input to quantized representation.
        """
        quantized, _, _, encoding_indices = self.encoder(x)
        return quantized, encoding_indices

    def decode(self, quantized):
        """
        Decode from quantized representation.
        """
        return self.decoder(quantized)

    def decode_from_indices(self, indices):
        """
        Decode directly from codebook indices.

        Args:
            indices: Tensor of shape (batch_size, seq_len) containing codebook indices

        Returns:
            Reconstructed tensor of shape (batch_size, seq_len, d1)
        """
        # Get embeddings from codebook
        quantized = self.encoder.quantizer.embedding(indices)

        # Decode
        return self.decoder(quantized)


class CpDataset(Dataset):
    """Dataset for loading token sequences from cached files"""

    def __init__(self, files: list[str]):
        self.files = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        return self.get(idx)

    def get(self, idx: int) -> torch.Tensor:
        """
        Get a single item by index

        Args:
            idx: Index of the item

        Returns:
            - data_tensor: Tensor of shape (seq_len, input_dim)
        """
        file = self.files[idx]
        try:
            data = load_musicxml_tokens(file)
        except Exception as e:
            logging.error(f"Error loading {file}: {e}")
            return self.__getitem__(np.random.randint(0, len(self.files)))

        data_tensor = torch.from_numpy(data).float()
        return data_tensor


@cache
def get_token_dims():
    """Get the input dimension of the tokenized data"""
    from ..test import BACH_C_MAJOR_PRELUDE
    tokens = notes_to_tokens(musicxml_to_notes(get_path(XML_ROOT, BACH_C_MAJOR_PRELUDE)))
    return tokens.shape[-1]


def load_musicxml_tokens(file_path: str) -> np.ndarray:
    """Load and tokenize MusicXML file"""
    try:
        notes = musicxml_to_notes(file_path)
        tokens = notes_to_tokens(notes)
        return tokens
    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
        return np.array([]).reshape(0, get_token_dims())


def get_model(config: CpConfig) -> VQVAE:
    """
    Create a VQ-VAE model with the given configuration.

    Args:
        config: Configuration object containing model parameters

    Returns:
        VQVAE model instance
    """
    return VQVAE(
        d1=get_token_dims(),
        d2=config.hidden_dims,
        num_embeddings=config.num_embeddings,
        n_encoder_blocks=config.n_encoder_blocks,
        n_decoder_blocks=config.n_decoder_blocks,
        n_heads=config.n_heads,
        dropout=config.dropout,
        use_dcae=config.use_dcae,
        temperature=config.temperature
    )


if __name__ == "__main__":
    config = CpConfig()
    model = get_model(config)

    # Generate random input
    x = torch.randn(
        config.batch_size,
        128,  # Sequence length
        get_token_dims()  # Input dimension
    )
    print("\n" + "=" * 50)
    print("dcae-based VQ-VAE:")
    print("=" * 50)

    # Forward pass with dcae model
    reconstructed_ce, quantized_ce, vq_loss_ce, perplexity_ce, indices_ce = model(x)
    losses_ce = model.compute_loss(x, reconstructed_ce, vq_loss_ce)

    print(f"Input shape: {x.shape}")
    print(f"Quantized shape: {quantized_ce.shape}")
    print(f"Reconstructed shape: {reconstructed_ce.shape}")
    print(f"Encoding indices shape: {indices_ce.shape}")
    print(f"Total loss: {losses_ce['total_loss'].item():.4f}")
    print(f"Reconstruction loss: {losses_ce['recon_loss'].item():.4f}")
    print(f"VQ loss: {losses_ce['vq_loss'].item():.4f}")
    print(f"Perplexity: {perplexity_ce.item():.2f}")

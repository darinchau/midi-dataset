# Training script for CP tokenizer
# python -m src.model.cp_tokenizer_train

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import logging
from tqdm import tqdm
import wandb
from datetime import datetime
import json

# Assuming these imports from your existing code
from .cp_tokenizer import (
    CpConfig, CpDataset, get_model, get_token_dims,
    VQVAE, MultiHeadSelfAttention, TransformerBlock
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training the CP tokenizer."""

    # Model configuration
    model_config: CpConfig = field(default_factory=CpConfig)

    # Training hyperparameters
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 0.01
    max_epochs: int = 100
    gradient_clip: float = 1.0
    warmup_steps: int = 1000

    # Loss weights
    reconstruction_weight: float = 1.0

    # Data configuration
    train_data_path: str = "data/train"
    val_data_path: str = "data/val"
    num_workers: int = 4

    # Checkpointing
    checkpoint_dir: str = "checkpoints/cp_tokenizer"
    save_every_n_epochs: int = 5
    keep_last_n_checkpoints: int = 3

    # Logging
    use_wandb: bool = True
    wandb_project: str = "cp-tokenizer"
    wandb_run_name: Optional[str] = None
    log_every_n_steps: int = 10
    val_every_n_epochs: int = 1

    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4


class AttentionMask:
    """Utility class for creating attention masks."""

    @staticmethod
    def create_padding_mask(lengths: torch.Tensor, max_len: int, device: torch.device) -> torch.Tensor:
        """
        Create a padding mask for variable length sequences.

        Args:
            lengths: Tensor of sequence lengths (batch_size,)
            max_len: Maximum sequence length in the batch
            device: Device to create the mask on

        Returns:
            Mask tensor of shape (batch_size, max_len) where True indicates valid positions
        """
        batch_size = lengths.size(0)
        mask = torch.arange(max_len, device=device).expand(batch_size, max_len) < lengths.unsqueeze(1)
        return mask


class CPDataCollator:
    """Custom data collator for variable-length sequences."""

    def __init__(self, pad_value: float = 0.0):
        self.pad_value = pad_value

    def __call__(self, batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of variable length sequences.

        Args:
            batch: List of tensors with shape (seq_len, input_dim)

        Returns:
            Dictionary containing:
                - input: Padded tensor (batch_size, max_seq_len, input_dim)
                - attention_mask: Boolean mask (batch_size, max_seq_len)
                - lengths: Original sequence lengths (batch_size,)
        """
        # Get sequence lengths
        lengths = torch.tensor([x.size(0) for x in batch])

        # Find max length in batch
        max_len = lengths.max().item()

        # Get input dimension
        input_dim = batch[0].size(-1)

        # Create padded tensor
        batch_size = len(batch)
        padded = torch.full((batch_size, max_len, input_dim), self.pad_value)

        # Fill in the sequences
        for i, seq in enumerate(batch):
            seq_len = seq.size(0)
            padded[i, :seq_len] = seq

        # Create attention mask
        attention_mask = AttentionMask.create_padding_mask(lengths, max_len, padded.device)

        return {
            'input': padded,
            'attention_mask': attention_mask,
            'lengths': lengths
        }


class VQVAEWithMasking(VQVAE):
    """Extended VQVAE model with attention masking support."""

    def forward(self, x, attention_mask=None):
        """
        Forward pass with optional attention masking.

        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            attention_mask: Boolean mask (batch_size, seq_len)

        Returns:
            Same as parent class but with masking applied
        """
        # Store original mask for reconstruction loss
        self.current_mask = attention_mask

        # Encode with masking
        quantized, vq_loss, perplexity, encoding_indices = self.encoder_with_mask(x, attention_mask)

        # Decode with masking
        reconstructed = self.decoder_with_mask(quantized, attention_mask)

        return reconstructed, quantized, vq_loss, perplexity, encoding_indices

    def encoder_with_mask(self, x, attention_mask=None):
        """Encoder forward pass with attention masking."""
        # Project input
        x = self.encoder.input_projection(x)

        # Apply self-attention blocks with masking
        for block in self.encoder.attention_blocks:
            x = self._apply_block_with_mask(block, x, attention_mask)

        # Normalize before quantization
        x = self.encoder.pre_quant_norm(x)

        # Apply mask before quantization (zero out padding positions)
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)

        # Quantize
        quantized, vq_loss, perplexity, encoding_indices = self.encoder.quantizer(x)

        return quantized, vq_loss, perplexity, encoding_indices

    def decoder_with_mask(self, x, attention_mask=None):
        """Decoder forward pass with attention masking."""
        # Apply self-attention blocks with masking
        for block in self.decoder.attention_blocks:
            x = self._apply_block_with_mask(block, x, attention_mask)

        # Project back to original dimension
        x = self.decoder.output_projection(x)

        return x

    def _apply_block_with_mask(self, block, x, attention_mask=None):
        """Apply transformer block with optional attention masking."""
        if attention_mask is None:
            return block(x)

        # Modify attention computation to include mask
        # This is a simplified version - you might need to modify the attention module itself
        # for proper masking implementation
        batch_size, seq_len, _ = x.shape

        # Apply attention with mask
        attn_output = block.attention(x)  # This would need modification in actual implementation

        # Apply mask to attention output
        if attention_mask is not None:
            attn_output = attn_output * attention_mask.unsqueeze(-1)

        x = block.norm1(x + attn_output)

        # Feed-forward with residual connection
        ff_output = block.feed_forward(x)
        if attention_mask is not None:
            ff_output = ff_output * attention_mask.unsqueeze(-1)

        x = block.norm2(x + ff_output)

        return x

    def compute_loss_with_mask(self, x, reconstructed, vq_loss, attention_mask=None, beta=1.0):
        """
        Compute loss with attention masking.

        Args:
            x: Original input tensor (batch_size, seq_len, input_dim)
            reconstructed: Reconstructed tensor (batch_size, seq_len, input_dim)
            vq_loss: Vector quantization loss
            attention_mask: Boolean mask (batch_size, seq_len)
            beta: Weight for reconstruction loss

        Returns:
            Dictionary containing losses
        """
        if attention_mask is not None:
            # Apply mask to both input and reconstruction
            mask_expanded = attention_mask.unsqueeze(-1).float()
            x_masked = x * mask_expanded
            reconstructed_masked = reconstructed * mask_expanded

            # Compute MSE only on valid positions
            diff = (x_masked - reconstructed_masked) ** 2
            recon_loss = diff.sum() / mask_expanded.sum()
        else:
            recon_loss = F.mse_loss(reconstructed, x)

        # Total loss
        total_loss = beta * recon_loss + vq_loss

        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'vq_loss': vq_loss
        }


class Trainer:
    """Trainer class for VQ-VAE model."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Setup model
        self.model = self._create_model()

        # Setup data
        self.train_loader, self.val_loader = self._setup_data()

        # Setup optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        total_steps = len(self.train_loader) * config.max_epochs
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=config.min_learning_rate
        )

        # Setup logging
        self._setup_logging()

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def _create_model(self) -> VQVAEWithMasking:
        """Create and initialize the model."""
        base_model = get_model(self.config.model_config)

        # Wrap the model with masking support
        model = VQVAEWithMasking(
            d1=get_token_dims(),
            d2=self.config.model_config.hidden_dims,
            num_embeddings=self.config.model_config.num_embeddings,
            n_encoder_blocks=self.config.model_config.n_encoder_blocks,
            n_decoder_blocks=self.config.model_config.n_decoder_blocks,
            n_heads=self.config.model_config.n_heads,
            dropout=self.config.model_config.dropout,
            use_dcae=self.config.model_config.use_dcae,
            temperature=self.config.model_config.temperature
        )

        return model.to(self.device)

    def _setup_data(self):
        """Setup data loaders."""
        # Get training files
        train_files = self._get_files(self.config.train_data_path)
        val_files = self._get_files(self.config.val_data_path)

        logger.info(f"Found {len(train_files)} training files")
        logger.info(f"Found {len(val_files)} validation files")

        # Create datasets
        train_dataset = CpDataset(train_files)
        val_dataset = CpDataset(val_files)

        # Create data collator
        collator = CPDataCollator()

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.model_config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=collator,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.model_config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=collator,
            pin_memory=True
        )

        return train_loader, val_loader

    def _get_files(self, path: str) -> List[str]:
        """Get all MusicXML files from a directory."""
        data_path = Path(path)
        if not data_path.exists():
            raise ValueError(f"Data path {path} does not exist")

        # Get all XML files recursively
        files = list(data_path.glob("**/*.xml")) + list(data_path.glob("**/*.musicxml"))
        return [str(f) for f in files]

    def _setup_logging(self):
        """Setup logging and wandb."""
        # Create checkpoint directory
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        # Setup wandb
        if self.config.use_wandb:
            run_name = self.config.wandb_run_name or f"cp_tokenizer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wandb.init(
                project=self.config.wandb_project,
                name=run_name,
                config=self.config.__dict__
            )

    def train(self):
        """Main training loop."""
        logger.info("Starting training...")

        for epoch in range(self.config.max_epochs):
            self.epoch = epoch

            # Training epoch
            train_metrics = self._train_epoch()

            # Validation
            if (epoch + 1) % self.config.val_every_n_epochs == 0:
                val_metrics = self._validate()

                # Early stopping check
                if self._check_early_stopping(val_metrics['total_loss']):
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break

            # Save checkpoint
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self._save_checkpoint()

            # Log metrics
            self._log_metrics(train_metrics, val_metrics if 'val_metrics' in locals() else None)

        logger.info("Training completed!")

        # Save final model
        self._save_checkpoint(final=True)

    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        epoch_losses = {
            'total_loss': 0.0,
            'recon_loss': 0.0,
            'vq_loss': 0.0,
            'perplexity': 0.0
        }

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            inputs = batch['input'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            # Forward pass
            reconstructed, quantized, vq_loss, perplexity, indices = self.model(inputs, attention_mask)

            # Compute loss
            losses = self.model.compute_loss_with_mask(
                inputs, reconstructed, vq_loss, attention_mask,
                beta=self.config.reconstruction_weight
            )

            # Backward pass
            self.optimizer.zero_grad()
            losses['total_loss'].backward()

            # Gradient clipping
            clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)

            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()

            # Update metrics
            for key in epoch_losses:
                if key == 'perplexity':
                    epoch_losses[key] += perplexity.item()
                else:
                    epoch_losses[key] += losses.get(key, 0).item()

            # Update progress bar
            progress_bar.set_postfix({
                'loss': losses['total_loss'].item(),
                'recon': losses['recon_loss'].item(),
                'vq': losses['vq_loss'].item(),
                'perp': perplexity.item()
            })

            # Log to wandb
            if self.config.use_wandb and batch_idx % self.config.log_every_n_steps == 0:
                wandb.log({
                    'train/total_loss': losses['total_loss'].item(),
                    'train/recon_loss': losses['recon_loss'].item(),
                    'train/vq_loss': losses['vq_loss'].item(),
                    'train/perplexity': perplexity.item(),
                    'train/learning_rate': self.scheduler.get_last_lr()[0],
                    'step': self.global_step
                })

            self.global_step += 1

        # Average losses
        num_batches = len(self.train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    def _validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()

        val_losses = {
            'total_loss': 0.0,
            'recon_loss': 0.0,
            'vq_loss': 0.0,
            'perplexity': 0.0
        }

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                inputs = batch['input'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                # Forward pass
                reconstructed, quantized, vq_loss, perplexity, indices = self.model(inputs, attention_mask)

                # Compute loss
                losses = self.model.compute_loss_with_mask(
                    inputs, reconstructed, vq_loss, attention_mask,
                    beta=self.config.reconstruction_weight
                )

                # Update metrics
                for key in val_losses:
                    if key == 'perplexity':
                        val_losses[key] += perplexity.item()
                    else:
                        val_losses[key] += losses.get(key, 0).item()

        # Average losses
        num_batches = len(self.val_loader)
        for key in val_losses:
            val_losses[key] /= num_batches

        # Log to wandb
        if self.config.use_wandb:
            wandb.log({
                'val/total_loss': val_losses['total_loss'],
                'val/recon_loss': val_losses['recon_loss'],
                'val/vq_loss': val_losses['vq_loss'],
                'val/perplexity': val_losses['perplexity'],
                'epoch': self.epoch
            })

        logger.info(f"Validation - Loss: {val_losses['total_loss']:.4f}, "
                    f"Recon: {val_losses['recon_loss']:.4f}, "
                    f"VQ: {val_losses['vq_loss']:.4f}, "
                    f"Perplexity: {val_losses['perplexity']:.2f}")

        return val_losses

    def _check_early_stopping(self, val_loss: float) -> bool:
        """Check if early stopping should be triggered."""
        if val_loss < self.best_val_loss - self.config.early_stopping_min_delta:
            self.best_val_loss = val_loss
            self.patience_counter = 0

            # Save best model
            self._save_checkpoint(best=True)
        else:
            self.patience_counter += 1

        return self.patience_counter >= self.config.early_stopping_patience

    def _save_checkpoint(self, best: bool = False, final: bool = False):
        """Save model checkpoint."""
        if best:
            checkpoint_path = os.path.join(self.config.checkpoint_dir, "best_model.pt")
        elif final:
            checkpoint_path = os.path.join(self.config.checkpoint_dir, "final_model.pt")
        else:
            checkpoint_path = os.path.join(
                self.config.checkpoint_dir,
                f"checkpoint_epoch_{self.epoch}.pt"
            )

        torch.save({
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config.__dict__
        }, checkpoint_path)

        logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Clean up old checkpoints
        if not best and not final:
            self._cleanup_checkpoints()

    def _cleanup_checkpoints(self):
        """Remove old checkpoints, keeping only the last N."""
        checkpoint_files = sorted([
            f for f in os.listdir(self.config.checkpoint_dir)
            if f.startswith('checkpoint_epoch_')
        ])

        if len(checkpoint_files) > self.config.keep_last_n_checkpoints:
            for f in checkpoint_files[:-self.config.keep_last_n_checkpoints]:
                os.remove(os.path.join(self.config.checkpoint_dir, f))

    def _log_metrics(self, train_metrics: Dict[str, float], val_metrics: Optional[Dict[str, float]] = None):
        """Log metrics to console."""
        log_str = f"Epoch {self.epoch} - Train Loss: {train_metrics['total_loss']:.4f}"
        if val_metrics:
            log_str += f", Val Loss: {val_metrics['total_loss']:.4f}"
        logger.info(log_str)


def main():
    """Main training entry point."""
    # Parse arguments (you can add argparse here if needed)
    config = TrainingConfig()

    # Create trainer
    trainer = Trainer(config)

    # Start training
    trainer.train()

    # Close wandb
    if config.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()

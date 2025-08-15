#!/usr/bin/env python3
"""
Training script for VQ-VAE model for MIDI tokenization.
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
import json
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split

import wandb
from tqdm import tqdm
import logging
from typing import Dict, Optional, Tuple

# Import your model components (adjust import paths as needed)
from src.model.cp_tokenizer import (
    CpConfig, VQVAE, CpDataset, CpDataCollator,
    get_model, create_dataloader, train_step, get_token_dims
)

from src.util import get_all_xml_paths


def setup_logging(log_dir: Path):
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_train_val_dataloaders(config: CpConfig, val_split: float = 0.1) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.

    Args:
        config: Model configuration
        val_split: Fraction of data to use for validation

    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    files = get_all_xml_paths()
    total_files = len(files)
    val_size = int(total_files * val_split)
    train_size = total_files - val_size

    # Shuffle files and split
    random.shuffle(files)
    train_files = files[:train_size]
    val_files = files[train_size:]

    # Create datasets
    train_dataset = CpDataset(train_files, max_seq_length=config.max_seq_length, on_too_long='truncate')
    val_dataset = CpDataset(val_files, max_seq_length=config.max_seq_length, on_too_long='truncate')

    # Create collator
    collator = CpDataCollator(pad_value=0.0)

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True
    )

    return train_dataloader, val_dataloader


def validate(model: VQVAE, val_dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
    """
    Run validation loop.

    Args:
        model: VQVAE model
        val_dataloader: Validation dataloader
        device: Device to run on

    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_vq_loss = 0
    total_perplexity = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validation", leave=False):
            # Move batch to device
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Forward pass
            reconstructed, quantized, vq_loss, perplexity, encoding_indices = model(inputs, attention_mask)

            # Compute losses
            losses = model.compute_loss(inputs, reconstructed, vq_loss, attention_mask)

            # Accumulate metrics
            total_loss += losses['total_loss'].item()
            total_recon_loss += losses['recon_loss'].item()
            total_vq_loss += losses['vq_loss'].item()
            total_perplexity += perplexity.item()
            num_batches += 1

    # Compute averages
    metrics = {
        'val_loss': total_loss / num_batches,
        'val_recon_loss': total_recon_loss / num_batches,
        'val_vq_loss': total_vq_loss / num_batches,
        'val_perplexity': total_perplexity / num_batches
    }

    return metrics


def save_checkpoint(model: VQVAE, optimizer: torch.optim.Optimizer,
                    scheduler: Optional[object], epoch: int,
                    metrics: Dict[str, float], checkpoint_dir: Path,
                    is_best: bool = False):
    """
    Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Learning rate scheduler state (optional)
        epoch: Current epoch
        metrics: Current metrics
        checkpoint_dir: Directory to save checkpoints
        is_best: Whether this is the best model so far
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    # Save regular checkpoint
    checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch:04d}.pt'
    torch.save(checkpoint, checkpoint_path)

    # Save best model
    if is_best:
        best_path = checkpoint_dir / 'best_model.pt'
        torch.save(checkpoint, best_path)

    # Save latest checkpoint (for resuming)
    latest_path = checkpoint_dir / 'latest_checkpoint.pt'
    torch.save(checkpoint, latest_path)

    logging.info(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint(checkpoint_path: Path, model: VQVAE,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    scheduler: Optional[object] = None) -> int:
    """
    Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)

    Returns:
        Starting epoch number
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    logging.info(f"Loaded checkpoint from {checkpoint_path}")
    return checkpoint.get('epoch', 0) + 1


def train(config: CpConfig, args: argparse.Namespace):
    """
    Main training loop.

    Args:
        config: Model configuration
        args: Command line arguments
    """
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    # Create directories
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / 'checkpoints'
    log_dir = output_dir / 'logs'

    # Setup logging
    logger = setup_logging(log_dir)
    logger.info(f"Starting training with config: {config}")
    logger.info(f"Using device: {device}")

    # Initialize wandb
    run_name = args.run_name or f"vqvae_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    wandb_config = {
        **config.__dict__,
        'run_name': run_name,
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs,
        'val_split': args.val_split,
        'seed': args.seed,
        'gradient_clip': args.gradient_clip,
        'warmup_epochs': args.warmup_epochs,
        'scheduler': args.scheduler,
        'beta': args.beta
    }

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config=wandb_config,
        tags=args.wandb_tags.split(',') if args.wandb_tags else None,
        mode='online' if not args.no_wandb else 'disabled'
    )

    # Create model
    model = get_model(config)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dataloaders
    train_dataloader, val_dataloader = create_train_val_dataloaders(config, args.val_split)
    logger.info(f"Train batches: {len(train_dataloader)}, Val batches: {len(val_dataloader)}")

    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # Create learning rate scheduler
    scheduler = None
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.num_epochs,
            eta_min=args.learning_rate * 0.01
        )
    elif args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume_from:
        start_epoch = load_checkpoint(
            Path(args.resume_from),
            model, optimizer, scheduler
        )

    # Training loop
    best_val_loss = float('inf')
    global_step = 0

    for epoch in range(start_epoch, args.num_epochs):
        epoch_start_time = datetime.now()

        # Training phase
        model.train()
        train_losses = {
            'train_loss': 0,
            'train_recon_loss': 0,
            'train_vq_loss': 0,
            'train_perplexity': 0
        }
        num_train_batches = 0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Learning rate warmup
            if epoch < args.warmup_epochs:
                warmup_factor = (epoch * len(train_dataloader) + batch_idx + 1) / (args.warmup_epochs * len(train_dataloader))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.learning_rate * warmup_factor

            # Forward pass
            optimizer.zero_grad()
            reconstructed, quantized, vq_loss, perplexity, encoding_indices = model(inputs, attention_mask)

            # Compute losses with beta weighting
            losses = model.compute_loss(inputs, reconstructed, vq_loss, attention_mask, beta=args.beta)

            # Backward pass
            losses['total_loss'].backward()

            # Gradient clipping
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)

            optimizer.step()

            # Accumulate losses
            train_losses['train_loss'] += losses['total_loss'].item()
            train_losses['train_recon_loss'] += losses['recon_loss'].item()
            train_losses['train_vq_loss'] += losses['vq_loss'].item()
            train_losses['train_perplexity'] += perplexity.item()
            num_train_batches += 1
            global_step += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': losses['total_loss'].item(),
                'recon': losses['recon_loss'].item(),
                'vq': losses['vq_loss'].item(),
                'ppl': perplexity.item()
            })

            # Log to wandb
            if global_step % args.log_interval == 0:
                wandb.log({
                    'train/loss': losses['total_loss'].item(),
                    'train/recon_loss': losses['recon_loss'].item(),
                    'train/vq_loss': losses['vq_loss'].item(),
                    'train/perplexity': perplexity.item(),
                    'train/learning_rate': optimizer.param_groups[0]['lr'],
                    'global_step': global_step
                })

        # Average training losses
        for key in train_losses:
            train_losses[key] /= num_train_batches

        # Validation phase
        val_metrics = validate(model, val_dataloader, device)

        # Update learning rate scheduler
        if scheduler is not None:
            if args.scheduler == 'plateau':
                scheduler.step(val_metrics['val_loss'])
            else:
                scheduler.step()

        # Log epoch metrics
        epoch_metrics = {**train_losses, **val_metrics}
        epoch_metrics['epoch'] = epoch + 1
        epoch_metrics['learning_rate'] = optimizer.param_groups[0]['lr']

        wandb.log(epoch_metrics)

        # Check if best model
        is_best = val_metrics['val_loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['val_loss']
            logger.info(f"New best model! Val loss: {best_val_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0 or is_best:
            save_checkpoint(
                model, optimizer, scheduler, epoch + 1,
                epoch_metrics, checkpoint_dir, is_best
            )

        # Log epoch summary
        epoch_time = (datetime.now() - epoch_start_time).total_seconds()
        logger.info(
            f"Epoch {epoch+1}/{args.num_epochs} - "
            f"Train Loss: {train_losses['train_loss']:.4f}, "
            f"Val Loss: {val_metrics['val_loss']:.4f}, "
            f"Val Perplexity: {val_metrics['val_perplexity']:.2f}, "
            f"Time: {epoch_time:.1f}s"
        )

        # Early stopping check
        if args.early_stopping_patience > 0:
            if epoch - best_val_loss > args.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

    # Final save
    save_checkpoint(
        model, optimizer, scheduler, args.num_epochs,
        epoch_metrics, checkpoint_dir, is_best=False
    )

    # Close wandb
    wandb.finish()

    logger.info("Training completed!")


def main():
    parser = argparse.ArgumentParser(description="Train VQ-VAE model for MIDI tokenization")

    # Model configuration
    parser.add_argument('--hidden_dims', type=int, default=512,
                        help='Hidden dimension size')
    parser.add_argument('--num_embeddings', type=int, default=8192,
                        help='Number of codebook entries')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--n_encoder_blocks', type=int, default=8,
                        help='Number of encoder blocks')
    parser.add_argument('--n_decoder_blocks', type=int, default=8,
                        help='Number of decoder blocks')
    parser.add_argument('--use_dcae', action='store_true',
                        help='Use DCAE for vector quantization')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for DCAE')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')

    # Training configuration
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--max_seq_length', type=int, default=2048,
                        help='Maximum sequence length')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                        help='Gradient clipping value')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Weight for reconstruction loss')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Number of warmup epochs')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['none', 'cosine', 'plateau'],
                        help='Learning rate scheduler')

    # Data configuration
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation split fraction')

    # Logging configuration
    parser.add_argument('--output_dir', type=str, default='outputs/vqvae',
                        help='Output directory')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Logging interval (steps)')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Checkpoint save interval (epochs)')

    # Wandb configuration
    parser.add_argument('--run_name', type=str, default=None,
                        help='Wandb run name')
    parser.add_argument('--wandb_project', type=str, default='vqvae-midi',
                        help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='Wandb entity')
    parser.add_argument('--wandb_tags', type=str, default=None,
                        help='Wandb tags (comma-separated)')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable wandb logging')

    # Other configuration
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--early_stopping_patience', type=int, default=0,
                        help='Early stopping patience (0 to disable)')

    args = parser.parse_args()

    # Create configuration
    config = CpConfig(
        hidden_dims=args.hidden_dims,
        num_embeddings=args.num_embeddings,
        n_heads=args.n_heads,
        n_encoder_blocks=args.n_encoder_blocks,
        n_decoder_blocks=args.n_decoder_blocks,
        use_dcae=args.use_dcae,
        dropout=args.dropout,
        temperature=args.temperature,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length
    )

    # Run training
    train(config, args)


if __name__ == '__main__':
    main()

"""
Training script for VQ-VAE model for MIDI tokenization.
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
import json
import random
import numpy as np
from dataclasses import dataclass, asdict, field
from typing import Optional, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler

import wandb
from tqdm import tqdm
import logging
from typing import Dict, Optional, Tuple

from src.model.cp_tokenizer import CpConfig, VQVAE, CpDataset, CpDataCollator, get_model
from src.utils.model import print_model_hierarchy
from src.utils.gpu_monitor import wait_until_gpu_drops_below_temp
from src.utils import clear_cuda


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for training the VQ-VAE model."""

    # Model configuration
    hidden_dims: int
    num_embeddings: int
    n_heads: int
    n_encoder_blocks: int
    n_decoder_blocks: int
    use_dcae: bool
    temperature: float
    dropout: float
    use_checkpoint: bool  # Whether to use gradient checkpointing

    # Training configuration
    batch_size: int
    grad_accumulation_steps: int
    max_seq_length: int
    num_epochs: int
    learning_rate: float
    weight_decay: float
    gradient_clip: float
    beta: float
    warmup_steps: int
    scheduler: str  # Options: 'none', 'cosine', 'plateau'
    num_workers: int

    # Data configuration
    val_split: float
    val_size: Optional[int]
    val_interval: int
    max_train_steps: Optional[int]

    # Logging configuration
    output_dir: str
    log_interval: int
    save_interval: int

    # Wandb configuration
    run_name: Optional[str]
    wandb_project: str
    wandb_entity: Optional[str]
    wandb_tags: Optional[List[str]]
    no_wandb: bool

    # Mixed precision configuration
    mixed_precision: str  # Options: 'no', 'fp16', 'bf16'

    # Other configuration
    seed: int
    resume_from: Optional[str]
    limit_gpu_temp: int | str  # Temperature in Celsius to wait for GPU to cool down

    def to_cp_config(self) -> CpConfig:
        """Convert to CpConfig for model initialization."""
        return CpConfig(
            hidden_dims=self.hidden_dims,
            num_embeddings=self.num_embeddings,
            n_heads=self.n_heads,
            n_encoder_blocks=self.n_encoder_blocks,
            n_decoder_blocks=self.n_decoder_blocks,
            use_dcae=self.use_dcae,
            dropout=self.dropout,
            temperature=self.temperature,
            batch_size=self.batch_size,
            max_seq_length=self.max_seq_length,
            use_checkpoint=self.use_checkpoint,
        )

    def to_dict(self) -> dict:
        """Convert config to dictionary for wandb logging."""
        return asdict(self)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> TrainingConfig:
        """Create TrainingConfig from parsed arguments."""
        # Handle wandb_tags separately since it needs to be converted from string to list
        wandb_tags = None
        if args.wandb_tags:
            wandb_tags = args.wandb_tags.split(',')

        return cls(
            hidden_dims=args.hidden_dims,
            num_embeddings=args.num_embeddings,
            n_heads=args.n_heads,
            n_encoder_blocks=args.n_encoder_blocks,
            n_decoder_blocks=args.n_decoder_blocks,
            use_dcae=args.use_dcae,
            temperature=args.temperature,
            dropout=args.dropout,
            use_checkpoint=args.use_checkpoint,
            batch_size=args.batch_size,
            grad_accumulation_steps=args.grad_accumulation_steps,
            max_seq_length=args.max_seq_length,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            gradient_clip=args.gradient_clip,
            beta=args.beta,
            warmup_steps=args.warmup_steps,
            scheduler=args.scheduler,
            num_workers=args.num_workers,
            val_split=args.val_split,
            val_size=args.val_size if args.val_size is not None else None,
            val_interval=args.val_interval,
            max_train_steps=args.max_train_steps if args.max_train_steps is not None else None,
            output_dir=args.output_dir,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            run_name=args.run_name,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_tags=wandb_tags,
            no_wandb=args.no_wandb,
            mixed_precision=args.mixed_precision,
            seed=args.seed,
            resume_from=args.resume_from,
            limit_gpu_temp=args.limit_gpu_temp
        )

    def save(self, path: Path):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> TrainingConfig:
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


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

    # Set subloggers in src to warn and error only
    for name in ['src.model', 'src.util', 'src.extract', 'src.model.cp_tokenizer']:
        logging.getLogger(name).setLevel(logging.WARNING)

    return logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_train_val_dataloaders(config: TrainingConfig) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.

    Args:
        config: Training configuration

    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    from src.utils import get_all_xml_paths

    total_files = len(get_all_xml_paths())
    val_size = int(total_files * config.val_split) if config.val_size is None else config.val_size
    train_size = total_files - val_size

    # Shuffle files and split
    train_split = (config.seed, 0, train_size)
    val_split = (config.seed, train_size, train_size + val_size)

    # Create datasets
    train_dataset = CpDataset(
        split=train_split,
        max_seq_length=config.max_seq_length,
        on_too_long='skip'
    )
    val_dataset = CpDataset(
        split=val_split,
        max_seq_length=config.max_seq_length,
        on_too_long='skip'
    )

    num_workers = config.num_workers

    # Create collator
    collator = CpDataCollator(pad_value=0.0)

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_dataloader, val_dataloader


def validate(model: VQVAE, val_dataloader: DataLoader, device: torch.device, use_amp: bool) -> Dict[str, float]:
    """
    Run validation loop.

    Args:
        model: VQVAE model
        val_dataloader: Validation dataloader
        device: Device to run on
        use_amp: Whether to use automatic mixed precision

    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_vq_loss = 0
    total_perplexity = 0
    num_batches = 0

    # Determine dtype for autocast
    amp_dtype = torch.float16 if use_amp else None

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validation", leave=False):
            # Move to device
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Forward pass with or without autocast
            if use_amp and amp_dtype is not None:
                with autocast(device_type='cuda', dtype=amp_dtype):
                    reconstructed, quantized, vq_loss, perplexity, encoding_indices = model(inputs, attention_mask)
                    losses = model.compute_loss(inputs, reconstructed, vq_loss, attention_mask)
            else:
                reconstructed, quantized, vq_loss, perplexity, encoding_indices = model(inputs, attention_mask)
                losses = model.compute_loss(inputs, reconstructed, vq_loss, attention_mask)

            # Accumulate metrics
            total_loss += losses['total_loss'].item()
            total_recon_loss += losses['recon_loss'].item()
            total_vq_loss += losses['vq_loss'].item()
            total_perplexity += perplexity.item()
            num_batches += 1

    # Compute averages
    metrics = {
        'val/loss': total_loss / num_batches,
        'val/recon_loss': total_recon_loss / num_batches,
        'val/vq_loss': total_vq_loss / num_batches,
        'val/perplexity': total_perplexity / num_batches
    }

    return metrics


def save_checkpoint(
    model: VQVAE,
    optimizer: torch.optim.Optimizer,
    scheduler,
    training_step: int,
    global_step: int,
    checkpoint_dir: Path,
    config: TrainingConfig,
    scaler: Optional[GradScaler] = None,
):
    """
    Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Learning rate scheduler state (optional)
        training_step: Current training step
        global_step: Current global step
        checkpoint_dir: Directory to save checkpoints
        config: Training configuration
        scaler: GradScaler state (optional)
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'training_step': training_step,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config.to_dict(),
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()

    # Save regular checkpoint
    checkpoint_path = checkpoint_dir / f'model_{global_step:05d}.pt'
    torch.save(checkpoint, checkpoint_path)

    # Save config alongside checkpoint
    config.save(checkpoint_dir / 'config.json')

    logging.info(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint(checkpoint_path: Path, model: VQVAE,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    scheduler=None, scaler: Optional[GradScaler] = None) -> Tuple[int, int]:
    """
    Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        scaler: GradScaler to load state into (optional)

    Returns:
        Tuple of (training step, global step)
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    logging.info(f"Loaded checkpoint from {checkpoint_path}")

    training_step = checkpoint.get('training_step', 0)
    global_step = checkpoint.get('global_step', 0)

    return training_step, global_step


def get_warmup_lr(base_lr: float, current_step: int, warmup_steps: int) -> float:
    """
    Calculate learning rate during warmup period.

    Args:
        base_lr: Base learning rate after warmup
        current_step: Current training step
        warmup_steps: Total number of warmup steps

    Returns:
        Adjusted learning rate
    """
    if current_step < warmup_steps:
        return base_lr * (current_step / warmup_steps)
    return base_lr


def train(config: TrainingConfig):
    """
    Main training loop.

    Args:
        config: Training configuration containing all hyperparameters
    """
    # Setup
    set_seed(config.seed)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create directories
    output_dir = Path(config.output_dir)
    checkpoint_dir = output_dir / 'checkpoints'
    log_dir = output_dir / 'logs'

    # Setup logging
    logger = setup_logging(log_dir)
    logger.info(f"Starting training with config:")
    logger.info(json.dumps(config.to_dict(), indent=2))
    logger.info(f"Using device: {device}")
    logger.info(f"Mixed precision: {config.mixed_precision}")

    # Save config
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir / 'config.json')

    # Initialize wandb
    if not config.no_wandb:
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.run_name,
            tags=config.wandb_tags,
            config=config.to_dict()
        )

    # Create model
    cp_config = config.to_cp_config()
    model = get_model(cp_config)
    model.to(device)
    model.train()

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dataloaders
    train_dataloader, val_dataloader = create_train_val_dataloaders(config)
    logger.info(f"Train batches: {len(train_dataloader)}, Val batches: {len(val_dataloader)}")

    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # Create learning rate scheduler
    scheduler = None
    if config.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.num_epochs,
            eta_min=config.learning_rate * 0.01
        )
    elif config.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
        )

    # Setup mixed precision
    use_amp = config.mixed_precision != 'no'
    scaler = GradScaler() if use_amp else None

    # Determine dtype for autocast
    if config.mixed_precision == 'bf16':
        amp_dtype = torch.bfloat16
    elif config.mixed_precision == 'fp16':
        amp_dtype = torch.float16
    else:
        amp_dtype = None

    # Load checkpoint if resuming
    training_step = 0
    global_step = 0
    if config.resume_from:
        training_step, global_step = load_checkpoint(
            Path(config.resume_from),
            model, optimizer, scheduler, scaler
        )

    # Print model hierarchy
    print_model_hierarchy(model)

    # Training loop
    progress_bar = tqdm(
        train_dataloader,
        desc=f"Training model...",
    )

    stop_training = False
    accumulation_steps = 0

    while not stop_training:
        for _, batch in enumerate(progress_bar):
            wait_until_gpu_drops_below_temp(config.limit_gpu_temp)

            # Move to device
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Learning rate warmup based on global steps
            if global_step < config.warmup_steps:
                warmup_lr = get_warmup_lr(config.learning_rate, global_step, config.warmup_steps)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr

            training_step += 1

            # Forward pass with or without autocast
            if use_amp and amp_dtype is not None:
                with autocast(device_type='cuda', dtype=amp_dtype):
                    reconstructed, quantized, vq_loss, perplexity, encoding_indices = model(inputs, attention_mask)
                    losses = model.compute_loss(inputs, reconstructed, vq_loss, attention_mask, beta=config.beta)
                    # Scale loss for gradient accumulation
                    loss = losses['total_loss'] / config.grad_accumulation_steps
            else:
                reconstructed, quantized, vq_loss, perplexity, encoding_indices = model(inputs, attention_mask)
                losses = model.compute_loss(inputs, reconstructed, vq_loss, attention_mask, beta=config.beta)
                # Scale loss for gradient accumulation
                loss = losses['total_loss'] / config.grad_accumulation_steps

            # Backward pass
            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accumulation_steps += 1

            # Update weights after gradient accumulation
            if accumulation_steps >= config.grad_accumulation_steps:
                if use_amp and scaler is not None:
                    # Unscale gradients and clip
                    scaler.unscale_(optimizer)
                    if config.gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                    # Step optimizer
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Gradient clipping
                    if config.gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                    optimizer.step()

                clear_cuda()
                optimizer.zero_grad()
                global_step += 1
                accumulation_steps = 0

            # Update progress bar
            current_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': losses['total_loss'].detach().item(),
                'lr': f'{current_lr:.2e}',
                'global_step': global_step
            })

            metrics = {
                'train/loss': losses['total_loss'].item(),
                'train/recon_loss': losses['recon_loss'].item(),
                'train/vq_loss': losses['vq_loss'].item(),
                'train/perplexity': perplexity.item(),
                'train/learning_rate': optimizer.param_groups[0]['lr'],
                'global_step': global_step
            }

            # Log to wandb
            if training_step % config.log_interval == 0 and not config.no_wandb:
                wandb.log(metrics, step=training_step)

            # Validation phase
            if global_step % config.val_interval == 0 and global_step > 0 and accumulation_steps == 0:
                val_metrics = validate(model, val_dataloader, device, use_amp)
                logger.info(f"Validation metrics at step {global_step}: {val_metrics}")
                if not config.no_wandb:
                    wandb.log(val_metrics, step=training_step)
                # Set model back to training mode
                model.train()

            # Save checkpoint
            if global_step % config.save_interval == 0 and global_step > 0 and accumulation_steps == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    training_step=training_step,
                    global_step=global_step,
                    checkpoint_dir=checkpoint_dir,
                    config=config,
                    scaler=scaler,
                )

            # Update learning rate scheduler (after warmup is complete)
            if scheduler is not None and global_step >= config.warmup_steps and accumulation_steps == 0:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(metrics['train/loss'])
                else:
                    scheduler.step()

            # Check max training steps
            if config.max_train_steps is not None and global_step >= config.max_train_steps:
                stop_training = True
                break

    # Final save
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_step=training_step,
        global_step=global_step,
        checkpoint_dir=checkpoint_dir,
        config=config,
        scaler=scaler,
    )

    # End wandb run
    if not config.no_wandb:
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
    parser.add_argument('--use_checkpoint', action='store_true',
                        help='Use gradient checkpointing to save memory')

    # Training configuration
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size')
    parser.add_argument('--grad_accumulation_steps', type=int, default=32,
                        help='Gradient accumulation steps')
    parser.add_argument('--max_seq_length', type=int, default=8192,
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
    parser.add_argument('--warmup_steps', type=int, default=1000,
                        help='Number of warmup steps')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['none', 'cosine', 'plateau'],
                        help='Learning rate scheduler')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')

    # Data configuration
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation split fraction')
    parser.add_argument('--val_size', type=int, default=1000,
                        help='Size of validation set (taken after split, to ensure fixed size)')
    parser.add_argument('--val_interval', type=int, default=500,
                        help='Validation interval (steps)')
    parser.add_argument('--max_train_steps', type=int, default=None,
                        help='Maximum number of training steps (overrides num_epochs if set)')

    # Logging configuration
    parser.add_argument('--output_dir', type=str, default='outputs/vqvae',
                        help='Output directory')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Logging interval (steps)')
    parser.add_argument('--save_interval', type=int, default=1000,
                        help='Checkpoint save interval (steps)')

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

    # Mixed precision configuration
    parser.add_argument('--mixed_precision', type=str, default='fp16',
                        choices=['no', 'fp16', 'bf16'],
                        help='Mixed precision training mode')

    # Other configuration
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--limit_gpu_temp', default=100,
                        help='Temperature in Celsius to wait for GPU to cool down before training '
                        '(defaults to 100 which if your GPU is above 100C you have bigger problems) '
                        'If input is a string it will be interpreted as a path to a JSON file '
                        'where the key "max_temp" will be used as the temperature threshold.'
                        'This is useful for running with dynamic GPU load.')

    # Config file support
    parser.add_argument('--config_file', type=str, default=None,
                        help='Load configuration from JSON file')

    args = parser.parse_args()

    # Load config from file if provided, otherwise create from args
    if args.config_file:
        config = TrainingConfig.load(Path(args.config_file))
        # Override with any command line arguments
        for key, value in vars(args).items():
            if key != 'config_file' and value is not None:
                setattr(config, key, value)
    else:
        config = TrainingConfig.from_args(args)

    # Run training
    train(config)


if __name__ == '__main__':
    main()

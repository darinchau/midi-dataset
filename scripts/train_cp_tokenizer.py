"""
Training script for VQ-VAE model for MIDI tokenization with HF Accelerate support.
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

# Import Accelerate
from accelerate import Accelerator
from accelerate.utils import set_seed as accelerate_set_seed

import wandb
from tqdm import tqdm
import logging
from typing import Dict, Optional, Tuple

from src.model.cp_tokenizer import CpConfig, VQVAE, CpDataset, CpDataCollator, get_model


@dataclass
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

    # Data configuration
    val_split: float
    val_size: Optional[int]

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

    # Accelerate configuration
    mixed_precision: str  # Options: 'no', 'fp16', 'bf16'

    # Other configuration
    seed: int
    resume_from: Optional[str]
    early_stopping_patience: int

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
            max_seq_length=self.max_seq_length
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
            val_split=args.val_split,
            val_size=args.val_size if args.val_size is not None else None,
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
            early_stopping_patience=args.early_stopping_patience,
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


def setup_logging(log_dir: Path, accelerator: Accelerator):
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)

    # Only log on main process
    if accelerator.is_main_process:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'training.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )

        # Set subloggers in src to warn and error only
        for name in ['src.model', 'src.util', 'src.extract']:
            logging.getLogger(name).setLevel(logging.WARNING)
    else:
        logging.basicConfig(level=logging.ERROR)

    return logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_train_val_dataloaders(config: TrainingConfig) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.

    Args:
        config: Training configuration

    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    from src.util import get_all_xml_paths

    files = get_all_xml_paths()
    total_files = len(files)
    val_size = int(total_files * config.val_split)
    train_size = total_files - val_size

    # Shuffle files and split
    random.shuffle(files)
    train_files = files[:train_size]
    val_files = files[train_size:]

    # Create datasets
    train_dataset = CpDataset(
        train_files,
        max_seq_length=config.max_seq_length,
        on_too_long='skip'
    )
    val_dataset = CpDataset(
        val_files,
        max_seq_length=config.max_seq_length,
        on_too_long='skip'
    )

    num_workers = 0

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


def validate(model: VQVAE, val_dataloader: DataLoader, accelerator: Accelerator) -> Dict[str, float]:
    """
    Run validation loop.

    Args:
        model: VQVAE model
        val_dataloader: Validation dataloader
        accelerator: Accelerator instance

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
        for batch in tqdm(val_dataloader, desc="Validation", leave=False, disable=not accelerator.is_local_main_process):
            # No need to move to device, accelerator handles it
            inputs = batch['input_ids']
            attention_mask = batch['attention_mask']

            # Forward pass
            reconstructed, quantized, vq_loss, perplexity, encoding_indices = model(inputs, attention_mask)

            # Compute losses
            losses = model.compute_loss(inputs, reconstructed, vq_loss, attention_mask)

            # Gather losses across all processes
            losses_gathered = accelerator.gather_for_metrics({
                'total_loss': losses['total_loss'],
                'recon_loss': losses['recon_loss'],
                'vq_loss': losses['vq_loss'],
                'perplexity': perplexity
            })

            # Accumulate metrics
            total_loss += losses_gathered['total_loss'].mean().item()  # type: ignore
            total_recon_loss += losses_gathered['recon_loss'].mean().item()  # type: ignore
            total_vq_loss += losses_gathered['vq_loss'].mean().item()  # type: ignore
            total_perplexity += losses_gathered['perplexity'].mean().item()  # type: ignore
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
                    scheduler, epoch: int, global_step: int,
                    metrics: Dict[str, float], checkpoint_dir: Path,
                    config: TrainingConfig, accelerator: Accelerator,
                    is_best: bool = False):
    """
    Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Learning rate scheduler state (optional)
        epoch: Current epoch
        global_step: Current global step
        metrics: Current metrics
        checkpoint_dir: Directory to save checkpoints
        config: Training configuration
        accelerator: Accelerator instance
        is_best: Whether this is the best model so far
    """
    # Only save on main process
    if not accelerator.is_main_process:
        return

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Unwrap model if wrapped by accelerator
    unwrapped_model = accelerator.unwrap_model(model)

    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': unwrapped_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config.to_dict()
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    # Save regular checkpoint
    checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch:04d}.pt'
    accelerator.save(checkpoint, checkpoint_path)

    # Save best model
    if is_best:
        best_path = checkpoint_dir / 'best_model.pt'
        accelerator.save(checkpoint, best_path)

    # Save latest checkpoint (for resuming)
    latest_path = checkpoint_dir / 'latest_checkpoint.pt'
    accelerator.save(checkpoint, latest_path)

    # Save config alongside checkpoint
    config.save(checkpoint_dir / 'config.json')

    logging.info(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint(checkpoint_path: Path, model: VQVAE,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    scheduler=None, accelerator: Optional[Accelerator] = None) -> Tuple[int, int]:
    """
    Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        accelerator: Accelerator instance

    Returns:
        Tuple of (starting epoch number, global step)
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Unwrap model if using accelerator
    if accelerator is not None:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    logging.info(f"Loaded checkpoint from {checkpoint_path}")

    start_epoch = checkpoint.get('epoch', 0) + 1
    global_step = checkpoint.get('global_step', 0)

    return start_epoch, global_step


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
    Main training loop with Accelerate support.

    Args:
        config: Training configuration containing all hyperparameters
    """
    # Initialize accelerator with mixed precision
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.grad_accumulation_steps,
        log_with="wandb" if not config.no_wandb else None,
        project_dir=config.output_dir
    )

    # Setup
    accelerate_set_seed(config.seed)

    # Create directories
    output_dir = Path(config.output_dir)
    checkpoint_dir = output_dir / 'checkpoints'
    log_dir = output_dir / 'logs'

    # Setup logging (only on main process)
    logger = setup_logging(log_dir, accelerator)

    if accelerator.is_main_process:
        logger.info(f"Starting training with config:")
        logger.info(json.dumps(config.to_dict(), indent=2))
        logger.info(f"Using device: {accelerator.device}")
        logger.info(f"Mixed precision: {config.mixed_precision}")
        logger.info(f"Num processes: {accelerator.num_processes}")

        # Save config
        output_dir.mkdir(parents=True, exist_ok=True)
        config.save(output_dir / 'config.json')

    # Initialize wandb through accelerator
    if not config.no_wandb:
        wandb_config = config.to_dict()
        accelerator.init_trackers(
            project_name=config.wandb_project,
            config=wandb_config,
            init_kwargs={
                "wandb": {
                    "entity": config.wandb_entity,
                    "name": config.run_name,
                    "tags": config.wandb_tags,
                }
            }
        )

    # Create model
    cp_config = config.to_cp_config()
    model = get_model(cp_config)

    if accelerator.is_main_process:
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dataloaders
    train_dataloader, val_dataloader = create_train_val_dataloaders(config)

    if accelerator.is_main_process:
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

    # Prepare everything with accelerator
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    if scheduler is not None and config.scheduler != 'plateau':
        # Only prepare non-ReduceLROnPlateau schedulers
        scheduler = accelerator.prepare(scheduler)

    # Load checkpoint if resuming
    start_epoch = 0
    global_step = 0
    if config.resume_from:
        start_epoch, global_step = load_checkpoint(
            Path(config.resume_from),
            model, optimizer, scheduler, accelerator
        )

    # Training loop
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    epoch_metrics: dict[str, float] = {}

    for epoch in range(start_epoch, config.num_epochs):
        epoch_start_time = datetime.now()

        # Training phase
        model.train()
        train_losses = {
            'train_loss': 0.,
            'train_recon_loss': 0.,
            'train_vq_loss': 0.,
            'train_perplexity': 0.
        }
        num_train_batches = 0

        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch+1}/{config.num_epochs}",
            disable=not accelerator.is_local_main_process
        )

        for batch_idx, batch in enumerate(progress_bar):
            # Accelerator handles device placement
            inputs = batch['input_ids']
            attention_mask = batch['attention_mask']

            # Learning rate warmup based on global steps
            if global_step < config.warmup_steps:
                warmup_lr = get_warmup_lr(config.learning_rate, global_step, config.warmup_steps)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr

            # Use accelerator's accumulate context manager
            with accelerator.accumulate(model):
                # Forward pass
                reconstructed, quantized, vq_loss, perplexity, encoding_indices = model(inputs, attention_mask)

                # Compute losses with beta weighting
                losses = model.compute_loss(inputs, reconstructed, vq_loss, attention_mask, beta=config.beta)

                # Backward pass (accelerator handles scaling)
                accelerator.backward(losses['total_loss'])

                # Gradient clipping
                if config.gradient_clip > 0 and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.gradient_clip)

                optimizer.step()
                optimizer.zero_grad()

            # Accumulate losses
            train_losses['train_loss'] += losses['total_loss'].item()
            train_losses['train_recon_loss'] += losses['recon_loss'].item()
            train_losses['train_vq_loss'] += losses['vq_loss'].item()
            train_losses['train_perplexity'] += perplexity.item()
            num_train_batches += 1

            # Update global step
            if accelerator.sync_gradients:
                global_step += 1

            # Update progress bar
            if accelerator.is_local_main_process:
                current_lr = optimizer.param_groups[0]['lr']
                progress_bar.set_postfix({
                    'loss': losses['total_loss'].item(),
                    'recon': losses['recon_loss'].item(),
                    'vq': losses['vq_loss'].item(),
                    'ppl': perplexity.item(),
                    'lr': f'{current_lr:.2e}'
                })

            # Log to wandb
            if global_step % config.log_interval == 0 and accelerator.is_main_process:
                accelerator.log({
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
        val_metrics = validate(model, val_dataloader, accelerator)

        # Update learning rate scheduler (after warmup is complete)
        if scheduler is not None and global_step >= config.warmup_steps:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics['val_loss'])
            else:
                scheduler.step()

        # Log epoch metrics
        epoch_metrics = {**train_losses, **val_metrics}
        epoch_metrics['epoch'] = epoch + 1
        epoch_metrics['learning_rate'] = optimizer.param_groups[0]['lr']
        epoch_metrics['global_step'] = global_step

        if accelerator.is_main_process:
            accelerator.log(epoch_metrics)

        # Check if best model
        is_best = val_metrics['val_loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['val_loss']
            epochs_without_improvement = 0
            if accelerator.is_main_process:
                logger.info(f"New best model! Val loss: {best_val_loss:.4f}")
        else:
            epochs_without_improvement += 1

        # Save checkpoint
        if (epoch + 1) % config.save_interval == 0 or is_best:
            accelerator.wait_for_everyone()
            save_checkpoint(
                model, optimizer, scheduler, epoch + 1, global_step,
                epoch_metrics, checkpoint_dir, config, accelerator, is_best
            )

        # Log epoch summary
        if accelerator.is_main_process:
            epoch_time = (datetime.now() - epoch_start_time).total_seconds()
            logger.info(
                f"Epoch {epoch+1}/{config.num_epochs} - "
                f"Train Loss: {train_losses['train_loss']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"Val Perplexity: {val_metrics['val_perplexity']:.2f}, "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}, "
                f"Global Step: {global_step}, "
                f"Time: {epoch_time:.1f}s"
            )

        # Early stopping check
        if config.early_stopping_patience > 0:
            if epochs_without_improvement >= config.early_stopping_patience:
                if accelerator.is_main_process:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

    # Final save
    accelerator.wait_for_everyone()
    save_checkpoint(
        model, optimizer, scheduler, config.num_epochs, global_step,
        epoch_metrics, checkpoint_dir, config, accelerator, is_best=False
    )

    # End tracking
    accelerator.end_training()

    if accelerator.is_main_process:
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

    # Data configuration
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation split fraction')
    parser.add_argument('--val_size', type=int, default=100,
                        help='Size of validation set (taken after split, to ensure fixed size)')

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

    # Accelerate configuration
    parser.add_argument('--mixed_precision', type=str, default='fp16',
                        choices=['no', 'fp16', 'bf16'],
                        help='Mixed precision training mode')

    # Other configuration
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--early_stopping_patience', type=int, default=0,
                        help='Early stopping patience (0 to disable)')

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

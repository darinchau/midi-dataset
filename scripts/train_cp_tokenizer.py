import wandb
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from torch.optim import AdamW
import torch.nn.functional as F
import torch.nn as nn
from typing import Dict, Any, Optional, Callable
from pathlib import Path
from datetime import datetime
import logging
import json
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional, Union
import os
import numpy as np
from typing import List, Tuple
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Optional
from functools import partial
import pickle
from accelerate import Accelerator

from src.model.cp_tokenizer import ReformerCompressor, ModelConfig
from src.extract import musicxml_to_notes
from src.extract.tokenize import musicxml_to_tokens
from src.util import get_all_xml_paths
from tqdm.auto import tqdm

from dotenv import load_dotenv
from torch.utils.data import random_split
load_dotenv()


@dataclass(frozen=True)
class DataConfig:
    """Configuration for data loading and processing"""
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = True
    drop_last: bool = False
    input_dim: int = 275
    padding_value: float = 0.0
    gradient_accumulation_steps: int = 1
    chunk_length: int = 256


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for training"""
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_epochs: int = 100
    gradient_clip_val: float = 1.0

    # Loss weights
    discrete_loss_weight: float = 1.0
    continuous_loss_weight: float = 2.0  # Time features are more important
    vq_loss_weight: float = 1.0

    # Scheduler
    scheduler_type: str = "cosine"  # "cosine", "onecycle", or "none"

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_every_n_epochs: int = 5
    keep_last_n_checkpoints: int = 3

    # Logging
    log_every_n_steps: int = 50
    val_every_n_epochs: int = 1
    use_wandb: bool = True
    wandb_project: str = "reformer-compression"

    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4

    # Device - removed as Accelerate handles this
    # device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: str = "fp16"  # Changed to string for accelerate: "no", "fp16", "bf16"

    # Resume training
    resume_from_checkpoint: Optional[str] = None


class TokenDataset(Dataset):
    """Dataset for loading token sequences from cached files"""

    def __init__(
        self,
        files: list[str],
        config: DataConfig,
        is_training: bool = True
    ):
        self.config = config
        self.is_training = is_training
        self.files = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int]]:
        file = self.files[idx]
        try:
            data = load_musicxml_tokens(file)
        except Exception as e:
            logging.error(f"Error loading {file}: {e}")
            return self.__getitem__(np.random.randint(0, len(self.files)))

        data_tensor = torch.from_numpy(data).float()
        return {
            'input': data_tensor,
            'length': len(data_tensor)
        }

    def get(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        Get a single item by index

        Args:
            idx: Index of the item

        Returns:
            Tuple containing:
                - input: Tensor of shape (seq_len, input_dim)
                - length: Length of the sequence
        """
        item = self.__getitem__(idx)
        return item['input'], item['length']  # type: ignore[return-value]


def collate_fn(
    batch: List[Dict[str, Union[torch.Tensor, int]]],
    padding_value: float = 0.0,
    chunk_length: int = 256
) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to handle variable-length sequences
    Pads sequences to the nearest multiple of chunk_length

    Args:
        batch: List of dictionaries from dataset
        padding_value: Value to use for padding
        chunk_length: Pad sequences to multiples of this value

    Returns:
        Dictionary containing:
            - input: Padded sequences (batch_size, padded_seq_len, input_dim)
            - attention_mask: Binary mask (batch_size, padded_seq_len)
            - lengths: Original sequence lengths (batch_size,)
    """
    # Extract inputs and lengths
    inputs: list[torch.Tensor] = [item['input'] for item in batch]  # type: ignore
    lengths = torch.LongTensor([item['length'] for item in batch])

    # Find max length and pad to nearest multiple of chunk_length
    max_length = max(len(inp) for inp in inputs)
    padded_length = ((max_length + chunk_length - 1) // chunk_length) * chunk_length

    # Pad sequences to the padded_length
    padded_inputs = []
    for inp in inputs:
        if len(inp) < padded_length:
            padding = torch.full(
                (padded_length - len(inp), inp.shape[1]),
                padding_value,
                dtype=inp.dtype
            )
            padded_inp = torch.cat([inp, padding], dim=0)
        else:
            padded_inp = inp[:padded_length]
        padded_inputs.append(padded_inp)

    padded_inputs = torch.stack(padded_inputs, dim=0)

    # Create attention mask
    batch_size, max_seq_len = padded_inputs.shape[:2]
    attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)

    for i, length in enumerate(lengths):
        attention_mask[i, :length] = True

    return {
        'input': padded_inputs,
        'attention_mask': attention_mask,
        'lengths': lengths
    }


class ChunkedSampler(torch.utils.data.Sampler):
    """
    Sampler that groups sequences of similar lengths together
    to minimize padding within batches
    """

    def __init__(
        self,
        dataset: TokenDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Group sequences by length
        self.length_to_indices: Dict[int, List[int]] = {}

        print("Grouping sequences by length...")
        for idx in tqdm(range(len(dataset))):
            item, length = dataset.get(idx)

            # Group by length buckets (e.g., 0-100, 100-200, etc.)
            bucket = (length // 100) * 100

            if bucket not in self.length_to_indices:
                self.length_to_indices[bucket] = []
            self.length_to_indices[bucket].append(idx)

        # Create batches
        self.batches = []
        for bucket, indices in self.length_to_indices.items():
            if self.shuffle:
                np.random.shuffle(indices)

            # Create batches from this bucket
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    self.batches.append(batch)

        if self.shuffle:
            np.random.shuffle(self.batches)

    def __iter__(self):
        for batch in self.batches:
            yield from batch

    def __len__(self):
        return sum(len(batch) for batch in self.batches)


def create_dataloader(
    dataset: TokenDataset,
    config: DataConfig,
    is_training: bool = True,
    use_length_sampling: bool = True
) -> DataLoader:
    """
    Create a DataLoader with appropriate settings

    Args:
        dataset: Dataset instance
        config: DataConfig instance
        is_training: Whether this is for training or validation
        use_length_sampling: Whether to use length-based sampling to minimize padding

    Returns:
        DataLoader configured for the dataset
    """
    # Create custom sampler if using length-based sampling
    sampler = None
    if use_length_sampling and is_training:
        sampler = ChunkedSampler(
            dataset,
            config.batch_size,
            shuffle=config.shuffle,
            drop_last=config.drop_last
        )
        shuffle = False  # Sampler handles shuffling
    else:
        shuffle = config.shuffle and is_training

    collate = partial(
        collate_fn,
        padding_value=config.padding_value,
        chunk_length=config.chunk_length
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size if sampler is None else 1,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last if sampler is None else False,
        collate_fn=collate,
        persistent_workers=config.num_workers > 0
    )

    return dataloader


class TokenDataModule:
    """Data module that manages train/val/test datasets and dataloaders"""

    def __init__(
        self,
        config: DataConfig,
        n_val: int = 1000,
        n_test: int = 1000,
        random_seed: int = 42,
    ):
        self.config = config
        self.files = get_all_xml_paths()

        # Split files into train/val/test
        n_files = len(self.files)
        n_train = n_files - n_test - n_val

        if n_train <= 0 or n_val <= 0 or n_test <= 0:
            raise ValueError(f"Not enough files to create train/val/test splits. Ensure you have at least {n_val + n_test + 1} files.")

        # Set random seed for reproducible splits
        torch.manual_seed(random_seed)

        # Shuffle files and split
        shuffled_files = self.files.copy()
        torch.manual_seed(random_seed)
        indices = torch.randperm(len(shuffled_files)).tolist()
        shuffled_files = [shuffled_files[i] for i in indices]

        train_files = shuffled_files[:n_train]
        val_files = shuffled_files[n_train:n_train + n_val]
        test_files = shuffled_files[n_train + n_val:]

        # Create datasets
        train_dataset = TokenDataset(train_files, config, is_training=True)
        val_dataset = TokenDataset(val_files, config, is_training=False)
        test_dataset = TokenDataset(test_files, config, is_training=False)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def train_dataloader(self) -> DataLoader:
        return create_dataloader(
            self.train_dataset,
            self.config,
            is_training=True,
            use_length_sampling=False
        )

    def val_dataloader(self) -> DataLoader:
        return create_dataloader(
            self.val_dataset, self.config,
            is_training=False, use_length_sampling=False
        )

    def test_dataloader(self) -> DataLoader:
        return create_dataloader(
            self.test_dataset, self.config,
            is_training=False, use_length_sampling=False
        )


def create_model(config: Optional[ModelConfig] = None) -> ReformerCompressor:
    """Create a ReformerCompressor model with the given configuration"""
    if config is None:
        config = ModelConfig()

    model = ReformerCompressor(config)

    total_params: int = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params:,} parameters")
    print(f"Theoretical compression ratio: {model.compress_ratio():.2f}x")

    return model


class CompressionLoss(nn.Module):
    """Combined loss for compression task"""

    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.discrete_weight = config.discrete_loss_weight
        self.continuous_weight = config.continuous_loss_weight
        self.vq_weight = config.vq_loss_weight

    def forward(
        self,
        reconstructed: torch.Tensor,
        target: torch.Tensor,
        vq_loss: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute reconstruction loss with proper masking

        Args:
            reconstructed: Model output (batch, seq_len, dim)
            target: Ground truth (batch, seq_len, dim)
            vq_loss: Vector quantization loss from model
            attention_mask: Boolean mask for valid positions
        """
        # Split discrete and continuous parts
        discrete_dim = target.shape[-1] - 2  # Last 2 dims are continuous

        discrete_recon = reconstructed[..., :discrete_dim]
        continuous_recon = reconstructed[..., discrete_dim:]

        discrete_target = target[..., :discrete_dim]
        continuous_target = target[..., discrete_dim:]

        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask to match feature dimensions
            mask_discrete = attention_mask.unsqueeze(-1)
            mask_continuous = attention_mask.unsqueeze(-1)

            # Masked MSE loss
            discrete_loss = (F.mse_loss(discrete_recon, discrete_target, reduction='none') * mask_discrete).sum()
            discrete_loss = discrete_loss / (mask_discrete.sum() + 1e-8)

            continuous_loss = (F.mse_loss(continuous_recon, continuous_target, reduction='none') * mask_continuous).sum()
            continuous_loss = continuous_loss / (mask_continuous.sum() + 1e-8)
        else:
            discrete_loss = F.mse_loss(discrete_recon, discrete_target)
            continuous_loss = F.mse_loss(continuous_recon, continuous_target)

        # Total loss
        total_loss = (
            self.discrete_weight * discrete_loss +
            self.continuous_weight * continuous_loss +
            self.vq_weight * vq_loss
        )

        loss_dict = {
            'loss/total': total_loss.item(),
            'loss/discrete': discrete_loss.item(),
            'loss/continuous': continuous_loss.item(),
            'loss/vq': vq_loss.item() if isinstance(vq_loss, torch.Tensor) else vq_loss
        }

        return total_loss, loss_dict


class ModelCheckpoint:
    """Handle model checkpointing"""

    def __init__(self, config: TrainingConfig, accelerator: Accelerator):
        self.config = config
        self.accelerator = accelerator
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_loss = float('inf')

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        epoch: int,
        step: int,
        val_loss: float,
        model_config: ModelConfig,
        data_config: DataConfig
    ) -> None:
        """Save model checkpoint"""
        # Only save on main process
        if self.accelerator.is_main_process:
            checkpoint = {
                'epoch': epoch,
                'step': step,
                'model_state_dict': self.accelerator.unwrap_model(model).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_loss': val_loss,
                'model_config': model_config,
                'data_config': data_config,
                'training_config': self.config,
            }

            # Save latest checkpoint
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
            self.accelerator.save(checkpoint, checkpoint_path)

            # Save best checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_path = self.checkpoint_dir / 'best_model.pt'
                self.accelerator.save(checkpoint, best_path)
                logging.info(f"New best model saved with val_loss: {val_loss:.4f}")

            # Clean up old checkpoints
            self._cleanup_old_checkpoints(epoch)

    def _cleanup_old_checkpoints(self, current_epoch: int) -> None:
        """Keep only the last N checkpoints"""
        if self.accelerator.is_main_process:
            checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
            if len(checkpoints) > self.config.keep_last_n_checkpoints:
                for ckpt in checkpoints[:-self.config.keep_last_n_checkpoints]:
                    ckpt.unlink()

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Load checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return checkpoint


class Trainer:
    """Main trainer class"""

    def __init__(
        self,
        model: ReformerCompressor,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model_config: ModelConfig,
        data_config: DataConfig,
        training_config: TrainingConfig
    ):
        # Initialize accelerator
        self.accelerator = Accelerator(
            mixed_precision=training_config.mixed_precision,
            gradient_accumulation_steps=data_config.gradient_accumulation_steps,
            log_with="wandb" if training_config.use_wandb else None,
            project_dir=training_config.checkpoint_dir
        )

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_config = model_config
        self.data_config = data_config
        self.config = training_config

        # Setup loss
        self.criterion = CompressionLoss(training_config)

        # Setup optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay
        )

        # Setup scheduler
        self.scheduler = self._create_scheduler()

        # Prepare with accelerator
        self.model, self.optimizer, self.train_loader, self.val_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.val_loader
        )

        if self.scheduler is not None:
            self.scheduler = self.accelerator.prepare(self.scheduler)

        # Setup checkpointing
        self.checkpoint_manager = ModelCheckpoint(training_config, self.accelerator)

        # Setup logging
        self._setup_logging()

        # State
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Resume from checkpoint if specified
        if training_config.resume_from_checkpoint:
            self._resume_from_checkpoint(training_config.resume_from_checkpoint)

    def _create_scheduler(self) -> Optional[Any]:
        """Create learning rate scheduler"""
        if self.config.scheduler_type == "cosine":
            return CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=len(self.train_loader) * 10,  # Restart every 10 epochs
                T_mult=2
            )
        elif self.config.scheduler_type == "onecycle":
            return OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                total_steps=len(self.train_loader) * self.config.max_epochs // self.data_config.gradient_accumulation_steps,
                pct_start=0.1
            )
        return None

    def _setup_logging(self) -> None:
        """Setup logging"""
        if self.accelerator.is_main_process:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )

            if self.config.use_wandb:
                self.accelerator.init_trackers(
                    project_name=self.config.wandb_project,
                    config={
                        'model_config': self.model_config.__dict__,
                        'data_config': self.data_config.__dict__,
                        'training_config': self.config.__dict__
                    }
                )

    def _resume_from_checkpoint(self, checkpoint_path: str) -> None:
        """Resume training from checkpoint"""
        logging.info(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = self.checkpoint_manager.load_checkpoint(
            checkpoint_path, self.model, self.optimizer, self.scheduler
        )
        self.current_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['step']
        self.best_val_loss = checkpoint.get('val_loss', float('inf'))

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = []

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch}/{self.config.max_epochs}",
            disable=not self.accelerator.is_local_main_process
        )

        for batch_idx, batch in enumerate(progress_bar):
            # Move to device is handled by accelerator
            inputs = batch['input']
            attention_mask = batch['attention_mask']

            # Forward pass
            with self.accelerator.accumulate(self.model):
                reconstructed, indices, vq_loss = self.model(inputs)

                # Compute loss
                loss, loss_dict = self.criterion(
                    reconstructed, inputs, vq_loss, attention_mask
                )

                # Scale loss by gradient accumulation steps
                loss = loss / self.data_config.gradient_accumulation_steps

                # Backward pass
                self.accelerator.backward(loss)

                # Gradient clipping
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_val
                    )

                # Optimizer step (only when gradients are synced)
                self.optimizer.step()

                # Update scheduler for OneCycle
                if self.scheduler and self.config.scheduler_type == "onecycle":
                    self.scheduler.step()

                # Zero gradients
                self.optimizer.zero_grad()

            # Logging
            epoch_losses.append(loss_dict['loss/total'])

            if self.global_step % self.config.log_every_n_steps == 0:
                # Add learning rate to logs
                loss_dict['lr'] = self.optimizer.param_groups[0]['lr']

                # Update progress bar
                progress_bar.set_postfix(
                    loss=loss_dict['loss/total'],
                    lr=loss_dict['lr']
                )

                # Log to wandb
                if self.config.use_wandb and self.accelerator.is_main_process:
                    self.accelerator.log(loss_dict, step=self.global_step)

            self.global_step += 1

        # Update scheduler (for epoch-based schedulers)
        if self.scheduler and self.config.scheduler_type == "cosine":
            self.scheduler.step()

        return {'train_loss': np.mean(epoch_losses).item()}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        val_losses = []

        progress_bar = tqdm(
            self.val_loader,
            desc="Validation",
            disable=not self.accelerator.is_local_main_process
        )

        for batch in progress_bar:
            # Move to device is handled by accelerator
            inputs = batch['input']
            attention_mask = batch['attention_mask']

            # Forward pass
            reconstructed, indices, vq_loss = self.model(inputs)

            # Compute loss
            loss, loss_dict = self.criterion(
                reconstructed, inputs, vq_loss, attention_mask
            )

            # Gather losses across devices
            gathered_loss: torch.Tensor = self.accelerator.gather(loss)  # type: ignore

            if self.accelerator.is_main_process:
                val_losses.extend(gathered_loss.cpu().numpy())

            progress_bar.set_postfix(loss=loss_dict['loss/total'])

        avg_val_loss = np.mean(val_losses).item() if val_losses else float('inf')

        # Log validation metrics
        if self.config.use_wandb and self.accelerator.is_main_process:
            self.accelerator.log({
                'val/loss': avg_val_loss,
                'epoch': self.current_epoch
            }, step=self.global_step)

        return {'val_loss': avg_val_loss}

    def train(self) -> None:
        """Main training loop"""
        if self.accelerator.is_main_process:
            logging.info("Starting training...")

        for epoch in range(self.current_epoch, self.config.max_epochs):
            self.current_epoch = epoch

            # Train epoch
            train_metrics = self.train_epoch()
            if self.accelerator.is_main_process:
                logging.info(f"Epoch {epoch} - Train loss: {train_metrics['train_loss']:.4f}")

            # Validate
            if epoch % self.config.val_every_n_epochs == 0:
                val_metrics = self.validate()
                if self.accelerator.is_main_process:
                    logging.info(f"Epoch {epoch} - Val loss: {val_metrics['val_loss']:.4f}")

                    # Early stopping
                    if val_metrics['val_loss'] < self.best_val_loss - self.config.early_stopping_min_delta:
                        self.best_val_loss = val_metrics['val_loss']
                        self.patience_counter = 0
                    else:
                        self.patience_counter += 1

                    if self.patience_counter >= self.config.early_stopping_patience:
                        logging.info(f"Early stopping triggered after {epoch + 1} epochs")
                        break

                # Save checkpoint
                if epoch % self.config.save_every_n_epochs == 0:
                    self.checkpoint_manager.save_checkpoint(
                        self.model, self.optimizer, self.scheduler,
                        epoch, self.global_step, val_metrics['val_loss'],
                        self.model_config, self.data_config
                    )

        if self.accelerator.is_main_process:
            logging.info("Training completed!")

            # Save final model
            self.checkpoint_manager.save_checkpoint(
                self.model, self.optimizer, self.scheduler,
                self.current_epoch, self.global_step, self.best_val_loss,
                self.model_config, self.data_config
            )

        self.accelerator.end_training()


def load_musicxml_tokens(file_path: str) -> np.ndarray:
    """Load and tokenize MusicXML file"""
    try:
        # This returns tokens as numpy array of shape (T, 275)
        notes = musicxml_to_notes(file_path)
        tokens = musicxml_to_tokens(notes)
        return tokens
    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
        # Return empty array on error
        return np.array([]).reshape(0, 275)


def test_model(trainer: Trainer, data_module: TokenDataModule) -> None:
    # Test the model after training
    if not trainer.accelerator.is_main_process:
        return
    logging.info("Testing final model...")
    test_loader = data_module.test_dataloader()
    test_loader = trainer.accelerator.prepare(test_loader)
    test_metrics = trainer.validate()  # Use same validation logic for test
    logging.info(f"Test loss: {test_metrics['val_loss']:.4f}")

    # Calculate and log compression statistics
    trainer.model.eval()
    with torch.no_grad():
        sample_batch = next(iter(test_loader))
        inputs = sample_batch['input']

        # Get compressed representation
        compressed, indices = trainer.model.get_compressed_representation(inputs)

        # Log compression statistics
        original_size = inputs.numel() * 4  # float32 = 4 bytes
        compressed_size = indices.numel() * 2  # int16 indices = 2 bytes
        actual_ratio = original_size / compressed_size

    logging.info(f"Theoretical compression ratio: {trainer.model.compress_ratio():.2f}x")
    logging.info(f"Actual compression ratio: {actual_ratio:.2f}x")


def main():
    """Main training script"""
    # Setup configurations
    model_config = ModelConfig(
        total_input_dim=275,
        discrete_dim=273,
        continuous_dim=2,
        compressed_dim=128,
        num_layers=6,
        num_quantizers=8,
        codebook_size=8192,
        num_heads=8,
        feed_forward_size=2048
    )

    data_config = DataConfig(
        batch_size=8,
        num_workers=4,
        gradient_accumulation_steps=4,
        input_dim=model_config.total_input_dim,
        chunk_length=model_config.lsh_attn_chunk_length
    )

    training_config = TrainingConfig(
        learning_rate=1e-4,
        max_epochs=100,
        checkpoint_dir="./checkpoints/reformer_compression",
        use_wandb=True,
        wandb_project="music-compression-1",
        early_stopping_patience=15,
        val_every_n_epochs=1,
        save_every_n_epochs=5,
        mixed_precision="bf16"
    )

    # Create data module
    data_module = TokenDataModule(
        config=data_config,
        n_val=1000,
        n_test=1000,
        random_seed=1943
    )

    # Create model
    model = create_model(model_config)

    # Get dataloaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_config=model_config,
        data_config=data_config,
        training_config=training_config
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()

"""
Training module for pose estimation models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
import time
import os
from pathlib import Path
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

from ..models import BasePoseModel
from ..data import PoseDataset
from .losses import PoseLoss
from .metrics import PoseMetrics
from .callbacks import TrainingCallbacks


class PoseTrainer:
    """Main training class for pose estimation models."""
    
    def __init__(
        self,
        model: BasePoseModel,
        train_dataset: PoseDataset,
        val_dataset: PoseDataset,
        config: Dict[str, Any],
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize trainer.
        
        Args:
            model: Pose estimation model
            train_dataset: Training dataset
            val_dataset: Validation dataset
            config: Training configuration
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.device = device
        
        # Initialize loss function
        self.criterion = PoseLoss(
            loss_type=config.get('loss_type', 'mse'),
            keypoint_weight=config.get('keypoint_weight', 1.0),
            heatmap_weight=config.get('heatmap_weight', 1.0),
            paf_weight=config.get('paf_weight', 1.0)
        )
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Initialize metrics
        self.metrics = PoseMetrics()
        
        # Initialize callbacks
        self.callbacks = TrainingCallbacks(config.get('callbacks', {}))
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        # Create output directory
        self.output_dir = Path(config.get('output_dir', 'output/training'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config."""
        optimizer_type = self.config.get('optimizer', 'adam')
        learning_rate = self.config.get('learning_rate', 0.001)
        weight_decay = self.config.get('weight_decay', 0.0001)
        
        if optimizer_type.lower() == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler based on config."""
        scheduler_type = self.config.get('scheduler', None)
        
        if scheduler_type is None:
            return None
        
        if scheduler_type.lower() == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('scheduler_step_size', 30),
                gamma=self.config.get('scheduler_gamma', 0.1)
            )
        elif scheduler_type.lower() == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('epochs', 100)
            )
        elif scheduler_type.lower() == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=self.config.get('scheduler_patience', 10)
            )
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_type}")
    
    def train(self, epochs: int) -> Dict[str, Any]:
        """
        Train the model for specified number of epochs.
        
        Args:
            epochs: Number of epochs to train
        
        Returns:
            Training history
        """
        print(f"Starting training for {epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")
        
        # Create data loaders
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True if self.device == 'cuda' else False
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True if self.device == 'cuda' else False
        )
        
        # Training loop
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 50)
            
            # Train for one epoch
            train_loss, train_metrics = self._train_epoch(train_loader)
            
            # Validate
            val_loss, val_metrics = self._validate_epoch(val_loader)
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Record history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_metrics'].append(train_metrics)
            self.training_history['val_metrics'].append(val_metrics)
            
            # Print epoch results
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Train PCK: {train_metrics.get('pck', 0):.4f}")
            print(f"Val PCK: {val_metrics.get('pck', 0):.4f}")
            
            # Callbacks
            self.callbacks.on_epoch_end(
                epoch, 
                train_loss, val_loss, 
                train_metrics, val_metrics
            )
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint('best_model.pth', epoch, val_loss)
            
            # Save periodic checkpoint
            if (epoch + 1) % self.config.get('save_interval', 10) == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth', epoch, val_loss)
        
        # Save final model
        self._save_checkpoint('final_model.pth', epochs - 1, val_loss)
        
        # Save training history
        self._save_training_history()
        
        # Plot training curves
        self._plot_training_curves()
        
        print(f"\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        return self.training_history
    
    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_metrics = {}
        num_batches = len(train_loader)
        
        pbar = tqdm(train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            images = batch['image'].to(self.device)
            keypoints = batch['keypoints'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Calculate loss
            loss = self.criterion(outputs, keypoints)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['grad_clip']
                )
            
            self.optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                metrics = self.metrics.calculate_metrics(outputs, keypoints)
            
            # Update totals
            total_loss += loss.item()
            for key, value in metrics.items():
                if key not in total_metrics:
                    total_metrics[key] = 0.0
                total_metrics[key] += value
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'PCK': f"{metrics.get('pck', 0):.4f}"
            })
        
        # Calculate averages
        avg_loss = total_loss / num_batches
        avg_metrics = {key: value / num_batches for key, value in total_metrics.items()}
        
        return avg_loss, avg_metrics
    
    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_metrics = {}
        num_batches = len(val_loader)
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            
            for batch in pbar:
                # Move batch to device
                images = batch['image'].to(self.device)
                keypoints = batch['keypoints'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate loss
                loss = self.criterion(outputs, keypoints)
                
                # Calculate metrics
                metrics = self.metrics.calculate_metrics(outputs, keypoints)
                
                # Update totals
                total_loss += loss.item()
                for key, value in metrics.items():
                    if key not in total_metrics:
                        total_metrics[key] = 0.0
                    total_metrics[key] += value
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'PCK': f"{metrics.get('pck', 0):.4f}"
                })
        
        # Calculate averages
        avg_loss = total_loss / num_batches
        avg_metrics = {key: value / num_batches for key, value in total_metrics.items()}
        
        return avg_loss, avg_metrics
    
    def _save_checkpoint(self, filename: str, epoch: int, val_loss: float):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
            'training_history': self.training_history
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        checkpoint_path = self.output_dir / filename
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
    
    def _save_training_history(self):
        """Save training history to JSON."""
        history_path = self.output_dir / 'training_history.json'
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_history = {}
        for key, value in self.training_history.items():
            if isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], (np.ndarray, np.floating)):
                    serializable_history[key] = [float(v) for v in value]
                elif isinstance(value[0], dict):
                    serializable_history[key] = value
                else:
                    serializable_history[key] = value
            else:
                serializable_history[key] = value
        
        with open(history_path, 'w') as f:
            json.dump(serializable_history, f, indent=2)
    
    def _plot_training_curves(self):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss curves
        axes[0, 0].plot(self.training_history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.training_history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # PCK curves
        train_pck = [m.get('pck', 0) for m in self.training_history['train_metrics']]
        val_pck = [m.get('pck', 0) for m in self.training_history['val_metrics']]
        
        axes[0, 1].plot(train_pck, label='Train PCK')
        axes[0, 1].plot(val_pck, label='Val PCK')
        axes[0, 1].set_title('PCK (Percentage of Correct Keypoints)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('PCK')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # MPJPE curves
        train_mpjpe = [m.get('mpjpe', 0) for m in self.training_history['train_metrics']]
        val_mpjpe = [m.get('mpjpe', 0) for m in self.training_history['val_metrics']]
        
        axes[1, 0].plot(train_mpjpe, label='Train MPJPE')
        axes[1, 0].plot(val_mpjpe, label='Val MPJPE')
        axes[1, 0].set_title('MPJPE (Mean Per Joint Position Error)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MPJPE')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate curve
        if hasattr(self.optimizer, 'param_groups'):
            lr_history = []
            for group in self.optimizer.param_groups:
                lr_history.append(group['lr'])
            
            axes[1, 1].plot(lr_history)
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved training curves: {plot_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['val_loss']
        self.training_history = checkpoint.get('training_history', self.training_history)
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
        print(f"Validation loss: {self.best_val_loss:.4f}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model and training information."""
        return {
            'model_info': self.model.get_model_info(),
            'config': self.config,
            'device': self.device,
            'current_epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'output_dir': str(self.output_dir)
        }

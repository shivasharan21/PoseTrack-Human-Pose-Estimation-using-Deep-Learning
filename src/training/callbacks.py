"""
Training callbacks for monitoring and control.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Callable
import time
import os
from pathlib import Path
import matplotlib.pyplot as plt


class TrainingCallbacks:
    """Collection of training callbacks."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize callbacks.
        
        Args:
            config: Callback configuration dictionary
        """
        self.config = config
        self.callbacks = {}
        
        # Initialize enabled callbacks
        self._initialize_callbacks()
    
    def _initialize_callbacks(self):
        """Initialize enabled callbacks."""
        # Early stopping callback
        if self.config.get('early_stopping', {}).get('enabled', False):
            self.callbacks['early_stopping'] = EarlyStoppingCallback(
                **self.config['early_stopping']
            )
        
        # Model checkpoint callback
        if self.config.get('model_checkpoint', {}).get('enabled', False):
            self.callbacks['model_checkpoint'] = ModelCheckpointCallback(
                **self.config['model_checkpoint']
            )
        
        # Learning rate scheduler callback
        if self.config.get('lr_scheduler', {}).get('enabled', False):
            self.callbacks['lr_scheduler'] = LRSchedulerCallback(
                **self.config['lr_scheduler']
            )
        
        # TensorBoard callback
        if self.config.get('tensorboard', {}).get('enabled', False):
            self.callbacks['tensorboard'] = TensorBoardCallback(
                **self.config['tensorboard']
            )
        
        # Progress callback
        if self.config.get('progress', {}).get('enabled', True):
            self.callbacks['progress'] = ProgressCallback(
                **self.config['progress']
            )
    
    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any]):
        """Called at the beginning of each epoch."""
        for callback in self.callbacks.values():
            if hasattr(callback, 'on_epoch_begin'):
                callback.on_epoch_begin(epoch, logs)
    
    def on_epoch_end(
        self, 
        epoch: int, 
        train_loss: float, 
        val_loss: float, 
        train_metrics: Dict[str, float], 
        val_metrics: Dict[str, float]
    ):
        """Called at the end of each epoch."""
        logs = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }
        
        for callback in self.callbacks.values():
            if hasattr(callback, 'on_epoch_end'):
                callback.on_epoch_end(epoch, logs)
    
    def on_batch_begin(self, batch: int, logs: Dict[str, Any]):
        """Called at the beginning of each batch."""
        for callback in self.callbacks.values():
            if hasattr(callback, 'on_batch_begin'):
                callback.on_batch_begin(batch, logs)
    
    def on_batch_end(self, batch: int, logs: Dict[str, Any]):
        """Called at the end of each batch."""
        for callback in self.callbacks.values():
            if hasattr(callback, 'on_batch_end'):
                callback.on_batch_end(batch, logs)
    
    def on_train_begin(self, logs: Dict[str, Any]):
        """Called at the beginning of training."""
        for callback in self.callbacks.values():
            if hasattr(callback, 'on_train_begin'):
                callback.on_train_begin(logs)
    
    def on_train_end(self, logs: Dict[str, Any]):
        """Called at the end of training."""
        for callback in self.callbacks.values():
            if hasattr(callback, 'on_train_end'):
                callback.on_train_end(logs)


class BaseCallback:
    """Base callback class."""
    
    def __init__(self):
        self.model = None
        self.optimizer = None
    
    def set_model(self, model):
        """Set the model."""
        self.model = model
    
    def set_optimizer(self, optimizer):
        """Set the optimizer."""
        self.optimizer = optimizer


class EarlyStoppingCallback(BaseCallback):
    """Early stopping callback to prevent overfitting."""
    
    def __init__(
        self, 
        monitor: str = 'val_loss',
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min',
        restore_best_weights: bool = True
    ):
        """
        Initialize early stopping callback.
        
        Args:
            monitor: Metric to monitor
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' for improvement direction
            restore_best_weights: Whether to restore best weights
        """
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.wait = 0
        self.best_weights = None
        self.stopped_epoch = 0
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]):
        """Check for early stopping condition."""
        current_score = self._get_monitor_value(logs)
        
        if current_score is None:
            return
        
        if self.best_score is None:
            self.best_score = current_score
            if self.restore_best_weights:
                self.best_weights = self.model.state_dict().copy()
        else:
            if self._is_improvement(current_score):
                self.best_score = current_score
                self.wait = 0
                if self.restore_best_weights:
                    self.best_weights = self.model.state_dict().copy()
            else:
                self.wait += 1
                
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    if self.restore_best_weights and self.best_weights is not None:
                        self.model.load_state_dict(self.best_weights)
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    print(f"Restored best weights from epoch {epoch - self.patience + 1}")
    
    def _get_monitor_value(self, logs: Dict[str, Any]) -> Optional[float]:
        """Get the monitored value from logs."""
        if self.monitor in logs:
            return logs[self.monitor]
        elif 'val_metrics' in logs and self.monitor in logs['val_metrics']:
            return logs['val_metrics'][self.monitor]
        elif 'train_metrics' in logs and self.monitor in logs['train_metrics']:
            return logs['train_metrics'][self.monitor]
        return None
    
    def _is_improvement(self, current_score: float) -> bool:
        """Check if current score is an improvement."""
        if self.mode == 'min':
            return current_score < (self.best_score - self.min_delta)
        else:
            return current_score > (self.best_score + self.min_delta)


class ModelCheckpointCallback(BaseCallback):
    """Model checkpoint callback to save best models."""
    
    def __init__(
        self,
        filepath: str = 'best_model.pth',
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True,
        save_freq: int = 1
    ):
        """
        Initialize model checkpoint callback.
        
        Args:
            filepath: Path to save the model
            monitor: Metric to monitor
            mode: 'min' or 'max' for improvement direction
            save_best_only: Whether to save only the best model
            save_freq: Frequency of saving (in epochs)
        """
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_freq = save_freq
        
        self.best_score = None
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]):
        """Save model checkpoint if conditions are met."""
        if (epoch + 1) % self.save_freq != 0:
            return
        
        current_score = self._get_monitor_value(logs)
        
        if current_score is None:
            return
        
        should_save = False
        
        if not self.save_best_only:
            should_save = True
        elif self.best_score is None:
            should_save = True
            self.best_score = current_score
        else:
            if self.mode == 'min':
                should_save = current_score < self.best_score
            else:
                should_save = current_score > self.best_score
            
            if should_save:
                self.best_score = current_score
        
        if should_save:
            self._save_model(epoch, logs)
    
    def _get_monitor_value(self, logs: Dict[str, Any]) -> Optional[float]:
        """Get the monitored value from logs."""
        if self.monitor in logs:
            return logs[self.monitor]
        elif 'val_metrics' in logs and self.monitor in logs['val_metrics']:
            return logs['val_metrics'][self.monitor]
        elif 'train_metrics' in logs and self.monitor in logs['train_metrics']:
            return logs['train_metrics'][self.monitor]
        return None
    
    def _save_model(self, epoch: int, logs: Dict[str, Any]):
        """Save the model."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'logs': logs
        }
        
        torch.save(checkpoint, self.filepath)
        print(f"Model saved: {self.filepath}")


class LRSchedulerCallback(BaseCallback):
    """Learning rate scheduler callback."""
    
    def __init__(
        self,
        scheduler_type: str = 'reduce_on_plateau',
        monitor: str = 'val_loss',
        patience: int = 5,
        factor: float = 0.5,
        min_lr: float = 1e-6
    ):
        """
        Initialize LR scheduler callback.
        
        Args:
            scheduler_type: Type of scheduler
            monitor: Metric to monitor
            patience: Patience for plateau detection
            factor: Factor to reduce LR by
            min_lr: Minimum learning rate
        """
        super().__init__()
        self.scheduler_type = scheduler_type
        self.monitor = monitor
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        
        self.scheduler = None
        self.wait = 0
        self.best_score = None
    
    def on_train_begin(self, logs: Dict[str, Any]):
        """Initialize scheduler."""
        if self.scheduler_type == 'reduce_on_plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.factor,
                patience=self.patience,
                min_lr=self.min_lr
            )
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]):
        """Update learning rate."""
        if self.scheduler is None:
            return
        
        current_score = self._get_monitor_value(logs)
        
        if current_score is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(current_score)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Learning rate: {current_lr:.6f}")
    
    def _get_monitor_value(self, logs: Dict[str, Any]) -> Optional[float]:
        """Get the monitored value from logs."""
        if self.monitor in logs:
            return logs[self.monitor]
        elif 'val_metrics' in logs and self.monitor in logs['val_metrics']:
            return logs['val_metrics'][self.monitor]
        elif 'train_metrics' in logs and self.monitor in logs['train_metrics']:
            return logs['train_metrics'][self.monitor]
        return None


class TensorBoardCallback(BaseCallback):
    """TensorBoard logging callback."""
    
    def __init__(self, log_dir: str = 'logs', log_freq: int = 1):
        """
        Initialize TensorBoard callback.
        
        Args:
            log_dir: Directory to save logs
            log_freq: Frequency of logging (in epochs)
        """
        super().__init__()
        self.log_dir = log_dir
        self.log_freq = log_freq
        
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
            self.enabled = True
        except ImportError:
            print("TensorBoard not available. Install with: pip install tensorboard")
            self.enabled = False
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]):
        """Log metrics to TensorBoard."""
        if not self.enabled or (epoch + 1) % self.log_freq != 0:
            return
        
        # Log scalar metrics
        self.writer.add_scalar('Loss/Train', logs['train_loss'], epoch)
        self.writer.add_scalar('Loss/Validation', logs['val_loss'], epoch)
        
        # Log training metrics
        if 'train_metrics' in logs:
            for key, value in logs['train_metrics'].items():
                self.writer.add_scalar(f'Train/{key}', value, epoch)
        
        # Log validation metrics
        if 'val_metrics' in logs:
            for key, value in logs['val_metrics'].items():
                self.writer.add_scalar(f'Val/{key}', value, epoch)
        
        # Log learning rate
        if self.optimizer:
            lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Learning_Rate', lr, epoch)
        
        self.writer.flush()
    
    def on_train_end(self, logs: Dict[str, Any]):
        """Close TensorBoard writer."""
        if self.enabled:
            self.writer.close()


class ProgressCallback(BaseCallback):
    """Progress monitoring callback."""
    
    def __init__(self, verbose: bool = True):
        """
        Initialize progress callback.
        
        Args:
            verbose: Whether to print progress information
        """
        super().__init__()
        self.verbose = verbose
        self.start_time = None
    
    def on_train_begin(self, logs: Dict[str, Any]):
        """Initialize progress tracking."""
        self.start_time = time.time()
        if self.verbose:
            print("Starting training...")
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]):
        """Print progress information."""
        if not self.verbose:
            return
        
        elapsed_time = time.time() - self.start_time
        avg_time_per_epoch = elapsed_time / (epoch + 1)
        
        print(f"Epoch {epoch + 1} completed in {avg_time_per_epoch:.2f}s")
        print(f"Total elapsed time: {elapsed_time:.2f}s")
    
    def on_train_end(self, logs: Dict[str, Any]):
        """Print training completion information."""
        if self.verbose and self.start_time:
            total_time = time.time() - self.start_time
            print(f"Training completed in {total_time:.2f}s")

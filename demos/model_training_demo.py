"""
Model training demo for pose estimation.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
import argparse
import json
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models import HRNetModel, OpenPoseModel
from src.training import PoseTrainer
from src.data import PoseDataset
from src.training.losses import PoseLoss
from src.training.metrics import PoseMetrics


def create_synthetic_dataset(num_samples=1000, num_keypoints=17):
    """Create a synthetic dataset for demonstration."""
    print("Creating synthetic dataset...")
    
    # Generate synthetic data
    images = torch.randn(num_samples, 3, 256, 256)
    keypoints = torch.randn(num_samples, num_keypoints, 3)  # x, y, visibility
    
    # Normalize keypoints to image coordinates
    keypoints[:, :, 0] = keypoints[:, :, 0] * 256  # x coordinates
    keypoints[:, :, 1] = keypoints[:, :, 1] * 256  # y coordinates
    keypoints[:, :, 2] = torch.sigmoid(keypoints[:, :, 2])  # visibility scores
    
    # Create dataset
    dataset = SyntheticPoseDataset(images, keypoints)
    
    return dataset


class SyntheticPoseDataset:
    """Synthetic pose dataset for demonstration."""
    
    def __init__(self, images, keypoints):
        self.images = images
        self.keypoints = keypoints
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return {
            'image': self.images[idx],
            'keypoints': self.keypoints[idx],
            'image_id': idx
        }


def main():
    """Main function for model training demo."""
    parser = argparse.ArgumentParser(description='Model Training Demo')
    parser.add_argument('--model', type=str, default='hrnet',
                       choices=['hrnet', 'openpose'],
                       help='Model type to train')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='output/training',
                       help='Output directory')
    parser.add_argument('--synthetic_data', action='store_true',
                       help='Use synthetic data for demonstration')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to real dataset')
    
    args = parser.parse_args()
    
    print("Model Training Demo")
    print("=" * 40)
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print("=" * 40)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create model
    if args.model == 'hrnet':
        model = HRNetModel(
            num_keypoints=17,
            input_size=(256, 256)
        )
    elif args.model == 'openpose':
        model = OpenPoseModel(
            num_keypoints=19,
            input_size=(368, 368)
        )
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    
    print(f"Created {args.model} model")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dataset
    if args.synthetic_data or args.data_path is None:
        print("Using synthetic data for demonstration")
        train_dataset = create_synthetic_dataset(num_samples=800)
        val_dataset = create_synthetic_dataset(num_samples=200)
    else:
        print(f"Loading dataset from {args.data_path}")
        # In practice, you would load real datasets here
        train_dataset = create_synthetic_dataset(num_samples=800)
        val_dataset = create_synthetic_dataset(num_samples=200)
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Validation dataset: {len(val_dataset)} samples")
    
    # Create training configuration
    config = {
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'optimizer': 'adam',
        'scheduler': 'step',
        'loss_type': 'mse',
        'output_dir': args.output_dir,
        'save_interval': 5,
        'callbacks': {
            'early_stopping': {
                'enabled': True,
                'monitor': 'val_loss',
                'patience': 5,
                'min_delta': 0.001
            },
            'model_checkpoint': {
                'enabled': True,
                'filepath': os.path.join(args.output_dir, 'best_model.pth'),
                'monitor': 'val_loss',
                'save_best_only': True
            },
            'tensorboard': {
                'enabled': True,
                'log_dir': os.path.join(args.output_dir, 'logs')
            }
        }
    }
    
    # Create trainer
    trainer = PoseTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config
    )
    
    print("Starting training...")
    
    # Train model
    training_history = trainer.train(epochs=args.epochs)
    
    print("Training completed!")
    
    # Save training results
    results_path = os.path.join(args.output_dir, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'config': config,
            'model_info': trainer.get_model_info(),
            'training_history': training_history,
            'final_metrics': {
                'best_val_loss': trainer.best_val_loss,
                'final_epoch': trainer.current_epoch
            }
        }, f, indent=2)
    
    print(f"Training results saved to: {results_path}")
    
    # Evaluate model
    print("Evaluating model...")
    evaluator = PoseEvaluator(model, val_dataset)
    evaluation_results = evaluator.evaluate(
        save_results=True,
        output_dir=os.path.join(args.output_dir, 'evaluation')
    )
    
    print("Evaluation completed!")
    print(f"Final validation loss: {trainer.best_val_loss:.4f}")
    
    # Print model performance
    print("\nModel Performance Summary:")
    print(f"  Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"  Training epochs: {trainer.current_epoch}")
    print(f"  Model size: {trainer.get_model_info()['model_info']['model_size_mb']:.2f} MB")


if __name__ == '__main__':
    main()

"""
Evaluation module for pose estimation models.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

from ..models import BasePoseModel
from ..data import PoseDataset
from .metrics import PoseMetrics
from ..inference import PoseDetector


class PoseEvaluator:
    """Comprehensive evaluator for pose estimation models."""
    
    def __init__(
        self,
        model: BasePoseModel,
        test_dataset: PoseDataset,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Trained pose estimation model
            test_dataset: Test dataset for evaluation
            device: Device to run evaluation on
        """
        self.model = model.to(device)
        self.test_dataset = test_dataset
        self.device = device
        self.metrics = PoseMetrics()
        
        # Set model to evaluation mode
        self.model.eval()
    
    def evaluate(
        self, 
        batch_size: int = 32,
        save_results: bool = True,
        output_dir: str = "evaluation_results"
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of the model.
        
        Args:
            batch_size: Batch size for evaluation
            save_results: Whether to save evaluation results
            output_dir: Directory to save results
        
        Returns:
            Dictionary containing evaluation results
        """
        print("Starting comprehensive evaluation...")
        
        # Create data loader
        from torch.utils.data import DataLoader
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if self.device == 'cuda' else False
        )
        
        # Run evaluation
        results = self._run_evaluation(test_loader)
        
        # Generate detailed analysis
        analysis = self._generate_analysis(results)
        
        # Combine results
        evaluation_results = {
            'metrics': results,
            'analysis': analysis,
            'model_info': self.model.get_model_info(),
            'dataset_info': {
                'num_samples': len(self.test_dataset),
                'num_keypoints': self.test_dataset.num_keypoints
            }
        }
        
        # Save results if requested
        if save_results:
            self._save_results(evaluation_results, output_dir)
        
        # Print summary
        self._print_evaluation_summary(results)
        
        return evaluation_results
    
    def _run_evaluation(self, test_loader) -> Dict[str, float]:
        """Run evaluation on test dataset."""
        all_metrics = []
        
        print("Evaluating on test dataset...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                # Move batch to device
                images = batch['image'].to(self.device)
                keypoints = batch['keypoints'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate metrics
                metrics = self.metrics.calculate_metrics(outputs, keypoints)
                all_metrics.append(metrics)
                
                # Print progress
                if (batch_idx + 1) % 10 == 0:
                    print(f"Processed {batch_idx + 1}/{len(test_loader)} batches")
        
        # Calculate average metrics
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [metrics[key] for metrics in all_metrics if key in metrics]
            avg_metrics[key] = np.mean(values)
        
        return avg_metrics
    
    def _generate_analysis(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Generate detailed analysis of evaluation results."""
        analysis = {}
        
        # Performance categorization
        analysis['performance_category'] = self._categorize_performance(metrics)
        
        # Keypoint analysis
        analysis['keypoint_analysis'] = self._analyze_keypoint_performance(metrics)
        
        # Error analysis
        analysis['error_analysis'] = self._analyze_errors(metrics)
        
        # Comparison with baselines
        analysis['baseline_comparison'] = self._compare_with_baselines(metrics)
        
        return analysis
    
    def _categorize_performance(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Categorize model performance."""
        pck_05 = metrics.get('pck_0.5', 0)
        mpjpe = metrics.get('mpjpe', float('inf'))
        
        if pck_05 >= 0.8 and mpjpe <= 50:
            category = "Excellent"
            color = "green"
        elif pck_05 >= 0.6 and mpjpe <= 80:
            category = "Good"
            color = "blue"
        elif pck_05 >= 0.4 and mpjpe <= 120:
            category = "Fair"
            color = "orange"
        else:
            category = "Poor"
            color = "red"
        
        return {
            'category': category,
            'color': color,
            'pck_05': pck_05,
            'mpjpe': mpjpe
        }
    
    def _analyze_keypoint_performance(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Analyze performance per keypoint."""
        keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        keypoint_errors = {}
        best_keypoints = []
        worst_keypoints = []
        
        for name in keypoint_names:
            error_key = f'{name}_error'
            if error_key in metrics:
                error = metrics[error_key]
                keypoint_errors[name] = error
        
        if keypoint_errors:
            # Sort by error
            sorted_keypoints = sorted(keypoint_errors.items(), key=lambda x: x[1])
            
            # Get best and worst performing keypoints
            best_keypoints = sorted_keypoints[:3]  # Top 3
            worst_keypoints = sorted_keypoints[-3:]  # Bottom 3
        
        return {
            'keypoint_errors': keypoint_errors,
            'best_keypoints': best_keypoints,
            'worst_keypoints': worst_keypoints,
            'average_error': np.mean(list(keypoint_errors.values())) if keypoint_errors else 0
        }
    
    def _analyze_errors(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Analyze error patterns."""
        error_analysis = {
            'mae': metrics.get('mae', 0),
            'rmse': metrics.get('rmse', 0),
            'mpjpe': metrics.get('mpjpe', 0),
            'median_pjpe': metrics.get('median_pjpe', 0)
        }
        
        # Error distribution analysis
        if error_analysis['mpjpe'] > 0:
            error_analysis['error_consistency'] = (
                error_analysis['median_pjpe'] / error_analysis['mpjpe']
            )
        else:
            error_analysis['error_consistency'] = 1.0
        
        return error_analysis
    
    def _compare_with_baselines(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Compare with baseline methods."""
        # Baseline results (example values - replace with actual baselines)
        baselines = {
            'MediaPipe': {'pck_0.5': 0.65, 'mpjpe': 45},
            'OpenPose': {'pck_0.5': 0.61, 'mpjpe': 52},
            'HRNet-W32': {'pck_0.5': 0.75, 'mpjpe': 38}
        }
        
        comparison = {}
        current_pck = metrics.get('pck_0.5', 0)
        current_mpjpe = metrics.get('mpjpe', float('inf'))
        
        for baseline_name, baseline_metrics in baselines.items():
            baseline_pck = baseline_metrics['pck_0.5']
            baseline_mpjpe = baseline_metrics['mpjpe']
            
            pck_improvement = (current_pck - baseline_pck) / baseline_pck * 100
            mpjpe_improvement = (baseline_mpjpe - current_mpjpe) / baseline_mpjpe * 100
            
            comparison[baseline_name] = {
                'pck_improvement': pck_improvement,
                'mpjpe_improvement': mpjpe_improvement,
                'better_pck': current_pck > baseline_pck,
                'better_mpjpe': current_mpjpe < baseline_mpjpe
            }
        
        return comparison
    
    def _save_results(self, results: Dict[str, Any], output_dir: str):
        """Save evaluation results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save metrics as JSON
        metrics_path = output_path / 'evaluation_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(results['metrics'], f, indent=2)
        
        # Save full results
        results_path = output_path / 'evaluation_results.json'
        
        # Convert numpy types for JSON serialization
        serializable_results = self._make_json_serializable(results)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Generate and save plots
        self._generate_evaluation_plots(results, output_path)
        
        print(f"Evaluation results saved to: {output_dir}")
    
    def _make_json_serializable(self, obj):
        """Convert numpy types to JSON serializable types."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, tuple):
            return list(obj)
        else:
            return obj
    
    def _generate_evaluation_plots(self, results: Dict[str, Any], output_path: Path):
        """Generate evaluation plots."""
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. PCK curve
        self._plot_pck_curve(results['metrics'], output_path)
        
        # 2. Per-keypoint error analysis
        self._plot_keypoint_errors(results['analysis']['keypoint_analysis'], output_path)
        
        # 3. Performance comparison
        self._plot_performance_comparison(results['analysis']['baseline_comparison'], output_path)
        
        # 4. Error distribution
        self._plot_error_distribution(results['metrics'], output_path)
    
    def _plot_pck_curve(self, metrics: Dict[str, float], output_path: Path):
        """Plot PCK curve."""
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
        pck_values = [metrics.get(f'pck_{t}', 0) for t in thresholds]
        
        plt.figure(figsize=(8, 6))
        plt.plot(thresholds, pck_values, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('PCK Threshold')
        plt.ylabel('PCK (%)')
        plt.title('PCK Curve')
        plt.grid(True, alpha=0.3)
        plt.xlim(0.05, 0.55)
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(output_path / 'pck_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_keypoint_errors(self, keypoint_analysis: Dict[str, Any], output_path: Path):
        """Plot per-keypoint error analysis."""
        keypoint_errors = keypoint_analysis['keypoint_errors']
        
        if not keypoint_errors:
            return
        
        names = list(keypoint_errors.keys())
        errors = list(keypoint_errors.values())
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(names, errors)
        
        # Color bars based on error level
        for bar, error in zip(bars, errors):
            if error < 30:
                bar.set_color('green')
            elif error < 60:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        plt.xlabel('Keypoint')
        plt.ylabel('Mean Error (pixels)')
        plt.title('Per-Keypoint Error Analysis')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'keypoint_errors.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_comparison(self, baseline_comparison: Dict[str, Any], output_path: Path):
        """Plot performance comparison with baselines."""
        methods = list(baseline_comparison.keys())
        pck_improvements = [baseline_comparison[method]['pck_improvement'] for method in methods]
        mpjpe_improvements = [baseline_comparison[method]['mpjpe_improvement'] for method in methods]
        
        x = np.arange(len(methods))
        width = 0.35
        
        plt.figure(figsize=(10, 6))
        bars1 = plt.bar(x - width/2, pck_improvements, width, label='PCK Improvement (%)')
        bars2 = plt.bar(x + width/2, mpjpe_improvements, width, label='MPJPE Improvement (%)')
        
        plt.xlabel('Baseline Method')
        plt.ylabel('Improvement (%)')
        plt.title('Performance Comparison with Baselines')
        plt.xticks(x, methods)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'baseline_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_error_distribution(self, metrics: Dict[str, float], output_path: Path):
        """Plot error distribution analysis."""
        error_metrics = ['mae', 'rmse', 'mpjpe', 'median_pjpe']
        error_values = [metrics.get(metric, 0) for metric in error_metrics]
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(error_metrics, error_values)
        
        # Color bars
        for bar in bars:
            bar.set_color('steelblue')
        
        plt.xlabel('Error Metric')
        plt.ylabel('Error Value')
        plt.title('Error Distribution Analysis')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'error_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _print_evaluation_summary(self, metrics: Dict[str, float]):
        """Print evaluation summary."""
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        
        # Key metrics
        print(f"PCK@0.5: {metrics.get('pck_0.5', 0):.3f}")
        print(f"MPJPE: {metrics.get('mpjpe', 0):.2f} pixels")
        print(f"MAE: {metrics.get('mae', 0):.2f} pixels")
        print(f"RMSE: {metrics.get('rmse', 0):.2f} pixels")
        
        # OKS metrics
        if 'oks_mean' in metrics:
            print(f"OKS Mean: {metrics['oks_mean']:.3f}")
        
        print("="*50)
    
    def evaluate_on_images(
        self, 
        image_paths: List[str],
        save_visualizations: bool = True,
        output_dir: str = "image_evaluation"
    ) -> List[Dict[str, Any]]:
        """
        Evaluate model on individual images.
        
        Args:
            image_paths: List of image paths
            save_visualizations: Whether to save visualizations
            output_dir: Output directory for results
        
        Returns:
            List of evaluation results per image
        """
        detector = PoseDetector(model_type='custom', device=self.device)
        detector.model = self.model
        
        results = []
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for i, image_path in enumerate(image_paths):
            print(f"Evaluating image {i+1}/{len(image_paths)}: {image_path}")
            
            try:
                # Detect poses
                result = detector.detect_poses(image_path, return_visualization=True)
                results.append(result)
                
                # Save visualization
                if save_visualizations and 'visualization' in result:
                    vis_path = output_path / f"result_{i:03d}.jpg"
                    import cv2
                    vis_image = cv2.cvtColor(result['visualization'], cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(vis_path), vis_image)
            
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append({'error': str(e)})
        
        return results

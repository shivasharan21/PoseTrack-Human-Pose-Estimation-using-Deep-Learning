"""
Script to download pre-trained model weights.
"""

import os
import urllib.request
import argparse
from pathlib import Path
import hashlib


def download_file(url, filename, expected_hash=None):
    """Download a file with optional hash verification."""
    print(f"Downloading {filename}...")
    
    try:
        urllib.request.urlretrieve(url, filename)
        
        if expected_hash:
            # Verify file hash
            with open(filename, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            
            if file_hash != expected_hash:
                print(f"Warning: Hash mismatch for {filename}")
                print(f"Expected: {expected_hash}")
                print(f"Got: {file_hash}")
            else:
                print(f"Hash verification passed for {filename}")
        
        print(f"Downloaded {filename} successfully")
        return True
    
    except Exception as e:
        print(f"Failed to download {filename}: {e}")
        return False


def main():
    """Main function to download model weights."""
    parser = argparse.ArgumentParser(description='Download Pre-trained Model Weights')
    parser.add_argument('--model', type=str, default='all',
                       choices=['all', 'hrnet', 'openpose', 'mediapipe'],
                       help='Model type to download')
    parser.add_argument('--output_dir', type=str, default='weights',
                       help='Output directory for weights')
    
    args = parser.parse_args()
    
    print("Downloading Pre-trained Model Weights")
    print("=" * 40)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Model weights URLs and hashes (example URLs - replace with actual ones)
    model_weights = {
        'hrnet': {
            'url': 'https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/releases/download/v1.0/pose_hrnet_w32_256x192.pth',
            'filename': 'hrnet_w32_256x192.pth',
            'hash': None  # Add actual hash if available
        },
        'openpose': {
            'url': 'https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases/download/v1.7.0/pose_iter_440000.caffemodel',
            'filename': 'openpose_pose_iter_440000.caffemodel',
            'hash': None
        }
    }
    
    # Download weights based on selection
    if args.model == 'all':
        models_to_download = list(model_weights.keys())
    else:
        models_to_download = [args.model]
    
    success_count = 0
    total_count = len(models_to_download)
    
    for model_name in models_to_download:
        if model_name in model_weights:
            weight_info = model_weights[model_name]
            output_path = output_dir / weight_info['filename']
            
            if download_file(
                weight_info['url'], 
                str(output_path), 
                weight_info['hash']
            ):
                success_count += 1
        else:
            print(f"No weights available for model: {model_name}")
    
    print(f"\nDownload completed: {success_count}/{total_count} models downloaded successfully")
    
    if success_count == total_count:
        print("All model weights downloaded successfully!")
    else:
        print("Some downloads failed. Please check the URLs and try again.")


if __name__ == '__main__':
    main()

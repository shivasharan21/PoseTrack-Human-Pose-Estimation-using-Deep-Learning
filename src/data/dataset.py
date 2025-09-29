"""
Dataset classes for pose estimation training and evaluation.
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
from typing import Dict, List, Tuple, Optional, Union
import albumentations as A
from albumentations.pytorch import ToTensorV2


class PoseDataset(Dataset):
    """Base class for pose estimation datasets."""
    
    def __init__(
        self,
        data_dir: str,
        annotations_file: str,
        image_size: Tuple[int, int] = (256, 256),
        num_keypoints: int = 17,
        augmentation: Optional[A.Compose] = None,
        is_training: bool = True
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing images
            annotations_file: Path to annotations JSON file
            image_size: Target image size (height, width)
            num_keypoints: Number of keypoints to detect
            augmentation: Albumentations augmentation pipeline
            is_training: Whether dataset is for training
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.num_keypoints = num_keypoints
        self.is_training = is_training
        
        # Load annotations
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Set up augmentations
        if augmentation is None:
            augmentation = self._get_default_augmentations()
        self.augmentation = augmentation
        
        # Process annotations
        self.samples = self._process_annotations()
    
    def _get_default_augmentations(self) -> A.Compose:
        """Get default augmentation pipeline."""
        if self.is_training:
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.RandomSaturation(p=0.3),
                A.RandomGamma(p=0.3),
                A.GaussNoise(p=0.2),
                A.Blur(p=0.2),
                A.Rotate(limit=30, p=0.5),
                A.RandomScale(scale_limit=0.2, p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        else:
            return A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    def _process_annotations(self) -> List[Dict]:
        """Process annotations into samples."""
        samples = []
        
        for annotation in self.annotations['annotations']:
            image_id = annotation['image_id']
            image_info = next(img for img in self.annotations['images'] if img['id'] == image_id)
            
            sample = {
                'image_id': image_id,
                'image_path': os.path.join(self.data_dir, image_info['file_name']),
                'keypoints': annotation['keypoints'],
                'bbox': annotation['bbox'],
                'area': annotation['area'],
                'iscrowd': annotation['iscrowd']
            }
            samples.append(sample)
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset."""
        sample = self.samples[idx]
        
        # Load image
        image = cv2.imread(sample['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
        
        # Prepare keypoints
        keypoints = np.array(sample['keypoints']).reshape(-1, 3)  # [x, y, visibility]
        
        # Normalize keypoints to image coordinates
        h, w = self.image_size
        keypoints[:, 0] = keypoints[:, 0] * w / sample['bbox'][2]  # x
        keypoints[:, 1] = keypoints[:, 1] * h / sample['bbox'][3]  # y
        
        # Apply augmentations
        augmented = self.augmentation(image=image, keypoints=keypoints)
        
        return {
            'image': augmented['image'],
            'keypoints': torch.tensor(augmented['keypoints'], dtype=torch.float32),
            'image_id': sample['image_id']
        }


class COCODataset(PoseDataset):
    """COCO pose estimation dataset."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_keypoints = 17  # COCO format
    
    def _process_annotations(self) -> List[Dict]:
        """Process COCO annotations."""
        samples = []
        
        for annotation in self.annotations['annotations']:
            if annotation['num_keypoints'] == 0:
                continue
                
            image_id = annotation['image_id']
            image_info = next(img for img in self.annotations['images'] if img['id'] == image_id)
            
            sample = {
                'image_id': image_id,
                'image_path': os.path.join(self.data_dir, image_info['file_name']),
                'keypoints': annotation['keypoints'],
                'bbox': annotation['bbox'],
                'area': annotation['area'],
                'iscrowd': annotation['iscrowd'],
                'num_keypoints': annotation['num_keypoints']
            }
            samples.append(sample)
        
        return samples


class MPIIDataset(PoseDataset):
    """MPII pose estimation dataset."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_keypoints = 16  # MPII format
    
    def _process_annotations(self) -> List[Dict]:
        """Process MPII annotations."""
        samples = []
        
        for annotation in self.annotations:
            sample = {
                'image_id': annotation['image'],
                'image_path': os.path.join(self.data_dir, annotation['image']),
                'keypoints': annotation['joints'],
                'bbox': annotation['bbox'],
                'area': annotation['area'],
                'iscrowd': False,
                'scale': annotation['scale'],
                'center': annotation['center']
            }
            samples.append(sample)
        
        return samples

"""
Data preprocessing and augmentation utilities for pose estimation.
"""

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import List, Tuple, Optional, Dict, Any
import torch
from PIL import Image
import imgaug as ia
from imgaug import augmenters as iaa


class PosePreprocessor:
    """Preprocessing utilities for pose estimation data."""
    
    def __init__(
        self,
        input_size: Tuple[int, int] = (256, 256),
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225]
    ):
        """
        Initialize preprocessor.
        
        Args:
            input_size: Target image size (height, width)
            mean: Normalization mean values
            std: Normalization std values
        """
        self.input_size = input_size
        self.mean = mean
        self.std = std
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] and apply mean/std normalization."""
        image = image.astype(np.float32) / 255.0
        
        # Apply normalization
        image = (image - np.array(self.mean)) / np.array(self.std)
        
        return image
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target size."""
        return cv2.resize(image, (self.input_size[1], self.input_size[0]))
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Complete preprocessing pipeline for an image."""
        # Resize
        image = self.resize_image(image)
        
        # Normalize
        image = self.normalize_image(image)
        
        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        return image
    
    def preprocess_keypoints(
        self, 
        keypoints: np.ndarray, 
        original_size: Tuple[int, int],
        target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Preprocess keypoints for training/inference.
        
        Args:
            keypoints: Array of keypoints [N, 3] (x, y, visibility)
            original_size: Original image size (height, width)
            target_size: Target image size (height, width)
        
        Returns:
            Processed keypoints
        """
        if target_size is None:
            target_size = self.input_size
        
        # Scale keypoints to target size
        scale_x = target_size[1] / original_size[1]
        scale_y = target_size[0] / original_size[0]
        
        keypoints = keypoints.copy()
        keypoints[:, 0] *= scale_x  # x
        keypoints[:, 1] *= scale_y  # y
        
        return keypoints


class ImageAugmentation:
    """Advanced image augmentation for pose estimation."""
    
    def __init__(self, is_training: bool = True):
        """
        Initialize augmentation pipeline.
        
        Args:
            is_training: Whether to apply training augmentations
        """
        self.is_training = is_training
        self._setup_albumentations()
        self._setup_imgaug()
    
    def _setup_albumentations(self):
        """Setup Albumentations augmentation pipeline."""
        if self.is_training:
            self.albumentations = A.Compose([
                # Geometric transformations
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=30, p=0.5),
                A.RandomScale(scale_limit=0.2, p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.2,
                    rotate_limit=30,
                    p=0.5
                ),
                A.ElasticTransform(p=0.3),
                A.Perspective(p=0.3),
                
                # Color transformations
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.RandomSaturation(p=0.3),
                A.RandomGamma(p=0.3),
                A.HueSaturationValue(p=0.3),
                A.CLAHE(p=0.3),
                
                # Noise and blur
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.Blur(blur_limit=3, p=0.3),
                A.MotionBlur(blur_limit=3, p=0.3),
                
                # Weather effects
                A.RandomRain(p=0.2),
                A.RandomShadow(p=0.2),
                
                # Normalization
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        else:
            self.albumentations = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    def _setup_imgaug(self):
        """Setup imgaug augmentation pipeline."""
        if self.is_training:
            self.imgaug = iaa.Sequential([
                # Geometric transformations
                iaa.Sometimes(0.5, iaa.Fliplr(1.0)),
                iaa.Sometimes(0.5, iaa.Affine(
                    rotate=(-30, 30),
                    scale=(0.8, 1.2),
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}
                )),
                iaa.Sometimes(0.3, iaa.ElasticTransformation(alpha=50, sigma=5)),
                iaa.Sometimes(0.3, iaa.PerspectiveTransform(scale=(0.01, 0.1))),
                
                # Color transformations
                iaa.Sometimes(0.5, iaa.MultiplyBrightness((0.8, 1.2))),
                iaa.Sometimes(0.5, iaa.MultiplySaturation((0.8, 1.2))),
                iaa.Sometimes(0.3, iaa.MultiplyHue((0.8, 1.2))),
                iaa.Sometimes(0.3, iaa.GammaContrast((0.8, 1.2))),
                iaa.Sometimes(0.3, iaa.CLAHE()),
                
                # Noise and blur
                iaa.Sometimes(0.3, iaa.AdditiveGaussianNoise(scale=(10, 50))),
                iaa.Sometimes(0.3, iaa.GaussianBlur(sigma=(0, 1.0))),
                iaa.Sometimes(0.3, iaa.MotionBlur(k=3)),
                
                # Weather effects
                iaa.Sometimes(0.2, iaa.Rain(speed=(0.1, 0.3))),
                iaa.Sometimes(0.2, iaa.Snowflakes(flake_size=(0.1, 0.4), speed=(0.01, 0.05))),
            ], random_order=True)
        else:
            self.imgaug = iaa.Identity()
    
    def apply_albumentations(
        self, 
        image: np.ndarray, 
        keypoints: Optional[List[Tuple[float, float]]] = None
    ) -> Dict[str, Any]:
        """
        Apply Albumentations augmentations.
        
        Args:
            image: Input image
            keypoints: List of keypoint tuples (x, y)
        
        Returns:
            Dictionary with augmented image and keypoints
        """
        if keypoints is not None:
            return self.albumentations(image=image, keypoints=keypoints)
        else:
            return self.albumentations(image=image)
    
    def apply_imgaug(
        self, 
        image: np.ndarray, 
        keypoints: Optional[ia.KeypointsOnImage] = None
    ) -> Dict[str, Any]:
        """
        Apply imgaug augmentations.
        
        Args:
            image: Input image
            keypoints: imgaug KeypointsOnImage object
        
        Returns:
            Dictionary with augmented image and keypoints
        """
        if keypoints is not None:
            return self.imgaug(image=image, keypoints=keypoints)
        else:
            return self.imgaug(image=image)
    
    def create_heatmap_augmentation(self) -> A.Compose:
        """Create augmentation pipeline for heatmap generation."""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.RandomScale(scale_limit=0.2, p=0.5),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


def create_training_augmentations(
    image_size: Tuple[int, int] = (256, 256),
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> A.Compose:
    """
    Create training augmentation pipeline.
    
    Args:
        image_size: Target image size
        mean: Normalization mean
        std: Normalization std
    
    Returns:
        Albumentations compose object
    """
    return A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.RandomSaturation(p=0.3),
        A.RandomGamma(p=0.3),
        A.GaussNoise(p=0.2),
        A.Blur(p=0.2),
        A.Rotate(limit=30, p=0.5),
        A.RandomScale(scale_limit=0.2, p=0.5),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


def create_validation_augmentations(
    image_size: Tuple[int, int] = (256, 256),
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> A.Compose:
    """
    Create validation augmentation pipeline.
    
    Args:
        image_size: Target image size
        mean: Normalization mean
        std: Normalization std
    
    Returns:
        Albumentations compose object
    """
    return A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

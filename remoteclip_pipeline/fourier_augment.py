#!/usr/bin/env python3
"""
Fourier Amplitude Swap Augmentation

Domain generalization technique that swaps low-frequency amplitude
components between images while preserving phase (structure).
"""
from __future__ import annotations

import random
from typing import Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Optional dependencies
np: Any = None
Image: Any = None
torch: Any = None

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class FourierAmplitudeSwap:
    """
    Fourier Amplitude Swap Augmentation.
    
    Swaps low-frequency amplitude components between two images while
    preserving phase information. This helps with domain generalization
    by mixing the style of one image with the structure of another.
    
    Args:
        swap_ratio: Ratio of low-frequency components to swap (0.0 to 1.0)
        probability: Probability of applying the augmentation
    """
    
    def __init__(self, swap_ratio: float = 0.5, probability: float = 1.0):
        if not HAS_NUMPY:
            raise RuntimeError("numpy required for FourierAmplitudeSwap")
        
        self.swap_ratio = swap_ratio
        self.probability = probability
    
    def __call__(
        self,
        source_image: Any,
        target_image: Any
    ) -> Any:
        """
        Apply Fourier amplitude swap.
        
        Args:
            source_image: Source image (structure preserved)
            target_image: Target image (style transferred)
        
        Returns:
            Augmented image with source structure and mixed style
        """
        if random.random() > self.probability:
            return source_image
        
        if Image is None:
            return source_image
        
        # Convert to numpy
        if isinstance(source_image, Image.Image):
            src_arr = np.array(source_image).astype(np.float32)
            was_pil = True
        else:
            src_arr = np.array(source_image).astype(np.float32)
            was_pil = False
        
        if isinstance(target_image, Image.Image):
            tgt_arr = np.array(target_image).astype(np.float32)
        else:
            tgt_arr = np.array(target_image).astype(np.float32)
        
        # Resize target to match source if needed
        if src_arr.shape != tgt_arr.shape:
            tgt_pil = Image.fromarray(tgt_arr.astype(np.uint8))
            tgt_pil = tgt_pil.resize((src_arr.shape[1], src_arr.shape[0]))
            tgt_arr = np.array(tgt_pil).astype(np.float32)
        
        # Handle grayscale
        if len(src_arr.shape) == 2:
            src_arr = src_arr[:, :, np.newaxis]
            tgt_arr = tgt_arr[:, :, np.newaxis]
        
        result = np.zeros_like(src_arr)
        
        for c in range(src_arr.shape[2]):
            result[:, :, c] = self._swap_channel(
                src_arr[:, :, c],
                tgt_arr[:, :, c]
            )
        
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        if result.shape[2] == 1:
            result = result.squeeze(-1)
        
        if was_pil:
            return Image.fromarray(result)
        return result
    
    def _swap_channel(
        self,
        source: Any,
        target: Any
    ) -> Any:
        """Swap amplitude for single channel."""
        h, w = source.shape
        
        # FFT
        src_fft = np.fft.fft2(source)
        tgt_fft = np.fft.fft2(target)
        
        # Shift zero-frequency to center
        src_fft_shifted = np.fft.fftshift(src_fft)
        tgt_fft_shifted = np.fft.fftshift(tgt_fft)
        
        # Get amplitude and phase
        src_amp = np.abs(src_fft_shifted)
        src_phase = np.angle(src_fft_shifted)
        tgt_amp = np.abs(tgt_fft_shifted)
        
        # Create low-frequency mask
        mask = self._create_low_freq_mask(h, w, self.swap_ratio)
        
        # Swap low-frequency amplitude
        mixed_amp = src_amp.copy()
        mixed_amp[mask] = (
            (1 - self.swap_ratio) * src_amp[mask] + 
            self.swap_ratio * tgt_amp[mask]
        )
        
        # Reconstruct
        mixed_fft = mixed_amp * np.exp(1j * src_phase)
        mixed_fft = np.fft.ifftshift(mixed_fft)
        result = np.fft.ifft2(mixed_fft)
        
        return np.real(result)
    
    def _create_low_freq_mask(
        self,
        h: int,
        w: int,
        ratio: float
    ) -> Any:
        """Create mask for low-frequency region."""
        center_h, center_w = h // 2, w // 2
        
        # Radius for low-frequency region
        radius_h = int(h * ratio / 2)
        radius_w = int(w * ratio / 2)
        
        y, x = np.ogrid[:h, :w]
        mask = ((y - center_h) ** 2 / (radius_h + 1) ** 2 + 
                (x - center_w) ** 2 / (radius_w + 1) ** 2) <= 1
        
        return mask


class FourierMixUp:
    """
    Fourier-based MixUp augmentation.
    
    Creates interpolated samples in the frequency domain.
    """
    
    def __init__(
        self,
        alpha: float = 0.4,
        freq_ratio: float = 0.5
    ):
        if not HAS_NUMPY:
            raise RuntimeError("numpy required for FourierMixUp")
        
        self.alpha = alpha
        self.freq_ratio = freq_ratio
    
    def __call__(
        self,
        images: Any,
        labels: Optional[Any] = None
    ) -> Tuple[Any, Optional[Any]]:
        """
        Apply Fourier MixUp to batch.
        
        Args:
            images: Batch of images [B, C, H, W] (torch tensor)
            labels: Optional labels [B, ...] (torch tensor)
        
        Returns:
            Mixed images and mixed labels
        """
        if torch is None or not HAS_TORCH:
            return images, labels
        
        batch_size = images.shape[0]
        
        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Random permutation for pairs
        indices = torch.randperm(batch_size)
        
        # Mix in frequency domain
        mixed = self._frequency_mix(images, images[indices], lam)
        
        # Mix labels if provided
        if labels is not None:
            mixed_labels = lam * labels + (1 - lam) * labels[indices]
            return mixed, mixed_labels
        
        return mixed, None
    
    def _frequency_mix(
        self,
        x1: Any,
        x2: Any,
        lam: float
    ) -> Any:
        """Mix two batches in frequency domain."""
        # FFT along spatial dimensions
        x1_fft = torch.fft.fft2(x1)
        x2_fft = torch.fft.fft2(x2)
        
        # Mix amplitude
        amp1 = torch.abs(x1_fft)
        amp2 = torch.abs(x2_fft)
        phase1 = torch.angle(x1_fft)
        
        # Create frequency mask
        h, w = x1.shape[-2:]
        mask = self._torch_freq_mask(h, w, self.freq_ratio, x1.device)
        
        # Mix low frequencies
        mixed_amp = amp1.clone()
        mixed_amp = torch.where(
            mask.unsqueeze(0).unsqueeze(0),
            lam * amp1 + (1 - lam) * amp2,
            amp1
        )
        
        # Reconstruct
        mixed_fft = mixed_amp * torch.exp(1j * phase1)
        mixed = torch.fft.ifft2(mixed_fft)
        
        return torch.real(mixed)
    
    def _torch_freq_mask(
        self,
        h: int,
        w: int,
        ratio: float,
        device: Any
    ) -> Any:
        """Create low-frequency mask as torch tensor."""
        center_h, center_w = h // 2, w // 2
        radius_h = int(h * ratio / 2)
        radius_w = int(w * ratio / 2)
        
        y = torch.arange(h, device=device).float()
        x = torch.arange(w, device=device).float()
        
        y = (y - center_h) ** 2 / (radius_h + 1) ** 2
        x = (x - center_w) ** 2 / (radius_w + 1) ** 2
        
        mask = y.unsqueeze(1) + x.unsqueeze(0)
        mask = mask <= 1
        
        return mask


class AdaptiveFourierAugment:
    """
    Adaptive Fourier augmentation with learnable parameters.
    
    Adjusts swap ratio based on domain gap statistics.
    """
    
    def __init__(
        self,
        initial_ratio: float = 0.5,
        min_ratio: float = 0.1,
        max_ratio: float = 0.9,
        ema_decay: float = 0.99
    ):
        if not HAS_NUMPY:
            raise RuntimeError("numpy required")
        
        self.swap_ratio = initial_ratio
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.ema_decay = ema_decay
        
        # Statistics for adaptation
        self.domain_gap_ema: Optional[float] = None
        self.base_augmentor = FourierAmplitudeSwap(swap_ratio=initial_ratio)
    
    def __call__(
        self,
        source_image: Any,
        target_image: Any
    ) -> Any:
        """Apply adaptive augmentation."""
        self.base_augmentor.swap_ratio = self.swap_ratio
        return self.base_augmentor(source_image, target_image)
    
    def update_from_gap(self, domain_gap: float):
        """
        Update swap ratio based on domain gap.
        
        Higher domain gap → higher swap ratio to close the gap.
        """
        if self.domain_gap_ema is None:
            self.domain_gap_ema = domain_gap
        else:
            self.domain_gap_ema = (
                self.ema_decay * self.domain_gap_ema + 
                (1 - self.ema_decay) * domain_gap
            )
        
        # Adjust ratio based on gap
        # Higher gap = more aggressive swapping
        new_ratio = self.min_ratio + (self.max_ratio - self.min_ratio) * self.domain_gap_ema
        self.swap_ratio = np.clip(new_ratio, self.min_ratio, self.max_ratio)
    
    def get_stats(self) -> dict:
        """Get current statistics."""
        return {
            'swap_ratio': self.swap_ratio,
            'domain_gap_ema': self.domain_gap_ema
        }

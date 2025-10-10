"""
Audio quality metrics for evaluation: SI-SNR.
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, Optional, Tuple
import warnings

logger = logging.getLogger(__name__)

try:
    from torchmetrics import ScaleInvariantSignalNoiseRatio
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False
    logger.warning("torchmetrics not available. SI-SNR will use custom implementation.")


class AudioMetrics:
    """Audio quality metrics calculator for SI-SNR."""
    
    def __init__(self, sample_rate: int = 24000, device: str = 'cpu'):
        """
        Initialize audio metrics calculator.
        
        Args:
            sample_rate: Sample rate of audio signals
            device: Device to run computations on
        """
        self.sample_rate = sample_rate
        self.device = device
        
        # Initialize SI-SNR metric
        if TORCHMETRICS_AVAILABLE:
            self.si_snr_metric = ScaleInvariantSignalNoiseRatio().to(device)
        else:
            self.si_snr_metric = None
    
    
    def compute_si_snr(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Compute Scale-Invariant Signal-to-Noise Ratio.
        
        Args:
            pred: Predicted audio [batch, channels, time] or [channels, time]
            target: Target audio [batch, channels, time] or [channels, time]
            
        Returns:
            SI-SNR value
        """
        # Ensure same shape
        if pred.shape != target.shape:
            min_length = min(pred.shape[-1], target.shape[-1])
            pred = pred[..., :min_length]
            target = target[..., :min_length]
        
        # Convert to mono if multi-channel
        if pred.dim() > 2:
            pred = pred.mean(dim=-2)  # Average across channels
        if target.dim() > 2:
            target = target.mean(dim=-2)  # Average across channels
        
        # Flatten batch dimension if present
        if pred.dim() > 1:
            pred = pred.view(-1, pred.shape[-1])
            target = target.view(-1, target.shape[-1])
        
        if TORCHMETRICS_AVAILABLE and self.si_snr_metric is not None:
            # Use torchmetrics implementation
            with torch.no_grad():
                si_snr = self.si_snr_metric(pred, target)
                return si_snr.item()
        else:
            # Custom SI-SNR implementation
            return self._compute_si_snr_custom(pred, target)
    
    def _compute_si_snr_custom(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Custom SI-SNR implementation."""
        # Ensure same length
        min_length = min(pred.shape[-1], target.shape[-1])
        pred = pred[..., :min_length]
        target = target[..., :min_length]
        
        # Compute SI-SNR
        eps = 1e-8
        
        # Zero-mean
        pred = pred - pred.mean(dim=-1, keepdim=True)
        target = target - target.mean(dim=-1, keepdim=True)
        
        # Compute s_target
        s_target = (target * pred).sum(dim=-1, keepdim=True) / (target * target).sum(dim=-1, keepdim=True).clamp(min=eps)
        s_target = s_target * target
        
        # Compute SI-SNR
        si_snr = 10 * torch.log10(
            (s_target * s_target).sum(dim=-1) / 
            ((pred - s_target) * (pred - s_target)).sum(dim=-1).clamp(min=eps)
        )
        
        return si_snr.mean().item()
    
    def compute_batch_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """
        Compute SI-SNR metrics for a batch.
        
        Args:
            pred: Predicted audio [batch, channels, time]
            target: Target audio [batch, channels, time]
            
        Returns:
            Dictionary with metric values
        """
        metrics = {}
        
        # Compute SI-SNR
        try:
            si_snr = self.compute_si_snr(pred, target)
            metrics['si_snr'] = si_snr
        except Exception as e:
            logger.warning(f"SI-SNR computation failed: {e}")
            metrics['si_snr'] = 0.0
        
        return metrics


def create_audio_metrics(sample_rate: int = 24000, device: str = 'cpu') -> AudioMetrics:
    """
    Create audio metrics calculator.
    
    Args:
        sample_rate: Sample rate of audio signals
        device: Device to run computations on
        
    Returns:
        AudioMetrics instance
    """
    return AudioMetrics(sample_rate=sample_rate, device=device)

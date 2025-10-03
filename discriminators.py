# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Discriminators for adversarial training of EnCodec models.
These discriminators work on reconstructed audio (continuous embeddings → decoder → audio).
"""

import typing as tp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from einops import rearrange


class NormConv2d(nn.Module):
    """2D convolution with normalization."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 groups: int = 1, bias: bool = True, norm: str = 'weight_norm'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                              padding, dilation, groups, bias)
        
        if norm == 'weight_norm':
            self.conv = nn.utils.weight_norm(self.conv)
        elif norm == 'spectral_norm':
            self.conv = nn.utils.spectral_norm(self.conv)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class NormConv1d(nn.Module):
    """1D convolution with normalization."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 groups: int = 1, bias: bool = True, norm: str = 'weight_norm'):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, 
                              padding, dilation, groups, bias)
        
        if norm == 'weight_norm':
            self.conv = nn.utils.weight_norm(self.conv)
        elif norm == 'spectral_norm':
            self.conv = nn.utils.spectral_norm(self.conv)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class MultiDiscriminator(nn.Module):
    """Base class for multi-discriminator systems."""
    def __init__(self):
        super().__init__()
    
    @property
    def num_discriminators(self) -> int:
        """Number of discriminators."""
        raise NotImplementedError
    
    def forward(self, x: torch.Tensor) -> tp.Tuple[tp.List[torch.Tensor], tp.List[torch.Tensor]]:
        """
        Forward pass through all discriminators.
        
        Args:
            x: Input audio tensor (B, C, T)
            
        Returns:
            logits: List of discriminator outputs
            fmaps: List of feature maps from each discriminator
        """
        raise NotImplementedError


def get_2d_padding(kernel_size: tp.Tuple[int, int], dilation: tp.Tuple[int, int] = (1, 1)):
    """Calculate 2D padding for given kernel size and dilation."""
    return (((kernel_size[0] - 1) * dilation[0]) // 2, ((kernel_size[1] - 1) * dilation[1]) // 2)


class DiscriminatorSTFT(nn.Module):
    """STFT sub-discriminator for MS-STFT discriminator.
    
    Args:
        filters (int): Number of filters in convolutions
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        n_fft (int): Size of FFT for this scale
        hop_length (int): Length of hop between STFT windows
        win_length (int): Window size for STFT
        max_filters (int): Maximum number of filters
        filters_scale (int): Growth factor for filters
        kernel_size (tuple): Inner Conv2d kernel sizes (3, 8) as per paper
        dilations (list): Dilation rates for time dimension [1, 2, 4]
        stride (tuple): Stride over frequency axis (1, 2)
        normalized (bool): Whether to normalize STFT
        norm (str): Normalization method
        activation (str): Activation function
        activation_params (dict): Activation parameters
    """
    def __init__(self, filters: int, in_channels: int = 1, out_channels: int = 1,
                 n_fft: int = 1024, hop_length: int = 256, win_length: int = 1024, 
                 max_filters: int = 1024, filters_scale: int = 1, 
                 kernel_size: tp.Tuple[int, int] = (3, 8), 
                 dilations: tp.List = [1, 2, 4], stride: tp.Tuple[int, int] = (1, 2), 
                 normalized: bool = True, norm: str = 'weight_norm',
                 activation: str = 'LeakyReLU', 
                 activation_params: dict = {'negative_slope': 0.2}):
        super().__init__()
        
        assert len(kernel_size) == 2
        assert len(stride) == 2
        
        self.filters = filters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.normalized = normalized
        
        # Activation function
        self.activation = getattr(torch.nn, activation)(**activation_params)
        
        # STFT transform
        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            win_length=self.win_length, 
            window_fn=torch.hann_window,
            normalized=self.normalized, 
            center=False, 
            pad_mode=None, 
            power=None
        )
        
        # STFT gives complex output, so we have 2 channels per input channel
        spec_channels = 2 * self.in_channels
        
        # Build convolutional layers
        self.convs = nn.ModuleList()
        
        # First conv: spec_channels -> filters
        self.convs.append(
            NormConv2d(spec_channels, self.filters, kernel_size=kernel_size, 
                      padding=get_2d_padding(kernel_size), norm=norm)
        )
        
        # Middle convs with increasing dilation and filters
        in_chs = min(filters_scale * self.filters, max_filters)
        for i, dilation in enumerate(dilations):
            out_chs = min((filters_scale ** (i + 1)) * self.filters, max_filters)
            self.convs.append(
                NormConv2d(in_chs, out_chs, kernel_size=kernel_size, 
                          stride=stride, dilation=(dilation, 1), 
                          padding=get_2d_padding(kernel_size, (dilation, 1)), 
                          norm=norm)
            )
            in_chs = out_chs
        
        # Final conv before output
        out_chs = min((filters_scale ** (len(dilations) + 1)) * self.filters, max_filters)
        self.convs.append(
            NormConv2d(in_chs, out_chs, kernel_size=(kernel_size[0], kernel_size[0]),
                      padding=get_2d_padding((kernel_size[0], kernel_size[0])), norm=norm)
        )
        
        # Output conv
        self.conv_post = NormConv2d(out_chs, self.out_channels,
                                   kernel_size=(kernel_size[0], kernel_size[0]),
                                   padding=get_2d_padding((kernel_size[0], kernel_size[0])),
                                   norm=norm)

    def forward(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, tp.List[torch.Tensor]]:
        """
        Forward pass through STFT discriminator.
        
        Args:
            x: Input audio tensor (B, C, T)
            
        Returns:
            logit: Final discriminator output
            fmap: List of feature maps from each layer
        """
        fmap = []
        
        # Convert audio to STFT spectrogram
        z = self.spec_transform(x)  # [B, 2, Freq, Frames, 2] (complex)
        
        # Separate real and imaginary parts and concatenate
        z = torch.cat([z.real, z.imag], dim=1)
        
        # Rearrange: (B, C, Freq, Time) -> (B, C, Time, Freq)
        z = rearrange(z, 'b c w t -> b c t w')
        
        # Pass through convolutional layers
        for i, layer in enumerate(self.convs):
            z = layer(z)
            z = self.activation(z)
            fmap.append(z)
        
        # Final output layer
        z = self.conv_post(z)
        fmap.append(z)
        
        return z, fmap


class MultiScaleSTFTDiscriminator(MultiDiscriminator):
    """Multi-Scale STFT (MS-STFT) discriminator as described in the EnCodec paper.
    
    Args:
        filters (int): Number of filters in convolutions (32 as per paper)
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        sep_channels (bool): Separate channels for stereo support
        n_ffts (Sequence[int]): FFT sizes for each scale [2048, 1024, 512, 256, 128]
        hop_lengths (Sequence[int]): Hop lengths for each scale
        win_lengths (Sequence[int]): Window sizes for each scale
        sequential (bool): Whether to process discriminators sequentially (reduces memory usage)
        **kwargs: Additional args for DiscriminatorSTFT
    """
    def __init__(self, filters: int = 32, in_channels: int = 1, out_channels: int = 1, 
                 sep_channels: bool = False,
                 n_ffts: tp.List[int] = [2048, 1024, 512, 256, 128], 
                 hop_lengths: tp.List[int] = [512, 256, 128, 64, 32],
                 win_lengths: tp.List[int] = [2048, 1024, 512, 256, 128], 
                 sequential: bool = True,
                 **kwargs):
        super().__init__()
        
        assert len(n_ffts) == len(hop_lengths) == len(win_lengths)
        
        self.sep_channels = sep_channels
        self.sequential = sequential
        
        # Create sub-discriminators for each scale
        self.discriminators = nn.ModuleList([
            DiscriminatorSTFT(
                filters=filters, 
                in_channels=in_channels, 
                out_channels=out_channels,
                n_fft=n_ffts[i], 
                win_length=win_lengths[i], 
                hop_length=hop_lengths[i], 
                **kwargs
            ) for i in range(len(n_ffts))
        ])

    @property
    def num_discriminators(self) -> int:
        return len(self.discriminators)

    def _separate_channels(self, x: torch.Tensor) -> torch.Tensor:
        """Separate channels for stereo processing."""
        B, C, T = x.shape
        return x.view(-1, 1, T)

    def forward(self, x: torch.Tensor) -> tp.Tuple[tp.List[torch.Tensor], tp.List[torch.Tensor]]:
        """
        Forward pass through all STFT discriminators.
        
        Args:
            x: Input audio tensor (B, C, T)
            
        Returns:
            logits: List of discriminator outputs from each scale
            fmaps: List of feature maps from each discriminator
        """
        logits = []
        fmaps = []
        
        if self.sequential:
            # Process each discriminator sequentially to reduce CUDA memory usage
            for i, disc in enumerate(self.discriminators):
                # Process one discriminator at a time
                logit, fmap = disc(x)
                logits.append(logit)
                fmaps.append(fmap)
                
                # Clear intermediate activations to free memory
                # This helps reduce peak memory usage during sequential processing
                if i < len(self.discriminators) - 1:  # Don't clear on last iteration
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
        else:
            # Process all discriminators in parallel (original behavior)
            for disc in self.discriminators:
                logit, fmap = disc(x)
                logits.append(logit)
                fmaps.append(fmap)
        
        return logits, fmaps


# Convenience function to create MS-STFT discriminator with paper defaults
def create_ms_stft_discriminator(in_channels: int = 1, out_channels: int = 1, 
                                filters: int = 32, sep_channels: bool = False,
                                sequential: bool = True) -> MultiScaleSTFTDiscriminator:
    """Create MS-STFT discriminator with EnCodec paper defaults.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        filters: Number of filters in convolutions
        sep_channels: Whether to separate channels for stereo support
        sequential: Whether to process discriminators sequentially (reduces memory usage)
    """
    return MultiScaleSTFTDiscriminator(
        filters=filters,
        in_channels=in_channels,
        out_channels=out_channels,
        sep_channels=sep_channels,
        n_ffts=[2048, 1024, 512, 256, 128],
        hop_lengths=[512, 256, 128, 64, 32],
        win_lengths=[2048, 1024, 512, 256, 128],
        sequential=sequential
    )

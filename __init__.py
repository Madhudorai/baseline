
"""
Baseline Autoencoder Training Package

A modular implementation of baseline autoencoder training with SEANet encoder/decoder.
"""

# Core components
from model import EncodecModel, CompressionModel

# SEANet architecture
from seanet import (
    SEANetEncoder,
    SEANetDecoder,
    SEANetResnetBlock,
    StreamableConv1d,
    StreamableConvTranspose1d,
    StreamableLSTM
)

# Model building
from model_builder import (
    build_encodec_model
)

# Training - using custom training loop in main.py

# Losses
from losses import ReconstructionLoss

# Discriminators
from discriminators import (
    MultiScaleSTFTDiscriminator,
    DiscriminatorSTFT,
    create_ms_stft_discriminator
)

# Adversarial losses
from adversarial_losses import (
    AdversarialLoss,
    FeatureMatchingLoss,
    create_adversarial_loss,
    get_adv_criterion,
    get_fake_criterion,
    get_real_criterion
)

# Data loading
from dataloader import (
    MultiChannelAudioDataset,
    create_dataloader
)

# Utilities
from utils import (
    pad_for_conv1d,
    pad1d,
    unpad1d,
    NormConv1d,
    NormConv2d,
    NormConvTranspose1d,
    NormConvTranspose2d,
    StreamableConv1d,
    StreamableConvTranspose1d
)

__version__ = "1.0.0"
__all__ = [
    # Core models
    "EncodecModel",
    "CompressionModel",
    
    # SEANet components
    "SEANetEncoder",
    "SEANetDecoder", 
    "SEANetResnetBlock",
    "StreamableConv1d",
    "StreamableConvTranspose1d",
    "StreamableLSTM",
    
    # Model building
    "build_encodec_model",
    
    # Training - using custom training loop in main.py
    
    # Losses
    "ReconstructionLoss",
    
    # Discriminators
    "MultiScaleSTFTDiscriminator",
    "DiscriminatorSTFT",
    "create_ms_stft_discriminator",
    
    # Adversarial losses
    "AdversarialLoss",
    "FeatureMatchingLoss",
    "create_adversarial_loss",
    "get_adv_criterion",
    "get_fake_criterion",
    "get_real_criterion",
    
    # Data loading
    "MultiChannelAudioDataset",
    "create_dataloader",
    
    # Utilities
    "pad_for_conv1d",
    "pad1d",
    "unpad1d",
    "NormConv1d",
    "NormConv2d",
    "NormConvTranspose1d",
    "NormConvTranspose2d",
    "StreamableConv1d",
    "StreamableConvTranspose1d",
]

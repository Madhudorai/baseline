# Baseline Autoencoder Training

A modular implementation of baseline autoencoder training with SEANet encoder/decoder for multichannel audio compression.

## Installation

```bash
# Install from source
git clone https://github.com/yourusername/encodec-training.git
cd encodec-training
pip install -e .

# All dependencies (including wandb) will be installed automatically
```

## Quick Start

```python
from baseline_training import build_encodec_model, create_dataloader, train_model

# Build model
model = build_encodec_model(
    sample_rate=24000,
    channels=32,
    n_filters=4,
    n_residual_layers=1,
    dimension=32
)

# Create dataloader
train_loader = create_dataloader(
    audio_dir="/path/to/audio",
    batch_size=8,
    channels=32,
    segment_duration=10.0
)

# Advanced: Get multiple samples from each file
large_loader = create_dataloader(
    audio_dir="/path/to/audio",
    batch_size=8,
    channels=32,
    segment_duration=10.0,
    dataset_size=1000,        # Get 1000 samples total
    min_file_duration=5.0,    # Only files longer than 5s
    random_crop=True          # Random segments for variety
)

# Train
train_model(model, train_loader, val_loader, num_epochs=100)
```

## Advanced Usage: Adversarial Training with Weighted Loss

```python
from baseline_training import (
    build_encodec_model, 
    create_ms_stft_discriminator,
    create_adversarial_loss
)

# Build model and discriminator
model = build_encodec_model(sample_rate=24000, channels=32)
discriminator = create_ms_stft_discriminator(in_channels=32)

# Create adversarial loss
adversarial_loss = create_adversarial_loss(
    discriminator=discriminator,
    optimizer=disc_optimizer,
    loss_type='hinge'
)

# Loss weights (paper parameters)
reconstruction_weight = 1.0      # Main loss
adversarial_weight = 3.0         # Adversarial loss  
feature_matching_weight = 3.0    # Feature matching loss

# Training loop with weighted loss
total_loss = (reconstruction_weight * recon_loss + 
             adversarial_weight * adv_loss + 
             feature_matching_weight * feat_loss)

total_loss.backward()
```

## Features

- **Baseline Autoencoder**: Continuous embedding encoder-decoder without quantization
- **SEANet Architecture**: State-of-the-art neural audio architecture  
- **Reconstruction Losses**: Time domain (L1) + Frequency domain (multi-scale mel-spectrogram)
- **MS-STFT Discriminator**: Multi-scale STFT-based discriminator for adversarial training
- **Adversarial Training**: Complete GAN-style training with hinge losses
- **Weighted Loss**: Simple weighted combination of reconstruction, adversarial, and feature matching losses
- **Smart Dataloader**: Get multiple samples from each audio file with random segments
- **Multichannel Audio Support**: Load and process multichannel .wav files (1, 2, 32, 64+ channels)
- **Modular Design**: Clean separation of concerns for easy customization

## Requirements

- Python 3.9+
- PyTorch 2.0.0+
- All dependencies are automatically installed via `setup.py`

## License

MIT License - see LICENSE file for details.

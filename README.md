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

**You must specify either `--1ch` or `--32ch` when running the training script:**

```bash
# For 1-channel (mono) training
python main.py --1ch

# For 32-channel (multichannel) training  
python main.py --32ch
```

## Configuration Differences

### 1-Channel Mode (`--1ch`)
- **Channels**: 1 (mono audio)
- **Batch Size**: 32 (larger batches for better gradient estimates)
- **Data Loading**: Randomly selects one channel from 32-channel source files
- **Model**: SEANet encoder/decoder configured for 1 input channel
- **Discriminator**: MS-STFT discriminator with 1 input channel
- **Use Case**: Standard mono audio compression, faster training

### 32-Channel Mode (`--32ch`)
- **Channels**: 32 (multichannel audio)
- **Batch Size**: 4 (smaller batches due to memory constraints)
- **Data Loading**: Uses all 32 channels from source files
- **Model**: SEANet encoder/decoder configured for 32 input channels
- **Discriminator**: MS-STFT discriminator with 32 input channels
- **Use Case**: Multichannel audio compression, spatial audio processing

## Training Parameters

Both modes use the same core training parameters:
- **Sample Rate**: 24kHz
- **Segment Duration**: 1 second
- **Epochs**: 300
- **Updates per Epoch**: 2000
- **Learning Rate**: 3e-4
- **Loss Weights**: λt=0.1, λf=1.0, λg=3.0, λfeat=3.0

## Data Loading Strategy

The dataloader automatically handles channel conversion:
- **1-channel mode**: Randomly picks one of the 32 available channels from each file
- **32-channel mode**: Uses all 32 channels from each file
- **Random sampling**: Each training sample comes from a random file and random time segment
- **Fixed validation**: Validation uses the same segments every epoch for consistent evaluation

## Training Process

The training script automatically handles:
- **Model Building**: Creates SEANet encoder/decoder with appropriate channel configuration
- **Discriminator Setup**: MS-STFT discriminator with matching input channels
- **Loss Balancing**: Gradient-balanced loss combination with paper-accurate weights
- **Wandb Logging**: Automatic experiment tracking and visualization
- **Model Saving**: Best model checkpointing based on validation loss

## Data Requirements

- **Audio Directory**: Place your audio files in `/scratch/eigenscape/`
- **File Format**: `.wav` files (any sample rate, will be resampled to 24kHz)
- **Channel Count**: Source files should have 32 channels for proper operation
- **Duration**: Files should be at least 1 second long (configurable)

## Features

- **Dual Channel Modes**: Support for both 1-channel (mono) and 32-channel (multichannel) training
- **Baseline Autoencoder**: Continuous embedding encoder-decoder without quantization
- **SEANet Architecture**: State-of-the-art neural audio architecture  
- **Reconstruction Losses**: Time domain (L1) + Frequency domain (multi-scale mel-spectrogram)
- **MS-STFT Discriminator**: Multi-scale STFT-based discriminator for adversarial training
- **Adversarial Training**: Complete GAN-style training with hinge losses
- **Gradient Balancing**: Advanced loss balancing for stable training
- **Smart Dataloader**: Random channel selection for 1ch mode, random file/segment sampling
- **Wandb Integration**: Automatic experiment tracking and visualization
- **Paper-Accurate**: Implements EnCodec paper parameters (300 epochs, 2000 updates/epoch)

## Requirements

- Python 3.9+
- PyTorch 2.0.0+
- All dependencies are automatically installed via `setup.py`

## License

MIT License - see LICENSE file for details.

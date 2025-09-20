
"""
Baseline autoencoder training with paper-accurate parameters and Weights & Biases logging.
Based on the EnCodec paper: 300 epochs, 2000 updates per epoch, batch size 64.
"""

import logging
import wandb
import torch
import torch.nn.functional as F
import argparse
import time
import soundfile as sf
import numpy as np
from pathlib import Path
import os

# Set CUDA memory configuration to avoid fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from model_builder import build_encodec_model, build_quantized_encodec_model
from dataloader import create_train_test_dataloaders, create_folder_based_dataloaders
from discriminators import create_ms_stft_discriminator
from adversarial_losses import create_adversarial_loss
from balancer import Balancer

logger = logging.getLogger(__name__)


def get_discriminator_input(audio_tensor, channels, channel_idx=None):
    """Get input for discriminator - use random channel for 32ch mode to save memory."""
    if channels == 32:
        # For 32ch mode, use a random channel for discriminator
        if channel_idx is None:
            # Pick a random channel if not specified
            channel_idx = torch.randint(0, audio_tensor.shape[1], (1,)).item()
        return audio_tensor[:, channel_idx:channel_idx + 1, :]  # Keep channel dimension
    else:
        # For 1ch mode, use as is
        return audio_tensor


def save_audio_samples_to_wandb(original_audio, reconstructed_audio, epoch, sample_rate=24000):
    """Save audio samples as wandb artifacts for monitoring reconstruction quality.
    
    Args:
        original_audio: Original audio tensor (batch, channels, samples)
        reconstructed_audio: Reconstructed audio tensor (batch, channels, samples)
        epoch: Current epoch number
        sample_rate: Audio sample rate
    """
    import tempfile
    import os
    
    # Take first 5 samples from the batch
    num_samples = min(5, original_audio.shape[0])
    
    # Create temporary directory for audio files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Prepare audio files for wandb
        audio_files = []
        
        for i in range(num_samples):
            # Convert to numpy and transpose to (samples, channels) for soundfile
            orig_np = original_audio[i].detach().cpu().numpy().T
            recon_np = reconstructed_audio[i].detach().cpu().numpy().T
            
            # Save original audio
            orig_filename = temp_path / f"epoch_{epoch:03d}_sample_{i}_original.wav"
            sf.write(orig_filename, orig_np, sample_rate)
            
            # Save reconstructed audio
            recon_filename = temp_path / f"epoch_{epoch:03d}_sample_{i}_reconstructed.wav"
            sf.write(recon_filename, recon_np, sample_rate)
            
            # Add to wandb files list
            audio_files.append(str(orig_filename))
            audio_files.append(str(recon_filename))
            
            print(f"Prepared audio samples: {orig_filename.name}, {recon_filename.name}")
        
        # Create wandb artifact
        artifact = wandb.Artifact(
            name=f"audio_samples_epoch_{epoch:03d}",
            type="audio_samples",
            description=f"Original and reconstructed audio samples from epoch {epoch}"
        )
        
        # Add all audio files to the artifact
        for audio_file in audio_files:
            artifact.add_file(audio_file)
        
        # Log the artifact to wandb
        wandb.log_artifact(artifact)
        print(f"Uploaded audio samples for epoch {epoch} to wandb as artifact: audio_samples_epoch_{epoch:03d}")
        
        # Also log individual audio files to wandb media for easy playback
        wandb_media = {}
        for i in range(num_samples):
            orig_file = temp_path / f"epoch_{epoch:03d}_sample_{i}_original.wav"
            recon_file = temp_path / f"epoch_{epoch:03d}_sample_{i}_reconstructed.wav"
            
            wandb_media[f"audio/epoch_{epoch:03d}_sample_{i}_original"] = wandb.Audio(str(orig_file), sample_rate=sample_rate)
            wandb_media[f"audio/epoch_{epoch:03d}_sample_{i}_reconstructed"] = wandb.Audio(str(recon_file), sample_rate=sample_rate)
        
        # Log audio media to wandb
        wandb.log(wandb_media)


def parse_args():
    """Parse command line arguments for 32ch vs 1ch vs 1chtoken vs 2branch configuration."""
    parser = argparse.ArgumentParser(description='Baseline autoencoder training')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--32ch', action='store_true', help='Use 32-channel configuration')
    group.add_argument('--1ch', action='store_true', help='Use 1-channel configuration')
    group.add_argument('--1chtoken', action='store_true', help='Use 1-channel configuration with quantization/tokenization')
    group.add_argument('--2branch', action='store_true', help='Use 2-branch mode with shared first codebook and diverse second codebook')
    return parser.parse_args()


def setup_wandb():
    """Setup wandb for baseline autoencoder training - handles login and configuration."""
    
    print("=" * 60)
    print("WEIGHTS & BIASES (WANDB) SETUP")
    print("=" * 60)
    
    # Check if wandb is installed
    try:
        import wandb
        print("✅ wandb is already installed")
    except ImportError:
        print("❌ wandb is not installed. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "wandb"])
        import wandb
        print("✅ wandb installed successfully")
    
    # Check if user is logged in
    try:
        api = wandb.Api()
        print("✅ You are logged in to wandb")
        print(f"   Username: {api.default_entity}")
    except Exception:
        print("❌ You are not logged in to wandb")
        print("\nTo login:")
        print("1. Go to https://wandb.ai/authorize")
        print("2. Copy your API key")
        print("3. Run: wandb login")
        print("4. Paste your API key when prompted")
        
        # Try to login automatically
        try:
            import getpass
            api_key = getpass.getpass("Enter your wandb API key (or press Enter to skip): ")
            if api_key.strip():
                subprocess.check_call(["wandb", "login", api_key])
                print("✅ wandb login successful!")
            else:
                print("⏭️  Skipping wandb login")
                print("   Please run 'wandb login' manually before training")
        except Exception as e:
            print(f"⚠️  Could not login automatically: {e}")
            print("   Please run 'wandb login' manually")
 
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("1. Make sure you're logged in: wandb login")
    print("2. Run your training: python3 main.py")
    print("3. View results at: https://wandb.ai/[your-username]/baseline-autoencoder-training")
    print("=" * 60)


def train_baseline_with_wandb(model, discriminator, train_loader, val_loader, 
                            model_optimizer, adversarial_loss,
                            num_epochs, updates_per_epoch, save_path, device, batch_size, channels, use_quantization=False, use_2branch=False):
    """Custom training loop with comprehensive wandb logging for baseline autoencoder."""
    
    from losses import ReconstructionLoss
    import torch
    
    # Create reconstruction loss
    reconstruction_loss = ReconstructionLoss(sample_rate=24000)
    
    # Create loss balancer with paper weights: λt=0.1, λf=1, λg=3, λfeat=3
    loss_weights = {
        'time_loss': 0.1,      # λt - time domain reconstruction
        'freq_loss': 1.0,      # λf - frequency domain reconstruction  
        'adv_loss': 3.0,       # λg - adversarial loss
        'feat_loss': 3.0       # λfeat - feature matching loss
    }
    
    # Note: quantization penalty (commitment loss) is NOT included in loss weights
    # as it's handled separately per EnCodec paper
    
    balancer = Balancer(
        weights=loss_weights,
        balance_grads=True,     # Enable gradient balancing
        total_norm=1.0,         # Reference norm for scaling
        ema_decay=0.999,        # EMA decay for gradient norm tracking
        per_batch_item=True,    # Compute norms per batch item
        epsilon=1e-12,          # Numerical stability
        monitor=True            # Store gradient ratio metrics
    )
    
    # Training state
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1}/{num_epochs}")
        
        # Training phase
        model.train()
        discriminator.train()
        
        train_metrics = {
            'time_reconstruction_loss': 0.0,
            'freq_reconstruction_loss': 0.0,
            'reconstruction_loss': 0.0,
            'adversarial_loss': 0.0,
            'feature_matching_loss': 0.0,
            'total_loss': 0.0,
            'discriminator_loss': 0.0,
            'learning_rate': model_optimizer.param_groups[0]['lr']
        }
        
        # Add quantization metrics if using quantization
        if use_quantization or use_2branch:
            train_metrics.update({
                'quantization_penalty': 0.0,
                'bandwidth_kbps': 0.0,
                'codes_entropy': 0.0
            })
            if use_2branch:
                train_metrics.update({
                    'diversity_loss': 0.0,
                    'consistency_loss': 0.0
                })
        
        # Training loop for this epoch
        for update in range(updates_per_epoch):
            try:
                # Debug: Batch loading
                print(f"DEBUG: Getting batch {update + 1}/{updates_per_epoch} for epoch {epoch + 1}")
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Get batch
                batch = next(iter(train_loader))
                print(f"DEBUG: Got batch with shape {batch.shape}")
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Check batch size (use dynamic batch size instead of hardcoded 4)
                if batch.shape[0] != batch_size:  # Skip incomplete batches
                    print(f"DEBUG: Skipping incomplete batch with size {batch.shape[0]} (expected {batch_size})")
                    continue
                
                # Move batch to device
                print(f"DEBUG: Moved batch to device {device}")
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}")
                batch = batch.to(device)
                
                # Forward pass: continuous embeddings → decoder → reconstructed audio
                print(f"DEBUG: Starting forward pass for batch {update + 1}")
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                if use_2branch:
                    # For 2-branch quantized model, use forward() to get TwoBranchQuantizedResult
                    quantized_result = model(batch)
                    reconstructed_audio = quantized_result.x
                    quantization_penalty = quantized_result.penalty
                    diversity_loss = quantized_result.diversity_loss
                    consistency_loss = quantized_result.consistency_loss
                    codes = quantized_result.codes
                    bandwidth = quantized_result.bandwidth
                    print(f"DEBUG: 2-Branch quantized batch, reconstructed shape: {reconstructed_audio.shape}")
                    print(f"DEBUG: Codes shape: {codes.shape}, Penalty: {quantization_penalty.item():.6f}")
                    print(f"DEBUG: Diversity loss: {diversity_loss.item():.6f}")
                    print(f"DEBUG: Consistency loss: {consistency_loss.item():.6f}")
                    print(f"DEBUG: Bandwidth: {bandwidth.item():.6f} kbps")
                elif use_quantization:
                    # For quantized model, use forward() to get QuantizedResult
                    quantized_result = model(batch)
                    reconstructed_audio = quantized_result.x
                    quantization_penalty = quantized_result.penalty
                    diversity_loss = torch.tensor(0.0, device=device)
                    consistency_loss = torch.tensor(0.0, device=device)
                    codes = quantized_result.codes
                    bandwidth = quantized_result.bandwidth
                    print(f"DEBUG: Quantized batch, reconstructed shape: {reconstructed_audio.shape}")
                    print(f"DEBUG: Codes shape: {codes.shape}, Penalty: {quantization_penalty.item():.6f}")
                    print(f"DEBUG: Bandwidth: {bandwidth.item():.6f} kbps")
                else:
                    # For baseline model, use encode/decode
                    continuous_embeddings = model.encode(batch)
                    print(f"DEBUG: Encoded batch, embeddings shape: {continuous_embeddings.shape}")
                    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    reconstructed_audio = model.decode(continuous_embeddings)
                    quantization_penalty = torch.tensor(0.0, device=device)
                    diversity_loss = torch.tensor(0.0, device=device)
                    consistency_loss = torch.tensor(0.0, device=device)
                    codes = None
                    bandwidth = torch.tensor(0.0, device=device)
                
                print(f"DEBUG: Decoded audio, shape: {reconstructed_audio.shape}")
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Ensure same length
                if reconstructed_audio.shape[-1] > batch.shape[-1]:
                    reconstructed_audio = reconstructed_audio[..., :batch.shape[-1]]
                elif reconstructed_audio.shape[-1] < batch.shape[-1]:
                    pad_length = batch.shape[-1] - reconstructed_audio.shape[-1]
                    reconstructed_audio = torch.nn.functional.pad(
                        reconstructed_audio, (0, pad_length), mode='reflect'
                    )
                
                # Step 1: Train discriminator
                print(f"DEBUG: Training discriminator for batch {update + 1}")
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Use same random channel for both original and reconstructed in 32ch mode
                if channels == 32:
                    channel_idx = torch.randint(0, batch.shape[1], (1,)).item()
                    disc_fake = get_discriminator_input(reconstructed_audio.detach(), channels, channel_idx)
                    disc_real = get_discriminator_input(batch, channels, channel_idx)
                else:
                    disc_fake = get_discriminator_input(reconstructed_audio.detach(), channels)
                    disc_real = get_discriminator_input(batch, channels)
                
                disc_loss = adversarial_loss.train_adv(
                    fake=disc_fake,
                    real=disc_real
                )
                print(f"DEBUG: Discriminator loss: {disc_loss.item():.6f}")
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Step 2: Train generator (baseline autoencoder) with loss balancer
                print(f"DEBUG: Training generator for batch {update + 1}")
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                model_optimizer.zero_grad()
                
                # Compute individual losses
                print(f"DEBUG: Computing reconstruction loss")
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}")
                recon_loss, recon_metrics = reconstruction_loss(reconstructed_audio, batch)
                
                print(f"DEBUG: Computing adversarial loss")
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}")
                # Use same random channel for both original and reconstructed in 32ch mode
                if channels == 32:
                    # Use the same channel_idx that was used for discriminator training
                    adv_fake = get_discriminator_input(reconstructed_audio, channels, channel_idx)
                    adv_real = get_discriminator_input(batch, channels, channel_idx)
                else:
                    adv_fake = get_discriminator_input(reconstructed_audio, channels)
                    adv_real = get_discriminator_input(batch, channels)
                adv_loss, feat_loss = adversarial_loss(adv_fake, adv_real)
                
                # Split reconstruction loss into time and frequency components
                time_loss = recon_metrics['time_loss']
                freq_loss = recon_metrics['freq_loss']
                
                # DEBUG: Print raw losses before balancer
                print(f"DEBUG - Raw losses for batch {update + 1}:")
                print(f"  recon_loss: {recon_loss.item():.6f}")
                print(f"  time_loss: {time_loss.item():.6f}")
                print(f"  freq_loss: {freq_loss.item():.6f}")
                print(f"  adv_loss: {adv_loss.item():.6f}")
                print(f"  feat_loss: {feat_loss.item():.6f}")
                print(f"  disc_loss: {disc_loss.item():.6f}")
                if use_quantization:
                    print(f"  quantization_penalty: {quantization_penalty.item():.6f}")
                print(f"  batch stats: min={batch.min().item():.6f}, max={batch.max().item():.6f}, mean={batch.mean().item():.6f}")
                print(f"  reconstructed stats: min={reconstructed_audio.min().item():.6f}, max={reconstructed_audio.max().item():.6f}, mean={reconstructed_audio.mean().item():.6f}")
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Use loss balancer for gradient balancing
                balanced_losses = {
                    'time_loss': time_loss,
                    'freq_loss': freq_loss,
                    'adv_loss': adv_loss,
                    'feat_loss': feat_loss
                }
                
                # Note: quantization_penalty (commitment loss) is NOT included in balancer
                # as per EnCodec paper - it applies only to encoder, not model output
                
                # Apply balancer - this handles gradient balancing and backward pass
                print(f"DEBUG: Applying loss balancer for batch {update + 1}")
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                effective_loss = balancer.backward(balanced_losses, reconstructed_audio)
                print(f"DEBUG: Effective loss: {effective_loss.item():.6f}")
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Add commitment loss, diversity loss, and consistency loss separately (not in balancer per EnCodec paper)
                if use_quantization or use_2branch:
                    effective_loss = effective_loss + quantization_penalty
                    print(f"DEBUG: Added commitment loss: {quantization_penalty.item():.6f}")
                    if use_2branch:
                        effective_loss = effective_loss + diversity_loss
                        effective_loss = effective_loss + consistency_loss
                        print(f"DEBUG: Added diversity loss: {diversity_loss.item():.6f}")
                        print(f"DEBUG: Added consistency loss: {consistency_loss.item():.6f}")
                    print(f"DEBUG: Total loss with commitment: {effective_loss.item():.6f}")
                
                # DEBUG: Print effective loss and balancer metrics
                print(f"DEBUG - Loss weights: λt=0.1, λf=1.0, λg=3.0, λfeat=3.0")
                print(f"DEBUG - Balancer metrics: {balancer.metrics}")
                
                # Gradient clipping
                print(f"DEBUG: Applying gradient clipping")
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}")
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Step the optimizer
                print(f"DEBUG: Stepping optimizer for batch {update + 1}")
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}")
                model_optimizer.step()
                
                # Update metrics
                train_metrics['time_reconstruction_loss'] += time_loss.item()
                train_metrics['freq_reconstruction_loss'] += freq_loss.item()
                train_metrics['reconstruction_loss'] += recon_loss.item()  # Keep total for compatibility
                train_metrics['adversarial_loss'] += adv_loss.item()
                train_metrics['feature_matching_loss'] += feat_loss.item()
                train_metrics['total_loss'] += effective_loss.item()
                train_metrics['discriminator_loss'] += disc_loss.item()
                
                # Update quantization metrics if using quantization
                if use_quantization or use_2branch:
                    train_metrics['quantization_penalty'] += quantization_penalty.item()
                    train_metrics['bandwidth_kbps'] += bandwidth.item()
                    
                    if use_2branch:
                        train_metrics['diversity_loss'] += diversity_loss.item()
                        train_metrics['consistency_loss'] += consistency_loss.item()
                    
                    # Calculate codes entropy (diversity measure)
                    if codes is not None:
                        # Calculate entropy for each codebook
                        codes_flat = codes.reshape(-1, codes.shape[1])  # [B*T, n_q]
                        entropy_sum = 0.0
                        for q in range(codes_flat.shape[1]):
                            code_counts = torch.bincount(codes_flat[:, q], minlength=model.quantizer.codebook_size)
                            probs = code_counts.float() / code_counts.sum()
                            probs = probs[probs > 0]  # Remove zero probabilities
                            entropy = -(probs * torch.log2(probs)).sum()
                            entropy_sum += entropy.item()
                        train_metrics['codes_entropy'] += entropy_sum / codes_flat.shape[1]  # Average across codebooks
                
                print(f"DEBUG: Completed batch {update + 1}/{updates_per_epoch} for epoch {epoch + 1}")
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}")
                print("=" * 50)
                
                # Clear memory every 5 batches to prevent memory leaks
                if (update + 1) % 5 == 0:
                    torch.cuda.empty_cache()
                    if torch.cuda.is_available():
                        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                        memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
                        print(f"DEBUG: Cleared CUDA cache after batch {update + 1}")
                        print(f"DEBUG: Memory - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB")
                
                # Force garbage collection every 20 batches
                if (update + 1) % 20 == 0:
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                
                # Check memory usage and warn if getting close to limit
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                    memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
                    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                    
                    # Warn if using more than 90% of GPU memory
                    if memory_allocated / total_memory > 0.9:
                        print(f"⚠️  WARNING: High GPU memory usage! {memory_allocated:.2f}GB / {total_memory:.2f}GB ({memory_allocated/total_memory*100:.1f}%)")
                        # Force aggressive cleanup
                        torch.cuda.empty_cache()
                        import gc
                        gc.collect()
                
                # Log to wandb every 100 steps
                if global_step % 100 == 0:
                    log_dict = {
                        'train/time_reconstruction_loss': time_loss.item(),
                        'train/freq_reconstruction_loss': freq_loss.item(),
                        'train/reconstruction_loss': recon_loss.item(),
                        'train/adversarial_loss': adv_loss.item(),
                        'train/feature_matching_loss': feat_loss.item(),
                        'train/effective_loss': effective_loss.item(),
                        'train/discriminator_loss': disc_loss.item(),
                        'train/learning_rate': model_optimizer.param_groups[0]['lr'],
                        'epoch': epoch,
                        'global_step': global_step,
                        'update': update
                    }
                    
                    # Add quantization metrics if using quantization
                    if use_quantization or use_2branch:
                        log_dict.update({
                            'train/quantization_penalty': quantization_penalty.item(),
                            'train/bandwidth_kbps': bandwidth.item(),
                        })
                        
                        if use_2branch:
                            log_dict['train/diversity_loss'] = diversity_loss.item()
                            log_dict['train/consistency_loss'] = consistency_loss.item()
                        
                        # Add codes entropy if available
                        if codes is not None:
                            codes_flat = codes.reshape(-1, codes.shape[1])
                            entropy_sum = 0.0
                            for q in range(codes_flat.shape[1]):
                                code_counts = torch.bincount(codes_flat[:, q], minlength=model.quantizer.codebook_size)
                                probs = code_counts.float() / code_counts.sum()
                                probs = probs[probs > 0]
                                entropy = -(probs * torch.log2(probs)).sum()
                                entropy_sum += entropy.item()
                            log_dict['train/codes_entropy'] = entropy_sum / codes_flat.shape[1]
                    
                    # Add balancer metrics if available
                    if balancer.metrics:
                        for key, value in balancer.metrics.items():
                            log_dict[f'train/balancer_{key}'] = value
                    
                    wandb.log(log_dict)
                                   
                global_step += 1
                
            except torch.cuda.OutOfMemoryError as e:
                print(f"❌ CUDA OOM Error at batch {update + 1}: {e}")
                print("🔄 Attempting recovery...")
                
                # Clear all caches
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                
                # Try to continue with next batch
                print("⚠️  Skipping this batch due to OOM")
                continue
            except Exception as e:
                logger.error(f"Error in training step: {e}")
                continue
        
        # Average metrics over the epoch
        for key in train_metrics:
            if key != 'learning_rate':
                train_metrics[key] /= updates_per_epoch
        
        # Validation phase
        model.eval()
        discriminator.eval()
        
        val_metrics = {
            'val_time_reconstruction_loss': 0.0,
            'val_freq_reconstruction_loss': 0.0,
            'val_reconstruction_loss': 0.0,
            'val_adversarial_loss': 0.0,
            'val_feature_matching_loss': 0.0,
            'val_total_loss': 0.0
        }
        
        # Add quantization metrics if using quantization
        if use_quantization or use_2branch:
            val_metrics.update({
                'val_quantization_penalty': 0.0,
                'val_bandwidth_kbps': 0.0,
                'val_codes_entropy': 0.0
            })
            if use_2branch:
                val_metrics.update({
                    'val_diversity_loss': 0.0,
                    'val_consistency_loss': 0.0
                })
        
        val_batches = 0
        first_batch_audio = None  # Store first batch for audio saving
        first_batch_reconstructed = None
        
        # Validation metrics for different audio consistency check
        different_audio_consistency = 0.0  # Should be HIGH (different first codebook tokens)
        different_audio_samples = 0
        
        # Store fixed different audio samples for consistent validation across epochs
        if not hasattr(train_baseline_with_wandb, 'fixed_different_audio_samples'):
            train_baseline_with_wandb.fixed_different_audio_samples = None
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch.shape[0] != batch_size:  # Skip incomplete batches
                    continue
                
                # Move batch to device
                batch = batch.to(device)
                
                # Forward pass
                if use_2branch:
                    # For 2-branch quantized model, use forward() to get TwoBranchQuantizedResult
                    quantized_result = model(batch)
                    reconstructed_audio = quantized_result.x
                    quantization_penalty = quantized_result.penalty
                    diversity_loss = quantized_result.diversity_loss
                    consistency_loss = quantized_result.consistency_loss
                    codes = quantized_result.codes
                    bandwidth = quantized_result.bandwidth
                elif use_quantization:
                    # For quantized model, use forward() to get QuantizedResult
                    quantized_result = model(batch)
                    reconstructed_audio = quantized_result.x
                    quantization_penalty = quantized_result.penalty
                    diversity_loss = torch.tensor(0.0, device=device)
                    consistency_loss = torch.tensor(0.0, device=device)
                    codes = quantized_result.codes
                    bandwidth = quantized_result.bandwidth
                else:
                    # For baseline model, use encode/decode
                    continuous_embeddings = model.encode(batch)
                    reconstructed_audio = model.decode(continuous_embeddings)
                    quantization_penalty = torch.tensor(0.0, device=device)
                    diversity_loss = torch.tensor(0.0, device=device)
                    consistency_loss = torch.tensor(0.0, device=device)
                    codes = None
                    bandwidth = torch.tensor(0.0, device=device)
                
                # Ensure same length
                if reconstructed_audio.shape[-1] > batch.shape[-1]:
                    reconstructed_audio = reconstructed_audio[..., :batch.shape[-1]]
                elif reconstructed_audio.shape[-1] < batch.shape[-1]:
                    pad_length = batch.shape[-1] - reconstructed_audio.shape[-1]
                    reconstructed_audio = torch.nn.functional.pad(
                        reconstructed_audio, (0, pad_length), mode='reflect'
                    )
                
                # Store first batch for audio saving (every 10 epochs)
                if batch_idx == 0 and (epoch + 1) % 10 == 0:
                    first_batch_audio = batch.clone()
                    first_batch_reconstructed = reconstructed_audio.clone()
                
                # Compute losses
                recon_loss, recon_metrics = reconstruction_loss(reconstructed_audio, batch)
                # Use same random channel for both original and reconstructed in 32ch mode
                if channels == 32:
                    channel_idx = torch.randint(0, batch.shape[1], (1,)).item()
                    adv_fake = get_discriminator_input(reconstructed_audio, channels, channel_idx)
                    adv_real = get_discriminator_input(batch, channels, channel_idx)
                else:
                    adv_fake = get_discriminator_input(reconstructed_audio, channels)
                    adv_real = get_discriminator_input(batch, channels)
                adv_loss, feat_loss = adversarial_loss(adv_fake, adv_real)
                
                # Split reconstruction loss into time and frequency components
                time_loss = recon_metrics['time_loss']
                freq_loss = recon_metrics['freq_loss']
                
                val_metrics['val_time_reconstruction_loss'] += time_loss.item()
                val_metrics['val_freq_reconstruction_loss'] += freq_loss.item()
                val_metrics['val_reconstruction_loss'] += recon_loss.item()
                val_metrics['val_adversarial_loss'] += adv_loss.item()
                val_metrics['val_feature_matching_loss'] += feat_loss.item()
                
                # Update quantization metrics if using quantization
                if use_quantization or use_2branch:
                    val_metrics['val_quantization_penalty'] += quantization_penalty.item()
                    val_metrics['val_bandwidth_kbps'] += bandwidth.item()
                    
                    if use_2branch:
                        val_metrics['val_diversity_loss'] += diversity_loss.item()
                        val_metrics['val_consistency_loss'] += consistency_loss.item()
                    
                    # Calculate codes entropy (diversity measure)
                    if codes is not None:
                        codes_flat = codes.reshape(-1, codes.shape[1])
                        entropy_sum = 0.0
                        for q in range(codes_flat.shape[1]):
                            code_counts = torch.bincount(codes_flat[:, q], minlength=model.quantizer.codebook_size)
                            probs = code_counts.float() / code_counts.sum()
                            probs = probs[probs > 0]
                            entropy = -(probs * torch.log2(probs)).sum()
                            entropy_sum += entropy.item()
                        val_metrics['val_codes_entropy'] += entropy_sum / codes_flat.shape[1]
                
                # For validation, compute effective loss manually without using balancer.backward
                # since we don't want to perform actual backward passes during validation
                balanced_losses = {
                    'time_loss': time_loss,
                    'freq_loss': freq_loss,
                    'adv_loss': adv_loss,
                    'feat_loss': feat_loss
                }
                
                # Note: quantization_penalty (commitment loss) is NOT included in balancer
                # as per EnCodec paper - it applies only to encoder, not model output
                
                # Compute effective loss using the same weights as the balancer
                total_weights = sum(balancer.weights.values())
                val_total = torch.tensor(0., device=batch.device, dtype=batch.dtype)
                for name, loss in balanced_losses.items():
                    weight = balancer.weights.get(name, 0.0)
                    val_total += (weight / total_weights) * loss.detach()
                
                # Add commitment loss, diversity loss, and consistency loss separately (not in balancer per EnCodec paper)
                if use_quantization or use_2branch:
                    val_total = val_total + quantization_penalty.detach()
                    if use_2branch:
                        val_total = val_total + diversity_loss.detach()
                        val_total = val_total + consistency_loss.detach()
                
                val_metrics['val_total_loss'] += val_total.item()
                
                val_batches += 1
                
                # Additional validation: Test different audio samples (same channel, different audio)
                if use_2branch and batch_idx == 0:  # Only process first batch to find different samples
                    # Find and store different audio samples for consistent validation across epochs
                    if train_baseline_with_wandb.fixed_different_audio_samples is None:
                        print("DEBUG: Finding different audio samples for validation...")
                        different_samples_found = False
                        
                        # Try to find different audio samples by comparing with next few batches
                        for next_batch_idx in range(1, min(10, len(val_loader))):
                            try:
                                next_batch = val_loader.dataset[next_batch_idx * batch_size:(next_batch_idx + 1) * batch_size]
                                if len(next_batch) == batch_size:
                                    next_batch = next_batch.to(device)
                                    
                                    # Check if inputs are actually different (different volumes, different sounds)
                                    current_audio_flat = batch.view(batch.shape[0], -1)
                                    next_audio_flat = next_batch.view(next_batch.shape[0], -1)
                                    
                                    # Normalize for cosine similarity
                                    current_audio_norm = F.normalize(current_audio_flat, p=2, dim=1)
                                    next_audio_norm = F.normalize(next_audio_flat, p=2, dim=1)
                                    
                                    # Audio similarity (we want this to be LOW for different audio)
                                    audio_similarity = (current_audio_norm * next_audio_norm).sum(dim=1).mean()
                                    
                                    # Check if audio samples are different enough (similarity < 0.8)
                                    if audio_similarity.item() < 0.8:
                                        # Store the different audio samples for future epochs
                                        train_baseline_with_wandb.fixed_different_audio_samples = {
                                            'sample_1': batch.clone().detach(),
                                            'sample_2': next_batch.clone().detach(),
                                            'audio_similarity': audio_similarity.item()
                                        }
                                        different_samples_found = True
                                        print(f"DEBUG: Found different audio samples with similarity: {audio_similarity.item():.4f}")
                                        break
                                    else:
                                        print(f"DEBUG: Batch {next_batch_idx} too similar: {audio_similarity.item():.4f}")
                            except Exception as e:
                                print(f"DEBUG: Error checking batch {next_batch_idx}: {e}")
                                continue
                        
                        if not different_samples_found:
                            print("DEBUG: Warning - Could not find sufficiently different audio samples for validation")
                    
                    # Use the fixed different audio samples for validation
                    if train_baseline_with_wandb.fixed_different_audio_samples is not None:
                        fixed_samples = train_baseline_with_wandb.fixed_different_audio_samples
                        sample_1 = fixed_samples['sample_1'].to(device)
                        sample_2 = fixed_samples['sample_2'].to(device)
                        
                        # Process both samples
                        result_1 = model(sample_1)
                        result_2 = model(sample_2)
                        
                        codes_1 = result_1.codes
                        codes_2 = result_2.codes
                        
                        # Reshape to pairs: [B//2, 2, n_q, T]
                        codes_1_pairs = codes_1.view(codes_1.shape[0] // 2, 2, codes_1.shape[1], -1)
                        codes_2_pairs = codes_2.view(codes_2.shape[0] // 2, 2, codes_2.shape[1], -1)
                        
                        # Test: Different audio should have different first codebook tokens
                        # Compare first codebook tokens between different audio samples
                        first_codebook_1 = codes_1_pairs[:, 0, 0, :]  # [B//2, T] - first codebook, first sample
                        first_codebook_2 = codes_2_pairs[:, 0, 0, :]  # [B//2, T] - first codebook, second sample
                        
                        # Calculate similarity (we want this to be LOW for different audio)
                        first_flat_1 = first_codebook_1.float().view(first_codebook_1.shape[0], -1)
                        first_flat_2 = first_codebook_2.float().view(first_codebook_2.shape[0], -1)
                        
                        # Normalize for cosine similarity
                        first_norm_1 = F.normalize(first_flat_1, p=2, dim=1)
                        first_norm_2 = F.normalize(first_flat_2, p=2, dim=1)
                        
                        # Cosine similarity (we want this to be LOW for different audio)
                        different_audio_sim = (first_norm_1 * first_norm_2).sum(dim=1).mean()
                        
                        # Different audio consistency: 1 - similarity (higher = more different)
                        different_audio_consistency += (1.0 - different_audio_sim.item())
                        different_audio_samples += 1
                        
                        # Debug: Print audio similarity to verify we're comparing different audio
                        print(f"DEBUG: Using fixed different audio samples with similarity: {fixed_samples['audio_similarity']:.4f}")
                        print(f"DEBUG: First codebook similarity: {different_audio_sim.item():.4f} (should be LOW for different audio)")
        
        # Average validation metrics
        if val_batches > 0:
            for key in val_metrics:
                val_metrics[key] /= val_batches
                
        # Average different audio metrics
        if different_audio_samples > 0:
            different_audio_consistency /= different_audio_samples
            val_metrics['val_different_audio_consistency'] = different_audio_consistency
        
        # Save audio samples every 10 epochs
        if (epoch + 1) % 10 == 0 and first_batch_audio is not None and first_batch_reconstructed is not None:
            print(f"Saving audio samples for epoch {epoch + 1}...")
            save_audio_samples_to_wandb(
                original_audio=first_batch_audio,
                reconstructed_audio=first_batch_reconstructed,
                epoch=epoch + 1,
                sample_rate=24000
            )
        
        # Log epoch metrics to wandb
        wandb.log({
            'epoch': epoch,
            'train/epoch_time_reconstruction_loss': train_metrics['time_reconstruction_loss'],
            'train/epoch_freq_reconstruction_loss': train_metrics['freq_reconstruction_loss'],
            'train/epoch_reconstruction_loss': train_metrics['reconstruction_loss'],
            'train/epoch_adversarial_loss': train_metrics['adversarial_loss'],
            'train/epoch_feature_matching_loss': train_metrics['feature_matching_loss'],
            'train/epoch_total_loss': train_metrics['total_loss'],
            'train/epoch_discriminator_loss': train_metrics['discriminator_loss'],
            **val_metrics
        })
        
        # Save best model (simplified - no wandb config to avoid loading issues)
        if val_metrics['val_total_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_total_loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'model_optimizer_state_dict': model_optimizer.state_dict(),
                'val_loss': best_val_loss
            }, save_path)
            print(f"Saved best model with validation loss: {best_val_loss:.6f}")
        
        # Log epoch summary
        print(f"Epoch {epoch + 1}/{num_epochs} completed:")
        print(f"  Train - Recon: {train_metrics['reconstruction_loss']:.6f}, "
                   f"Adv: {train_metrics['adversarial_loss']:.6f}, "
                   f"Feat: {train_metrics['feature_matching_loss']:.6f}")
        print(f"  Val - Total: {val_metrics['val_total_loss']:.6f}")
        
        # Log 2-branch specific metrics
        if use_2branch:
            print(f"  2-Branch Metrics:")
            print(f"    Consistency Loss: {val_metrics.get('val_consistency_loss', 0.0):.6f} (should decrease)")
            print(f"    Diversity Loss: {val_metrics.get('val_diversity_loss', 0.0):.6f} (should decrease)")
            print(f"    Different Audio Consistency: {val_metrics.get('val_different_audio_consistency', 0.0):.6f} (should be HIGH)")
        
        # Clear memory after each epoch to prevent memory leaks
        torch.cuda.empty_cache()
        print(f"DEBUG: Cleared CUDA cache after epoch {epoch + 1}")


def main():
    """Baseline autoencoder training with paper-accurate parameters and wandb logging."""
    
    # Parse command line arguments
    args = parse_args()
    
    # Configure based on channel selection
    if args.__dict__['32ch']:
        channels = 32
        batch_size = 4
        config_name = "baseline-32ch-24khz"
        use_quantization = False
        use_2branch = False
        print("Using 32-channel configuration")
    elif args.__dict__['1chtoken']:
        channels = 1
        batch_size = 1  # Reduced to 1 for testing
        config_name = "baseline-1chtoken-24khz"
        use_quantization = True
        use_2branch = False
        print("Using 1-channel configuration with quantization/tokenization")
    elif args.__dict__['2branch']:
        channels = 1
        batch_size = 2  # Need pairs for 2-branch mode
        config_name = "baseline-2branch-24khz"
        use_quantization = True
        use_2branch = True
        print("Using 2-branch mode with shared first codebook and diverse second codebook")
    else:  # 1ch
        channels = 1
        batch_size = 32
        config_name = "baseline-1ch-24khz"
        use_quantization = False
        use_2branch = False
        print("Using 1-channel configuration")
    
    # Setup wandb (will handle login if needed)
    setup_wandb()
    
    # Initialize wandb
    wandb_config = {
        "sample_rate": 24000,
        "channels": channels,
        "n_filters": 4,
        "n_residual_layers": 1,
        "dimension": 32,
        "causal": False,
        "epochs": 300,
        "updates_per_epoch": 2000,
        "batch_size": batch_size,
        "learning_rate": 3e-4,
        "beta1": 0.5,
        "beta2": 0.9,
        "segment_duration": 1.0,
        "use_quantization": use_quantization,
        "loss_weights": {
            "time_reconstruction": 0.1,  # λt (time domain)
            "freq_reconstruction": 1.0,  # λf (frequency domain)
            "adversarial": 3.0,          # λg (generator)
            "feature_matching": 3.0      # λfeat (feature matching)
        },
        "balancer": {
            "balance_grads": True,       # Enable gradient balancing
            "total_norm": 1.0,          # Reference norm for scaling
            "ema_decay": 0.999,         # EMA decay for gradient norm tracking
            "per_batch_item": True,     # Compute norms per batch item
            "epsilon": 1e-12,           # Numerical stability
            "monitor": True             # Store gradient ratio metrics
        }
    }
    
    # Add quantization parameters if using quantization
    if use_quantization or use_2branch:
        wandb_config["loss_weights"]["quantization_penalty"] = 1.0  # λq (quantization penalty)
        wandb_config["quantization"] = {
            "n_q": 2,                    # Number of codebooks
            "codebook_size": 1024,       # Size of each codebook
            "commitment_weight": 1.0,    # Weight for commitment loss
            "decay": 0.99,              # EMA decay for codebook updates
            "kmeans_init": True,        # Use k-means initialization
            "kmeans_iters": 50,         # Number of k-means iterations
            "threshold_ema_dead_code": 2.0  # Threshold for dead code replacement
        }
        if use_2branch:
            wandb_config["loss_weights"]["diversity_loss"] = 1.0  # λd (diversity loss)
            wandb_config["loss_weights"]["consistency_loss"] = 1.0  # λc (consistency loss)
            wandb_config["quantization"]["diversity_weight"] = 1.0  # Weight for diversity loss
    
    wandb.init(
        project="baseline-autoencoder-training",
        name=config_name,
        config=wandb_config
    )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Build the model
    if use_2branch:
        print("Building 2-branch quantized autoencoder model...")
        from model_builder import build_2branch_quantized_encodec_model
        model = build_2branch_quantized_encodec_model(
            sample_rate=24000,
            channels=channels,
            n_filters=4,
            n_residual_layers=1,
            dimension=32,
            causal=False,
            n_q=2,  # Number of codebooks
            codebook_size=1024,  # Size of each codebook
            commitment_weight=1.0  # Weight for commitment loss
        ).to(device)
    elif use_quantization:
        print("Building quantized autoencoder model...")
        model = build_quantized_encodec_model(
            sample_rate=24000,
            channels=channels,
            n_filters=4,
            n_residual_layers=1,
            dimension=32,
            causal=False,
            n_q=2,  # Number of codebooks
            codebook_size=1024,  # Size of each codebook
            commitment_weight=1.0  # Weight for commitment loss
        ).to(device)
    else:
        print("Building baseline autoencoder model...")
        model = build_encodec_model(
            sample_rate=24000,
            channels=channels,
            n_filters=4,
            n_residual_layers=1,
            dimension=32,
            causal=False
        ).to(device)
    
    # Build MS-STFT discriminator
    print("Building MS-STFT discriminator...")
    # Use 1 channel for discriminator input to save memory (even for 32ch mode)
    disc_input_channels = 1 if channels == 32 else channels
    discriminator = create_ms_stft_discriminator(
        in_channels=disc_input_channels,
        out_channels=1,
        filters=32
    ).to(device)
    
    # Create optimizers with paper parameters
    print("Creating optimizers...")
    model_optimizer = torch.optim.Adam(
        model.parameters(),
        lr=3e-4,      # Paper: 3 · 10^-4
        betas=(0.5, 0.9),  # Paper: β1 = 0.5, β2 = 0.9
        eps=1e-8
    )
    
    disc_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=3e-4,
        betas=(0.5, 0.9),
        eps=1e-8
    )
    
    # Create adversarial loss
    print("Creating adversarial loss...")
    adversarial_loss = create_adversarial_loss(
        discriminator=discriminator,
        optimizer=disc_optimizer,
        loss_type='hinge',  # Paper uses hinge loss
        use_feature_matching=True
    )
    
    # Loss weights from paper: λt=0.1, λf=1, λg=3, λfeat=3 (24kHz model)
    print("Using paper loss weights: λt=0.1, λf=1, λg=3, λfeat=3 (24kHz model)")
    
    # Create dataloaders with folder-based split
    print("Creating dataloaders...")
    
    # Define folder-based train/validation split
    train_folders = ["Beach", "Busy Street", "Park", "Pedestrian Zone", "Quiet Street", "Shopping Centre"]
    val_folders = ["Woodland", "Train Station"]
    
    # Create folder-based dataloaders with dynamic configuration
    if use_2branch:
        from dataloader import create_2branch_dataloaders
        train_loader, val_loader = create_2branch_dataloaders(
            audio_dir="/scratch/eigenscape/",
            train_folders=train_folders,
            val_folders=val_folders,
            batch_size=batch_size,           
            sample_rate=24000,
            segment_duration=1.0,    # 1 second segments
            channels=channels,
            train_dataset_size=8000,  # 2000 updates × batch_size = 8000 samples per epoch
            val_dataset_size=2000,    # 2000 validation samples (fixed each epoch)
            min_file_duration=1.0,    
            random_crop=True,
            num_workers=4,  # Enable multiprocessing for faster data loading
            pin_memory=True,  # Enable pinned memory for faster GPU transfer
            persistent_workers=True  # Keep workers alive between epochs
        )
    else:
        train_loader, val_loader = create_folder_based_dataloaders(
            audio_dir="/scratch/eigenscape/",
            train_folders=train_folders,
            val_folders=val_folders,
            batch_size=batch_size,           
            sample_rate=24000,
            segment_duration=1.0,    # 1 second segments
            channels=channels,
            train_dataset_size=8000,  # 2000 updates × batch_size = 8000 samples per epoch
            val_dataset_size=2000,    # 2000 validation samples (fixed each epoch)
            min_file_duration=1.0,    
            random_crop=True,
            num_workers=4,  # Enable multiprocessing for faster data loading
            pin_memory=True,  # Enable pinned memory for faster GPU transfer
            persistent_workers=True  # Keep workers alive between epochs
        )
    
    # Log dataset information
    print(f"Training dataset size: {len(train_loader.dataset)} samples")
    print(f"Validation dataset size: {len(val_loader.dataset)} samples")
    print(f"Training batches per epoch: {len(train_loader)}")
    print(f"Validation batches per epoch: {len(val_loader)}")
    
    # Paper: 300 epochs, 2000 updates per epoch
    num_epochs = 300
    updates_per_epoch = 2000  # Paper: 2000 updates per epoch
    
    print(f"Starting training for {num_epochs} epochs...")
    print(f"Each epoch has {updates_per_epoch} updates (8k samples)")
    print(f"Validation uses FIXED 2k samples (same segments every epoch)")
    print(f"Total updates: {num_epochs * updates_per_epoch}")
    print(f"Training folders: {train_folders}")
    print(f"Validation folders: {val_folders}")
    
    # Custom training loop with wandb logging
    # Use different model names for different channel modes
    if channels == 32:
        save_path = Path("bestmodel_32ch.pth")
    elif use_2branch:  # 2branch
        save_path = Path("bestmodel_2branch.pth")
    elif use_quantization:  # 1chtoken
        save_path = Path("bestmodel_1chtoken.pth")
    else:  # 1ch
        save_path = Path("bestmodel_1ch.pth")
    
    train_baseline_with_wandb(
        model=model,
        discriminator=discriminator,
        train_loader=train_loader,
        val_loader=val_loader,
        model_optimizer=model_optimizer,
        adversarial_loss=adversarial_loss,
        num_epochs=num_epochs,
        updates_per_epoch=updates_per_epoch,
        save_path=save_path,
        device=device,
        batch_size=batch_size,
        channels=channels,
        use_quantization=use_quantization,
        use_2branch=use_2branch
    )
    
    print("Training completed!")
    wandb.finish()


if __name__ == "__main__":
    main()

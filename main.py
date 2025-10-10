
"""
Baseline autoencoder training with paper-accurate parameters and Weights & Biases logging.
Based on the EnCodec paper: 300 epochs, 2000 updates per epoch, batch size 64.
"""

import logging
import wandb
import torch
import argparse
import time
import numpy as np
import os
from pathlib import Path

from model_builder import build_encodec_model
from dataloader import create_train_test_dataloaders, create_folder_based_dataloaders
from discriminators import create_ms_stft_discriminator
from adversarial_losses import create_adversarial_loss
from balancer import Balancer
from metrics import create_audio_metrics

logger = logging.getLogger(__name__)


def get_discriminator_input(audio_tensor):
    """Get input for discriminator - use audio as is."""
    return audio_tensor


def load_checkpoint(checkpoint_path, model, discriminator, model_optimizer, adversarial_loss, device):
    """Load checkpoint and return training state."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model states
    model.load_state_dict(checkpoint['model_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    model_optimizer.load_state_dict(checkpoint['model_optimizer_state_dict'])
    adversarial_loss.optimizer.load_state_dict(checkpoint['disc_optimizer_state_dict'])
    
    # Get training state
    start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
    best_val_loss = checkpoint['val_loss']
    
    print(f"✅ Checkpoint loaded successfully!")
    print(f"   Resuming from epoch: {start_epoch}")
    print(f"   Best validation loss: {best_val_loss:.6f}")
    print(f"   Model weights: ✅")
    print(f"   Discriminator weights: ✅")
    print(f"   Generator optimizer: ✅")
    print(f"   Discriminator optimizer: ✅")
    
    return start_epoch, best_val_loss




def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Baseline autoencoder training')
    parser.add_argument('--resume', type=str, default=None, 
                       help='Path to checkpoint file to resume from (e.g., bestmodel_epoch_25_loss_1.2345.pth)')
    parser.add_argument('--checkpoint-dir', type=str, default='.', 
                       help='Directory to save checkpoints (default: current directory)')
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


def train_baseline_with_wandb(model, discriminator, train_loader, val_loader, test_loader,
                            model_optimizer, adversarial_loss,
                            num_epochs, updates_per_epoch, device, batch_size, 
                            start_epoch=0, best_val_loss=float('inf'), checkpoint_dir='.'):
    """Custom training loop with comprehensive wandb logging for baseline autoencoder."""
    
    from losses import ReconstructionLoss
    import torch
    
    # Create reconstruction loss
    reconstruction_loss = ReconstructionLoss(sample_rate=24000)
    
    # Create audio quality metrics
    audio_metrics = create_audio_metrics(sample_rate=24000, device=device)
    
    # Create loss balancer with paper weights: λt=0.1, λf=1, λg=3, λfeat=3
    loss_weights = {
        'time_loss': 0.1,      # λt - time domain reconstruction
        'freq_loss': 1.0,      # λf - frequency domain reconstruction  
        'adv_loss': 3.0,       # λg - adversarial loss
        'feat_loss': 3.0       # λfeat - feature matching loss
    }
    
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
    
    print(f"Starting training from epoch {start_epoch + 1} to {num_epochs}")
    print(f"Initial best validation loss: {best_val_loss:.6f}")
    
    for epoch in range(start_epoch, num_epochs):
        print(f"Starting epoch {epoch + 1}/{num_epochs}")
        
        # Training phase
        model.train()
        discriminator.train()
        
        train_metrics = {
            'loss_t': 0.0,        # Time domain reconstruction loss
            'loss_f': 0.0,        # Frequency domain reconstruction loss
            'loss_feat': 0.0,     # Feature matching loss
            'loss_g': 0.0,        # Adversarial loss for generator
            'loss_G': 0.0,        # Effective loss (LG)
            'loss_disc': 0.0,     # Discriminator loss
            'learning_rate': model_optimizer.param_groups[0]['lr']
        }
        
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
                
                continuous_embeddings = model.encode(batch)
                print(f"DEBUG: Encoded batch, embeddings shape: {continuous_embeddings.shape}")
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                reconstructed_audio = model.decode(continuous_embeddings)
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
                
                # Prepare discriminator inputs (call once, use everywhere)
                print(f"DEBUG: Preparing discriminator inputs for batch {update + 1}")
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Step 1: Train generator first (like original encodec-pytorch)
                print(f"DEBUG: Training generator for batch {update + 1}")
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                model_optimizer.zero_grad()
                
                # Compute individual losses
                print(f"DEBUG: Computing reconstruction loss")
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}")
                recon_loss, recon_metrics = reconstruction_loss(reconstructed_audio, batch)
                
                print(f"DEBUG: Computing adversarial loss")
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}")
                # Call discriminator for generator training (2 calls: real + fake) - like original encodec-pytorch
                logits_real, fmap_real = discriminator(batch)
                logits_fake, fmap_fake = discriminator(reconstructed_audio)
                
                # Compute adversarial losses using pre-computed discriminator outputs
                adv_loss, feat_loss = adversarial_loss.compute_losses_from_outputs(
                    logits_real, logits_fake, fmap_real, fmap_fake
                )
                
                # Split reconstruction loss into time and frequency components
                time_loss = recon_metrics['time_loss']
                freq_loss = recon_metrics['freq_loss']
                
                # Combine losses for balancer
                losses = {
                    'loss_t': time_loss,
                    'loss_f': freq_loss, 
                    'loss_g': adv_loss,
                    'loss_feat': feat_loss
                }
                
                # Use loss balancer to compute effective loss
                print(f"DEBUG: Computing balanced loss")
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}")
                effective_loss = balancer.backward(losses, reconstructed_audio)
                
                # Gradient clipping
                print(f"DEBUG: Applying gradient clipping")
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}")
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Step the optimizer
                print(f"DEBUG: Stepping optimizer for batch {update + 1}")
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}")
                model_optimizer.step()
                
                # Step 2: Train discriminator (every batch for 24kHz audio as per paper)
                print(f"DEBUG: Training discriminator for batch {update + 1}")
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}")
                # Use .detach() to avoid backpropagation to generator (like original encodec-pytorch)
                disc_loss = adversarial_loss.train_adv(
                    fake=reconstructed_audio.detach(),
                    real=batch.detach()
                )
                print(f"DEBUG: Discriminator loss: {disc_loss.item():.6f}")
                print(f"DEBUG: Number of discriminator scales: {discriminator.num_discriminators}")
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # DEBUG: Print raw losses before balancer (compute once to avoid memory leaks)
                recon_loss_val = recon_loss.item()
                time_loss_val = time_loss.item()
                freq_loss_val = freq_loss.item()
                adv_loss_val = adv_loss.item()
                feat_loss_val = feat_loss.item()
                disc_loss_val = disc_loss.item()
                batch_min, batch_max, batch_mean = batch.min().item(), batch.max().item(), batch.mean().item()
                recon_min, recon_max, recon_mean = reconstructed_audio.min().item(), reconstructed_audio.max().item(), reconstructed_audio.mean().item()
                
                print(f"DEBUG - Raw losses for batch {update + 1}:")
                print(f"  loss_t: {time_loss_val:.6f}      # Time domain reconstruction loss")
                print(f"  loss_f: {freq_loss_val:.6f}      # Frequency domain reconstruction loss")
                print(f"  loss_feat: {feat_loss_val:.6f}   # Feature matching loss")
                print(f"  loss_g: {adv_loss_val:.6f}       # Adversarial loss for generator")
                print(f"  loss_disc: {disc_loss_val:.6f}   # Discriminator loss")
                print(f"  batch stats: min={batch_min:.6f}, max={batch_max:.6f}, mean={batch_mean:.6f}")
                print(f"  reconstructed stats: min={recon_min:.6f}, max={recon_max:.6f}, mean={recon_mean:.6f}")
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Use loss balancer for gradient balancing
                balanced_losses = {
                    'time_loss': time_loss,
                    'freq_loss': freq_loss,
                    'adv_loss': adv_loss,
                    'feat_loss': feat_loss
                }
                
                # Apply balancer - this handles gradient balancing and backward pass
                print(f"DEBUG: Applying loss balancer for batch {update + 1}")
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                effective_loss = balancer.backward(balanced_losses, reconstructed_audio)
                print(f"DEBUG: loss_G (effective loss): {effective_loss.item():.6f}")
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}")
                
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
                
                # Update metrics (use pre-computed values to avoid memory leaks)
                train_metrics['loss_t'] += time_loss_val      # Time domain reconstruction loss
                train_metrics['loss_f'] += freq_loss_val      # Frequency domain reconstruction loss
                train_metrics['loss_feat'] += feat_loss_val   # Feature matching loss
                train_metrics['loss_g'] += adv_loss_val       # Adversarial loss for generator
                train_metrics['loss_G'] += effective_loss.item()  # Effective loss (LG)
                train_metrics['loss_disc'] += disc_loss_val   # Discriminator loss
                
                # Clean up intermediate tensors to free memory
                del continuous_embeddings, reconstructed_audio
                del logits_real, logits_fake, fmap_real, fmap_fake  # Discriminator outputs
                del recon_loss, time_loss, freq_loss, adv_loss, feat_loss, disc_loss
                del recon_metrics, balanced_losses, losses
                torch.cuda.empty_cache()  # Clear GPU cache
                
                print(f"DEBUG: Completed batch {update + 1}/{updates_per_epoch} for epoch {epoch + 1}")
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}")
                print("=" * 50)
                
                # Log to wandb every 100 steps (use pre-computed values)
                if global_step % 100 == 0:
                    log_dict = {
                        'train/loss_t': time_loss_val,        # Time domain reconstruction loss
                        'train/loss_f': freq_loss_val,        # Frequency domain reconstruction loss
                        'train/loss_feat': feat_loss_val,     # Feature matching loss
                        'train/loss_g': adv_loss_val,         # Adversarial loss for generator
                        'train/loss_G': effective_loss.item(), # Effective loss (LG)
                        'train/loss_disc': disc_loss_val,     # Discriminator loss
                        'train/learning_rate': model_optimizer.param_groups[0]['lr'],
                        'epoch': epoch,
                        'global_step': global_step,
                        'update': update
                    }
                    
                    # Add balancer metrics if available
                    if balancer.metrics:
                        for key, value in balancer.metrics.items():
                            log_dict[f'train/balancer_{key}'] = value
                    
                    wandb.log(log_dict)
                                   
                global_step += 1
                
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
            'val_loss_t': 0.0,        # Time domain reconstruction loss
            'val_loss_f': 0.0,        # Frequency domain reconstruction loss
            'val_loss_feat': 0.0,     # Feature matching loss
            'val_loss_g': 0.0,        # Adversarial loss for generator
            'val_loss_G': 0.0,        # Effective loss (LG)
            'val_loss_disc': 0.0      # Discriminator loss
        }
        
        val_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch.shape[0] != batch_size:  # Skip incomplete batches
                    continue
                
                # Move batch to device
                batch = batch.to(device)
                
                # Forward pass
                continuous_embeddings = model.encode(batch)
                reconstructed_audio = model.decode(continuous_embeddings)
                
                # Ensure same length
                if reconstructed_audio.shape[-1] > batch.shape[-1]:
                    reconstructed_audio = reconstructed_audio[..., :batch.shape[-1]]
                elif reconstructed_audio.shape[-1] < batch.shape[-1]:
                    pad_length = batch.shape[-1] - reconstructed_audio.shape[-1]
                    reconstructed_audio = torch.nn.functional.pad(
                        reconstructed_audio, (0, pad_length), mode='reflect'
                    )
                
                
                # Compute losses
                recon_loss, recon_metrics = reconstruction_loss(reconstructed_audio, batch)
                # Prepare discriminator inputs for validation
                adv_fake = get_discriminator_input(reconstructed_audio)
                adv_real = get_discriminator_input(batch)
                adv_loss, feat_loss = adversarial_loss(adv_fake, adv_real)
                
                # Compute discriminator loss for validation
                disc_loss = adversarial_loss.train_adv(adv_fake, adv_real)
                
                # Split reconstruction loss into time and frequency components
                time_loss = recon_metrics['time_loss']
                freq_loss = recon_metrics['freq_loss']
                
                val_metrics['val_loss_t'] += time_loss.item()        # Time domain reconstruction loss
                val_metrics['val_loss_f'] += freq_loss.item()        # Frequency domain reconstruction loss
                val_metrics['val_loss_feat'] += feat_loss.item()     # Feature matching loss
                val_metrics['val_loss_g'] += adv_loss.item()         # Adversarial loss for generator
                val_metrics['val_loss_disc'] += disc_loss.item()     # Discriminator loss
                
                # For validation, compute effective loss manually without using balancer.backward
                # since we don't want to perform actual backward passes during validation
                balanced_losses = {
                    'time_loss': time_loss,
                    'freq_loss': freq_loss,
                    'adv_loss': adv_loss,
                    'feat_loss': feat_loss
                }
                
                # Compute effective loss using the same weights as the balancer
                total_weights = sum(balancer.weights.values())
                val_total = torch.tensor(0., device=batch.device, dtype=batch.dtype)
                for name, loss in balanced_losses.items():
                    weight = balancer.weights.get(name, 0.0)
                    val_total += (weight / total_weights) * loss.detach()
                
                val_metrics['val_loss_G'] += val_total.item()        # Effective loss (LG)
                
                val_batches += 1
        
        # Average validation metrics
        if val_batches > 0:
            for key in val_metrics:
                val_metrics[key] /= val_batches
        
        # Test phase - objective metrics evaluation (fixed 1000 samples every 25 epochs)
        test_metrics = {
            'test_si_snr': 0.0,        # Scale-Invariant Signal-to-Noise Ratio
        }
        
        # Only run testing every 25 epochs
        if (epoch + 1) % 25 == 0 or epoch == 0:  # Test on epoch 1, 25, 50, 75, etc.
            model.eval()
            test_batches = 0
            max_test_batches = 1000 // batch_size  # Limit to 1000 samples
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_loader):
                    if batch_idx >= max_test_batches:  # Stop after 1000 samples
                        break
                        
                    if batch.shape[0] != batch_size:  # Skip incomplete batches
                        continue
                    
                    # Move batch to device
                    batch = batch.to(device)
                    
                    # Forward pass
                    continuous_embeddings = model.encode(batch)
                    reconstructed_audio = model.decode(continuous_embeddings)
                    
                    # Ensure same length
                    if reconstructed_audio.shape[-1] > batch.shape[-1]:
                        reconstructed_audio = reconstructed_audio[..., :batch.shape[-1]]
                    elif reconstructed_audio.shape[-1] < batch.shape[-1]:
                        pad_length = batch.shape[-1] - reconstructed_audio.shape[-1]
                        reconstructed_audio = torch.nn.functional.pad(
                            reconstructed_audio, (0, pad_length), mode='reflect'
                        )
                    
                    # Compute objective audio quality metrics
                    try:
                        obj_metrics = audio_metrics.compute_batch_metrics(reconstructed_audio, batch)
                        if 'si_snr' in obj_metrics:
                            test_metrics['test_si_snr'] += obj_metrics['si_snr']
                    except Exception as e:
                        logger.warning(f"Failed to compute objective metrics: {e}")
                    
                    test_batches += 1
            
            # Average test metrics
            if test_batches > 0:
                for key in test_metrics:
                    test_metrics[key] /= test_batches
        
        # Log epoch metrics to wandb
        log_dict = {
            'epoch': epoch,
            'train/epoch_loss_t': train_metrics['loss_t'],        # Time domain reconstruction loss
            'train/epoch_loss_f': train_metrics['loss_f'],        # Frequency domain reconstruction loss
            'train/epoch_loss_feat': train_metrics['loss_feat'],  # Feature matching loss
            'train/epoch_loss_g': train_metrics['loss_g'],        # Adversarial loss for generator
            'train/epoch_loss_G': train_metrics['loss_G'],        # Effective loss (LG)
            'train/epoch_loss_disc': train_metrics['loss_disc'],  # Discriminator loss
            **val_metrics,  # val_loss_t, val_loss_f, val_loss_feat, val_loss_g, val_loss_G, val_loss_disc
        }
        
        # Only log test metrics when testing is performed
        if (epoch + 1) % 25 == 0 or epoch == 0:
            log_dict.update(test_metrics)  # test_si_snr
        
        wandb.log(log_dict)
        
        # Save best model (simplified - no wandb config to avoid loading issues)
        if val_metrics['val_loss_G'] < best_val_loss:
            best_val_loss = val_metrics['val_loss_G']
            
            # Remove previous bestmodel files
            import glob
            old_checkpoints = glob.glob(os.path.join(checkpoint_dir, "bestmodel_*.pth"))
            for old_file in old_checkpoints:
                try:
                    os.remove(old_file)
                    print(f"Removed old checkpoint: {old_file}")
                except OSError:
                    pass
            
            # Save new checkpoint with epoch number
            new_save_path = os.path.join(checkpoint_dir, f"bestmodel_epoch_{epoch+1}_loss_{best_val_loss:.4f}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'model_optimizer_state_dict': model_optimizer.state_dict(),
                'disc_optimizer_state_dict': adversarial_loss.optimizer.state_dict(),
                'val_loss': best_val_loss
            }, new_save_path)
            print(f"Saved best model: {new_save_path} (validation loss: {best_val_loss:.6f})")
        
        # Log epoch summary
        print(f"Epoch {epoch + 1}/{num_epochs} completed:")
        print(f"  Train - Loss_G: {train_metrics['loss_G']:.6f}, "
                   f"Adv: {train_metrics['loss_g']:.6f}, "
                   f"Feat: {train_metrics['loss_feat']:.6f}")
        print(f"  Val - Loss_G: {val_metrics['val_loss_G']:.6f}")
        
        # Only print test metrics when testing is performed
        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"  Test - SI-SNR: {test_metrics['test_si_snr']:.2f}")
        else:
            print(f"  Test - Skipped (next test at epoch {((epoch + 1) // 25 + 1) * 25})")


def main():
    """Baseline autoencoder training with paper-accurate parameters and wandb logging."""
    
    # Parse command line arguments
    args = parse_args()
    
    # Use 1-channel configuration
    channels = 1
    batch_size = 32  # Reduced to prevent overfitting with small dataset
    config_name = "baseline-24khz"
    print("Using 1-channel configuration")
    
    # Handle resume functionality
    start_epoch = 0
    best_val_loss = float('inf')
    checkpoint_dir = args.checkpoint_dir
    
    if args.resume:
        print(f"🔄 Resume mode: Loading checkpoint from {args.resume}")
        # We'll load the checkpoint after creating the models
    
    # Setup wandb (will handle login if needed)
    setup_wandb()
    
    # Initialize wandb
    wandb.init(
        project="baseline-autoencoder-training",
        name=config_name,
        config={
            "sample_rate": 24000,
            "channels": channels,
            "n_filters": 32,  # Paper: C = 32
            "n_residual_layers": 1,
            "dimension": 32,  # Paper: D = 32
            "causal": False,
            "ratios": [2, 4, 5, 8],  # Paper: B = 4 blocks with strides (2, 4, 5, 8)
            "lstm_layers": 2,  # Paper: "two-layer LSTM"
            "epochs": 300,
            "updates_per_epoch": 2000,  # Paper: 2000 updates per epoch
            "batch_size": batch_size,
            "learning_rate": 3e-4,
            "beta1": 0.5,
            "beta2": 0.9,
            "segment_duration": 1.0,
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
    )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Build the model
    print("Building baseline autoencoder model...")
    model = build_encodec_model(
        sample_rate=24000,
        channels=channels,
        n_filters=32,  # Paper: C = 32
        n_residual_layers=1,
        dimension=32,  # Paper: D = 32
        causal=False,
        custom_ratios=[2, 4, 5, 8]  # Paper: B = 4 blocks with strides (2, 4, 5, 8)
    ).to(device)
    
    # Build MS-STFT discriminator
    print("Building MS-STFT discriminator...")
    discriminator = create_ms_stft_discriminator(
        in_channels=channels,
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
    
    # Load checkpoint if resuming
    if args.resume:
        start_epoch, best_val_loss = load_checkpoint(
            args.resume, model, discriminator, model_optimizer, adversarial_loss, device
        )
    
    # Loss weights from paper: λt=0.1, λf=1, λg=3, λfeat=3 (24kHz model)
    print("Using paper loss weights: λt=0.1, λf=1, λg=3, λfeat=3 (24kHz model)")
    
    # Create dataloaders with folder-based split
    print("Creating dataloaders...")
    
    # Define folder-based train/validation/test split
    train_folders = ["Beach", "Busy Street", "Park", "Pedestrian Zone", "Quiet Street", "Shopping Centre"]
    val_folders = ["Woodland"]
    test_folders = ["Train Station"]
    
    # Create folder-based dataloaders with dynamic configuration
    train_loader, val_loader = create_folder_based_dataloaders(
        audio_dir="/scratch/eigenscape/",
        train_folders=train_folders,
        val_folders=val_folders,
        batch_size=batch_size,           
        sample_rate=24000,
        segment_duration=1.0,    # 1 second segments
        channels=channels,
        train_dataset_size=64000,  # 2000 updates × batch_size = 64k samples per epoch
        val_dataset_size=8000,     # 8k validation samples (fixed each epoch)
        min_file_duration=1.0,    
        random_crop=True,
        num_workers=4,  # Enable multiprocessing for faster data loading
        pin_memory=True,  # Enable pinned memory for faster GPU transfer
        persistent_workers=True  # Keep workers alive between epochs
    )
    
    # Create fixed test dataset for objective metrics evaluation (1000 samples)
    test_loader, _ = create_folder_based_dataloaders(
        audio_dir="/scratch/eigenscape/",
        train_folders=test_folders,
        val_folders=[],  # Empty validation folders
        batch_size=batch_size,
        sample_rate=24000,
        segment_duration=1.0,
        channels=channels,
        train_dataset_size=1000,  # Fixed 1000 samples for objective metrics
        val_dataset_size=0,       # No validation needed
        min_file_duration=1.0,
        random_crop=False,  # Fixed segments for consistent evaluation
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Log dataset information
    print(f"Training dataset size: {len(train_loader.dataset)} samples")
    print(f"Validation dataset size: {len(val_loader.dataset)} samples")
    print(f"Test dataset size: {len(test_loader.dataset)} samples")
    print(f"Training batches per epoch: {len(train_loader)}")
    print(f"Validation batches per epoch: {len(val_loader)}")
    print(f"Test batches per epoch: {len(test_loader)}")
    
    # Paper: 300 epochs, 2000 updates per epoch
    num_epochs = 300
    updates_per_epoch = 2000  # Paper: 2000 updates per epoch
    
    print(f"Starting training for {num_epochs} epochs...")
    print(f"Each epoch has {updates_per_epoch} updates ({updates_per_epoch * batch_size}k samples)")
    print(f"Validation uses FIXED 8k samples (same segments every epoch)")
    print(f"Test uses FIXED 1k samples for objective metrics (SI-SNR) - every 25 epochs")
    print(f"Total updates: {num_epochs * updates_per_epoch}")
    print(f"Training folders: {train_folders}")
    print(f"Validation folders: {val_folders}")
    print(f"Test folders: {test_folders}")
    
    # Custom training loop with wandb logging
    train_baseline_with_wandb(
        model=model,
        discriminator=discriminator,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model_optimizer=model_optimizer,
        adversarial_loss=adversarial_loss,
        num_epochs=num_epochs,
        updates_per_epoch=updates_per_epoch,
        device=device,
        batch_size=batch_size,
        start_epoch=start_epoch,
        best_val_loss=best_val_loss,
        checkpoint_dir=checkpoint_dir
    )
    
    print("Training completed!")
    wandb.finish()


if __name__ == "__main__":
    main()

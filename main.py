
"""
Baseline autoencoder training with paper-accurate parameters and Weights & Biases logging.
Based on the EnCodec paper: 300 epochs, 2000 updates per epoch, batch size 64.
"""

import logging
import wandb
import torch
from pathlib import Path

from model_builder import build_encodec_model
from dataloader import create_train_test_dataloaders, create_folder_based_dataloaders
from discriminators import create_ms_stft_discriminator
from adversarial_losses import create_adversarial_loss
from balancer import Balancer

logger = logging.getLogger(__name__)


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
                            num_epochs, updates_per_epoch, save_path, device):
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
        
        # Training loop for this epoch
        for update in range(updates_per_epoch):
            try:
                # Get batch
                batch = next(iter(train_loader))
                if batch.shape[0] != 4:  # Skip incomplete batches
                    continue
                
                # Move batch to device
                batch = batch.to(device)
                
                # Forward pass: continuous embeddings → decoder → reconstructed audio
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
                
                # Step 1: Train discriminator
                disc_loss = adversarial_loss.train_adv(
                    fake=reconstructed_audio.detach(),
                    real=batch
                )
                
                # Step 2: Train generator (baseline autoencoder) with loss balancer
                model_optimizer.zero_grad()
                
                # Compute individual losses
                recon_loss, recon_metrics = reconstruction_loss(reconstructed_audio, batch)
                adv_loss, feat_loss = adversarial_loss(reconstructed_audio, batch)
                
                # Split reconstruction loss into time and frequency components
                time_loss = recon_metrics['time_loss']
                freq_loss = recon_metrics['freq_loss']
                
                # DEBUG: Print raw losses before balancer
                if update == 0:  # Only print for first update
                    print(f"DEBUG - Raw losses:")
                    print(f"  recon_loss: {recon_loss.item():.6f}")
                    print(f"  time_loss: {time_loss.item():.6f}")
                    print(f"  freq_loss: {freq_loss.item():.6f}")
                    print(f"  adv_loss: {adv_loss.item():.6f}")
                    print(f"  feat_loss: {feat_loss.item():.6f}")
                    print(f"  batch stats: min={batch.min().item():.6f}, max={batch.max().item():.6f}, mean={batch.mean().item():.6f}")
                    print(f"  reconstructed stats: min={reconstructed_audio.min().item():.6f}, max={reconstructed_audio.max().item():.6f}, mean={reconstructed_audio.mean().item():.6f}")
                
                # Use loss balancer for gradient balancing
                balanced_losses = {
                    'time_loss': time_loss,
                    'freq_loss': freq_loss,
                    'adv_loss': adv_loss,
                    'feat_loss': feat_loss
                }
                
                # Apply balancer - this handles gradient balancing and backward pass
                effective_loss = balancer.backward(balanced_losses, reconstructed_audio)
                
                # DEBUG: Print effective loss and balancer metrics
                if update == 0:
                    print(f"DEBUG - Effective loss: {effective_loss.item():.6f}")
                    print(f"DEBUG - Loss weights: λt=0.1, λf=1.0, λg=3.0, λfeat=3.0")
                    print(f"DEBUG - Balancer metrics: {balancer.metrics}")
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Step the optimizer
                model_optimizer.step()
                
                # Update metrics
                train_metrics['time_reconstruction_loss'] += time_loss.item()
                train_metrics['freq_reconstruction_loss'] += freq_loss.item()
                train_metrics['reconstruction_loss'] += recon_loss.item()  # Keep total for compatibility
                train_metrics['adversarial_loss'] += adv_loss.item()
                train_metrics['feature_matching_loss'] += feat_loss.item()
                train_metrics['total_loss'] += effective_loss.item()
                train_metrics['discriminator_loss'] += disc_loss.item()
                
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
            'val_time_reconstruction_loss': 0.0,
            'val_freq_reconstruction_loss': 0.0,
            'val_reconstruction_loss': 0.0,
            'val_adversarial_loss': 0.0,
            'val_feature_matching_loss': 0.0,
            'val_total_loss': 0.0
        }
        
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                if batch.shape[0] != 4:  # Skip incomplete batches
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
                adv_loss, feat_loss = adversarial_loss(reconstructed_audio, batch)
                
                # Split reconstruction loss into time and frequency components
                time_loss = recon_metrics['time_loss']
                freq_loss = recon_metrics['freq_loss']
                
                val_metrics['val_time_reconstruction_loss'] += time_loss.item()
                val_metrics['val_freq_reconstruction_loss'] += freq_loss.item()
                val_metrics['val_reconstruction_loss'] += recon_loss.item()
                val_metrics['val_adversarial_loss'] += adv_loss.item()
                val_metrics['val_feature_matching_loss'] += feat_loss.item()
                
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
                
                val_metrics['val_total_loss'] += val_total.item()
                
                val_batches += 1
        
        # Average validation metrics
        if val_batches > 0:
            for key in val_metrics:
                val_metrics[key] /= val_batches
        
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
        
        # Save best model
        if val_metrics['val_total_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_total_loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'model_optimizer_state_dict': model_optimizer.state_dict(),
                'val_loss': best_val_loss,
                'config': wandb.config
            }, save_path)
            print(f"Saved best model with validation loss: {best_val_loss:.6f}")
        
        # Log epoch summary
        print(f"Epoch {epoch + 1}/{num_epochs} completed:")
        print(f"  Train - Recon: {train_metrics['reconstruction_loss']:.6f}, "
                   f"Adv: {train_metrics['adversarial_loss']:.6f}, "
                   f"Feat: {train_metrics['feature_matching_loss']:.6f}")
        print(f"  Val - Total: {val_metrics['val_total_loss']:.6f}")


def main():
    """Baseline autoencoder training with paper-accurate parameters and wandb logging."""
    
    # Setup wandb (will handle login if needed)
    setup_wandb()
    
    # Initialize wandb
    wandb.init(
        project="baseline-autoencoder-training",
        name="baseline-24khz-paper-params",
        config={
            "sample_rate": 24000,
            "channels": 32,
            "n_filters": 4,
            "n_residual_layers": 1,
            "dimension": 32,
            "causal": False,
            "epochs": 300,
            "updates_per_epoch": 2000,
            "batch_size": 4,
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
        channels=32,
        n_filters=4,
        n_residual_layers=1,
        dimension=32,
        causal=False
    ).to(device)
    
    # Build MS-STFT discriminator
    print("Building MS-STFT discriminator...")
    discriminator = create_ms_stft_discriminator(
        in_channels=32,
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
    
    # Create folder-based dataloaders with 8k train samples and 2k validation samples
    train_loader, val_loader = create_folder_based_dataloaders(
        audio_dir="/scratch/eigenscape/",
        train_folders=train_folders,
        val_folders=val_folders,
        batch_size=4,           
        sample_rate=24000,
        segment_duration=1.0,    
        channels=32,
        train_dataset_size=8000,  # 2000 updates × 4 batch size = 8000 samples per epoch
        val_dataset_size=2000,    # 2000 validation samples (fixed each epoch)
        min_file_duration=1.0,    
        random_crop=True,
        num_workers=0  # Disable multiprocessing to avoid crashes
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
    train_baseline_with_wandb(
        model=model,
        discriminator=discriminator,
        train_loader=train_loader,
        val_loader=val_loader,
        model_optimizer=model_optimizer,
        adversarial_loss=adversarial_loss,
        num_epochs=num_epochs,
        updates_per_epoch=updates_per_epoch,
        save_path=Path("best_model.pth"),
        device=device
    )
    
    print("Training completed!")
    wandb.finish()


if __name__ == "__main__":
    main()

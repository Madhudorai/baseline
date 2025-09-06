
"""
Baseline autoencoder training with paper-accurate parameters and Weights & Biases logging.
Based on the EnCodec paper: 300 epochs, 2000 updates per epoch, batch size 64.
"""

import logging
import wandb
import torch
from pathlib import Path

from model_builder import build_encodec_model
from dataloader import create_train_test_dataloaders
from discriminators import create_ms_stft_discriminator
from adversarial_losses import create_adversarial_loss
from loss_balancer import create_loss_balancer

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
                            model_optimizer, adversarial_loss, loss_balancer,
                            num_epochs, updates_per_epoch, save_path):
    """Custom training loop with comprehensive wandb logging for baseline autoencoder."""
    
    from losses import ReconstructionLoss
    import torch
    
    # Create reconstruction loss
    reconstruction_loss = ReconstructionLoss(sample_rate=24000)
    
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
                if batch.shape[0] != 64:  # Skip incomplete batches
                    continue
                
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
                
                # Use loss balancer for gradient balancing
                losses = {
                    'time_reconstruction': time_loss,
                    'freq_reconstruction': freq_loss,
                    'adversarial': adv_loss,
                    'feature_matching': feat_loss
                }
                
                # The balancer handles gradient computation and backward pass
                effective_loss = loss_balancer.backward(losses, reconstructed_audio)
                
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
                    wandb.log({
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
                    })
                    
                    # Log balancer metrics
                    if loss_balancer.metrics:
                        for name, ratio in loss_balancer.metrics.items():
                            wandb.log({f'train/balancer_{name}': ratio})
                
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
                if batch.shape[0] != 64:  # Skip incomplete batches
                    continue
                
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
                
                # Use same weights as training for validation
                val_total = (loss_balancer.weights['time_reconstruction'] * time_loss + 
                           loss_balancer.weights['freq_reconstruction'] * freq_loss +
                           loss_balancer.weights['adversarial'] * adv_loss + 
                           loss_balancer.weights['feature_matching'] * feat_loss)
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
            "updates_per_epoch": 200,
            "batch_size": 64,
            "learning_rate": 3e-4,
            "beta1": 0.5,
            "beta2": 0.9,
            "segment_duration": 1.0,
            "loss_weights": {
                "reconstruction": 1.0,      # λf (frequency domain)
                "adversarial": 3.0,         # λg (generator)
                "feature_matching": 3.0     # λfeat (feature matching)
            }
        }
    )
    
    # Build the model
    print("Building baseline autoencoder model...")
    model = build_encodec_model(
        sample_rate=24000,
        channels=32,
        n_filters=4,
        n_residual_layers=1,
        dimension=32,
        causal=False
    )
    
    # Build MS-STFT discriminator
    print("Building MS-STFT discriminator...")
    discriminator = create_ms_stft_discriminator(
        in_channels=32,
        out_channels=1,
        filters=32
    )
    
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
    
    # Create loss balancer with paper weights
    print("Creating loss balancer with paper weights: λt=0.1, λf=1, λg=3, λfeat=3 (24kHz model)")
    loss_balancer = create_loss_balancer(
        time_reconstruction_weight=0.1,    # λt = 0.1 (time domain)
        freq_reconstruction_weight=1.0,    # λf = 1 (frequency domain)
        adversarial_weight=3.0,            # λg = 3 (generator)
        feature_matching_weight=3.0,       # λfeat = 3 (feature matching)
        balance_grads=True,                # Enable gradient balancing
        total_norm=1.0,                    # R = 1
        ema_decay=0.999                    # β = 0.999
    )
    
    print(f"Loss balancer weights: {loss_balancer.weights}")
    print("Paper weights: λt=0.1, λf=1, λg=3, λfeat=3 (24kHz model)")
    
    # Create dataloaders with paper-accurate parameters
    print("Creating dataloaders...")
    
    # So we need updates_per_epoch * batch_size samples per epoch
    samples_per_epoch = 200 * 64 
    val_dataset_size = samples_per_epoch // 4  # 1/4th of train dataset size
    
    # Create train/test dataloaders from single directory with 80/20 split
    train_loader, val_loader = create_train_test_dataloaders(
        audio_dir="/scratch/eigenscape/",
        train_ratio=0.8,          
        batch_size=64,           
        sample_rate=24000,
        segment_duration=1.0,    
        channels=32,
        train_dataset_size=samples_per_epoch,  # 12,800 samples per epoch
        test_dataset_size=val_dataset_size,    # 3,200 validation samples (1/4th)
        min_file_duration=1.0,    
        random_crop=True     
    )
    
    # Log dataset information
    print(f"Training dataset size: {len(train_loader.dataset)} samples")
    print(f"Validation dataset size: {len(val_loader.dataset)} samples")
    print(f"Training batches per epoch: {len(train_loader)}")
    print(f"Validation batches per epoch: {len(val_loader)}")
    
    # Paper: 300 epochs, 2000 updates per epoch
    num_epochs = 300
    updates_per_epoch = 200
    
    print(f"Starting training for {num_epochs} epochs...")
    print(f"Each epoch has {updates_per_epoch} updates")
    print(f"Total updates: {num_epochs * updates_per_epoch}")
    
    # Custom training loop with wandb logging
    train_baseline_with_wandb(
        model=model,
        discriminator=discriminator,
        train_loader=train_loader,
        val_loader=val_loader,
        model_optimizer=model_optimizer,
        adversarial_loss=adversarial_loss,
        loss_balancer=loss_balancer,
        num_epochs=num_epochs,
        updates_per_epoch=updates_per_epoch,
        save_path=Path("best_model.pth")
    )
    
    print("Training completed!")
    wandb.finish()


if __name__ == "__main__":
    main()

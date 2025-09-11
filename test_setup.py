#!/usr/bin/env python3
"""
Comprehensive test setup for baseline autoencoder training pipeline.
Tests dataloader, model loading, losses, and backpropagation.
"""

import sys
import torch
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all required modules can be imported."""
    print("=" * 60)
    print("TESTING IMPORTS")
    print("=" * 60)
    
    try:
        from model_builder import build_encodec_model
        from dataloader import create_train_test_dataloaders
        from discriminators import create_ms_stft_discriminator
        from adversarial_losses import create_adversarial_loss
        from losses import ReconstructionLoss
        from balancer import Balancer
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_dataloader():
    """Test dataloader creation and data loading."""
    print("\n" + "=" * 60)
    print("TESTING DATALOADER")
    print("=" * 60)
    
    try:
        from dataloader import create_train_test_dataloaders
        
        # Test directory (you can change this to your actual directory)
        test_dir = "/scratch/eigenscape/"
        
        print(f"Testing dataloader with directory: {test_dir}")
        print("Note: This will fail if the directory doesn't exist or has no audio files")
        
        # Create small test dataloaders
        train_loader, test_loader = create_train_test_dataloaders(
            audio_dir=test_dir,
            train_ratio=0.8,
            batch_size=2,  # Small batch for testing
            sample_rate=24000,
            segment_duration=1.0,
            channels=32,
            train_dataset_size=10,  # Small dataset for testing
            test_dataset_size=5,
            min_file_duration=1.0,
            random_crop=True
        )
        
        print(f"✅ Dataloaders created successfully")
        print(f"   Train dataset size: {len(train_loader.dataset)} samples")
        print(f"   Test dataset size: {len(test_loader.dataset)} samples")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Test batches: {len(test_loader)}")
        
        # Test loading a batch
        print("\nTesting batch loading...")
        train_batch = next(iter(train_loader))
        test_batch = next(iter(test_loader))
        
        print(f"✅ Train batch shape: {train_batch.shape}")
        print(f"✅ Test batch shape: {test_batch.shape}")
        
        # Verify shapes
        expected_shape = (2, 32, 24000)  # (batch_size, channels, samples)
        if train_batch.shape == expected_shape:
            print("✅ Train batch has correct shape")
        else:
            print(f"❌ Train batch shape incorrect. Expected {expected_shape}, got {train_batch.shape}")
            return False
            
        if test_batch.shape == expected_shape:
            print("✅ Test batch has correct shape")
        else:
            print(f"❌ Test batch shape incorrect. Expected {expected_shape}, got {test_batch.shape}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Dataloader test failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        return False

def test_model_loading():
    """Test model creation and forward pass."""
    print("\n" + "=" * 60)
    print("TESTING MODEL LOADING")
    print("=" * 60)
    
    try:
        from model_builder import build_encodec_model
        
        # Create model with paper parameters
        model = build_encodec_model(
            sample_rate=24000,
            channels=32,
            n_filters=4,
            n_residual_layers=1,
            dimension=32,
            causal=False
        )
        
        print("✅ Model created successfully")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Sample rate: {model.sample_rate}")
        print(f"   Channels: {model.channels}")
        print(f"   Frame rate: {model.frame_rate}")
        
        # Test forward pass
        print("\nTesting forward pass...")
        batch_size = 2
        channels = 32
        samples = 24000  # 1 second at 24kHz
        
        # Create dummy input
        dummy_input = torch.randn(batch_size, channels, samples)
        print(f"   Input shape: {dummy_input.shape}")
        
        # Forward pass
        with torch.no_grad():
            output = model(dummy_input)
            print(f"   Output shape: {output.shape}")
        
        # Test encode/decode
        print("\nTesting encode/decode...")
        with torch.no_grad():
            # Encode
            codes = model.encode(dummy_input)
            print(f"   Encoded shape: {codes.shape}")
            
            # Decode
            reconstructed = model.decode(codes)
            print(f"   Decoded shape: {reconstructed.shape}")
        
        print("✅ Model forward pass successful")
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        return False

def test_losses():
    """Test loss functions and backpropagation."""
    print("\n" + "=" * 60)
    print("TESTING LOSSES")
    print("=" * 60)
    
    try:
        from losses import ReconstructionLoss
        from discriminators import create_ms_stft_discriminator
        from adversarial_losses import create_adversarial_loss
        
        # Create test data
        batch_size = 2
        channels = 32
        samples = 24000
        real_audio = torch.randn(batch_size, channels, samples)
        fake_audio = torch.randn(batch_size, channels, samples)
        
        print(f"Test data shape: {real_audio.shape}")
        
        # Test reconstruction loss
        print("\nTesting reconstruction loss...")
        recon_loss = ReconstructionLoss(sample_rate=24000)
        loss_value, metrics = recon_loss(fake_audio, real_audio)
        print(f"✅ Reconstruction loss: {loss_value.item():.6f}")
        print(f"   Metrics: {metrics}")
        
        # Test discriminator
        print("\nTesting discriminator...")
        discriminator = create_ms_stft_discriminator(
            in_channels=32,
            out_channels=1,
            filters=32
        )
        
        # Test discriminator forward pass
        with torch.no_grad():
            disc_real = discriminator(real_audio)
            disc_fake = discriminator(fake_audio)
            print(f"✅ Discriminator real output: {len(disc_real[0])} scales")
            print(f"✅ Discriminator fake output: {len(disc_fake[0])} scales")
        
        # Test adversarial loss
        print("\nTesting adversarial loss...")
        disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=3e-4)
        adv_loss = create_adversarial_loss(
            discriminator=discriminator,
            optimizer=disc_optimizer,
            loss_type='hinge',
            use_feature_matching=True
        )
        
        # Test adversarial loss forward pass
        adv_loss_val, feat_loss_val = adv_loss(fake_audio, real_audio)
        print(f"✅ Adversarial loss: {adv_loss_val.item():.6f}")
        print(f"✅ Feature matching loss: {feat_loss_val.item():.6f}")
        
        # Test basic loss combination (without balancer)
        print("\nTesting basic loss combination...")
        reconstruction_weight = 1.0
        adversarial_weight = 3.0
        feature_matching_weight = 3.0
        
        total_loss = (reconstruction_weight * loss_value + 
                     adversarial_weight * adv_loss_val + 
                     feature_matching_weight * feat_loss_val)
        print(f"✅ Total loss: {total_loss.item():.6f}")
        print(f"   Reconstruction: {loss_value.item():.6f}")
        print(f"   Adversarial: {adv_loss_val.item():.6f}")
        print(f"   Feature matching: {feat_loss_val.item():.6f}")
        
        print("✅ All losses working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Loss test failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        return False

def test_backpropagation():
    """Test that gradients flow correctly through the model."""
    print("\n" + "=" * 60)
    print("TESTING BACKPROPAGATION")
    print("=" * 60)
    
    try:
        from model_builder import build_encodec_model
        from losses import ReconstructionLoss
        from discriminators import create_ms_stft_discriminator
        from adversarial_losses import create_adversarial_loss
        
        # Create model and components
        model = build_encodec_model(
            sample_rate=24000,
            channels=32,
            n_filters=4,
            n_residual_layers=1,
            dimension=32,
            causal=False
        )
        
        discriminator = create_ms_stft_discriminator(
            in_channels=32,
            out_channels=1,
            filters=32
        )
        
        disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=3e-4)
        adv_loss = create_adversarial_loss(
            discriminator=discriminator,
            optimizer=disc_optimizer,
            loss_type='hinge',
            use_feature_matching=True
        )
        
        recon_loss = ReconstructionLoss(sample_rate=24000)
        
        # Loss weights (paper parameters)
        reconstruction_weight = 1.0
        adversarial_weight = 3.0
        feature_matching_weight = 3.0
        
        # Create test data
        batch_size = 2
        channels = 32
        samples = 24000
        real_audio = torch.randn(batch_size, channels, samples)
        
        print(f"Test data shape: {real_audio.shape}")
        
        # Forward pass
        print("\nTesting forward pass...")
        model.train()
        discriminator.train()
        
        # Encode and decode
        continuous_embeddings = model.encode(real_audio)
        reconstructed_audio = model.decode(continuous_embeddings)
        
        # Ensure same length
        if reconstructed_audio.shape[-1] > real_audio.shape[-1]:
            reconstructed_audio = reconstructed_audio[..., :real_audio.shape[-1]]
        elif reconstructed_audio.shape[-1] < real_audio.shape[-1]:
            pad_length = real_audio.shape[-1] - reconstructed_audio.shape[-1]
            reconstructed_audio = torch.nn.functional.pad(
                reconstructed_audio, (0, pad_length), mode='reflect'
            )
        
        print(f"✅ Forward pass successful")
        print(f"   Input shape: {real_audio.shape}")
        print(f"   Reconstructed shape: {reconstructed_audio.shape}")
        
        # Test discriminator training
        print("\nTesting discriminator training...")
        disc_loss = adv_loss.train_adv(
            fake=reconstructed_audio.detach(),
            real=real_audio
        )
        print(f"✅ Discriminator loss: {disc_loss.item():.6f}")
        
        # Test generator training
        print("\nTesting generator training...")
        model_optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        model_optimizer.zero_grad()
        
        # Compute individual losses
        recon_loss_val, recon_metrics = recon_loss(reconstructed_audio, real_audio)
        adv_loss_val, feat_loss_val = adv_loss(reconstructed_audio, real_audio)
        
        # Simple weighted loss combination
        effective_loss = (reconstruction_weight * recon_loss_val + 
                         adversarial_weight * adv_loss_val + 
                         feature_matching_weight * feat_loss_val)
        
        # Backward pass
        effective_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        model_optimizer.step()
        
        print(f"✅ Generator training successful")
        print(f"   Reconstruction loss: {recon_loss_val.item():.6f}")
        print(f"   Adversarial loss: {adv_loss_val.item():.6f}")
        print(f"   Feature matching loss: {feat_loss_val.item():.6f}")
        print(f"   Effective loss: {effective_loss.item():.6f}")
        
        # Check gradients
        print("\nChecking gradients...")
        has_gradients = False
        for name, param in model.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_gradients = True
                break
        
        if has_gradients:
            print("✅ Gradients are flowing through the model")
        else:
            print("❌ No gradients found in model parameters")
            return False
        
        print("✅ Backpropagation test successful")
        return True
        
    except Exception as e:
        print(f"❌ Backpropagation test failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        return False

def test_balancer_integration():
    """Test loss balancer integration with the complete training pipeline."""
    print("\n" + "=" * 60)
    print("TESTING LOSS BALANCER INTEGRATION")
    print("=" * 60)
    
    try:
        from balancer import Balancer
        from model_builder import build_encodec_model
        from losses import ReconstructionLoss
        from discriminators import create_ms_stft_discriminator
        from adversarial_losses import create_adversarial_loss
        
        # Create model and components (same as main training)
        print("Setting up training components...")
        model = build_encodec_model(
            sample_rate=24000,
            channels=32,
            n_filters=4,
            n_residual_layers=1,
            dimension=32,
            causal=False
        )
        
        discriminator = create_ms_stft_discriminator(
            in_channels=32,
            out_channels=1,
            filters=32
        )
        
        disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=3e-4)
        adv_loss = create_adversarial_loss(
            discriminator=discriminator,
            optimizer=disc_optimizer,
            loss_type='hinge',
            use_feature_matching=True
        )
        
        recon_loss = ReconstructionLoss(sample_rate=24000)
        
        # Create balancer with paper weights (same as main.py)
        print("Creating loss balancer with paper weights...")
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
        
        print("✅ Balancer created successfully")
        print(f"   Weights: {loss_weights}")
        print(f"   Balance grads: {balancer.balance_grads}")
        print(f"   Total norm: {balancer.total_norm}")
        print(f"   EMA decay: {balancer.averager.decay}")
        
        # Create test data
        batch_size = 2
        channels = 32
        samples = 24000
        real_audio = torch.randn(batch_size, channels, samples)
        
        print(f"\nTest data shape: {real_audio.shape}")
        
        # Test complete training step with balancer (like in main.py)
        print("\nTesting complete training step with balancer...")
        model.train()
        discriminator.train()
        
        # Forward pass: encode → decode (like in main.py)
        continuous_embeddings = model.encode(real_audio)
        reconstructed_audio = model.decode(continuous_embeddings)
        
        # Ensure same length (like in main.py)
        if reconstructed_audio.shape[-1] > real_audio.shape[-1]:
            reconstructed_audio = reconstructed_audio[..., :real_audio.shape[-1]]
        elif reconstructed_audio.shape[-1] < real_audio.shape[-1]:
            pad_length = real_audio.shape[-1] - reconstructed_audio.shape[-1]
            reconstructed_audio = torch.nn.functional.pad(
                reconstructed_audio, (0, pad_length), mode='reflect'
            )
        
        print(f"✅ Forward pass successful")
        print(f"   Input shape: {real_audio.shape}")
        print(f"   Reconstructed shape: {reconstructed_audio.shape}")
        
        # Test discriminator training (like in main.py)
        print("\nTesting discriminator training...")
        disc_loss = adv_loss.train_adv(
            fake=reconstructed_audio.detach(),
            real=real_audio
        )
        print(f"✅ Discriminator loss: {disc_loss.item():.6f}")
        
        # Test generator training with balancer (like in main.py)
        print("\nTesting generator training with balancer...")
        model_optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        model_optimizer.zero_grad()
        
        # Ensure reconstructed_audio has gradients enabled BEFORE computing losses
        reconstructed_audio.requires_grad_(True)
        
        # Compute individual losses (like in main.py)
        recon_loss_val, recon_metrics = recon_loss(reconstructed_audio, real_audio)
        adv_loss_val, feat_loss_val = adv_loss(reconstructed_audio, real_audio)
        
        # Split reconstruction loss into time and frequency components
        time_loss = recon_metrics['time_loss']
        freq_loss = recon_metrics['freq_loss']
        
        print(f"   Time loss: {time_loss.item():.6f}")
        print(f"   Freq loss: {freq_loss.item():.6f}")
        print(f"   Adv loss: {adv_loss_val.item():.6f}")
        print(f"   Feat loss: {feat_loss_val.item():.6f}")
        
        # Use balancer for gradient balancing (like in main.py)
        
        balanced_losses = {
            'time_loss': time_loss,
            'freq_loss': freq_loss,
            'adv_loss': adv_loss_val,
            'feat_loss': feat_loss_val
        }
        
        # Apply balancer - this handles gradient balancing and backward pass
        effective_loss = balancer.backward(balanced_losses, reconstructed_audio)
        
        print(f"✅ Balancer backward pass successful")
        print(f"   Effective loss: {effective_loss.item():.6f}")
        print(f"   Balancer metrics: {balancer.metrics}")
        
        # Gradient clipping (like in main.py)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        model_optimizer.step()
        
        print("✅ Generator training with balancer successful")
        
        # Test multiple training steps to verify balancer stability
        print("\nTesting balancer stability over multiple steps...")
        for step in range(3):
            # New random data for each step
            real_audio = torch.randn(batch_size, channels, samples)
            
            # Forward pass
            continuous_embeddings = model.encode(real_audio)
            reconstructed_audio = model.decode(continuous_embeddings)
            
            # Ensure same length
            if reconstructed_audio.shape[-1] > real_audio.shape[-1]:
                reconstructed_audio = reconstructed_audio[..., :real_audio.shape[-1]]
            elif reconstructed_audio.shape[-1] < real_audio.shape[-1]:
                pad_length = real_audio.shape[-1] - reconstructed_audio.shape[-1]
                reconstructed_audio = torch.nn.functional.pad(
                    reconstructed_audio, (0, pad_length), mode='reflect'
                )
            
            # Ensure reconstructed_audio has gradients enabled BEFORE computing losses
            reconstructed_audio.requires_grad_(True)
            
            # Compute losses
            recon_loss_val, recon_metrics = recon_loss(reconstructed_audio, real_audio)
            adv_loss_val, feat_loss_val = adv_loss(reconstructed_audio, real_audio)
            
            time_loss = recon_metrics['time_loss']
            freq_loss = recon_metrics['freq_loss']
            
            # Use balancer
            
            balanced_losses = {
                'time_loss': time_loss,
                'freq_loss': freq_loss,
                'adv_loss': adv_loss_val,
                'feat_loss': feat_loss_val
            }
            
            effective_loss = balancer.backward(balanced_losses, reconstructed_audio)
            
            print(f"   Step {step + 1}: Effective loss = {effective_loss.item():.6f}")
            if balancer.metrics:
                print(f"   Gradient ratios: {balancer.metrics}")
        
        print("✅ Balancer stability test completed")
        
        # Test validation with balancer (like in main.py)
        print("\nTesting validation with balancer...")
        model.eval()
        discriminator.eval()
        
        with torch.no_grad():
            # Forward pass
            continuous_embeddings = model.encode(real_audio)
            reconstructed_audio = model.decode(continuous_embeddings)
            
            # Ensure same length
            if reconstructed_audio.shape[-1] > real_audio.shape[-1]:
                reconstructed_audio = reconstructed_audio[..., :real_audio.shape[-1]]
            elif reconstructed_audio.shape[-1] < real_audio.shape[-1]:
                pad_length = real_audio.shape[-1] - reconstructed_audio.shape[-1]
                reconstructed_audio = torch.nn.functional.pad(
                    reconstructed_audio, (0, pad_length), mode='reflect'
                )
            
            # Use balancer for validation (temporarily enable gradients)
            reconstructed_audio.requires_grad_(True)
            
            # Compute losses
            recon_loss_val, recon_metrics = recon_loss(reconstructed_audio, real_audio)
            adv_loss_val, feat_loss_val = adv_loss(reconstructed_audio, real_audio)
            
            time_loss = recon_metrics['time_loss']
            freq_loss = recon_metrics['freq_loss']
            balanced_losses = {
                'time_loss': time_loss,
                'freq_loss': freq_loss,
                'adv_loss': adv_loss_val,
                'feat_loss': feat_loss_val
            }
            
            val_effective_loss = balancer.backward(balanced_losses, reconstructed_audio)
            reconstructed_audio.requires_grad_(False)
            
            print(f"✅ Validation with balancer successful")
            print(f"   Validation effective loss: {val_effective_loss.item():.6f}")
        
        print("✅ All balancer integration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Balancer integration test failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 STARTING BASELINE AUTOENCODER TRAINING PIPELINE TESTS")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Dataloader", test_dataloader),
        ("Model Loading", test_model_loading),
        ("Losses", test_losses),
        ("Backpropagation", test_backpropagation),
        ("Loss Balancer Integration", test_balancer_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Your setup is ready for training.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

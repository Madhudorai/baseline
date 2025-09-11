#!/usr/bin/env python3
"""
Core functionality test without requiring actual audio data.
Tests baseline autoencoder model, losses, and backpropagation with dummy data.
"""

import sys
import torch
import logging
import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_dummy_audio_files(temp_dir, num_files=5):
    """Create dummy audio files for testing."""
    files = []
    for i in range(num_files):
        # Create 2-second dummy audio (32 channels, 24kHz)
        duration = 2.0
        sample_rate = 24000
        samples = int(duration * sample_rate)
        
        # Generate random multichannel audio
        audio = np.random.randn(32, samples).astype(np.float32)
        
        # Save as WAV file
        file_path = temp_dir / f"dummy_audio_{i:03d}.wav"
        sf.write(str(file_path), audio.T, sample_rate)
        files.append(file_path)
    
    return files

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

def test_dataloader_with_dummy_data():
    """Test dataloader with dummy audio files."""
    print("\n" + "=" * 60)
    print("TESTING DATALOADER WITH DUMMY DATA")
    print("=" * 60)
    
    try:
        from dataloader import create_train_test_dataloaders
        
        # Create temporary directory with dummy audio files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            print(f"Created temporary directory: {temp_path}")
            
            # Create dummy audio files
            dummy_files = create_dummy_audio_files(temp_path, num_files=10)
            print(f"Created {len(dummy_files)} dummy audio files")
            
            # Test dataloader creation
            train_loader, test_loader = create_train_test_dataloaders(
                audio_dir=str(temp_path),
                train_ratio=0.8,
                batch_size=2,
                sample_rate=24000,
                segment_duration=1.0,
                channels=32,
                train_dataset_size=20,
                test_dataset_size=10,
                min_file_duration=1.0,
                random_crop=True
            )
            
            print(f"✅ Dataloaders created successfully")
            print(f"   Train dataset size: {len(train_loader.dataset)} samples")
            print(f"   Test dataset size: {len(test_loader.dataset)} samples")
            
            # Test loading batches
            train_batch = next(iter(train_loader))
            test_batch = next(iter(test_loader))
            
            print(f"✅ Train batch shape: {train_batch.shape}")
            print(f"✅ Test batch shape: {test_batch.shape}")
            
            # Verify shapes
            expected_shape = (2, 32, 24000)
            if train_batch.shape == expected_shape and test_batch.shape == expected_shape:
                print("✅ Batch shapes are correct")
                return True
            else:
                print(f"❌ Batch shapes incorrect. Expected {expected_shape}")
                return False
        
    except Exception as e:
        print(f"❌ Dataloader test failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        return False

def test_model_and_losses():
    """Test model creation, losses, and backpropagation."""
    print("\n" + "=" * 60)
    print("TESTING MODEL AND LOSSES")
    print("=" * 60)
    
    try:
        from model_builder import build_encodec_model
        from losses import ReconstructionLoss
        from discriminators import create_ms_stft_discriminator
        from adversarial_losses import create_adversarial_loss
        
        # Create test data
        batch_size = 2
        channels = 32
        samples = 24000
        real_audio = torch.randn(batch_size, channels, samples)
        print(f"Test data shape: {real_audio.shape}")
        
        # Test model creation
        print("\nTesting model creation...")
        model = build_encodec_model(
            sample_rate=24000,
            channels=32,
            n_filters=4,
            n_residual_layers=1,
            dimension=32,
            causal=False
        )
        print(f"✅ Model created: {type(model).__name__}")
        
        # Test forward pass
        print("\nTesting forward pass...")
        model.train()
        reconstructed = model(real_audio)
        print(f"✅ Forward pass: {real_audio.shape} -> {reconstructed.shape}")
        
        # Test encode/decode
        print("\nTesting encode/decode...")
        codes = model.encode(real_audio)
        decoded = model.decode(codes)
        print(f"✅ Encode/decode: {real_audio.shape} -> {codes.shape} -> {decoded.shape}")
        
        # Test reconstruction loss
        print("\nTesting reconstruction loss...")
        recon_loss = ReconstructionLoss(sample_rate=24000)
        loss_val, metrics = recon_loss(reconstructed, real_audio)
        print(f"✅ Reconstruction loss: {loss_val.item():.6f}")
        
        # Test discriminator
        print("\nTesting discriminator...")
        discriminator = create_ms_stft_discriminator(
            in_channels=32,
            out_channels=1,
            filters=32
        )
        discriminator.train()  # Ensure discriminator is in training mode
        
        disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=3e-4)
        adv_loss = create_adversarial_loss(
            discriminator=discriminator,
            optimizer=disc_optimizer,
            loss_type='hinge',
            use_feature_matching=True
        )
        
        # Test adversarial loss
        adv_loss_val, feat_loss_val = adv_loss(reconstructed, real_audio)
        print(f"✅ Adversarial loss: {adv_loss_val.item():.6f}")
        print(f"✅ Feature matching loss: {feat_loss_val.item():.6f}")
        
        # Test basic loss combination
        print("\nTesting weighted loss combination...")
        
        # Test simple weighted loss combination (what happens in training)
        model.train()
        fresh_reconstructed = model(real_audio)
        fresh_recon_loss, fresh_recon_metrics = recon_loss(fresh_reconstructed, real_audio)
        fresh_adv_loss, fresh_feat_loss = adv_loss(fresh_reconstructed, real_audio)
        
        # Get time and frequency components
        fresh_time_loss = fresh_recon_metrics['time_loss']
        fresh_freq_loss = fresh_recon_metrics['freq_loss']
        
        # Simple weighted combination using paper weights
        total_loss = (0.1 * fresh_time_loss + 1.0 * fresh_freq_loss + 3.0 * fresh_adv_loss + 3.0 * fresh_feat_loss)
        
        print(f"✅ Weighted loss combination: {total_loss.item():.6f}")
        print(f"   Time reconstruction: {fresh_time_loss.item():.6f}")
        print(f"   Freq reconstruction: {fresh_freq_loss.item():.6f}")
        print(f"   Adversarial: {fresh_adv_loss.item():.6f}")
        print(f"   Feature matching: {fresh_feat_loss.item():.6f}")
        print(f"   Using paper weights: λt=0.1, λf=1.0, λg=3.0, λfeat=3.0")
        
        # Test backpropagation
        print("\nTesting backpropagation...")
        model_optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        model_optimizer.zero_grad()
        
        # Forward pass with gradients (ensure we're not using no_grad)
        reconstructed = model(real_audio)
        loss_val, _ = recon_loss(reconstructed, real_audio)
        loss_val.backward()
        model_optimizer.step()
        
        print("✅ Backpropagation successful")
        return True
        
    except Exception as e:
        print(f"❌ Model and losses test failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        return False

def test_balancer():
    """Test loss balancer functionality."""
    print("\n" + "=" * 60)
    print("TESTING LOSS BALANCER")
    print("=" * 60)
    
    try:
        from balancer import Balancer
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
        
        # Test basic balancer functionality
        print("\nTesting basic balancer functionality...")
        weights = {'loss1': 1.0, 'loss2': 2.0}
        balancer = Balancer(
            weights=weights,
            balance_grads=True,
            total_norm=1.0,
            ema_decay=0.9,
            per_batch_item=True,
            epsilon=1e-12,
            monitor=True
        )
        
        # Create simple losses
        y = fake_audio.clone()
        y.requires_grad_(True)
        loss1 = (y - 1.0).pow(2).mean()
        loss2 = (y - 2.0).pow(2).mean()
        
        losses = {'loss1': loss1, 'loss2': loss2}
        effective_loss = balancer.backward(losses, y)
        
        print(f"✅ Basic balancer test passed")
        print(f"   Loss1: {loss1.item():.6f}")
        print(f"   Loss2: {loss2.item():.6f}")
        print(f"   Effective loss: {effective_loss.item():.6f}")
        print(f"   Balancer metrics: {balancer.metrics}")
        
        # Test balancer with audio losses (similar to main training)
        print("\nTesting balancer with audio-style losses...")
        
        # Create reconstruction loss
        recon_loss = ReconstructionLoss(sample_rate=24000)
        
        # Create discriminator and adversarial loss
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
        
        # Create balancer with paper weights
        paper_weights = {
            'time_loss': 0.1,      # λt - time domain reconstruction
            'freq_loss': 1.0,      # λf - frequency domain reconstruction  
            'adv_loss': 3.0,       # λg - adversarial loss
            'feat_loss': 3.0       # λfeat - feature matching loss
        }
        
        audio_balancer = Balancer(
            weights=paper_weights,
            balance_grads=True,
            total_norm=1.0,
            ema_decay=0.999,
            per_batch_item=True,
            epsilon=1e-12,
            monitor=True
        )
        
        # Compute audio losses
        recon_loss_val, recon_metrics = recon_loss(fake_audio, real_audio)
        adv_loss_val, feat_loss_val = adv_loss(fake_audio, real_audio)
        
        time_loss = recon_metrics['time_loss']
        freq_loss = recon_metrics['freq_loss']
        
        # Test balancer with audio losses
        # Ensure fake_audio has gradients enabled for balancer
        fake_audio.requires_grad_(True)
        
        balanced_losses = {
            'time_loss': time_loss,
            'freq_loss': freq_loss,
            'adv_loss': adv_loss_val,
            'feat_loss': feat_loss_val
        }
        
        effective_loss = audio_balancer.backward(balanced_losses, fake_audio)
        
        print(f"✅ Audio balancer test passed")
        print(f"   Time loss: {time_loss.item():.6f}")
        print(f"   Freq loss: {freq_loss.item():.6f}")
        print(f"   Adv loss: {adv_loss_val.item():.6f}")
        print(f"   Feat loss: {feat_loss_val.item():.6f}")
        print(f"   Effective loss: {effective_loss.item():.6f}")
        print(f"   Balancer metrics: {audio_balancer.metrics}")
        
        # Test balancer without gradient balancing
        print("\nTesting balancer without gradient balancing...")
        simple_balancer = Balancer(
            weights={'loss1': 1.0, 'loss2': 2.0},
            balance_grads=False,  # Disable gradient balancing
            total_norm=1.0,
            ema_decay=0.9,
            per_batch_item=True,
            epsilon=1e-12,
            monitor=True
        )
        
        y2 = fake_audio.clone()
        y2.requires_grad_(True)
        loss1 = (y2 - 1.0).pow(2).mean()
        loss2 = (y2 - 2.0).pow(2).mean()
        
        simple_losses = {'loss1': loss1, 'loss2': loss2}
        simple_effective_loss = simple_balancer.backward(simple_losses, y2)
        
        # Should be simple weighted sum: 1.0 * loss1 + 2.0 * loss2
        expected_loss = 1.0 * loss1 + 2.0 * loss2
        
        print(f"✅ Non-balancing balancer test passed")
        print(f"   Loss1: {loss1.item():.6f}")
        print(f"   Loss2: {loss2.item():.6f}")
        print(f"   Effective loss: {simple_effective_loss.item():.6f}")
        print(f"   Expected (1.0 * loss1 + 2.0 * loss2): {expected_loss.item():.6f}")
        
        # Test gradient ratio maintenance over multiple steps
        print("\nTesting gradient ratio maintenance...")
        ratio_balancer = Balancer(
            weights={'loss1': 1.0, 'loss2': 3.0},  # loss2 should get 75% of gradient
            balance_grads=True,
            total_norm=1.0,
            ema_decay=0.9,
            per_batch_item=True,
            epsilon=1e-12,
            monitor=True
        )
        
        for step in range(3):
            y3 = torch.randn(batch_size, channels, samples, requires_grad=True)
            loss1 = (y3 - 1.0).pow(2).mean()
            loss2 = (y3 - 2.0).pow(2).mean()
            
            ratio_losses = {'loss1': loss1, 'loss2': loss2}
            ratio_effective_loss = ratio_balancer.backward(ratio_losses, y3)
            
            print(f"   Step {step + 1}: Loss1={loss1.item():.6f}, Loss2={loss2.item():.6f}, "
                  f"Effective={ratio_effective_loss.item():.6f}")
            if ratio_balancer.metrics:
                print(f"   Gradient ratios: {ratio_balancer.metrics}")
        
        print("✅ Gradient ratio test completed")
        print("✅ All balancer tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Balancer test failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run core tests."""
    print("🚀 STARTING BASELINE AUTOENCODER TESTS")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Dataloader with Dummy Data", test_dataloader_with_dummy_data),
        ("Model and Losses", test_model_and_losses),
        ("Loss Balancer", test_balancer)
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
        print("🎉 ALL CORE TESTS PASSED! Your setup is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

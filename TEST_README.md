# Testing Setup

This directory contains comprehensive test scripts to verify your baseline autoencoder training pipeline is working correctly.

## Test Scripts

### 1. `test_core.py` - Core Functionality Test (Recommended)
**Use this first** - Tests core functionality without requiring actual audio data.

```bash
python3 test_core.py
```

**What it tests:**
- ✅ All imports work correctly
- ✅ Dataloader with dummy audio files
- ✅ Model creation and forward pass
- ✅ Loss functions (reconstruction, adversarial, feature matching)
- ✅ Backpropagation

**Requirements:** None (creates dummy data automatically)

### 2. `test_setup.py` - Full Pipeline Test
**Use this if you have actual audio data** - Tests the complete pipeline with real data.

```bash
python3 test_setup.py
```

**What it tests:**
- ✅ All imports work correctly
- ✅ Dataloader with real audio files from `/Users/swarsys/Documents/GitHub/eigenscape/`
- ✅ Model creation and forward pass
- ✅ All loss functions
- ✅ Discriminator training
- ✅ Generator training
- ✅ Complete backpropagation

**Requirements:** Audio files in the specified directory

## Expected Output

Both scripts will show:
- Detailed test progress
- ✅ or ❌ for each test component
- Final summary with pass/fail count
- Clear error messages if something fails

## Troubleshooting

### If `test_core.py` fails:
- Check that all dependencies are installed (torch, soundfile, etc.)
- Look at the specific error messages for guidance

### If `test_setup.py` fails:
- Make sure the audio directory exists: `/Users/swarsys/Documents/GitHub/eigenscape/`
- Ensure there are audio files (`.wav` format) in the directory
- Check that files are longer than 1 second
- Use `test_core.py` first to verify core functionality

## Success Criteria

All tests should pass before starting training. The tests verify:
1. **Data Loading**: Can load and process audio files correctly
2. **Model**: Can create, encode, and decode audio
3. **Losses**: All loss functions work and can backpropagate
4. **Training Loop**: Complete training step works end-to-end

## Next Steps

Once all tests pass:
1. Run your training with: `python3 main.py`
2. Monitor training progress in Weights & Biases
3. Check model checkpoints are being saved

---

**Note:** These test scripts are designed to be comprehensive but lightweight. They use small batch sizes and short audio segments to run quickly while still testing all the critical functionality.

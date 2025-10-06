#!/usr/bin/env python3
"""
Quick one-liner script to upload model to wandb.
"""

import wandb
import torch
from pathlib import Path

# Initialize wandb
wandb.init(project="baseline-autoencoder-training", job_type="model-upload")

# Create and upload artifact
artifact = wandb.Artifact(
    name="trained-model",
    type="model", 
    description="Trained baseline autoencoder model"
)
artifact.add_file("/user/i/iran/baseline/bestmodel.pth")
wandb.log_artifact(artifact)

print("✅ Model uploaded to wandb!")
wandb.finish()

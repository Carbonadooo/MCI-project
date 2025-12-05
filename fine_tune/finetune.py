#!/usr/bin/env python3
"""
Fine-tune IMU2CLIP on custom dataset with IMU↔Text alignment

Uses separate train and validation folders (no automatic splitting)
Configuration can be loaded from YAML and overridden via command-line arguments
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import argparse
import json
import yaml
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from inference_imu2clip import IMU2CLIPInference
from lib.imu_models import MW2StackRNNPooling, MW2StackRNNPooling9Ch, MW2StackRNNPooling18Ch, MW2StackRNNPooling21Ch
from lib.clip_model import ClipPLModel
from lib.train_modules import MultimodalContrastiveLearningModule
from fine_tune.dataset import IMUTextDataset
from fine_tune.text_alternatives import TEXT_ALTERNATIVES


def load_config(config_path=None):
    """Load configuration from YAML file"""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def symmetric_contrastive_loss(imu_emb, text_emb, temperature=0.07):
    """
    Symmetric IMU↔Text contrastive loss
    
    Args:
        imu_emb: (B, D) IMU embeddings
        text_emb: (B, D) Text embeddings
        temperature: Temperature parameter γ
        
    Returns:
        loss: Symmetric contrastive loss Li↔t
    """
    # Normalize embeddings with epsilon to prevent division by zero
    eps = 1e-8
    imu_emb_norm = torch.nn.functional.normalize(imu_emb, p=2, dim=-1, eps=eps)
    text_emb_norm = torch.nn.functional.normalize(text_emb, p=2, dim=-1, eps=eps)
    
    # Check for NaN or Inf in embeddings
    if torch.isnan(imu_emb_norm).any() or torch.isinf(imu_emb_norm).any():
        print("WARNING: NaN or Inf detected in IMU embeddings")
        return torch.tensor(0.0, device=imu_emb.device, requires_grad=True), \
               torch.tensor(0.0, device=imu_emb.device, requires_grad=True), \
               torch.tensor(0.0, device=imu_emb.device, requires_grad=True)
    
    if torch.isnan(text_emb_norm).any() or torch.isinf(text_emb_norm).any():
        print("WARNING: NaN or Inf detected in text embeddings")
        return torch.tensor(0.0, device=text_emb.device, requires_grad=True), \
               torch.tensor(0.0, device=text_emb.device, requires_grad=True), \
               torch.tensor(0.0, device=text_emb.device, requires_grad=True)
    
    # Compute similarity matrix
    logits = torch.matmul(imu_emb_norm, text_emb_norm.t()) / temperature  # (B, B)
    
    # Clamp logits to prevent overflow
    logits = torch.clamp(logits, min=-100, max=100)
    
    # Labels: diagonal elements are positive pairs
    labels = torch.arange(len(imu_emb), device=imu_emb.device)
    
    # IMU-to-Text loss (Li2t)
    loss_i2t = nn.functional.cross_entropy(logits, labels)
    
    # Text-to-IMU loss (Lt2i)
    loss_t2i = nn.functional.cross_entropy(logits.t(), labels)
    
    # Symmetric loss
    loss = (loss_i2t + loss_t2i) / 2.0
    
    # Check for NaN in loss
    if torch.isnan(loss):
        print("WARNING: NaN loss detected")
        return torch.tensor(0.0, device=loss.device, requires_grad=True), \
               torch.tensor(0.0, device=loss.device, requires_grad=True), \
               torch.tensor(0.0, device=loss.device, requires_grad=True)
    
    return loss, loss_i2t, loss_t2i


def finetune(config):
    """Main fine-tuning function"""
    
    # Create output directory with model name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = config['model'].get('name', 'MW2StackRNNPooling')
    
    # Determine channel suffix based on model name
    if "21Ch" in model_name or "21" in model_name:
        model_suffix = "21ch"
    elif "18Ch" in model_name or ("18" in model_name and "MW2" in model_name):
        model_suffix = "18ch"
    elif "9Ch" in model_name or ("9" in model_name and "MW2" in model_name):
        model_suffix = "9ch"
    else:
        model_suffix = "6ch"
    
    output_dir = Path(config['output']['dir']) / f"finetune_{model_suffix}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("IMU2CLIP Fine-tuning")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print()
    
    # Save config
    with open(output_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Device
    device = torch.device(config['device'])
    print(f"Device: {device}")
    print()
    
    # Load datasets from SEPARATE train and val folders
    print("=" * 70)
    print("Loading Training Dataset")
    print("=" * 70)
    train_dataset = IMUTextDataset(
        data_dir=config['data']['train_dir'],
        window_size=config['data']['window_size'],
        augmentation_multiplier=config['augmentation']['multiplier'],
        training=True,  # Enable augmentation
        max_per_class=config['data']['max_per_class'],
        aug_config=config['augmentation']['imu'],
        num_channels=int(config['data'].get('num_channels', 6))  # Default to 6 channels
    )
    print()
    
    print("=" * 70)
    print("Loading Validation Dataset")
    print("=" * 70)
    val_dataset = IMUTextDataset(
        data_dir=config['data']['val_dir'],
        window_size=config['data']['window_size'],
        augmentation_multiplier=1,  # No augmentation
        training=False,  # Disable augmentation
        max_per_class=config['data']['max_per_class'],
        aug_config=None,
        num_channels=int(config['data'].get('num_channels', 6))  # Default to 6 channels
    )
    print()
    
    print(f"Summary:")
    print(f"  Train: {len(train_dataset)} samples (after {config['augmentation']['multiplier']}x augmentation)")
    print(f"  Val:   {len(val_dataset)} samples (original data)")
    print()
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )
    
    # Load pre-trained model
    print("Loading pre-trained model...")
    
    # Select model based on config
    model_name = config['model'].get('name', 'MW2StackRNNPooling')  # Default to 6-channel model
    num_channels = int(config['data'].get('num_channels', 6))
    
    if model_name == 'MW2StackRNNPooling21Ch':
        if num_channels != 21:
            print(f"WARNING: Model '{model_name}' expects 21 channels but config specifies {num_channels}. Using 21 channels.")
            num_channels = 21
        imu_encoder = MW2StackRNNPooling21Ch(size_embeddings=config['model']['embedding_size'])
        print(f"  Model: {model_name} (21-channel: 9 IMU + 9 rotation + 3 world-frame accel)")
    elif model_name == 'MW2StackRNNPooling18Ch':
        if num_channels != 18:
            print(f"WARNING: Model '{model_name}' expects 18 channels but config specifies {num_channels}. Using 18 channels.")
            num_channels = 18
        imu_encoder = MW2StackRNNPooling18Ch(size_embeddings=config['model']['embedding_size'])
        print(f"  Model: {model_name} (18-channel: 9 IMU + 9 rotation features)")
    elif model_name == 'MW2StackRNNPooling9Ch':
        if num_channels != 9:
            print(f"WARNING: Model '{model_name}' expects 9 channels but config specifies {num_channels}. Using 9 channels.")
            num_channels = 9
        imu_encoder = MW2StackRNNPooling9Ch(size_embeddings=config['model']['embedding_size'])
        print(f"  Model: {model_name} (9-channel with learnable 9→6 projection)")
    elif model_name == 'MW2StackRNNPooling':
        if num_channels != 6:
            print(f"WARNING: Model '{model_name}' expects 6 channels but config specifies {num_channels}. Using 6 channels.")
            num_channels = 6
        imu_encoder = MW2StackRNNPooling(size_embeddings=config['model']['embedding_size'])
        print(f"  Model: {model_name} (6-channel)")
    else:
        raise ValueError(f"Unknown model name: {model_name}. Supported models: MW2StackRNNPooling, MW2StackRNNPooling9Ch, MW2StackRNNPooling18Ch, MW2StackRNNPooling21Ch")
    
    text_encoder = ClipPLModel(freeze=True)
    
    # Load checkpoint
    checkpoint = torch.load(config['model']['checkpoint'], map_location=device)
    
    # Load from LightningModule checkpoint
    if 'state_dict' in checkpoint:
        # Extract imu_encoder weights
        imu_state_dict = {}
        for key, value in checkpoint['state_dict'].items():
            if key.startswith('imu_encoder.'):
                new_key = key.replace('imu_encoder.', '')
                imu_state_dict[new_key] = value
        
        # Load weights (strict=False for 9/18/21-channel models to allow missing channel_projection)
        if model_name in ['MW2StackRNNPooling9Ch', 'MW2StackRNNPooling18Ch', 'MW2StackRNNPooling21Ch']:
            # For 9/18/21-channel models, only load the main network weights (not channel_projection)
            missing_keys, unexpected_keys = imu_encoder.load_state_dict(imu_state_dict, strict=False)
            print(f"✓ Loaded IMU encoder from checkpoint ({model_name}, channel_projection initialized randomly)")
            if missing_keys:
                projection_keys = [k for k in missing_keys if 'channel_projection' in k]
                if projection_keys:
                    print(f"  New layers (randomly initialized): {projection_keys}")
        else:
            # For 6-channel model, load all weights
            imu_encoder.load_state_dict(imu_state_dict, strict=True)
            print("✓ Loaded IMU encoder from checkpoint")
    else:
        # Direct state dict
        imu_encoder.load_state_dict(checkpoint)
        print("✓ Loaded IMU encoder")
    
    imu_encoder = imu_encoder.to(device)
    text_encoder = text_encoder.to(device)

    # Freeze all parameters first
    for name, param in imu_encoder.named_parameters():
        param.requires_grad = False
    
    # Freeze all layers except the final projection
    print("\nFreezing layers...")
    print("Available parameters:")
    all_params = list(imu_encoder.named_parameters())
    for name, param in all_params:
        print(f"  - {name}: {param.shape}")
    
    print("\nSetting trainable parameters...")
    # Only train the final RNN layer (and channel_projection for 9-channel model)
    trainable_layers = ['linear', 'fc', 'out', "net.5.weight_hh_l0", 'net.5.weight_ih_l0', 'channel_projection']
    
    for name, param in imu_encoder.named_parameters():
        # Make trainable if it's in the last layer or matches keywords
        if any(layer in name.lower() for layer in trainable_layers):
            param.requires_grad = True
            print(f"  ✓ {name} - Trainable ({param.numel():,} params)")
        else:
            param.requires_grad = False
            print(f"  ✓ {name} - Frozen ({param.numel():,} params)")
    print()
    
    # Fallback: if no trainable params, unfreeze last RNN layer
    trainable_count = sum(p.requires_grad for p in imu_encoder.parameters())
    if trainable_count == 0:
        print("\n⚠ No parameters matched keywords. Unfreezing last RNN layer...")
        n_to_unfreeze = min(3, len(all_params))
        for i in range(1, n_to_unfreeze + 1):
            param_name, param = all_params[-i]
            param.requires_grad = True
            print(f"  ✓ {param_name} - trainable ({param.numel():,} params)")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in imu_encoder.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in imu_encoder.parameters())
    print(f"\nTrainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")
    print()
    
    # Optimizer (only for trainable parameters)
    # Ensure numeric values are floats (in case YAML parsed them as strings)
    learning_rate = float(config['training']['learning_rate'])
    weight_decay = float(config['training']['weight_decay'])
    
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, imu_encoder.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    epochs = int(config['training']['epochs'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=epochs * len(train_loader)
    )
    
    # Pre-encode all text labels
    print("Pre-encoding text labels...")
    with torch.no_grad():
        # Encode all text labels at once (use training dataset's class info)
        all_text_embeddings = text_encoder.get_text_embeddings(train_dataset.text_labels, device=device)
    print(f"✓ Encoded {len(all_text_embeddings)} text labels: {all_text_embeddings.shape}")
    
    # Check text embeddings for NaN or Inf
    if torch.isnan(all_text_embeddings).any():
        print("❌ ERROR: NaN detected in text embeddings!")
        return
    if torch.isinf(all_text_embeddings).any():
        print("❌ ERROR: Inf detected in text embeddings!")
        return
    
    print(f"✓ Text embeddings stats:")
    print(f"    Mean: {all_text_embeddings.mean():.4f}")
    print(f"    Std:  {all_text_embeddings.std():.4f}")
    print(f"    Min:  {all_text_embeddings.min():.4f}")
    print(f"    Max:  {all_text_embeddings.max():.4f}")
    print()
    
    # Training loop
    print("=" * 70)
    print("Starting training...")
    temperature = float(config['training']['temperature'])
    early_stop_patience = int(config['training']['early_stop_patience'])
    
    if early_stop_patience > 0:
        print(f"Early stopping enabled (patience: {early_stop_patience} epochs)")
    else:
        print("Early stopping disabled")
    print("=" * 70)
    
    history = {
        'train_loss': [],
        'train_loss_i2t': [],
        'train_loss_t2i': [],
        'val_loss': [],
        'val_loss_i2t': [],
        'val_loss_t2i': [],
        'val_accuracy': [],
        'learning_rate': []
    }
    
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0
    
    # Debug: Check first batch
    print("Checking first training batch...")
    first_batch = next(iter(train_loader))
    imu_test = first_batch['imu'].to(device)
    print(f"  IMU input shape: {imu_test.shape}")
    print(f"  IMU stats: mean={imu_test.mean():.4f}, std={imu_test.std():.4f}, min={imu_test.min():.4f}, max={imu_test.max():.4f}")
    
    with torch.no_grad():
        imu_encoder.eval()
        imu_emb_test = imu_encoder(imu_test)
        print(f"  IMU embedding shape: {imu_emb_test.shape}")
        print(f"  IMU embedding stats: mean={imu_emb_test.mean():.4f}, std={imu_emb_test.std():.4f}")
        if torch.isnan(imu_emb_test).any():
            print("  ❌ ERROR: NaN in IMU embeddings on first batch!")
            return
        if torch.isinf(imu_emb_test).any():
            print("  ❌ ERROR: Inf in IMU embeddings on first batch!")
            return
        print("  ✓ First batch check passed")
    print()
    
    for epoch in range(config['training']['epochs']):
        # Training
        imu_encoder.train()
        train_losses = []
        train_losses_i2t = []
        train_losses_t2i = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, batch in enumerate(pbar):
            imu_data = batch['imu'].to(device)
            class_indices = batch['class_idx'].to(device)
            
            # Check for NaN in input data
            if torch.isnan(imu_data).any():
                print(f"\n❌ WARNING: NaN in IMU input data at batch {batch_idx}")
                continue
            
            # Get text embeddings for this batch
            text_emb = all_text_embeddings[class_indices]
            
            # Forward pass
            imu_emb = imu_encoder(imu_data)
            
            # Check for NaN in IMU embeddings
            if torch.isnan(imu_emb).any():
                print(f"\n❌ WARNING: NaN in IMU embeddings at batch {batch_idx}")
                continue
            
            # Compute loss
            loss, loss_i2t, loss_t2i = symmetric_contrastive_loss(
                imu_emb, text_emb, temperature=temperature
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent explosion
            if 'gradient_clip_norm' in config['training']:
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, imu_encoder.parameters()),
                    max_norm=config['training']['gradient_clip_norm']
                )
            
            optimizer.step()
            scheduler.step()
            
            # Record
            train_losses.append(loss.item())
            train_losses_i2t.append(loss_i2t.item())
            train_losses_t2i.append(loss_t2i.item())
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
            })
        
        # Validation
        imu_encoder.eval()
        val_losses = []
        val_losses_i2t = []
        val_losses_t2i = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                imu_data = batch['imu'].to(device)
                class_indices = batch['class_idx'].to(device)
                
                # Get text embeddings
                text_emb = all_text_embeddings[class_indices]
                
                # Forward
                imu_emb = imu_encoder(imu_data)
                
                # Loss
                loss, loss_i2t, loss_t2i = symmetric_contrastive_loss(
                    imu_emb, text_emb, temperature=temperature
                )
                
                val_losses.append(loss.item())
                val_losses_i2t.append(loss_i2t.item())
                val_losses_t2i.append(loss_t2i.item())
                
                # Accuracy: find nearest text for each IMU
                imu_emb_norm = imu_emb / imu_emb.norm(dim=-1, keepdim=True)
                all_text_norm = all_text_embeddings / all_text_embeddings.norm(dim=-1, keepdim=True)
                similarities = torch.matmul(imu_emb_norm, all_text_norm.t())
                predictions = similarities.argmax(dim=1)
                correct += (predictions == class_indices).sum().item()
                total += len(class_indices)
        
        # Compute epoch metrics
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_acc = correct / total if total > 0 else 0.0
        
        history['train_loss'].append(train_loss)
        history['train_loss_i2t'].append(np.mean(train_losses_i2t))
        history['train_loss_t2i'].append(np.mean(train_losses_t2i))
        history['val_loss'].append(val_loss)
        history['val_loss_i2t'].append(np.mean(val_losses_i2t))
        history['val_loss_t2i'].append(np.mean(val_losses_t2i))
        history['val_accuracy'].append(val_acc)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        print(f"  Train Loss: {train_loss:.4f} (i2t: {history['train_loss_i2t'][-1]:.4f}, t2i: {history['train_loss_t2i'][-1]:.4f})")
        print(f"  Val Loss:   {val_loss:.4f} (i2t: {history['val_loss_i2t'][-1]:.4f}, t2i: {history['val_loss_t2i'][-1]:.4f})")
        print(f"  Val Acc:    {val_acc:.2%}")
        print(f"  LR:         {optimizer.param_groups[0]['lr']:.2e}")
        print()
        
        # Save best model (in Lightning checkpoint format)

        # TODO: Use easy to distinguish name for finetuned model folder name. i.e. which model
        if val_loss < best_val_loss or (val_acc > (history['val_accuracy'][best_epoch] if history['val_accuracy'] else 0.0)):
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            
            # Create checkpoint in same format as input
            checkpoint_dict = {
                'state_dict': {
                    f'imu_encoder.{k}': v for k, v in imu_encoder.state_dict().items()
                },
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'hyper_parameters': config
            }
            torch.save(checkpoint_dict, output_dir / "best_model.ckpt")
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f}, val_acc: {val_acc:.2%})")
        else:
            epochs_without_improvement += 1
            print(f"  ⚠ No improvement for {epochs_without_improvement} epoch(s)")
            
            # Early stopping
            if early_stop_patience > 0 and epochs_without_improvement >= early_stop_patience:
                print(f"\n{'=' * 70}")
                print(f"Early stopping triggered after {epochs_without_improvement} epochs without improvement")
                print(f"Best val_loss: {best_val_loss:.4f} at epoch {best_epoch + 1}")
                print(f"{'=' * 70}\n")
                break
    
    # Save final model (in Lightning checkpoint format)
    final_checkpoint_dict = {
        'state_dict': {
            f'imu_encoder.{k}': v for k, v in imu_encoder.state_dict().items()
        },
        'epoch': config['training']['epochs'],
        'best_val_loss': best_val_loss,
        'final_val_accuracy': history['val_accuracy'][-1],
        'hyper_parameters': config
    }
    torch.save(final_checkpoint_dict, output_dir / "final_model.ckpt")
    
    # Save history
    with open(output_dir / "history.json", 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        history_json = {k: [float(v) for v in vals] for k, vals in history.items()}
        json.dump(history_json, f, indent=2)
    
    # Plot training curves
    print("\nGenerating training curves...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Val')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # IMU-to-Text loss
    axes[0, 1].plot(history['train_loss_i2t'], label='Train')
    axes[0, 1].plot(history['val_loss_i2t'], label='Val')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('IMU-to-Text Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Text-to-IMU loss
    axes[1, 0].plot(history['train_loss_t2i'], label='Train')
    axes[1, 0].plot(history['val_loss_t2i'], label='Val')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Text-to-IMU Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Accuracy and LR
    ax1 = axes[1, 1]
    ax2 = ax1.twinx()
    
    ax1.plot(history['val_accuracy'], 'b-', label='Val Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(history['learning_rate'], 'r-', label='Learning Rate')
    ax2.set_ylabel('Learning Rate', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    if max(history['learning_rate']) > 0:
        ax2.set_yscale('log')
    
    axes[1, 1].set_title('Validation Accuracy & Learning Rate')
    
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150)
    print(f"✓ Saved to {output_dir / 'training_curves.png'}")
    
    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best val loss: {best_val_loss:.4f} (epoch {best_epoch + 1})")
    print(f"Final val acc: {history['val_accuracy'][-1]:.2%}")
    print(f"Total epochs trained: {len(history['train_loss'])}")
    if len(history['train_loss']) < epochs:
        print(f"Early stopping triggered (patience: {early_stop_patience})")
    print(f"\nResults saved to: {output_dir}")
    print(f"  - best_model.ckpt (best validation loss - Lightning format)")
    print(f"  - final_model.ckpt (last epoch - Lightning format)")
    print(f"  - history.json (training metrics)")
    print(f"  - training_curves.png (visualization)")
    print(f"  - config.yaml (hyperparameters)")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune IMU2CLIP on custom dataset\n"
                    "Uses YAML config file with optional command-line overrides",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "config.yaml"),
        help="Path to YAML config file"
    )
    
    # Override options (optional - will override YAML config)
    parser.add_argument("--train_dir", type=str, help="Training data directory")
    parser.add_argument("--val_dir", type=str, help="Validation data directory")
    parser.add_argument("--checkpoint", type=str, help="Pre-trained model checkpoint")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--device", type=str, help="Device: cuda or cpu")
    
    args = parser.parse_args()
    
    # Load config from YAML
    config = load_config(args.config)
    
    # Override with command-line arguments if provided
    if args.train_dir:
        config['data']['train_dir'] = args.train_dir
    if args.val_dir:
        config['data']['val_dir'] = args.val_dir
    if args.checkpoint:
        config['model']['checkpoint'] = args.checkpoint
    if args.output_dir:
        config['output']['dir'] = args.output_dir
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    if args.device:
        config['device'] = args.device
    
    # Start fine-tuning
    finetune(config)


if __name__ == "__main__":
    main()


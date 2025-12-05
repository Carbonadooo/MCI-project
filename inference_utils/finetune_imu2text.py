#!/usr/bin/env python3
"""
Fine-tune IMU2CLIP on custom dataset with IMU↔Text alignment
Only fine-tunes the output projection layer
"""
import os
import glob
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import argparse
import json
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

from inference_imu2clip import IMU2CLIPInference
from inference_hdf5 import load_hdf5_imu, resample_imu
from lib.imu_models import MW2StackRNNPooling
from lib.clip_model import ClipPLModel
from lib.train_modules import MultimodalContrastiveLearningModule
import random


# Text alternatives for data augmentation (10 alternatives per action)
TEXT_ALTERNATIVES = {
    'brush_teeth': [
        'brush teeth', 'brushing my teeth', 'cleaning teeth with toothbrush',
        'doing tooth brushing', 'brushing dental surface', 'tooth-cleaning motion',
        'hygiene brushing routine', 'scrubbing teeth', 'brushing oral area',
        'morning brushing motion', 'performing tooth hygiene'
    ],
    'clap_once': [
        'clap once', 'single clap', 'clap one time', 'quick clap',
        'one-hand clap motion', 'short single clapping', 'do one clap',
        'clap briefly', 'a single hand clap', 'quick single applaud', 'fast one-time clap'
    ],
    'cut_vegetables_with_knife': [
        'cut vegetables with knife', 'chopping vegetables', 'slicing veggies with a knife',
        'cutting vegetables on board', 'veggie chopping motion', 'knife cutting vegetables',
        'preparing vegetables by cutting', 'slicing produce', 'chopping food items',
        'dicing vegetables', 'performing knife-cutting on vegetables'
    ],
    'drink_water': [
        'drink water', 'taking a sip of water', 'drinking from a cup', 'sipping water',
        'taking a drink', 'consuming water', 'hydrating with water', 'gulping water',
        'lifting cup to drink', 'drinking from bottle', 'taking a water gulp'
    ],
    'flick_switch_up_down': [
        'flick switch up down', 'flipping a switch', 'toggling switch up and down',
        'switching on/off', 'flicking the light switch', 'toggling the lever',
        'quick switch flick', 'flipping electrical switch', 'moving switch upward/downward',
        'performing switch toggle motion', 'pressing switch up and down'
    ],
    'fold_clothes': [
        'fold clothes', 'folding laundry', 'folding a shirt/pants', 'doing clothes folding',
        'tidying clothes by folding', 'garment folding motion', 'folding fabric items',
        'organizing clothes', 'folding garments neatly', 'performing laundry folding',
        'folding clothing items'
    ],
    'high_five_motion': [
        'high five motion', 'giving a high-five', 'raising hand for high-five',
        'high-five gesture', 'performing high-five action', 'slapping palms together',
        'friendly high-five', 'reaching out for high-five', 'celebratory high-five motion',
        'hand-to-hand high-five', 'quick high-five tap'
    ],
    'mimimi': [
        'mimimi', 'miming "mi-mi-mi"', 'making "mimimi" sound', 'repetitive "mi" vocal motion',
        'lip-moving mi-mi-mi gesture', 'doing mimimi mouth shape', 'small vocalization motion',
        'repeating syllable "mi"', 'low-intensity vocal gesture', 'rhythmic "mi mi mi" motion',
        'mimicking "mimi" sound pattern'
    ],
    'open_close_drawer': [
        'open close drawer', 'opening and shutting a drawer', 'pulling drawer out and pushing in',
        'drawer open-close motion', 'sliding drawer in/out', 'opening drawer then closing it',
        'drawer sliding motion', 'drawer pull-and-push action', 'accessing a drawer briefly',
        'toggling drawer open/closed', 'moving drawer in and out'
    ],
    'open_close_notebook': [
        'open close notebook', 'opening and shutting a notebook', 'flipping notebook open and closed',
        'notebook open-close gesture', 'open book then close it', 'flipping notebook cover',
        'notebook cover motion', 'opening notebook then closing', 'toggling notebook open/shut',
        'flipping notebook pages open/closed', 'performing notebook open-close action'
    ],
    'open_door_with_handle': [
        'open door with handle', 'opening a door', 'pulling door handle', 'door opening motion',
        'turning handle and opening', 'accessing through door', 'handle turning motion',
        'opening entry door', 'pulling door open', 'door handle operation', 'entering through door'
    ],
    'open_refrigerator': [
        'open refrigerator', 'opening fridge door', 'pulling refrigerator open',
        'accessing refrigerator', 'fridge door opening', 'opening the fridge',
        'pulling fridge handle', 'refrigerator access motion', 'opening cold storage',
        'fridge opening gesture', 'accessing food storage'
    ],
    'pick_and_place': [
        'pick and place', 'picking up and putting down', 'grabbing and placing object',
        'lift and place motion', 'object transfer action', 'pick up then set down',
        'grabbing and releasing object', 'moving object from A to B', 'object relocation',
        'lifting and positioning item', 'transferring object motion'
    ],
    'pour_water_into_cup': [
        'pour water into cup', 'pouring water', 'filling cup with water',
        'water pouring motion', 'pouring liquid into container', 'filling a cup',
        'transferring water to cup', 'pouring beverage', 'cup filling action',
        'liquid pouring gesture', 'water transfer motion'
    ],
    'punch_forward': [
        'punch forward', 'throwing a punch', 'forward punching motion', 'straight punch',
        'punching ahead', 'forward strike motion', 'extending fist forward',
        'performing forward punch', 'straight arm punch', 'forward jabbing motion',
        'front punch action'
    ],
    'screw_bottle_cap': [
        'screw bottle cap', 'twisting bottle cap', 'screwing cap on bottle',
        'tightening bottle cap', 'rotating cap motion', 'closing bottle with cap',
        'bottle cap screwing motion', 'twisting cap closed', 'cap tightening action',
        'screwing lid onto bottle', 'bottle sealing motion'
    ],
    'shake_bottle': [
        'shake bottle', 'shaking a bottle', 'bottle shaking motion', 'agitating bottle',
        'vigorous bottle shake', 'shaking container', 'bottle mixing motion',
        'rapid bottle movement', 'shaking liquid container', 'bottle agitation',
        'performing bottle shake'
    ],
    'squeeze_hand_sanitizer': [
        'squeeze hand sanitizer', 'dispensing hand sanitizer', 'pumping sanitizer',
        'squeezing sanitizer bottle', 'applying hand sanitizer', 'sanitizer dispensing motion',
        'pressing sanitizer pump', 'getting hand sanitizer', 'squeezing hygiene gel',
        'dispensing cleaning gel', 'sanitizer application motion'
    ],
    'stir_with_spoon': [
        'stir with spoon', 'stirring with a spoon', 'mixing with spoon', 'spoon stirring motion',
        'circular stirring action', 'mixing contents with spoon', 'rotating spoon in liquid',
        'stirring mixture', 'performing stirring motion', 'spoon mixing action',
        'circular spoon movement'
    ],
    'throw_small_object': [
        'throw small object', 'tossing a small item', 'throwing object forward',
        'small object toss', 'lobbing small item', 'throwing something small',
        'object throwing motion', 'tossing item away', 'small throw action',
        'underhand object toss', 'discarding by throwing'
    ],
    'twist_towel': [
        'twist towel', 'twisting a towel', 'wringing towel', 'towel twisting motion',
        'rotating towel fabric', 'twisting cloth', 'wringing out towel',
        'towel wringing action', 'rotating towel ends', 'performing towel twist',
        'squeezing towel by twisting'
    ],
    'unplug_usb_cable': [
        'unplug usb cable', 'removing USB cable', 'unplugging USB', 'pulling out USB cable',
        'disconnecting USB', 'USB cable removal', 'extracting USB connector',
        'unplugging cable', 'removing USB connection', 'pulling USB out',
        'USB disconnection motion'
    ],
    'use_hammer': [
        'use hammer', 'hammering motion', 'using a hammer', 'striking with hammer',
        'hammer hitting motion', 'performing hammer action', 'hammering downward',
        'tool hammering motion', 'hammer strike action', 'pounding with hammer',
        'downward hammer motion'
    ],
    'wave_hand_left_right': [
        'wave hand left right', 'waving hand', 'hand waving motion', 'left-right hand wave',
        'greeting wave gesture', 'waving hello', 'horizontal hand wave',
        'side-to-side hand motion', 'friendly wave gesture', 'waving hand back and forth',
        'hand oscillation motion'
    ],
    'wipe_table_back_forth': [
        'wipe table back forth', 'wiping table surface', 'cleaning table', 'back and forth wiping',
        'table cleaning motion', 'surface wiping action', 'wiping back and forth',
        'table surface cleaning', 'horizontal wiping motion', 'cleaning table top',
        'back-forth cleaning gesture'
    ]
}


def augment_imu_data(imu_data, training=True):
    """
    Apply data augmentation to IMU data
    
    Args:
        imu_data: numpy array (6, N)
        training: if True, apply augmentation; if False, return as-is
        
    Returns:
        augmented IMU data (6, N)
    """
    if not training:
        return imu_data
    
    imu_aug = imu_data.copy()
    
    # 1. Random scaling (0.7 to 1.3)
    scale = random.uniform(0.7, 1.3)
    imu_aug = imu_aug * scale
    
    # 2. Add small Gaussian noise (relative to signal magnitude)
    # Noise std is 1-3% of the signal's standard deviation
    signal_std = np.std(imu_aug, axis=1, keepdims=True)
    noise_factor = random.uniform(0.001, 0.003)  # 1-3% noise
    noise = np.random.randn(*imu_aug.shape) * signal_std * noise_factor
    imu_aug = imu_aug + noise
    
    # 3. Random start zero-padding (0-10% of data)
    n_samples = imu_aug.shape[1]
    start_zero_pct = random.uniform(0, 0.10)
    start_zero_len = int(n_samples * start_zero_pct)
    if start_zero_len > 0:
        imu_aug[:, :start_zero_len] = 0.0
    
    # 4. Random end zero-padding (0-10% of data)
    end_zero_pct = random.uniform(0, 0.10)
    end_zero_len = int(n_samples * end_zero_pct)
    if end_zero_len > 0:
        imu_aug[:, -end_zero_len:] = 0.0
    
    return imu_aug


class IMUTextDataset(Dataset):
    """Dataset for IMU-Text pairs from folder structure"""
    
    def __init__(self, data_dir=None, max_per_class=None, window_size=1000, augmentation_multiplier=50, training=True, file_list=None):
        """
        Args:
            data_dir: Root directory with class folders (if file_list is None)
            max_per_class: Limit samples per class
            window_size: Fixed window size (1000 = 5 seconds at 200 Hz)
            augmentation_multiplier: Number of augmented versions per original sample
            training: If True, apply augmentation; if False, use original data only
            file_list: Optional list of (file_path, class_name) tuples to use instead of scanning data_dir
        """
        self.window_size = window_size
        self.augmentation_multiplier = augmentation_multiplier if training else 1
        self.training = training
        self.samples = []
        
        # Use provided file list or scan directories
        if file_list is not None:
            # Use pre-split file list
            for file_path, class_name in file_list:
                self.samples.append({
                    'file': file_path,
                    'class_name': class_name,
                    'text': class_name.replace('_', ' ')
                })
        else:
            # Scan directories (legacy behavior)
            class_dirs = sorted([d for d in Path(data_dir).iterdir() if d.is_dir()])
            
            print(f"Loading dataset from {data_dir}")
            print(f"Found {len(class_dirs)} classes")
            
            for class_dir in class_dirs:
                class_name = class_dir.name
                hdf5_files = sorted(glob.glob(str(class_dir / "*.hdf5")))
                
                if max_per_class:
                    hdf5_files = hdf5_files[:max_per_class]
                
                print(f"  {class_name}: {len(hdf5_files)} samples")
                
                for hdf5_file in hdf5_files:
                    self.samples.append({
                        'file': hdf5_file,
                        'class_name': class_name,
                        'text': class_name.replace('_', ' ')  # "clap_once" -> "clap once"
                    })
        
        print(f"Original samples: {len(self.samples)}")
        
        # Expand dataset with augmentation multiplier
        # Training: Each original sample will have N augmented versions
        # Validation: No expansion (augmentation_multiplier=1)
        original_samples = self.samples.copy()
        self.samples = []
        for aug_idx in range(self.augmentation_multiplier):
            for sample in original_samples:
                augmented_sample = sample.copy()
                augmented_sample['aug_idx'] = aug_idx  # Track which augmentation this is
                self.samples.append(augmented_sample)
        
        if self.training:
            print(f"After {self.augmentation_multiplier}x augmentation: {len(self.samples)} total samples")
        else:
            print(f"Validation mode (no augmentation): {len(self.samples)} total samples")
        
        # Get unique class names
        self.class_names = sorted(list(set([s['class_name'] for s in original_samples])))
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        # Use first alternative as canonical text label for validation
        self.text_labels = []
        for name in self.class_names:
            if name in TEXT_ALTERNATIVES:
                self.text_labels.append(TEXT_ALTERNATIVES[name][0])  # First is canonical
            else:
                self.text_labels.append(name.replace('_', ' '))
        
        print(f"Unique classes: {len(self.class_names)}")
        print(f"Text augmentation: {sum(1 for name in self.class_names if name in TEXT_ALTERNATIVES)} classes with alternatives")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load IMU data
        try:
            data = load_hdf5_imu(sample['file'], verbose=False, resample_to_200hz=True)
            imu_data = data['imu_data']  # Shape: (6, N)
            
            # Extract random window of fixed size
            n_samples = imu_data.shape[1]
            
            if n_samples >= self.window_size:
                # Random start position
                start_idx = np.random.randint(0, n_samples - self.window_size + 1)
                window = imu_data[:, start_idx:start_idx + self.window_size]
            else:
                # Pad if too short
                padding = self.window_size - n_samples
                window = np.pad(imu_data, ((0, 0), (0, padding)), mode='constant')
            
            # Apply IMU data augmentation (only if training)
            window = augment_imu_data(window, training=self.training)
            
            # Convert to tensor
            imu_tensor = torch.from_numpy(window).float()
            
            # Get class index
            class_idx = self.class_to_idx[sample['class_name']]
            
            # Get text with augmentation (only during training)
            class_name = sample['class_name']
            if self.training and class_name in TEXT_ALTERNATIVES:
                # Training: randomly pick one alternative
                text = random.choice(TEXT_ALTERNATIVES[class_name])
            elif class_name in TEXT_ALTERNATIVES:
                # Validation: use canonical (first) alternative
                text = TEXT_ALTERNATIVES[class_name][0]
            else:
                # Fallback to default (replace underscores with spaces)
                text = sample['text']
            
            return {
                'imu': imu_tensor,
                'class_idx': class_idx,
                'text': text
            }
            
        except Exception as e:
            print(f"Error loading {sample['file']}: {e}")
            # Return a dummy sample
            return {
                'imu': torch.zeros(6, self.window_size),
                'class_idx': 0,
                'text': 'error'
            }


def load_and_split_data(data_dir, max_per_class=None, train_split=0.9, random_seed=42):
    """
    Load all HDF5 files and split into train/val sets (stratified by class)
    
    Args:
        data_dir: Root directory with class folders
        max_per_class: Limit samples per class
        train_split: Fraction of data for training (e.g., 0.8 = 80% train, 20% val)
        random_seed: Random seed for reproducibility
        
    Returns:
        train_files: List of (file_path, class_name) tuples for training
        val_files: List of (file_path, class_name) tuples for validation
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    class_dirs = sorted([d for d in Path(data_dir).iterdir() if d.is_dir()])
    
    print(f"Scanning dataset from {data_dir}")
    print(f"Found {len(class_dirs)} classes")
    print(f"Train/Val split: {train_split:.0%} / {1-train_split:.0%}")
    print()
    
    train_files = []
    val_files = []
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        hdf5_files = sorted(glob.glob(str(class_dir / "*.hdf5")))
        
        if max_per_class:
            hdf5_files = hdf5_files[:max_per_class]
        
        # Shuffle for random split
        random.shuffle(hdf5_files)
        
        # Split into train/val
        n_train = max(1, int(len(hdf5_files) * train_split))
        class_train = hdf5_files[:n_train]
        class_val = hdf5_files[n_train:]
        
        # Add to lists
        for f in class_train:
            train_files.append((f, class_name))
        for f in class_val:
            val_files.append((f, class_name))
        
        print(f"  {class_name}: {len(hdf5_files)} total → {len(class_train)} train, {len(class_val)} val")
    
    print()
    print(f"Total: {len(train_files)} train files, {len(val_files)} val files")
    print()
    
    return train_files, val_files


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
    # Normalize embeddings
    imu_emb = imu_emb / imu_emb.norm(dim=-1, keepdim=True)
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    
    # Compute similarity matrix
    logits = torch.matmul(imu_emb, text_emb.t()) / temperature  # (B, B)
    
    # Labels: diagonal elements are positive pairs
    labels = torch.arange(len(imu_emb), device=imu_emb.device)
    
    # IMU-to-Text loss (Li2t)
    loss_i2t = nn.functional.cross_entropy(logits, labels)
    
    # Text-to-IMU loss (Lt2i)
    loss_t2i = nn.functional.cross_entropy(logits.t(), labels)
    
    # Symmetric loss
    loss = (loss_i2t + loss_t2i) / 2.0
    
    return loss, loss_i2t, loss_t2i


def finetune(args):
    """Main fine-tuning function"""
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"finetune_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("IMU2CLIP Fine-tuning")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print()
    
    # Save config
    config = vars(args)
    config['timestamp'] = timestamp
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Device
    device = torch.device(args.device)
    print(f"Device: {device}")
    print()
    
    # Split data into train/val sets
    train_files, val_files = load_and_split_data(
        args.data_dir,
        max_per_class=args.max_per_class,
        train_split=args.train_split,
        random_seed=42
    )
    
    # Load training dataset (with augmentation)
    print("Loading training dataset...")
    train_dataset = IMUTextDataset(
        window_size=args.window_size,
        augmentation_multiplier=args.augmentation_multiplier,
        training=True,  # Enable augmentation
        file_list=train_files
    )
    
    # Load validation dataset (without augmentation)
    print("\nLoading validation dataset...")
    val_dataset = IMUTextDataset(
        window_size=args.window_size,
        augmentation_multiplier=1,  # No augmentation multiplier
        training=False,  # Disable augmentation
        file_list=val_files
    )
    
    # Get the base dataset for text labels (use training dataset's class info)
    dataset = train_dataset
    
    print(f"Train: {len(train_dataset)} samples (after {args.augmentation_multiplier}x augmentation)")
    print(f"Val:   {len(val_dataset)} samples (original data)")
    print()
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Load pre-trained model
    print("Loading pre-trained model...")
    imu_encoder = MW2StackRNNPooling(size_embeddings=512)
    text_encoder = ClipPLModel(freeze=True)
    
    modality_to_encoder = {
        "imu": imu_encoder,
        "text": text_encoder
    }
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Load from LightningModule checkpoint
    if 'state_dict' in checkpoint:
        # Extract imu_encoder weights
        imu_state_dict = {}
        for key, value in checkpoint['state_dict'].items():
            if key.startswith('imu_encoder.'):
                new_key = key.replace('imu_encoder.', '')
                imu_state_dict[new_key] = value
        imu_encoder.load_state_dict(imu_state_dict)
        print("✓ Loaded IMU encoder from checkpoint")
    else:
        # Direct state dict
        imu_encoder.load_state_dict(checkpoint)
        print("✓ Loaded IMU encoder")
    
    imu_encoder = imu_encoder.to(device)
    text_encoder = text_encoder.to(device)

    for name, param in imu_encoder.named_parameters():
        param.requires_grad = False
    
    # Freeze all layers except the final projection
    print("\nFreezing layers...")
    print("Available parameters:")
    all_params = list(imu_encoder.named_parameters())
    for name, param in all_params:
        print(f"  - {name}: {param.shape}")
    
    print("\nSetting trainable parameters...")
    # Find the last few layers to fine-tune
    # Strategy: unfreeze last 2 layers or layers with 'linear' or 'fc' or 'out' in final position
    # trainable_keywords = ['linear', 'fc', 'out', 'pool.linear', 'embedding']
    trainable_layers = ['linear', 'fc', 'out', "net.5.weight_hh_l0"]
    
    for name, param in imu_encoder.named_parameters():
        # Make trainable if it's in the last layer or matches keywords
        if any(layer in name.lower() for layer in trainable_layers):
            param.requires_grad = True
            print(f"  ✓ {name} - Trainable ({param.numel():,} params)")
        else:
            param.requires_grad = False
            print(f"  ✓ {name} - Frozen ({param.numel():,} params)")
    print()
    # If still no trainable params, just unfreeze the last RNN layer
    trainable_count = sum(p.requires_grad for p in imu_encoder.parameters())
    if trainable_count == 0:
        print("\n⚠ No parameters matched keywords. Unfreezing last RNN layer...")
        # Unfreeze last 2 parameters (RNN weights and biases)
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
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, imu_encoder.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs * len(train_loader)
    )
    
    # Pre-encode all text labels
    print("Pre-encoding text labels...")
    with torch.no_grad():
        # Encode all text labels at once
        all_text_embeddings = text_encoder.get_text_embeddings(dataset.text_labels, device=device)
    print(f"✓ Encoded {len(all_text_embeddings)} text labels: {all_text_embeddings.shape}")
    print()
    
    # Training loop
    print("=" * 70)
    print("Starting training...")
    if args.early_stop_patience > 0:
        print(f"Early stopping enabled (patience: {args.early_stop_patience} epochs)")
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
    
    for epoch in range(args.epochs):
        # Training
        imu_encoder.train()
        train_losses = []
        train_losses_i2t = []
        train_losses_t2i = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            imu_data = batch['imu'].to(device)
            class_indices = batch['class_idx'].to(device)
            
            # Get text embeddings for this batch
            text_emb = all_text_embeddings[class_indices]
            
            # Forward pass
            imu_emb = imu_encoder(imu_data)
            
            # Compute loss
            loss, loss_i2t, loss_t2i = symmetric_contrastive_loss(
                imu_emb, text_emb, temperature=args.temperature
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
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
                    imu_emb, text_emb, temperature=args.temperature
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
        val_acc = correct / total
        
        history['train_loss'].append(train_loss)
        history['train_loss_i2t'].append(np.mean(train_losses_i2t))
        history['train_loss_t2i'].append(np.mean(train_losses_t2i))
        history['val_loss'].append(val_loss)
        history['val_loss_i2t'].append(np.mean(val_losses_i2t))
        history['val_loss_t2i'].append(np.mean(val_losses_t2i))
        history['val_accuracy'].append(val_acc)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f} (i2t: {history['train_loss_i2t'][-1]:.4f}, t2i: {history['train_loss_t2i'][-1]:.4f})")
        print(f"  Val Loss:   {val_loss:.4f} (i2t: {history['val_loss_i2t'][-1]:.4f}, t2i: {history['val_loss_t2i'][-1]:.4f})")
        print(f"  Val Acc:    {val_acc:.2%}")
        print(f"  LR:         {optimizer.param_groups[0]['lr']:.2e}")
        print()
        
        # Save best model (in Lightning checkpoint format)
        if val_loss < best_val_loss:
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
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"  ⚠ No improvement for {epochs_without_improvement} epoch(s)")
            
            # Early stopping
            if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
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
        'epoch': args.epochs,
        'best_val_loss': best_val_loss,
        'final_val_accuracy': history['val_accuracy'][-1],
        'hyper_parameters': config
    }
    torch.save(final_checkpoint_dict, output_dir / "final_model.ckpt")
    
    # Save history
    with open(output_dir / "history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
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
    if len(history['train_loss']) < args.epochs:
        print(f"Early stopping triggered (patience: {args.early_stop_patience})")
    print(f"\nResults saved to: {output_dir}")
    print(f"  - best_model.ckpt (best validation loss - Lightning format)")
    print(f"  - final_model.ckpt (last epoch - Lightning format)")
    print(f"  - history.json (training metrics)")
    print(f"  - training_curves.png (visualization)")
    print(f"  - config.json (hyperparameters)")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune IMU2CLIP on custom dataset")
    
    # Data
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/chuye/Documents/MCI-project/data/nov2_set",
        help="Data directory with class folders"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/chuye/Documents/imu2clip/saved/i2c/i2c_s_i_t_t_se_mw2_w_2.5_master-epoch=01-val_loss=5.87.ckpt",
        help="Pre-trained model checkpoint"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./finetuned_models",
        help="Output directory for fine-tuned model"
    )
    
    # Training
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="Weight decay")
    parser.add_argument("--temperature", type=float, default=0.07, help="Temperature for contrastive loss")
    parser.add_argument("--early_stop_patience", type=int, default=20, help="Early stopping patience (0 to disable)")
    
    # Data
    parser.add_argument("--window_size", type=int, default=1000, help="IMU window size (samples)")
    parser.add_argument("--max_per_class", type=int, default=None, help="Max samples per class")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--augmentation_multiplier", type=int, default=50, help="Augmentation multiplier (dataset expansion factor)")
    parser.add_argument("--train_split", type=float, default=0.8, help="Fraction of data for training (e.g., 0.8 = 80%% train, 20%% val)")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    finetune(args)


if __name__ == "__main__":
    main()


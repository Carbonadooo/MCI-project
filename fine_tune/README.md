# Fine-tuning IMU2CLIP

Modular fine-tuning pipeline for IMU2CLIP with separate train/validation datasets.

## 📁 Structure

```
fine_tune/
├── finetune.py              # Main fine-tuning script
├── config.yaml              # Configuration parameters
├── dataset.py               # IMU-Text dataset loader
├── text_alternatives.py     # Text augmentation data
├── __init__.py              # Package init
└── README.md                # This file
```

---

## 🚀 Quick Start

### 1. Prepare Your Data

**Important**: Use **separate folders** for training and validation data:

```
/path/to/data/
├── train/                    # Training data
│   ├── class1/
│   │   ├── sample1.hdf5
│   │   ├── sample2.hdf5
│   │   └── ...
│   ├── class2/
│   └── ...
└── val/                      # Validation data (separate!)
    ├── class1/
    │   ├── sample1.hdf5
    │   └── ...
    ├── class2/
    └── ...
```

### 2. Configure Training

Edit `config.yaml`:

```yaml
data:
  train_dir: "/path/to/train"  # Your training folder
  val_dir: "/path/to/val"      # Your validation folder (different!)
```

### 3. Run Fine-tuning

```bash
# Basic usage (uses config.yaml)
python fine_tune/finetune.py

# Override specific parameters
python fine_tune/finetune.py --epochs 50 --batch_size 64

# Use different config file
python fine_tune/finetune.py --config my_config.yaml
```

---

## ⚙️ Configuration

### YAML Configuration

All parameters are in `config.yaml`:

```yaml
# Data paths (SEPARATE folders!)
data:
  train_dir: "/path/to/train"
  val_dir: "/path/to/val"
  window_size: 1000  # IMU window size (samples)
  max_per_class: null  # Limit samples (null = all)
  num_workers: 8

# Model
model:
  checkpoint: "/path/to/pretrained.ckpt"
  embedding_size: 512

# Training
training:
  epochs: 100
  batch_size: 32
  learning_rate: 5.0e-6
  weight_decay: 0.001
  temperature: 0.07
  early_stop_patience: 20

# Augmentation
augmentation:
  multiplier: 50  # Dataset expansion factor
  imu:
    scale_min: 0.7
    scale_max: 1.3
    noise_min: 0.001
    noise_max: 0.003
    zero_pad_max: 0.10

# Output
output:
  dir: "./finetuned_models"

# Device
device: "cuda"
```

### Command-Line Overrides

Override any parameter:

```bash
python fine_tune/finetune.py \
    --train_dir /custom/train \
    --val_dir /custom/val \
    --epochs 50 \
    --batch_size 64 \
    --learning_rate 1e-5 \
    --device cuda
```

---

## 📊 Data Augmentation

### IMU Augmentation (4 techniques)

1. **Random Scaling**: 0.7x - 1.3x
2. **Gaussian Noise**: 0.1-0.3% of signal std
3. **Start Zero-Padding**: 0-10% at beginning
4. **End Zero-Padding**: 0-10% at end

Configure in `config.yaml`:

```yaml
augmentation:
  imu:
    scale_min: 0.7     # Adjust scaling range
    scale_max: 1.3
    noise_min: 0.001   # Adjust noise level
    noise_max: 0.003
    zero_pad_max: 0.10 # Adjust padding
```

### Text Augmentation

Each class has 11 text alternatives in `text_alternatives.py`.

Example:
```python
'clap_once': [
    'clap once',           # Canonical
    'single clap',         # Alternative 1
    'clap one time',       # Alternative 2
    ...                    # 8 more alternatives
]
```

Training randomly selects one alternative per sample.

### Augmentation Multiplier

Expand dataset by N times:

```yaml
augmentation:
  multiplier: 50  # Each sample → 50 augmented versions
```

- 100 original samples × 50 = **5,000 training samples**
- Validation: **no augmentation** (clean evaluation)

---

## 📂 Module Files

### `finetune.py`

Main fine-tuning script:
- Loads config from YAML
- Creates separate train/val datasets
- Trains IMU encoder with contrastive loss
- Saves checkpoints and training curves

### `config.yaml`

Configuration parameters:
- Data paths (train/val folders)
- Training hyperparameters
- Augmentation settings
- Output directory

### `dataset.py`

Dataset loader:
- `IMUTextDataset`: Loads HDF5 files
- `augment_imu_data()`: IMU augmentation function
- Handles text augmentation
- Supports training/validation modes

### `text_alternatives.py`

Text augmentation data:
- `TEXT_ALTERNATIVES`: Dictionary of class → text list
- 25 action classes
- 11 alternatives per class
- 275 total text descriptions

---

## 🎯 Usage Examples

### Example 1: Basic Training

```bash
# Edit config.yaml first, then:
python fine_tune/finetune.py
```

### Example 2: Quick Test

```bash
python fine_tune/finetune.py \
    --epochs 10 \
    --batch_size 16 \
    --device cpu
```

### Example 3: Custom Data

```bash
python fine_tune/finetune.py \
    --train_dir /custom/train \
    --val_dir /custom/val \
    --output_dir ./my_finetuned_models
```

### Example 4: High Augmentation

Edit `config.yaml`:
```yaml
augmentation:
  multiplier: 100  # More augmentation
```

Then run:
```bash
python fine_tune/finetune.py --epochs 200
```

### Example 5: Custom Config

```bash
# Create custom config
cp config.yaml my_config.yaml
# Edit my_config.yaml...

# Use it
python fine_tune/finetune.py --config my_config.yaml
```

---

## 📈 Output

Training creates a timestamped folder:

```
finetuned_models/finetune_20241203_123456/
├── best_model.ckpt          # Best validation checkpoint
├── final_model.ckpt         # Final epoch checkpoint
├── config.yaml              # Configuration used
├── history.json             # Training metrics
└── training_curves.png      # Loss/accuracy plots
```

### Checkpoints

Both `.ckpt` files are PyTorch Lightning format:

```python
{
    'state_dict': {
        'imu_encoder.net.0.weight': ...,
        'imu_encoder.net.0.bias': ...,
        ...
    },
    'epoch': 42,
    'best_val_loss': 2.34,
    'hyper_parameters': {...}
}
```

Compatible with:
- `IMU2CLIPInference` (inference)
- `retrive.py` (retrieval evaluation)
- `knn_classifier.py` (classification)

---

## 🔧 Adding New Classes

### 1. Add Text Alternatives

Edit `text_alternatives.py`:

```python
TEXT_ALTERNATIVES = {
    ...
    'new_action': [
        'new action',              # Canonical
        'performing new action',   # Alternative 1
        'doing new action',        # Alternative 2
        ...                        # 8 more alternatives (11 total)
    ],
}
```

### 2. Add Training Data

Create folder structure:

```
train/
└── new_action/
    ├── sample1.hdf5
    ├── sample2.hdf5
    └── ...

val/
└── new_action/
    ├── sample1.hdf5
    └── ...
```

### 3. Run Fine-tuning

```bash
python fine_tune/finetune.py
```

The new class will be automatically detected!

---

## 🐛 Troubleshooting

### Issue: "No samples found"

**Cause**: Empty or missing data folders

**Solution**: Check `train_dir` and `val_dir` paths in `config.yaml`

### Issue: "Classes don't match"

**Cause**: Train and val folders have different classes

**Solution**: Ensure both folders have the same class names

### Issue: Low validation accuracy

**Causes**:
1. Too few validation samples
2. Train/val distribution mismatch
3. Underfitting

**Solutions**:
1. Add more validation data
2. Check train/val folders have similar distributions
3. Train longer or increase model capacity

### Issue: OOM (Out of Memory)

**Solution**: Reduce batch size in `config.yaml`:

```yaml
training:
  batch_size: 16  # Reduce from 32
```

---

## 📊 Monitoring Training

### Training Curves

`training_curves.png` shows:
- Total loss (train/val)
- IMU-to-Text loss
- Text-to-IMU loss
- Validation accuracy
- Learning rate schedule

### Early Stopping

Training stops automatically when validation loss plateaus:

```yaml
training:
  early_stop_patience: 20  # Stop after 20 epochs without improvement
```

### Metrics

`history.json` contains:
```json
{
  "train_loss": [3.5, 3.2, 2.9, ...],
  "val_loss": [3.0, 2.8, 2.7, ...],
  "val_accuracy": [0.15, 0.23, 0.31, ...],
  ...
}
```

---

## 🎯 Best Practices

### 1. Data Split

✅ **Do**:
- Use separate physical folders for train/val
- 80-90% train, 10-20% val
- Ensure both have all classes

❌ **Don't**:
- Mix train and val in same folder
- Use automatic splitting (user already split on disk)

### 2. Augmentation

✅ **Do**:
- Use moderate multiplier (50-100x)
- Adjust augmentation strength in config
- Verify augmentation doesn't distort motion

❌ **Don't**:
- Over-augment (>200x usually too much)
- Use same augmentation for val (it's disabled)

### 3. Training

✅ **Do**:
- Start with config defaults
- Enable early stopping
- Monitor validation metrics

❌ **Don't**:
- Train too long without early stopping
- Ignore validation accuracy trends
- Use learning rate >1e-4 (too aggressive)

---

## 🚀 Quick Reference

```bash
# Basic training
python fine_tune/finetune.py

# Custom data paths
python fine_tune/finetune.py \
    --train_dir /path/to/train \
    --val_dir /path/to/val

# Quick test (small batch, few epochs)
python fine_tune/finetune.py \
    --epochs 5 \
    --batch_size 8 \
    --device cpu

# Full training (recommended)
python fine_tune/finetune.py \
    --epochs 100 \
    --batch_size 32 \
    --device cuda
```

---

## 📝 Summary

| Feature | Description |
|---------|-------------|
| **Config** | YAML-based with command-line overrides |
| **Data** | Separate train/val folders (no auto-split) |
| **Augmentation** | 4 IMU + 11 text alternatives per class |
| **Modular** | Separate files for dataset, text, config |
| **Output** | Lightning-format `.ckpt` files |
| **Monitoring** | Curves, metrics, early stopping |

---

**For more details, see the main IMU2CLIP documentation.**


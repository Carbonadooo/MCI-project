# Using 9-Channel IMU Data with IMU2CLIP

This guide explains how to use all 9 IMU channels (accelerometer + gyroscope + magnetometer) with IMU2CLIP.

## Overview

The original IMU2CLIP model uses 6 IMU channels:
- **Channels 0-2**: Accelerometer (x, y, z)
- **Channels 3-5**: Gyroscope (x, y, z)

The new 9-channel model extends this to include:
- **Channels 0-2**: Accelerometer (x, y, z)
- **Channels 3-5**: Gyroscope (x, y, z)
- **Channels 6-8**: Magnetometer (x, y, z)

## Architecture: MW2StackRNNPooling9Ch

The 9-channel model adds a **learnable channel projection layer** at the beginning:

```
Input: (batch, 9, time_steps)
   ↓
Channel Projection: Conv1d(9 → 6)  [NEW LAYER]
   ↓
GroupNorm + Conv Blocks (same as original)
   ↓
GRU Layer
   ↓
Output: (batch, 512) CLIP embedding
```

### Key Features

1. **Learnable 9→6 projection**: The model learns how to best combine all 9 channels into 6 effective channels
2. **Transfer learning friendly**: Can initialize from 6-channel pre-trained weights
3. **Only trains final layers**: By default, only the RNN layer and channel projection are trained during fine-tuning

## Configuration Files

### For 6-Channel IMU (Original)

**File**: `config.yaml` or `plain_finetune.yaml`

```yaml
data:
  num_channels: 6  # Use only accel + gyro

model:
  name: "MW2StackRNNPooling"  # Original 6-channel model
  checkpoint: "path/to/pretrained_6ch.ckpt"
```

### For 9-Channel IMU (New)

**File**: `finetune_9ch.yaml`

```yaml
data:
  num_channels: 9  # Use accel + gyro + magnetometer

model:
  name: "MW2StackRNNPooling9Ch"  # 9-channel model with projection
  checkpoint: "path/to/pretrained_6ch.ckpt"  # Can use 6-channel pretrained!
```

## Output Directory Structure

Fine-tuned models are organized by channel count:

```
log/
├── finetuned_models_6ch/          # 6-channel models
│   ├── finetune_6ch_20251203_120000/
│   │   ├── best_model.ckpt
│   │   ├── final_model.ckpt
│   │   ├── training_curves.png
│   │   └── history.json
│   └── finetune_6ch_20251203_130000/
│       └── ...
└── finetuned_models_9ch/          # 9-channel models
    ├── finetune_9ch_20251203_140000/
    │   ├── best_model.ckpt
    │   ├── final_model.ckpt
    │   ├── training_curves.png
    │   └── history.json
    └── finetune_9ch_20251203_150000/
        └── ...
```

This structure makes it easy to distinguish between 6-channel and 9-channel fine-tuned models.

## Usage

### 1. Fine-tuning with 9-Channel Data

```bash
# Use the 9-channel configuration
# Output will be saved to: log/finetuned_models_9ch/finetune_9ch_TIMESTAMP/
python fine_tune/finetune.py --config fine_tune/finetune_9ch.yaml

# Or specify parameters via command line
python fine_tune/finetune.py \
    --config fine_tune/config.yaml \
    --model_name MW2StackRNNPooling9Ch \
    --num_channels 9
```

### Fine-tuning with 6-Channel Data

```bash
# Output will be saved to: log/finetuned_models_6ch/finetune_6ch_TIMESTAMP/
python fine_tune/finetune.py --config fine_tune/config.yaml
```

### 2. Inference with 9-Channel Model

```python
from inference_imu2clip import IMU2CLIPInference
import numpy as np

# Load 9-channel fine-tuned model
model = IMU2CLIPInference(
    checkpoint_path="log/finetuned_models_9ch/finetune_9ch_20251203_140000/best_model.ckpt",
    model_name="MW2StackRNNPooling9Ch",
    num_channels=9,
    device="cuda"
)

# Prepare 9-channel IMU data (9, N)
imu_data = np.random.randn(9, 1000)  # 9 channels, 1000 samples

# Encode to CLIP space
embedding = model.encode(imu_data)  # Returns (512,) vector
```

### 3. Using with HDF5 Files

```python
from inference_hdf5 import load_hdf5_imu
from inference_imu2clip import IMU2CLIPInference

# Load 9-channel data
data = load_hdf5_imu(
    "path/to/data.hdf5",
    num_channels=9,  # Load all 9 channels
    resample_to_200hz=True
)

# Initialize 9-channel fine-tuned model
model = IMU2CLIPInference(
    checkpoint_path="log/finetuned_models_9ch/finetune_9ch_20251203_140000/best_model.ckpt",
    model_name="MW2StackRNNPooling9Ch",
    num_channels=9
)

# Encode
embedding = model.encode(data['imu_data'])
```

## Training from Pre-trained 6-Channel Weights

The 9-channel model can be initialized from 6-channel pre-trained weights:

1. **Main network weights** (GroupNorm, Conv blocks, GRU): Loaded from pre-trained checkpoint
2. **Channel projection layer**: Randomly initialized (since it's new)

During fine-tuning:
- Freezes all early layers (GroupNorm, Conv blocks)
- **Trains**: GRU hidden-to-hidden weights + channel projection
- This allows the model to learn how to best use the magnetometer data

## Data Format Requirements

### HDF5 File Structure

```
your_data.hdf5
├── imu/
│   ├── data        # Shape: (N, 9) or (9, N)
│   │               # Columns: [accel_x, accel_y, accel_z,
│   │               #           gyro_x, gyro_y, gyro_z,
│   │               #           mag_x, mag_y, mag_z]
│   └── timestamps  # Shape: (N,)
```

### Expected Order

For 9-channel data, the channel order must be:
1. Accelerometer X, Y, Z
2. Gyroscope X, Y, Z  
3. Magnetometer X, Y, Z

## Model Selection Logic

The fine-tuning script automatically selects the correct model based on:

```python
if config['model']['name'] == 'MW2StackRNNPooling9Ch':
    # Use 9-channel model
    imu_encoder = MW2StackRNNPooling9Ch(size_embeddings=512)
    num_channels = 9
elif config['model']['name'] == 'MW2StackRNNPooling':
    # Use 6-channel model
    imu_encoder = MW2StackRNNPooling(size_embeddings=512)
    num_channels = 6
```

## Benefits of 9-Channel Model

1. **More information**: Magnetometer provides absolute orientation
2. **Rotation invariance**: Better handling of device orientation
3. **Complementary signals**: Magnetometer complements accel/gyro
4. **Transfer learning**: Can initialize from 6-channel pre-trained weights

## Comparison

| Feature | 6-Channel | 9-Channel |
|---------|-----------|-----------|
| Input channels | 6 | 9 |
| Model parameters | ~850K | ~850K + 54 (projection) |
| Trainable (fine-tune) | ~786K | ~786K + 54 |
| Pre-training | ✓ Available | Use 6-ch pretrained |
| Best for | General motion | Orientation-sensitive tasks |

## Troubleshooting

### Error: "Expected at least 9 IMU channels, got 6"

**Solution**: Your HDF5 files only have 6 channels. Either:
1. Use the 6-channel model configuration
2. Add magnetometer data to your HDF5 files

### Warning: "Model expects 9 channels but config specifies 6"

**Solution**: Make sure `data.num_channels` matches `model.name`:
- `MW2StackRNNPooling` → `num_channels: 6`
- `MW2StackRNNPooling9Ch` → `num_channels: 9`

### Missing keys when loading checkpoint

This is normal! The channel_projection layer is new and won't be in 6-channel pre-trained weights. It will be randomly initialized and trained during fine-tuning.

## Examples

See the example configurations:
- `fine_tune/config.yaml` - 6-channel model
- `fine_tune/plain_finetune.yaml` - 6-channel model (alternative)
- `fine_tune/finetune_9ch.yaml` - **9-channel model** ← Use this for 9-channel data!



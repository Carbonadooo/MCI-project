# 18-Channel IMU Model: Raw IMU + Rotation Features

This document explains the 18-channel IMU model, which combines raw 9-channel IMU data with computed rotation matrix features.

## Overview

The 18-channel model enhances the standard 9-channel IMU input by adding **explicit rotation features** derived from the sensor fusion of accelerometer, gyroscope, and magnetometer data.

### Channel Layout

**Channels 0-8**: Raw 9-channel IMU data
- 0-2: Accelerometer (ax, ay, az)
- 3-5: Gyroscope (gx, gy, gz)
- 6-8: Magnetometer (mx, my, mz)

**Channels 9-17**: Flattened 3×3 rotation matrix
- 9-11: R[0,:] - First row of rotation matrix
- 12-14: R[1,:] - Second row of rotation matrix
- 15-17: R[2,:] - Third row of rotation matrix

## Why Rotation Features?

1. **Explicit Orientation**: Raw IMU data requires the model to learn orientation implicitly. Rotation features provide it explicitly.
2. **Better Generalization**: Helps the model understand absolute orientation, not just relative movements.
3. **Complementary Information**: Combines raw sensor readings (dynamic) with computed pose (global).
4. **Sensor Fusion**: Leverages all three sensor types (accel, gyro, mag) through rotation calculation.

## How Rotation Features Are Computed

The rotation matrix is computed using:

1. **Initial Orientation**: Derived from accelerometer (gravity) and magnetometer (magnetic north)
2. **Gyroscope Integration**: Track rotation changes over time
3. **Fast Approximation**: For short windows (5 seconds), uses a single representative rotation matrix

```python
from lib.rotation_utils import compute_rotation_features_fast

# Input: (9, N) IMU data
imu_9ch = np.array([...])  # Shape: (9, 1000)

# Compute rotation features
rotation_features = compute_rotation_features_fast(imu_9ch, sample_rate=200)
# Output: (9, 1000) - flattened 3x3 rotation matrix repeated for all time steps

# Concatenate to get 18 channels
imu_18ch = np.concatenate([imu_9ch, rotation_features], axis=0)  # (18, 1000)
```

## Model Architecture

```
Input: (batch, 18, time_steps)
   ↓
Channel Projection: Conv1d(18 → 6)  [LEARNABLE]
   ↓
GroupNorm + Conv Blocks (same as original)
   ↓
GRU Layer
   ↓
Output: (batch, 512) CLIP embedding
```

The 18→6 projection layer learns to optimally combine:
- Raw sensor data (9 channels)
- Rotation features (9 channels)
- Into 6 effective channels for the downstream network

## Configuration

### File: `finetune_18ch.yaml`

```yaml
data:
  num_channels: 18  # 9 IMU + 9 rotation

model:
  name: "MW2StackRNNPooling18Ch"
  checkpoint: "path/to/pretrained_6ch.ckpt"  # Can use 6-channel pretrained!

output:
  dir: "./log/finetuned_models_18ch"
```

## Usage

### 1. Fine-tuning with 18-Channel Features

```bash
python fine_tune/finetune.py --config fine_tune/finetune_18ch.yaml
```

**Output structure**:
```
log/finetuned_models_18ch/
└── finetune_18ch_20251203_150000/
    ├── best_model.ckpt
    ├── final_model.ckpt
    ├── training_curves.png
    └── history.json
```

### 2. Inference with 18-Channel Model

```python
from inference_imu2clip import IMU2CLIPInference
from inference_hdf5 import load_hdf5_imu

# Load model
model = IMU2CLIPInference(
    checkpoint_path="log/finetuned_models_18ch/best_model.ckpt",
    model_name="MW2StackRNNPooling18Ch",
    num_channels=18,
    device="cuda"
)

# Load 9-channel data (rotation features computed automatically)
data = load_hdf5_imu("data.hdf5", num_channels=18)  # Loads 9, adds rotation later
imu_9ch = data['imu_data']  # (9, N)

# Encode (rotation features computed inside)
embedding = model.encode(imu_9ch)  # Returns (512,)
```

### 3. KNN Classification with 18-Channel Model

```bash
python knn_classifier.py /path/to/data \
    --checkpoint log/finetuned_models_18ch/best_model.ckpt \
    --num_channels 18 \
    --k 5 \
    --device cuda
```

**Note**: Auto-selects `MW2StackRNNPooling18Ch` when `--num_channels 18` is specified.

### 4. Retrieval Evaluation with 18-Channel Model

```bash
python retrive.py \
    --val_data_dir /path/to/validation/data \
    --checkpoint log/finetuned_models_18ch/best_model.ckpt \
    --num_channels 18 \
    --k_values 1 5 10 50 \
    --device cuda
```

## Model Comparison

| Model | Input Channels | Features | Parameters | Use Case |
|-------|---------------|----------|------------|----------|
| MW2StackRNNPooling | 6 | Accel + Gyro | 850,892 | General motion |
| MW2StackRNNPooling9Ch | 9 | + Magnetometer | 850,952 | Orientation-aware |
| **MW2StackRNNPooling18Ch** | **18** | **+ Rotation matrix** | **851,006** | **Orientation-critical tasks** |

## Training Tips

1. **Start from 6-channel pretrained**: The main network can be initialized from pretrained weights
2. **Channel projection trains from scratch**: The 18→6 projection layer is new
3. **Augmentation order matters**: 
   - Apply data augmentation to raw 9 channels first
   - Then compute rotation features
   - This ensures rotation features are consistent with augmented data
4. **Regularization**: Use weight decay and early stopping to prevent overfitting on rotation features

## Data Requirements

- HDF5 files must contain **9 channels** minimum (accel + gyro + mag)
- Rotation features are **computed automatically** during training/inference
- No need to pre-compute or store rotation matrices

## Expected Performance

The 18-channel model should perform better on tasks that require:
- ✅ Absolute orientation understanding
- ✅ Rotation-invariant recognition
- ✅ Complex 3D motions
- ✅ Orientation-sensitive actions (e.g., "flip phone", "pour water")

For simple acceleration-based motions (e.g., "clap"), the 6 or 9-channel models may be sufficient.

## Troubleshooting

### Error: "Expected at least 9 IMU channels"

**Solution**: 18-channel mode requires 9-channel HDF5 files. Your data only has 6 channels. Use the 6-channel model instead.

### Warning: "Rotation feature computation failed"

This is usually fine - the model will use zeros for rotation features and continue. Common causes:
- Magnetometer data has NaN values
- All-zero accelerometer readings
- Invalid sensor readings

The fallback (zeros) allows training to continue.

### Slower Training

Yes, computing rotation features adds ~10-20% computational overhead. This is acceptable for the improved performance on orientation-sensitive tasks.

## See Also

- `lib/rotation_utils.py` - Rotation matrix computation implementation
- `fine_tune/finetune_18ch.yaml` - Configuration template
- `fine_tune/README_9CHANNEL.md` - 9-channel model documentation



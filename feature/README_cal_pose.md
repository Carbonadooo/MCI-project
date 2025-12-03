# Pose Calculation and Visualization

## Overview

`cal_pose.py` now performs **two tasks**:
1. **Calculate pose data** (rotation matrices) → Save as `.npy` files
2. **Generate visualizations** (animated GIFs) → Save as `.gif` files

## Features

### Pose Calculation
- Reads IMU data from HDF5 files
- Calculates Roll, Pitch, Yaw from accelerometer and magnetometer
- Converts to 3D rotation matrices
- Saves as `.npy` files for later use

### Visualization
- Creates animated GIF showing sensor orientation over time
- Fixed origin at (0, 0, 0)
- Shows all 3 axes: X (red), Y (green), Z (blue)
- Includes time counter and frame number
- Configurable FPS and frame skip

## Configuration

At the top of `cal_pose.py`:

```python
# Enable/disable visualizations
GENERATE_VISUALIZATIONS = True  # Set to False to skip GIF generation

# Visualization settings
FPS = 30              # Frames per second
ARROW_LENGTH = 0.5    # Size of orientation arrows
FRAME_SKIP = 5        # Show every Nth frame (reduces file size)
```

## Usage

### Batch Process All Files

```bash
cd /home/chuye/Documents/MCI-project
python feature/cal_pose.py
```

**Outputs:**
- Pose data: `/home/chuye/Documents/MCI-project/data/nov2_set_pose_features/`
- Visualizations: `/home/chuye/Documents/MCI-project/data/nov2_set_pose_visualizations/`

### Process Single File (Python)

```python
from feature.cal_pose import process_single_file
from pathlib import Path

hdf5_file = Path('data/nov2_set/punch_forward/punch_forward_20251108_204358.hdf5')
output_dir = Path('data/pose_output')
viz_dir = Path('data/pose_viz')

process_single_file(hdf5_file, output_dir, viz_dir)
```

## Output Structure

```
nov2_set_pose_features/
├── punch_forward/
│   ├── punch_forward_20251108_204358_pose.npy
│   └── ...
├── high_five_motion/
│   └── ...
└── ...

nov2_set_pose_visualizations/
├── punch_forward/
│   ├── punch_forward_20251108_204358_pose.gif
│   └── ...
├── high_five_motion/
│   └── ...
└── ...
```

## Performance

### Single File (~7 seconds of data)
- Pose calculation: < 1 second
- Visualization generation: ~10-15 seconds
- **Total: ~15 seconds per file**

### Batch Processing
For 100 files:
- Without visualizations: ~2 minutes
- With visualizations: ~25 minutes

**Tip:** If you only need pose data, set `GENERATE_VISUALIZATIONS = False`

## File Sizes

- **Pose .npy file**: ~32 KB (for 1336 frames)
- **Visualization .gif file**: ~1-2 MB (depending on motion complexity)

## Dependencies

```bash
pip install numpy pandas h5py matplotlib pillow
```

## Troubleshooting

### Visualization Takes Too Long
- Increase `FRAME_SKIP` (e.g., 10 instead of 5)
- Decrease `FPS` (e.g., 20 instead of 30)
- Set `GENERATE_VISUALIZATIONS = False` and visualize only selected files

### Memory Issues
- Process files one at a time instead of batch
- Increase `FRAME_SKIP` to reduce frame count

### GIF Quality Issues
- Decrease `FRAME_SKIP` for smoother animation
- Increase DPI in `render_pose_frame()` (line ~208)

## Technical Details

### Rotation Matrix Format
- Shape: `(N, 3, 3)` where N is number of samples
- Each matrix R represents body-to-world rotation
- Columns: [X-axis, Y-axis, Z-axis] in world coordinates

### Visualization
- Each frame shows sensor orientation at that timestamp
- Origin fixed at (0, 0, 0)
- Arrows show where sensor axes point in world frame
- Red=X, Green=Y, Blue=Z

### Why This Approach?
✅ No trajectory integration → No drift errors
✅ Direct visualization of rotation matrices
✅ Reliable representation of sensor orientation
✅ Perfect for understanding gesture dynamics


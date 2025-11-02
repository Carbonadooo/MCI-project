# MCI Project

IMU-driven Query System for Daily Human Manipulation Video Retrieval

| index    | variable         | unit    |
| ---- | ---------- | ----- |
| 1    | timestamp  | s |
| 2-4  | ax, ay, az | g     |
| 5-7  | gx, gy, gz | °/s   |
| 8-10 | mx, my, mz | μT    |


### Record Data

```bash
python record.py
```

### View Data

use gui to load target hdf5 file

```bash
python collection/view_hdf5.py
```

### Data Structure

```
data/
    $experiment_name/
        $task_name/
            $task_name_20250101_120000.hdf5
            $task_name_20250101_120000.mp4
            $task_name_20250101_120000.csv
```

To create dataset folder structure:
```bash
mkdir -p clap_once throw_small_object punch_forward high_five_motion open_refrigerator unplug_usb_cable flick_switch_up_down \
open_close_notebook open_door_with_handle screw_bottle_cap squeeze_hand_sanitizer open_close_drawer twist_towel pick_and_place drink_water pour_water_into_cup fold_clothes \
use_hammer shake_bottle wipe_table_back_forth brush_teeth cut_vegetables_with_knife wave_hand_left_right stir_with_spoon
```
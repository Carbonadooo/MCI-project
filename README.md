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
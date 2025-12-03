# Pose Calculation Methods Comparison

## Overview

There are **two different methods** for calculating sensor orientation, each suitable for different scenarios:

### Method 1: Accelerometer-Based (Static Assumption)
**File:** `visualize_pose_sequence.py`
**Best for:** Static or slow motions with minimal acceleration

### Method 2: Gyroscope-Based (Dynamic Tracking)
**File:** `visualize_pose_gyro.py`  
**Best for:** Dynamic motions with significant acceleration

---

## Method 1: Accelerometer-Based Orientation

### How It Works
```
1. Assume accelerometer ONLY measures gravity
2. Calculate Roll/Pitch from gravity direction
3. Calculate Yaw from magnetometer
4. Convert to rotation matrix
```

### Assumptions
- ✅ Device is stationary or moving very slowly
- ✅ Linear acceleration << gravity (9.81 m/s²)
- ❌ **BREAKS during dynamic motions** (punching, waving, etc.)

### When Accelerometer Reads
| Motion State | Accelerometer Measures | Valid for Orientation? |
|--------------|----------------------|----------------------|
| At rest | Only gravity (9.81 m/s²) | ✅ YES |
| Slow motion | Gravity + small linear accel | ✅ YES |
| Fast motion | Gravity + large linear accel | ❌ NO |
| Free fall | Zero! | ❌ NO |

### Example Problem
```
During a punch:
- True: Hand accelerates forward at 10 m/s²
- Measured: Accel = gravity + 10 m/s² = [10, 0, 9.81]
- Calculated orientation: WRONG! (interprets acceleration as tilt)
```

### Pros & Cons
✅ Simple and fast  
✅ No drift over time  
❌ **Fails during acceleration**  
❌ Not suitable for your tasks!

---

## Method 2: Gyroscope-Based Orientation ⭐ **RECOMMENDED**

### How It Works
```
1. Initialize orientation from first sample (at rest)
2. Integrate gyroscope (angular velocity) over time
3. Apply small corrections from accelerometer (when not accelerating)
4. Apply small corrections from magnetometer (prevent yaw drift)
```

### Algorithm: Complementary Filter
```python
For each timestep:
  1. Rotate by gyroscope: R = R_prev * exp(ω * dt)
  2. If |accel| ≈ 9.81: correct tilt using accelerometer
  3. Correct yaw drift using magnetometer
```

### Key Features
- **Primary sensor:** Gyroscope (measures rotation rate)
- **Secondary sensors:** Accel/Mag (corrections only)
- **Accel correction:** Only applied when |accel| ≈ gravity (8-11 m/s²)
- **During high acceleration:** Trusts gyroscope, ignores accelerometer

### Parameters
```python
alpha_accel = 0.02  # Accelerometer correction weight (2%)
alpha_mag = 0.01    # Magnetometer correction weight (1%)
```

Smaller values = trust gyroscope more (less drift correction)  
Larger values = faster corrections (but noisier during motion)

### Pros & Cons
✅ **Works during acceleration!**  
✅ Accurate for dynamic motions  
✅ Suitable for all your tasks  
⚠️ Slight drift over long periods (>10s)  
⚠️ Requires good gyroscope calibration

---

## Comparison on Your Tasks

| Task | Method 1 (Accel-based) | Method 2 (Gyro-based) |
|------|----------------------|---------------------|
| Punch forward | ❌ Wrong during punch | ✅ Accurate |
| High five | ❌ Wrong during motion | ✅ Accurate |
| Wave hand | ❌ Wrong during wave | ✅ Accurate |
| Squeeze sanitizer | ⚠️ Maybe OK (small motion) | ✅ Better |
| Drink water | ❌ Wrong during lift/tilt | ✅ Accurate |

---

## Which Method to Use?

### Use Method 1 (Accelerometer) if:
- Device is stationary (calibration, testing)
- Very slow motions only
- No gyroscope available

### Use Method 2 (Gyroscope) if: ⭐
- **Any task with acceleration** (your case!)
- Dynamic hand gestures
- Fast motions
- Rotations during movement

---

## Technical Details

### Gyroscope Data
- **Raw format:** degrees/second
- **Converted to:** radians/second (multiply by π/180)
- **Meaning:** [gx, gy, gz] = rotation rate around [X, Y, Z] axes

### Integration Method
Uses **Rodrigues' rotation formula** for incremental rotations:
```
R(t+dt) = R(t) * exp(ω × dt)
```

Where `ω × dt` is the rotation vector (axis-angle representation).

### Complementary Filter
Combines high-frequency gyroscope with low-frequency accel/mag:
- **Gyroscope:** Fast response, accurate short-term, drifts long-term
- **Accelerometer:** Slow response, noisy, but no drift
- **Result:** Best of both worlds!

---

## Output Comparison

### Method 1 Output
```
File: *_pose.gif
Title: "Sensor Orientation"
Issue: Jittery during acceleration
```

### Method 2 Output
```
File: *_pose_gyro.gif
Title: "Sensor Orientation (Gyro-based)"
Benefit: Smooth and accurate during motion
```

---

## Recommendation

**For your MCI gesture recognition project:**

✅ **USE METHOD 2 (Gyroscope-based)**

All your tasks involve dynamic motions with significant acceleration. The gyroscope-based method is essential for accurate orientation tracking during these motions.

---

## Next Steps

1. ✅ Tested gyroscope method: Works!
2. 🔄 Update `cal_pose.py` to use gyroscope method
3. 🔄 Batch process all your data with new method
4. 🔄 Compare visualizations: old vs new method



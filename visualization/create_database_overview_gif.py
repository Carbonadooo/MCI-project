import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation

# --- CONFIGURATION ---
DATA_ROOT = Path('/home/chuye/Documents/MCI-project/data')
OUTPUT_PATH = Path('/home/chuye/Documents/MCI-project/data/dataset_overview_for_figure_2.gif')

TRAJECTORIES = [
    # 'all_data/twist_towel/twist_towel_20251108_215802.hdf5',
    # 'all_data/open_door_with_handle/open_door_with_handle_20251202_152210.hdf5',
    # 'train_set/fold_clothes/fold_clothes_20251203_145455.hdf5',
    # 'all_data/clap_once/clap_once_20251116_165218.hdf5',
    # 'validate_set/shake_bottle/shake_bottle_20251202_165123.hdf5', # <--- Added missing comma here
    # 'all_data/pour_water_into_cup/pour_water_into_cup_20251116_165720.hdf5'
    # '/home/chuye/Documents/MCI-project/data/all_data/shake_bottle/shake_bottle_20251108_204510.hdf5',
    '/home/chuye/Documents/MCI-project/data/all_data/shake_bottle/shake_bottle_20251108_213757.hdf5',
    # '/home/chuye/Documents/MCI-project/data/all_data/shake_bottle/shake_bottle_20251202_164724.hdf5',
    # '/home/chuye/Documents/MCI-project/data/all_data/shake_bottle/shake_bottle_20251202_165144.hdf5',
]

# GIF SETTINGS
FPS = 15             
FRAME_STRIDE = 2     
DPI = 80             
# ---------------------

def load_hdf5_data(hdf5_path: Path):
    with h5py.File(hdf5_path, 'r') as f:
        if 'imu' in f:
            imu_data = f['imu']['data'][()]
            imu_timestamps = f['imu']['timestamps'][()]
        else:
            raise ValueError(f"No IMU data in {hdf5_path.name}")
        
        if 'video' in f:
            video_frames = f['video']['frames'][()]
            video_timestamps = f['video']['timestamps'][()]
            
            # Convert BGR to RGB (video data is typically stored in BGR format)
            video_frames = video_frames[..., ::-1]
        else:
            raise ValueError(f"No video data in {hdf5_path.name}")
    
    return {
        'imu_data': imu_data,
        'imu_ts': imu_timestamps,
        'video_frames': video_frames,
        'video_ts': video_timestamps
    }

def setup_sensor_plot(ax, timestamps, data, colors=['r', 'g', 'b']):
    """Initializes the static sensor lines and returns the dynamic cursor line."""
    # Plot the static data traces
    for i, color in enumerate(colors):
        ax.plot(timestamps, data[:, i], color=color, linewidth=4.0, alpha=0.9) # Slightly reduced width for clarity
    
    # Create the moving vertical line (cursor)
    cursor_line = ax.axvline(x=timestamps[0], color='k', linestyle='-', linewidth=4.0, alpha=0.6)
    
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Remove spines to make it cleaner/compact
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set limits explicitly so the plot doesn't resize during animation
    ax.set_xlim(timestamps[0], timestamps[-1])
    ax.set_ylim(np.min(data)*1.1, np.max(data)*1.1)
    
    return cursor_line

def create_animation():
    # 1. Load all data into memory first
    print("="*60)
    print("Loading Data...")
    
    rows_data = []
    max_frames = 0
    
    for traj_path in TRAJECTORIES:
        full_path = DATA_ROOT / traj_path
        if not full_path.exists():
            print(f"⚠️  Skip (File Not Found): {traj_path}")
            continue
            
        try:
            d = load_hdf5_data(full_path)
            
            # Normalize timestamps
            t0 = d['video_ts'][0]
            d['video_ts'] -= t0
            d['imu_ts'] -= t0  # Align IMU to video start
            
            # Store processed chunks
            task_name = full_path.parent.name
            pretty_name = ' '.join(task_name.split('_')).title()
            
            row_entry = {
                'name': pretty_name,
                'video': d['video_frames'],
                'video_ts': d['video_ts'],
                'imu_ts': d['imu_ts'],
                'accel': d['imu_data'][:, 0:3],
                'gyro': d['imu_data'][:, 3:6],
                'mag': d['imu_data'][:, 6:9]
            }
            rows_data.append(row_entry)
            
            # Track longest video for animation duration
            if len(d['video_frames']) > max_frames:
                max_frames = len(d['video_frames'])
                
        except Exception as e:
            print(f"Error loading {traj_path}: {e}")

    num_rows = len(rows_data)
    if num_rows == 0:
        print("No valid data loaded. Exiting.")
        return

    # 2. Setup Figure
    # Adjust height dynamically based on rows to maintain aspect ratio
    fig_height = max(12, num_rows * 4) 
    fig = plt.figure(figsize=(30, fig_height))
    
    # --- UPDATED LAYOUT SETTINGS ---
    # hspace/wspace set to 0.03 for very tight margins
    gs = gridspec.GridSpec(num_rows, 4, figure=fig, 
                           hspace=0.03, wspace=0.03, 
                           left=0.08, right=0.99, top=0.98, bottom=0.02)

    # Lists to store updateable objects
    video_artists = []   
    cursor_artists = []  

    print(f"Setting up plots for {num_rows} trajectories...")

    for row_idx, data in enumerate(rows_data):
        # --- Column 0: Video ---
        ax_vid = fig.add_subplot(gs[row_idx, 0])
        initial_frame = data['video'][0]
        im_display = ax_vid.imshow(initial_frame)
        ax_vid.axis('off')
        
        # Add Row Label (Gesture Name) to the left of video
        # Adjusted position slightly for tight layout
        # ax_vid.text(-0.05, 0.5, data['name'], transform=ax_vid.transAxes, 
        #            fontsize=24, fontweight='bold', rotation=90, 
        #            va='center', ha='right')
        
        video_artists.append({
            'img': im_display,
            'frames': data['video']
        })

        # --- Sensor Columns ---
        sensors = [
            (1, data['accel'], ['#FF4444', '#44FF44', '#4444FF']), # Accel
            (2, data['gyro'],  ['#CC0000', '#00CC00', '#0000CC']), # Gyro
            (3, data['mag'],   ['#880000', '#008800', '#000088'])  # Mag
        ]
        
        for col_idx, sensor_data, colors in sensors:
            ax = fig.add_subplot(gs[row_idx, col_idx])
            
            cursor = setup_sensor_plot(ax, data['imu_ts'], sensor_data, colors)
            
            cursor_artists.append({
                'line': cursor,
                'imu_ts': data['imu_ts'],
                'video_ts': data['video_ts']
            })
            
            # Remove x-tick labels for all rows (including bottom for ultra-clean look)
            # If you want time on bottom row, change to: if row_idx < num_rows - 1:
            ax.set_xticklabels([])
            ax.set_yticklabels([]) # Optional: Remove Y labels too for maximum compactness

    # 3. Animation Update Function
    def update(frame_num):
        # Update Videos
        for vid_obj in video_artists:
            frames = vid_obj['frames']
            idx = min(frame_num, len(frames) - 1)
            vid_obj['img'].set_data(frames[idx])
            
        # Update Cursors
        idx_counter = 0
        for row_data in rows_data:
            vid_ts = row_data['video_ts']
            current_vid_idx = min(frame_num, len(vid_ts) - 1)
            current_time = vid_ts[current_vid_idx]
            
            for _ in range(3):
                artist_obj = cursor_artists[idx_counter]
                artist_obj['line'].set_xdata([current_time, current_time])
                idx_counter += 1
                
        return []

    # 4. Generate and Save
    print(f"Generating Animation ({max_frames} frames)...")
    
    frames_to_render = range(0, max_frames, FRAME_STRIDE)
    
    anim = FuncAnimation(fig, update, frames=frames_to_render, 
                         interval=1000/FPS, blit=False)
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    anim.save(OUTPUT_PATH, writer='pillow', fps=FPS, dpi=DPI)
    
    print("="*60)
    print(f"✅ GIF saved to: {OUTPUT_PATH}")
    print("="*60)
    plt.close()

if __name__ == "__main__":
    create_animation()
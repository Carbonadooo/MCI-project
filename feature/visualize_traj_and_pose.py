import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from mpl_toolkits.mplot3d import Axes3D

# --- CONFIGURATION ---
# The directory where the processed features were saved
INPUT_ROOT = '/home/chuye/Documents/MCI-project/data/nov2_set_pose_features' 
# Directory to save the output plots
OUTPUT_ROOT = '/home/chuye/Documents/MCI-project/data/nov2_set_visualizations' 
# We'll plot an arrow every N steps to keep the plot clean
ARROW_SKIP_RATE = 50 
# Arrow length scale for visualization
ARROW_SCALE = 0.1
# ---------------------

def load_features(file_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads trajectory and rotation matrices for a single recording.
    """
    base_name = file_path.stem.replace('_traj', '')
    
    # Path to Trajectory (N, 3)
    traj_path = file_path
    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")

    # Path to Pose (N, 3, 3)
    pose_path = file_path.with_name(base_name + '_pose.npy')
    if not pose_path.exists():
        raise FileNotFoundError(f"Pose file not found: {pose_path}")

    trajectory = np.load(traj_path)
    rotation_matrices = np.load(pose_path)
    
    # Ensure they match in length
    if len(trajectory) != len(rotation_matrices):
        raise ValueError(f"Feature length mismatch: Trajectory ({len(trajectory)}) vs. Pose ({len(rotation_matrices)})")
        
    return trajectory, rotation_matrices

def create_visualization(trajectory: np.ndarray, rotation_matrices: np.ndarray, output_path: Path, task_name: str):
    """
    Generates and saves a 3D plot of the trajectory with orientation arrows.
    Includes data cleaning to handle NaN/Inf values.
    """
    
    # --- NEW: Data Cleaning Step ---
    # 1. Check for NaN or Inf values in the trajectory
    if not np.all(np.isfinite(trajectory)):
        print(f"⚠️ WARNING: Non-finite values found in trajectory for {output_path.name}. Cleaning data...")
        
        # 1.1 Find where the data is finite
        finite_mask = np.all(np.isfinite(trajectory), axis=1)
        
        # 1.2 Filter the data arrays
        if np.sum(finite_mask) == 0:
            raise ValueError("Trajectory contains NO finite data points. Cannot plot.")
            
        trajectory = trajectory[finite_mask]
        rotation_matrices = rotation_matrices[finite_mask]
        
        # 1.3 Recalculate arrow positions based on the cleaned data
        
    # --- End Data Cleaning Step ---

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. Plot the Trajectory Line (using cleaned trajectory)
    ax.plot(
        trajectory[:, 0], # X coordinates
        trajectory[:, 1], # Y coordinates
        trajectory[:, 2], # Z coordinates
        label='Calculated Trajectory', 
        color='blue', 
        linewidth=2
    )
    
    # 2. Plot Orientation Arrows (Quiver Plot)
    
    # Get the starting position for the arrows
    X_pos = trajectory[::ARROW_SKIP_RATE, 0]
    Y_pos = trajectory[::ARROW_SKIP_RATE, 1]
    Z_pos = trajectory[::ARROW_SKIP_RATE, 2]
    
    # Get the corresponding rotation vectors
    direction_vectors = rotation_matrices[::ARROW_SKIP_RATE, :, 0] 
    U = direction_vectors[:, 0]
    V = direction_vectors[:, 1]
    W = direction_vectors[:, 2]

    ax.quiver(
        X_pos, Y_pos, Z_pos, 
        U, V, W, 
        length=ARROW_SCALE, 
        color='red', 
        arrow_length_ratio=0.5,
        normalize=False
    )

    # 3. Aesthetics and Labels (Using cleaned trajectory for limits)
    ax.set_title(f"Pose Feature Visualization: {task_name}", fontsize=14)
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    
    # Set equal scale for better perspective
    x_range = trajectory[:,0].max()-trajectory[:,0].min()
    y_range = trajectory[:,1].max()-trajectory[:,1].min()
    z_range = trajectory[:,2].max()-trajectory[:,2].min()
    
    max_range = np.array([x_range, y_range, z_range]).max() / 2.0
    
    mid_x = (trajectory[:,0].max()+trajectory[:,0].min()) * 0.5
    mid_y = (trajectory[:,1].max()+trajectory[:,1].min()) * 0.5
    mid_z = (trajectory[:,2].max()+trajectory[:,2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Save the plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close(fig)
    print(f"   -> Plot saved to: {output_path}")


def batch_visualize_features(root_input_dir: str, root_output_dir: str):
    """
    Finds all trajectory files and generates visualizations.
    """
    root_input_path = Path(root_input_dir)
    root_output_path = Path(root_output_dir)
    
    if not root_input_path.is_dir():
        print(f"❌ Error: Input features directory not found at {root_input_dir}")
        return

    # Find all trajectory files (assuming they end with '_traj.npy')
    traj_files = list(root_input_path.glob('**/*_traj.npy'))
    
    if not traj_files:
        print(f"⚠️ No '_traj.npy' files found in {root_input_dir} or its subdirectories.")
        return

    print("="*60)
    print(f"Starting visualization for {len(traj_files)} trajectories...")
    print("="*60)
    
    total_plots = 0
    for traj_file in traj_files:
        try:
            print(f"\n--- Visualizing: {traj_file.name} ---")
            
            # 1. Load Data
            trajectory, rotation_matrices = load_features(traj_file)
            
            # 2. Define Output Paths
            task_name = traj_file.parent.name
            output_base_name = traj_file.stem.replace('_traj', '')
            
            output_dir = root_output_path / task_name
            output_filename = output_base_name + '_pose_traj.png'
            output_path = output_dir / output_filename
            
            # 3. Create and Save Plot
            create_visualization(trajectory, rotation_matrices, output_path, f"{task_name} - {output_base_name}")
            total_plots += 1
            
        except Exception as e:
            print(f"❌ FAILED to visualize {traj_file.name}: {e}")
            
    print("\n" + "="*60)
    print(f"Visualization completed! Generated {total_plots} plots.")
    print("="*60)


if __name__ == "__main__":
    # ⚠️ Check the paths in the CONFIGURATION section at the top of the file!
    batch_visualize_features(INPUT_ROOT, OUTPUT_ROOT)
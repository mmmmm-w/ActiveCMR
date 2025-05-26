import json
import os
import glob

import torch
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import map_coordinates


#METRICS
def calculate_dice(ground_truth, generated):
    """
    Calculate Dice coefficient for each anatomical structure (LV, MYO, RV)
    
    Args:
        ground_truth: [4, D, H, W] tensor/array (background, LV, MYO, RV)
        generated: [4, D, H, W] tensor/array (background, LV, MYO, RV)
    Returns:
        dict: Dice scores for each structure and average
    """
    # Convert to numpy if tensors
    if torch.is_tensor(ground_truth):
        ground_truth = ground_truth.cpu().numpy()
    if torch.is_tensor(generated):
        generated = generated.cpu().numpy()
    
    dice_scores = {}
    # Calculate Dice for each structure (excluding background)
    for i, label in enumerate(['LV', 'MYO', 'RV']):
        # Convert to binary masks using 0.5 threshold
        gt_mask = (ground_truth[i+1] > 0.5).astype(float)
        gen_mask = (generated[i+1] > 0.5).astype(float)
        
        # Calculate Dice coefficient: 2|X∩Y|/(|X|+|Y|)
        intersection = np.sum(gt_mask * gen_mask)
        dice = (2.0 * intersection) / (np.sum(gt_mask) + np.sum(gen_mask) + 1e-6)
        dice_scores[label] = dice
    
    # Calculate average Dice
    dice_scores['average'] = np.mean(list(dice_scores.values()))
    
    return dice_scores

#VISUALIZATION
def extract_slice_and_meta(volume, center, normal, slice_shape=(128, 128), pixel_spacing=1.0, to_degrees=True):
    """
    Extracts a 2D slice from a 3D segmentation label map along a plane defined by
    a center (z, x, y) and a normal vector.
    
    Parameters:
      volume (np.ndarray): 3D segmentation label map with shape (D, H, W).
      center (array-like): Center of the slicing plane in (z, x, y) coordinates.
      normal (array-like): Normal vector of the slicing plane (defines orientation).
      slice_shape (tuple): Desired shape (height, width) of the 2D slice.
      pixel_spacing (float): Spacing between pixels in the extracted slice.
      to_degrees (bool): If True, returns theta and phi in degrees.
      
    Returns:
      slice_2d (np.ndarray): The 2D segmentation slice.
      meta (dict): Metadata containing:
                    - 'center': The center coordinate as [x, y, z].
                    - 'theta': Azimuth angle of the normal (from x-axis in the x-y plane).
                    - 'phi': Angle from the z-axis.
    """
    # Ensure inputs are numpy arrays and normalize the normal vector
    center = np.array(center, dtype=np.float32)
    normal = np.array(normal, dtype=np.float32)
    normal /= np.linalg.norm(normal)
    
    # Compute orientation angles from the normal vector:
    # theta: angle in the x-y plane from the x-axis, phi: angle from the z-axis.
    theta = np.arctan2(normal[1], normal[0])
    phi = np.arccos(normal[2])
    if to_degrees:
        theta = np.degrees(theta)
        phi = np.degrees(phi)
    
    # Compute two orthonormal vectors (v1, v2) that span the plane.
    # This ensures v1 and v2 are perpendicular to the normal.
    if abs(normal[0]) > abs(normal[1]):
        v1 = np.array([-normal[2], 0, normal[0]])
    else:
        v1 = np.array([0, normal[2], -normal[1]])
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(normal, v1)
    
    # Create a grid in the plane.
    h, w = slice_shape
    # Create grid coordinates (centered at 0,0) scaled by pixel_spacing.
    grid_y = (np.arange(h) - h / 2) * pixel_spacing
    grid_x = (np.arange(w) - w / 2) * pixel_spacing
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)
    
    # For each grid point, compute its corresponding 3D coordinate:
    # coord = center + (grid_x * v1) + (grid_y * v2)
    grid_points = center.reshape((1, 3)) + \
                  grid_x.flatten().reshape(-1, 1) * v1.reshape((1, 3)) + \
                  grid_y.flatten().reshape(-1, 1) * v2.reshape((1, 3))
    
    # Use map_coordinates for nearest-neighbor interpolation.
    # Note: The coordinates array must be given as a list for each dimension.
    slice_flat = map_coordinates(
        volume, 
        [grid_points[:, 0], grid_points[:, 1], grid_points[:, 2]],
        order=0,
        mode='nearest'
    )
    slice_2d = slice_flat.reshape(slice_shape)
    
    # Package meta data
    meta = {
        'center': center.tolist(),
        'theta': theta,
        'phi': phi,
    }
    
    return slice_2d, meta

def extract_slice_and_meta_torch(volume: torch.Tensor, 
                                 center: torch.Tensor, 
                                 normal: torch.Tensor, 
                                 slice_shape=(128, 128), 
                                 pixel_spacing=1.0, 
                                 to_degrees=True,
                                 device='cpu'):
    """
    Extracts a 2D slice from a 3D volume using PyTorch along a plane defined by
    a center (z, x, y) and a normal vector.

    Parameters:
      volume (torch.Tensor): 3D volume with shape (D, H, W) on the specified device.
      center (torch.Tensor): Center of the slicing plane in (z, x, y) coordinates (1D tensor of size 3).
      normal (torch.Tensor): Normal vector of the slicing plane (1D tensor of size 3).
      slice_shape (tuple): Desired shape (height, width) of the 2D slice.
      pixel_spacing (float): Spacing between pixels in the extracted slice.
      to_degrees (bool): If True, returns theta and phi in degrees.
      device (str or torch.device): Device to perform computations on ('cpu' or 'cuda').

    Returns:
      slice_2d (torch.Tensor): The 2D segmentation slice on the specified device.
      meta (dict): Metadata containing:
                    - 'center': The center coordinate as a list [z, x, y].
                    - 'theta': Azimuth angle of the normal (from x-axis in the x-y plane).
                    - 'phi': Angle from the z-axis.
    """
    volume = volume.to(device).float()
    center = torch.as_tensor(center, dtype=torch.float32, device=device)
    normal = torch.as_tensor(normal, dtype=torch.float32, device=device)
    normal = normal / torch.linalg.norm(normal)

    # Compute orientation angles
    theta = torch.arctan2(normal[1], normal[0])
    phi = torch.arccos(normal[2])
    if to_degrees:
        theta = torch.rad2deg(theta)
        phi = torch.rad2deg(phi)

    # Compute orthonormal vectors spanning the plane
    if abs(normal[0]) > abs(normal[1]):
        v1 = torch.tensor([-normal[2], 0, normal[0]], dtype=torch.float32, device=device)
    else:
        v1 = torch.tensor([0, normal[2], -normal[1]], dtype=torch.float32, device=device)
    v1 = v1 / torch.linalg.norm(v1)
    v2 = torch.linalg.cross(normal, v1)

    # Create grid coordinates in the plane
    h, w = slice_shape
    grid_y_coords = (torch.arange(h, device=device) - h / 2) * pixel_spacing
    grid_x_coords = (torch.arange(w, device=device) - w / 2) * pixel_spacing
    grid_x, grid_y = torch.meshgrid(grid_x_coords, grid_y_coords, indexing='xy') # HxW grids

    # Compute 3D coordinates for each grid point
    # grid_points shape: (H, W, 3) in (z, x, y) order
    grid_points = center.view(1, 1, 3) + \
                  grid_x.unsqueeze(-1) * v1.view(1, 1, 3) + \
                  grid_y.unsqueeze(-1) * v2.view(1, 1, 3)

    # --- Prepare for grid_sample ---
    # grid_sample expects input shape (N, C, D_in, H_in, W_in)
    # and grid shape (N, D_out, H_out, W_out, 3) with coordinates in (x, y, z) order [-1, 1]
    
    # Reshape volume: (D, H, W) -> (1, 1, D, H, W)
    volume_unsqueezed = volume.unsqueeze(0).unsqueeze(0)
    D, H_vol, W_vol = volume.shape

    # Normalize grid_points to [-1, 1] based on volume dimensions
    # Original grid_points are (z, x, y), need to map to grid_sample's (x_norm, y_norm, z_norm)
    normalized_grid = torch.zeros_like(grid_points)
    normalized_grid[..., 0] = (grid_points[..., 1] / (W_vol - 1)) * 2 - 1  # Normalize x -> W
    normalized_grid[..., 1] = (grid_points[..., 2] / (H_vol - 1)) * 2 - 1  # Normalize y -> H
    normalized_grid[..., 2] = (grid_points[..., 0] / (D - 1)) * 2 - 1      # Normalize z -> D

    # Reshape grid for grid_sample: (H_slice, W_slice, 3) -> (1, H_slice, W_slice, 1, 3)
    # We want a 2D slice, so D_out=1. grid_sample needs 5D grid for 3D input.
    sampling_grid = normalized_grid.unsqueeze(0).unsqueeze(-2) # Shape: (1, H, W, 1, 3)

    # Perform sampling
    # mode='nearest' corresponds to order=0 interpolation
    # padding_mode='border' clamps coordinates to the edge, similar to mode='nearest' in map_coordinates
    slice_sampled = F.grid_sample(
        volume_unsqueezed, 
        sampling_grid, 
        mode='nearest', 
        padding_mode='border', 
        align_corners=True # Important for consistency with how coordinates were calculated
    )

    # Reshape output: (1, 1, 1, H, W) -> (H, W)
    slice_2d = slice_sampled.squeeze()

    # Package metadata
    meta = {
        'center': center.cpu().tolist(),
        'theta': theta.item() if torch.is_tensor(theta) else theta,
        'phi': phi.item() if torch.is_tensor(phi) else phi,
    }

    return slice_2d, meta

def plot_loss_curves(history_file, save_dir=None):
    """
    Plot training and validation loss curves from the history file with log scale.
    Training metrics are plotted for every epoch, while validation metrics are plotted every 10 epochs.
    KL and reconstruction losses are plotted on different y-axes for better visualization.
    
    Args:
        history_file (str): Path to the JSON file containing loss history
        save_dir (str, optional): Directory to save the plot. If None, plot will be shown.
    """
    # Load the history
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot total loss with log scale
    ax1.semilogy(history['epochs'], history['train_loss'], label='Training Loss', alpha=0.7)
    val_epochs = history['epochs'][9::10]  # Every 10th epoch
    val_losses = history['val_loss']
    ax1.semilogy(val_epochs, val_losses, 'o-', label='Validation Loss', alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (log scale)')
    ax1.set_title('Total Loss Curves')
    ax1.legend()
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    
    # Plot reconstruction and KL losses with two y-axes
    ax2_kl = ax2.twinx()  # Create second y-axis
    
    # Plot reconstruction loss on left y-axis (log scale)
    recon_lines = []
    l1 = ax2.semilogy(history['epochs'], history['train_recon'], 
                      label='Training Recon Loss', alpha=0.7, color='C0')[0]
    l2 = ax2.semilogy(val_epochs, history['val_recon'], 'o-',
                      label='Validation Recon Loss', alpha=0.7, color='C1')[0]
    recon_lines.extend([l1, l2])
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Reconstruction Loss (log scale)', color='C0')
    ax2.tick_params(axis='y', labelcolor='C0')
    
    # Plot KL loss on right y-axis
    kl_lines = []
    l3 = ax2_kl.plot(history['epochs'], history['train_kl'],
                     label='Training KL Loss', alpha=0.7, color='C2')[0]
    l4 = ax2_kl.plot(val_epochs, history['val_kl'], 'o-',
                     label='Validation KL Loss', alpha=0.7, color='C3')[0]
    kl_lines.extend([l3, l4])
    ax2_kl.set_ylabel('KL Loss', color='C2')
    ax2_kl.tick_params(axis='y', labelcolor='C2')
    
    # Add combined legend
    all_lines = recon_lines + kl_lines
    all_labels = [line.get_label() for line in all_lines]
    ax2.legend(all_lines, all_labels, loc='upper right')
    
    ax2.set_title('Reconstruction and KL Loss Curves')
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.tight_layout()
    
    if save_dir:
        # Extract model config from filename
        filename = os.path.basename(history_file)
        plot_name = filename.replace('loss_history', 'loss_curves').replace('.json', '_log.png')
        save_path = os.path.join(save_dir, plot_name)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    return fig

def show_label_map_slices_XYZ(volume, axis='axial', num_slices=20, cmap='gray'):
    """
    Visualize slices of a 3D volume assumed to be in (X, Y, Z) order.
    
    Args:
        volume (np.ndarray or torch.Tensor): shape [X, Y, Z]
        axis (str): 'axial' (Z), 'coronal' (Y), or 'sagittal' (X)
        num_slices (int): number of slices to display
    """
    volume = volume.numpy() if isinstance(volume, torch.Tensor) else volume
    X, Y, Z = volume.shape

    if axis == 'axial':
        indices = np.linspace(0, Z-1, num_slices).astype(int)
        slices = [volume[:, :, z] for z in indices]
        title = "Axial (Z)"
    elif axis == 'coronal':
        indices = np.linspace(0, Y-1, num_slices).astype(int)
        slices = [volume[:, y, :] for y in indices]
        title = "Coronal (Y)"
    elif axis == 'sagittal':
        indices = np.linspace(0, X-1, num_slices).astype(int)
        slices = [volume[x, :, :] for x in indices]
        title = "Sagittal (X)"
    else:
        raise ValueError("Axis must be 'axial', 'coronal', or 'sagittal'")

    plt.figure(figsize=(15, 3))
    for i, s in enumerate(slices):
        plt.subplot(2, int(num_slices/2), i + 1)
        plt.imshow(s.T, cmap=cmap, origin='lower')  # transpose for correct orientation
        plt.title(f"{title} {indices[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_scanned_slices(scanned_slices):
    """
    Visualize all scanned slices in 2 rows, with scan position (z=meta[0]) as caption.

    Args:
        scanned_slices: list of dicts or objects with keys/attributes 'slice' ([4,128,128]) and 'meta' ([5])
    """
    n = len(scanned_slices)
    ncols = (n + 1) // 2
    fig, axes = plt.subplots(2, ncols, figsize=(4 * ncols, 8))
    axes = axes.flatten() if n > 1 else [axes]

    for i, scanned in enumerate(scanned_slices):
        # Get the first channel (or you can loop over all 4 channels if you want)
        img = torch.argmax(scanned[0], dim=0).cpu().numpy() if isinstance(scanned[0], torch.Tensor) else scanned[0]
        z = scanned[1][0].item() if isinstance(scanned[1], torch.Tensor) else scanned[1][0]
        ax = axes[i]
        im = ax.imshow(img, cmap='nipy_spectral')
        ax.set_title(f"Slice {i+1}, z = {z}")
        ax.axis('off')
    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()
    return fig

def visualize_scan_and_sample(sample, scanned_slices, threshold=0.5):
    """
    Visualize scan planes (as red planes) and generated sample (as 3D scatter)
    in the same 3D plot.
    
    Args:
        scanned_slices (list): List of tuples (slice_data, meta) from inference pipeline
        generated_sample (torch.Tensor): Generated 3D volume [C, D, H, W]
        threshold (float): Threshold for visualization (default: 0.5)
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot generated sample as scatter points
    # Create coordinates for each point
    x, y, z = np.where(sample > threshold)
    
    # Plot 3D scatter
    scatter = ax.scatter(x, y, z, 
                        c=sample[x, y, z],
                        cmap='viridis',
                        alpha=0.1,
                        marker='.')
    
    # Plot scan planes
    for _, meta in scanned_slices[:-1]:
        center = meta[:3].numpy()  # [z, x, y]
        
        # Create a planar surface at each scan position
        xx, yy = np.meshgrid(
            np.linspace(-64, 64, 10),
            np.linspace(-64, 64, 10)
        )
        zz = np.full_like(xx, center[0])  # z-position from meta
        
        # Plot the plane
        ax.plot_surface(
            xx + center[2],  # Center x
            yy + center[1],  # Center y
            zz,             # Fixed z position
            alpha=0.2,
            color='red'
        )
    ax.set_xlim(0, 128)
    ax.set_ylim(0, 128)
    ax.set_zlim(0, 64)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    return fig

def visualize_3d_volume_plotly(volume, threshold=0.5):
    # Create coordinates for each point
    x, y, z = np.where(volume > threshold)
    
    # Create 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=2,
            color=volume[x, y, z],  # color by intensity
            colorscale='Viridis',
            opacity=0.8
        )
    )])
    
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        width=800,
        height=800,
        title='3D Volume Visualization'
    )
    
    fig.show()
    return fig

# Method 2: Using Matplotlib (Multiple isosurfaces)
def visualize_3d_volume_matplotlib(volume, threshold=0.5):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create coordinates for each point
    x, y, z = np.where(volume > threshold)
    
    # Plot 3D scatter
    scatter = ax.scatter(x, y, z, 
                        c=volume[x, y, z],
                        cmap='viridis',
                        alpha=0.1,
                        marker='.')
    #force the plot to show 128*128*64
    ax.set_xlim(0, 128)
    ax.set_ylim(0, 128)
    ax.set_zlim(0, 64)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    return fig

#UTILS
def label2onehot(labelmap, classes=(0, 1, 2, 4)):
    """
    Convert label map [D, H, W] to one-hot [C, D, H, W] for given classes.

    Args:
        labelmap (Tensor): shape [D, H, W], int values in {0, 1, 2, 4}
        classes (tuple): class values to include as one-hot channels
    
    Returns:
        Tensor: shape [C, D, H, W], float32
    """
    labelmap[labelmap == 3] = 0 # remap class 3 to 0
    return torch.stack([(labelmap == c).float() for c in classes], dim=0)

def onehot2label(seg_onehot):
    """
    Convert one-hot [C, D, H, W] → label map [D, H, W] with values {0,1,2,4}.

    Args:
        seg_onehot (Tensor or np.ndarray): shape [4, D, H, W]
    
    Returns:
        Tensor or np.ndarray: shape [D, H, W], with labels {0,1,2,4}. 0 - bg, 1-LV, 2-MYO, 4-RV
    """
    if isinstance(seg_onehot, torch.Tensor):
        labelmap = torch.argmax(seg_onehot, dim=0)  # shape: [D, H, W]
        # remap class index 3 → label 4
        labelmap = labelmap.clone()  # avoid in-place overwrite
        labelmap[labelmap == 3] = 4
        return labelmap
    else:
        labelmap = np.argmax(seg_onehot, axis=0)
        labelmap = labelmap.copy()
        labelmap[labelmap == 3] = 4
        return labelmap

def get_label_center(volume):
    """
    Calculate the center of mass of the labeled regions in a 3D volume,
    excluding the background (label 0).
    
    Args:
        volume (np.ndarray or torch.Tensor): 3D volume with shape [X, Y, Z]
                                            containing label values {0,1,2,4}
    
    Returns:
        np.ndarray: Center coordinates [x, y, z] of the labeled regions
    """
    # Convert to numpy if tensor
    if isinstance(volume, torch.Tensor):
        volume = volume.numpy()
    
    # Create binary mask of all non-zero labels
    mask = volume > 0
    
    # Get coordinates of labeled voxels
    coords = np.array(np.where(mask)).T  # shape: [N, 3]
    
    if len(coords) == 0:
        raise ValueError("No labeled regions found in volume")
    
    # Calculate center of mass
    center = coords.mean(axis=0)
    
    return center

def crop_around_center(volume, crop_size=(128, 128, 64), center=None, pad_value=0):
    """
    Crop a 3D volume around a specified center point (or label center if not specified).
    
    Args:
        volume (np.ndarray or torch.Tensor): shape [X, Y, Z]
        crop_size (tuple): Desired output size (crop_X, crop_Y, crop_Z)
        center (array-like, optional): Center point [x, y, z] for cropping.
                                     If None, uses label center.
        pad_value (int): Value to use for padding if crop extends beyond volume
    
    Returns:
        Cropped volume of shape [crop_X, crop_Y, crop_Z]
    """
    # Convert to numpy if tensor
    is_tensor = isinstance(volume, torch.Tensor)
    if is_tensor:
        volume = volume.numpy()
    
    # Get center if not provided
    if center is None:
        center = get_label_center(volume)
    center = np.array(center, dtype=np.int32)
    
    # Calculate crop boundaries
    crop_x, crop_y, crop_z = crop_size
    start_x = center[0] - crop_x // 2
    start_y = center[1] - crop_y // 2
    start_z = center[2] - crop_z // 2
    
    end_x = start_x + crop_x
    end_y = start_y + crop_y
    end_z = start_z + crop_z
    
    # Get volume dimensions
    vol_x, vol_y, vol_z = volume.shape
    
    # Initialize output array with pad_value
    cropped = np.full(crop_size, pad_value, dtype=volume.dtype)
    
    # Calculate valid crop regions for both source and target
    src_start_x = max(0, start_x)
    src_start_y = max(0, start_y)
    src_start_z = max(0, start_z)
    
    src_end_x = min(vol_x, end_x)
    src_end_y = min(vol_y, end_y)
    src_end_z = min(vol_z, end_z)
    
    dst_start_x = max(0, -start_x)
    dst_start_y = max(0, -start_y)
    dst_start_z = max(0, -start_z)
    
    dst_end_x = crop_x - max(0, end_x - vol_x)
    dst_end_y = crop_y - max(0, end_y - vol_y)
    dst_end_z = crop_z - max(0, end_z - vol_z)
    
    # Copy valid region
    cropped[dst_start_x:dst_end_x,
            dst_start_y:dst_end_y,
            dst_start_z:dst_end_z] = volume[src_start_x:src_end_x,
                                          src_start_y:src_end_y,
                                          src_start_z:src_end_z]
    
    # Convert back to tensor if input was tensor
    if is_tensor:
        cropped = torch.from_numpy(cropped)
    
    return cropped

def center_crop(volume, crop_size=(128, 128, 64)):
    """
    Center crop a 3D volume (shape: [X, Y, Z]) to the given size.
    
    Args:
        volume (np.ndarray or torch.Tensor): shape [X, Y, Z]
        crop_size (tuple): (crop_X, crop_Y, crop_Z)
    
    Returns:
        Cropped volume of shape [crop_X, crop_Y, crop_Z]
    """
    if isinstance(volume, torch.Tensor):
        shape = volume.shape
    else:
        shape = np.array(volume.shape)

    crop_x, crop_y, crop_z = crop_size
    start_x = (shape[0] - crop_x) // 2
    start_y = (shape[1] - crop_y) // 2
    start_z = (shape[2] - crop_z) // 2

    if isinstance(volume, torch.Tensor):
        return volume[start_x:start_x+crop_x,
                      start_y:start_y+crop_y,
                      start_z:start_z+crop_z]
    else:
        return volume[start_x:start_x+crop_x,
                      start_y:start_y+crop_y,
                      start_z:start_z+crop_z]
    
# Test the functions
def create_fake_labelmap(shape=(128, 128, 64), seed=42):
    torch.manual_seed(seed)
    labelmap = torch.randint(0, 5, shape)
    labelmap[labelmap == 3] = 0  # skip label 3
    return labelmap

def test_onehot_roundtrip():
    labelmap = create_fake_labelmap()
    onehot = label2onehot(labelmap)
    reconstructed = onehot2label(onehot)
    
    assert labelmap.shape == reconstructed.shape, "Shape mismatch"
    # Check all label values match original (within valid class set)
    mismatch = (labelmap != reconstructed)
    mismatch_ratio = mismatch.sum().item() / labelmap.numel()
    print(f"Mismatch ratio: {mismatch_ratio:.6f}")
    assert mismatch_ratio < 1e-6, "One-hot roundtrip failed!"
    print("✅ onehot2label(label2onehot(labelmap)) passed.")

def test_center_crop():
    labelmap = create_fake_labelmap((150, 150, 80))
    cropped = center_crop(labelmap, (128, 128, 64))
    print(f"Original shape: {labelmap.shape}, Cropped shape: {cropped.shape}")
    assert cropped.shape == (128, 128, 64), "Crop size mismatch"
    print("✅ center_crop passed.")

def test_visualization():
    labelmap = create_fake_labelmap()
    show_label_map_slices_XYZ(labelmap, axis='axial', num_slices=8)
    print("✅ Visualization shown.")

def create_interactive_comparison(checkpoint_dir):
    """Creates an interactive comparison plot of different model configurations."""
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Total Loss Comparison', 'Reconstruction and KL Loss Comparison'),
        vertical_spacing=0.12
    )
    
    # Get all json files
    json_files = glob.glob(os.path.join(checkpoint_dir, "loss_history_*.json"))
    
    for json_file in json_files:
        # Extract configuration from filename
        filename = os.path.basename(json_file)
        config = filename.replace('loss_history_', '').replace('.json', '')
        
        with open(json_file, 'r') as f:
            history = json.load(f)
        
        # Plot total loss
        fig.add_trace(
            go.Scatter(
                x=history['epochs'],
                y=history['train_loss'],
                name=f'{config} (train)',
                mode='lines',
                line=dict(dash='solid'),
                legendgroup=config,
                visible='legendonly' if 'z64' not in config else True
            ),
            row=1, col=1
        )
        
        # Plot validation loss
        val_epochs = history['epochs'][9::10]
        fig.add_trace(
            go.Scatter(
                x=val_epochs,
                y=history['val_loss'],
                name=f'{config} (val)',
                mode='markers+lines',
                legendgroup=config,
                visible='legendonly' if 'z64' not in config else True
            ),
            row=1, col=1
        )
        
        # Plot reconstruction and KL losses
        fig.add_trace(
            go.Scatter(
                x=history['epochs'],
                y=history['train_recon'],
                name=f'{config} (train recon)',
                mode='lines',
                legendgroup=config,
                visible='legendonly' if 'z64' not in config else True
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=history['epochs'],
                y=history['train_kl'],
                name=f'{config} (train KL)',
                mode='lines',
                legendgroup=config,
                visible='legendonly' if 'z64' not in config else True
            ),
            row=2, col=1
        )

    # Update layout
    fig.update_layout(
        height=1000,
        title_text="Beta-VAE Training Curves Comparison",
        showlegend=True,
        legend=dict(
            groupclick="toggleitem",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05
        ),
        hovermode='x unified'
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=2, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=2, col=1)
    
    # Add buttons for different views
    button_layer_1_height = 1.12
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=list([
                    dict(
                        args=[{"visible": [True if "z64" in trace.name else False 
                                         for trace in fig.data]}],
                        label="z=64",
                        method="update"
                    ),
                    dict(
                        args=[{"visible": [True if "z32" in trace.name else False 
                                         for trace in fig.data]}],
                        label="z=32",
                        method="update"
                    ),
                    dict(
                        args=[{"visible": [True if "z16" in trace.name else False 
                                         for trace in fig.data]}],
                        label="z=16",
                        method="update"
                    ),
                    dict(
                        args=[{"visible": [True if "z8" in trace.name else False 
                                         for trace in fig.data]}],
                        label="z=8",
                        method="update"
                    ),
                    dict(
                        args=[{"visible": [True] * len(fig.data)}],
                        label="Show All",
                        method="update"
                    )
                ]),
                direction="right",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=button_layer_1_height,
                yanchor="top"
            )
        ]
    )
    
    return fig

def test_extract_slice():
    # Create a simple test volume with a recognizable pattern
    volume = np.zeros((64, 64, 64), dtype=np.int32)
    # Add a diagonal plane of 1s
    for i in range(64):
        volume[i, i, :] = 1
    
    # Test case 1: Extract axial slice (normal = [0, 0, 1])
    center = [32, 32, 32]  # Center of the volume
    normal = [0, 0, 1]     # Axial view
    slice_2d, meta = extract_slice_and_meta(
        volume, 
        center=center, 
        normal=normal, 
        slice_shape=(32, 32),
        pixel_spacing=1.0
    )
    
    # Verify metadata
    assert meta['center'] == center, "Center point mismatch"
    assert meta['phi'] == 0, "Phi angle should be 0 for axial view"
    assert slice_2d.shape == (32, 32), "Incorrect slice shape"
    
    # Test case 2: Extract sagittal slice (normal = [1, 0, 0])
    normal = [1, 0, 0]     # Sagittal view
    slice_2d, meta = extract_slice_and_meta(
        volume, 
        center=center, 
        normal=normal,
        slice_shape=(32, 32),
        pixel_spacing=1.0
    )
    
    assert meta['phi'] == 90, "Phi angle should be 90 for sagittal view"
    assert slice_2d.shape == (32, 32), "Incorrect slice shape"
    
    print("✅ extract_slice_and_meta tests passed.")


if __name__ == "__main__":
    print("=== Testing label2onehot / onehot2label ===")
    test_onehot_roundtrip()

    print("\n=== Testing center_crop ===")
    test_center_crop()

    print("\n=== Testing slice visualization ===")
    test_visualization()

    print("\n=== Testing slice extraction ===")
    test_extract_slice()
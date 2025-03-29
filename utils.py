import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import json
import os
import glob
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_loss_curves(history_file, save_dir=None):
    """
    Plot training and validation loss curves from the history file.
    Training metrics are plotted for every epoch, while validation metrics are plotted every 10 epochs.
    
    Args:
        history_file (str): Path to the JSON file containing loss history
        save_dir (str, optional): Directory to save the plot. If None, plot will be shown.
    """
    # Load the history
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot total loss
    # Training loss for all epochs
    ax1.plot(history['epochs'], history['train_loss'], label='Training Loss', alpha=0.7)
    # Validation loss every 10 epochs
    val_epochs = history['epochs'][9::10]  # Every 10th epoch
    val_losses = history['val_loss']  # Every 10th validation loss
    ax1.plot(val_epochs, val_losses, 'o-', label='Validation Loss', alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Total Loss Curves')
    ax1.legend()
    ax1.grid(True)
    
    # Plot reconstruction and KL losses
    # Training losses for all epochs
    ax2.plot(history['epochs'], history['train_recon'], label='Training Recon Loss', alpha=0.7)
    ax2.plot(history['epochs'], history['train_kl'], label='Training KL Loss', alpha=0.7)
    # Validation losses every 10 epochs
    val_recon = history['val_recon']  # Every 10th validation recon loss
    val_kl = history['val_kl']  # Every 10th validation KL loss
    ax2.plot(val_epochs, val_recon, 'o-', label='Validation Recon Loss', alpha=0.7)
    ax2.plot(val_epochs, val_kl, 'o-', label='Validation KL Loss', alpha=0.7)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Reconstruction and KL Loss Curves')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_dir:
        # Extract model config from filename
        filename = os.path.basename(history_file)
        plot_name = filename.replace('loss_history', 'loss_curves').replace('.json', '.png')
        save_path = os.path.join(save_dir, plot_name)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

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
    Convert one-hot [C, D, H, W] → label map [D, H, W] with values {0,1,2,4}

    Args:
        seg_onehot (Tensor or np.ndarray): shape [4, D, H, W]
    
    Returns:
        Tensor or np.ndarray: shape [D, H, W], with labels {0,1,2,4}
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

def show_interactive_comparison(checkpoint_dir, host='0.0.0.0', port=8050):
    """Shows the interactive comparison plot with specified host and port."""
    fig = create_interactive_comparison(checkpoint_dir)
    fig.show(host=host, port=port)

if __name__ == "__main__":
    print("=== Testing label2onehot / onehot2label ===")
    test_onehot_roundtrip()

    print("\n=== Testing center_crop ===")
    test_center_crop()

    print("\n=== Testing slice visualization ===")
    test_visualization()
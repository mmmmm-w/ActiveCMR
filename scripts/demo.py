
import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import io # Add this import

from active_cmr.dataset import CardiacSliceDataset
from active_cmr.model import GenVAE3D_conditional
from active_cmr.pipeline import InferencePipeline
from active_cmr.policy import HybridPolicy
from active_cmr.utils import onehot2label, label2onehot

import monai.metrics as mm

# Load model
z_dim = 128
beta = 0.001
model = GenVAE3D_conditional(
    img_size=128,
    depth=64,
    z_dim=z_dim,
    cond_emb_dim=128,
    n_heads=4
)

# Load trained weights
checkpoint_path = f"checkpoints/cvae/z{z_dim}_beta{beta}/best_cvae_z{z_dim}_beta{beta}.pth"
model.load_state_dict(torch.load(checkpoint_path))
device = torch.device("cuda:4")
model = model.to(device)
model.eval()

#pick a random sample from the dataset
dataset = CardiacSliceDataset(root_dir="Dataset", 
                              state="HR_ED",   
                              volume_size=(64, 128, 128),
                              num_slices=1,
                              direction="axial")
# index = np.random.randint(0, len(dataset))
index = 41
print(f"Scanning sample {index}")
random_sample = dataset[index]
ground_truth_volume = onehot2label(random_sample['volume']) #[64, 128, 128]
gt_onehot = random_sample['volume'].to(device, dtype=torch.bool).unsqueeze(0)

scan_budget = 10
pipeline = InferencePipeline(model=model, volume_size=(64,128,128), num_samples=16, temperature=0.1)
policy = HybridPolicy(volume_size=ground_truth_volume.shape, scan_budget=scan_budget)

z = policy.get_first_position()
gif_frames = []

for scan_idx in range(scan_budget):
    samples, dice_history, next_z, uncertainty_map = pipeline.process_single_scan(ground_truth_volume, policy, z, log=True)
    mean_sample = torch.mean(samples, dim=0)  # [C, D, H, W]
    pred_label = mean_sample.argmax(dim=0).permute(1,2,0).cpu()
    uncertainty = uncertainty_map.permute(1,2,0).cpu()

    pred_onehot = label2onehot(onehot2label(mean_sample)).to(bool).unsqueeze(0)  # add batch dim if needed
    
    dice_scores = mm.compute_dice(
        y_pred=pred_onehot,
        y=gt_onehot,
        include_background=False,  # exclude background if desired
    )
    hd_per_class = mm.compute_hausdorff_distance(pred_onehot, gt_onehot, 
                                                include_background=False, 
                                                directed=False, 
                                                distance_metric='euclidean',
                                                spacing=(2.,1.2,1.2),
                                                percentile=99.9)
    assd = mm.compute_average_surface_distance(
        y_pred=pred_onehot,
        y=gt_onehot,
        include_background=False,  # Set to True if you want to include the background class
        symmetric=True,
        distance_metric='euclidean',
        spacing=(2.,1.2,1.2)  # Options: 'euclidean', 'chessboard', 'taxicab'
    )
    class_names = ['LV', 'MYO', 'RV']

    fig = plt.figure(figsize=(30, 10))
    ax0 = fig.add_subplot(131, projection='3d')
    x, y, z = np.where(ground_truth_volume.permute(1,2,0) > 0.5)
    ax0.scatter(x, y, z, c=ground_truth_volume.permute(1,2,0)[x, y, z],cmap='viridis',alpha=0.1,marker='.')
    ax0.set_xlim(0, 128)
    ax0.set_ylim(0, 128)
    ax0.set_zlim(0, 64)
    ax0.set_xlabel('X')
    ax0.set_ylabel('Y')
    ax0.set_zlabel('Z')
    ax0.set_title('Predicted Segmentation')

    ax1 = fig.add_subplot(132, projection='3d')
    x, y, z = np.where( pred_label > 0.5)
    scatter = ax1.scatter(x, y, z, c=pred_label[x, y, z],cmap='viridis',alpha=0.1,marker='.')
    ax1.set_xlim(0, 128)
    ax1.set_ylim(0, 128)
    ax1.set_zlim(0, 64)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Predicted Segmentation')

    ax2 = fig.add_subplot(133, projection='3d')
    #plot 3d uncertainty with scatter plot
    ax2.scatter(x, y, z, c=uncertainty[x, y, z], cmap='hot', alpha=0.05, marker='.')
    ax2.set_xlim(0, 128)
    ax2.set_ylim(0, 128)
    ax2.set_zlim(0, 64)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Uncertainty')



    metrics_summary_lines = ["Metrics:"]
    for i, name in enumerate(class_names):
        # dice_scores, hd_per_class, assd are likely (1, num_classes) tensors
        dice_val = dice_scores[0, i].item() if dice_scores.ndim > 1 else dice_scores[i].item()
        hd_val = hd_per_class[0, i].item() if hd_per_class.ndim > 1 else hd_per_class[i].item()
        assd_val = assd[0, i].item() if assd.ndim > 1 else assd[i].item()
        metrics_summary_lines.append(f"  {name} - Dice: {dice_val:.3f}, HD: {hd_val:.2f}, ASSD: {assd_val:.2f}")
    metrics_text = "\n".join(metrics_summary_lines)

    fig.text(0.5, 0.02, metrics_text, ha='center', va='bottom', fontsize=18, 
             bbox=dict(boxstyle='round,pad=0.4', fc='lightgoldenrodyellow', alpha=0.8))

    # Adjust layout to make space for titles and metrics text
    # rect=[left, bottom, right, top] in normalized figure coordinates
    plt.tight_layout(rect=[0, 0.07, 1, 0.95])

    buf = io.BytesIO()
    fig.savefig(buf, format='png') # Save figure to a BytesIO buffer
    buf.seek(0) # Reset buffer position to the beginning
    image = imageio.imread(buf) # Read image from buffer
    gif_frames.append(image)
    z = next_z
    plt.close(fig)

imageio.mimsave('segmentation_progress.gif', gif_frames, fps=1) # Adjust fps as needed
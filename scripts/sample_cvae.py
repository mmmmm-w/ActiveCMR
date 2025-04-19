import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
from active_cmr.model import GenVAE3D_conditional
from active_cmr.dataset import CardiacSliceDataset
from active_cmr.utils import visualize_3d_volume_matplotlib, calculate_dice
from torch.utils.data import DataLoader


# Load model
z_dim = 64
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Load a single datapoint
# Replace this with your actual dataset loading code
dataset = CardiacSliceDataset(root_dir="Dataset", 
                              state="HR_ED",   
                              volume_size=(64, 128, 128),
                              num_slices=1,
                              direction="axial")
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
data = next(iter(dataloader))

# Assuming your data contains:
x = data['volume'].to(device)  # [1, 4, 64, 128, 128]
label_maps = data['slices'].to(device)  # [1, M, 4, 128, 128]
metas = data['meta'].to(device)  # [1, M, 5]

# Generate multiple samples
num_samples = 5
temperature = 1.0
with torch.no_grad():
    samples = model.inference(
        label_maps=label_maps,
        metas=metas,
        num_samples=num_samples,
        temperature=temperature
    )

# Calculate dice scores for all samples and average them
all_scores = []
for i in range(num_samples):
    scores = calculate_dice(x[0], samples[i])
    all_scores.append(scores)
    
# Calculate average scores across all samples
avg_scores = {
    'LV': sum(s['LV'] for s in all_scores) / num_samples,
    'MYO': sum(s['MYO'] for s in all_scores) / num_samples,
    'RV': sum(s['RV'] for s in all_scores) / num_samples,
    'average': sum(s['average'] for s in all_scores) / num_samples
}

print(f"Average dice scores across {num_samples} samples:")
print(f"LV={avg_scores['LV']:.3f}, MYO={avg_scores['MYO']:.3f}, RV={avg_scores['RV']:.3f}, Avg={avg_scores['average']:.3f}")


samples = torch.argmax(samples, dim=1)

ground_truth = torch.argmax(x[0].cpu(), dim=0)
# Plot ground truth and generated samples

# Ground truth
visualize_3d_volume_matplotlib(ground_truth.permute(1,2,0).numpy())

# Generated samples
for i in range(num_samples):
    visualize_3d_volume_matplotlib(samples[i].permute(1,2,0).cpu().numpy())
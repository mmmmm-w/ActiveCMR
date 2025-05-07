
import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from active_cmr.dataset import CardiacSliceDataset
from active_cmr.model import GenVAE3D_conditional
from active_cmr.pipeline import InferencePipeline
from active_cmr.policy import RandomPolicy, SequentialPolicy, HybridPolicy, SampleVariancePolicy
from active_cmr.utils import onehot2label, visualize_3d_volume_matplotlib, visualize_scan_and_sample, visualize_scanned_slices, label2onehot

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
device = torch.device("cuda:2")
model = model.to(device)
model.eval()

pipeline = InferencePipeline(model=model, volume_size=(64,128,128), num_samples=16, temperature=0.1)

dataset = CardiacSliceDataset(root_dir="Dataset", 
                              state="HR_ED",   
                              volume_size=(64, 128, 128),
                              num_slices=1,
                              direction="axial")

#pick a random sample from the dataset
index = np.random.randint(0, len(dataset))
index = 988
print(f"Scanning sample {index}")
random_sample = dataset[index]
ground_truth_volume = onehot2label(random_sample['volume']) #[64, 128, 128]
#visualize the ground truth volume
sample, dice_history, scanned_slices = pipeline.run_inference(ground_truth_volume, policy_class=HybridPolicy, scan_budget=10, log=True)
visualize_3d_volume_matplotlib(ground_truth_volume.permute(1,2,0))
visualize_scan_and_sample(ground_truth_volume.permute(1,2,0),scanned_slices)
visualize_scanned_slices(scanned_slices)
visualize_3d_volume_matplotlib(onehot2label(sample.cpu()).permute(1,2,0))

pred_onehot = label2onehot(onehot2label(sample)).to(bool).unsqueeze(0)  # add batch dim if needed
gt_onehot   = random_sample['volume'].to(device, dtype=torch.bool).unsqueeze(0)
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
                                            percentile=100)
assd = mm.compute_average_surface_distance(
    y_pred=pred_onehot,
    y=gt_onehot,
    include_background=False,  # Set to True if you want to include the background class
    symmetric=True,
    distance_metric='euclidean',
    spacing=(2.,1.2,1.2)  # Options: 'euclidean', 'chessboard', 'taxicab'
)
class_names = ['LV', 'MYO', 'RV']
#print metrics of each class
print(f"Dice scores: LV: {dice_scores[0][0]}, MYO: {dice_scores[0][1]}, RV: {dice_scores[0][2]}")
print(f"ASSD: LV: {assd[0][0]}, MYO: {assd[0][1]}, RV: {assd[0][2]}")
print(f"HD95: LV: {hd_per_class[0][0]}, MYO: {hd_per_class[0][1]}, RV: {hd_per_class[0][2]}")
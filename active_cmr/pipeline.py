import torch
import numpy as np
from active_cmr.utils import extract_slice_and_meta, label2onehot
from active_cmr.policy import ScanPolicy

class InferencePipeline:
    def __init__(self, model, volume_size=(64,128,128), slice_size=(128,128), 
                 num_samples=5, temperature=0.8):
        """
        Args:
            model: trained GenVAE3D_conditional model
            policy_class: class for scan policy (e.g., RandomPolicy, UncertaintyPolicy)
            volume_size: target volume size (D,H,W)
            slice_size: size of extracted slices (H,W)
            num_samples: number of samples to generate for uncertainty estimation
            temperature: sampling temperature for generation
        """
        self.model = model
        self.model.eval()
        self.volume_size = volume_size
        self.slice_size = slice_size
        self.num_samples = num_samples
        self.temperature = temperature
        
        # Store scan history
        self.scanned_slices = []
        self.dice_history = []

    def scan(self, volume, z_position):
        """
        Take a scan at specified z position
        
        Args:
            volume: [D, H, W] ground truth volume
            z_position: z coordinate to scan
        Returns:
            tuple: (slice_tensor, meta_tensor)
        """
        # Calculate center point (x,y,z)
        center = np.array([
            z_position,               # z (first dimension in volume)
            self.volume_size[1]//2,  # x (H/2)
            self.volume_size[2]//2   # y (W/2)
        ])
        
        # Normal vector for axial view [1, 0, 0]
        normal = np.array([1, 0, 0])
        
        # Extract slice
        slice_2d, meta = extract_slice_and_meta(
            volume.cpu().numpy(),
            center=center,
            normal=normal,
            slice_shape=self.slice_size
        )
        slice_tensor = label2onehot(torch.from_numpy(slice_2d))
        
        # Convert to tensors
        meta_tensor = torch.tensor([
                *meta['center'],  # x, y, z
                meta['theta'],    # azimuth angle
                meta['phi']       # elevation angle
            ])
        
        return slice_tensor, meta_tensor
 
    
    def calculate_dice(self, samples, ground_truth):
        """Calculate Dice score for each class across samples"""
        class_dice_scores = {1: [], 2: [], 3: []}  # Store dice scores for each class
        for sample in samples:
            # Convert to binary segmentation
            pred = torch.argmax(sample, dim=0)  # [D, H, W]
            gt = torch.argmax(ground_truth, dim=0)  # [D, H, W]
            
            # Calculate Dice for each class
            for c in range(1, 4):  # exclude background
                pred_c = (pred == c)
                gt_c = (gt == c)
                intersection = torch.sum(pred_c & gt_c)
                dice = (2.0 * intersection) / (torch.sum(pred_c) + torch.sum(gt_c) + 1e-6)
                class_dice_scores[c].append(dice.item())
        
        # Calculate mean dice for each class
        mean_dice_scores = {c: np.mean(scores) for c, scores in class_dice_scores.items()}
        
        return mean_dice_scores
    
    def clear_history(self):
        self.scanned_slices = []
        self.dice_history = []
    
    def run_inference(self, volume, policy_class: ScanPolicy, scan_budget: int):
        """Run inference pipeline"""
        self.clear_history()
        device = next(self.model.parameters()).device
        volume_onehot = label2onehot(volume)
        policy = policy_class(self.volume_size, scan_budget)
        
        # First scan
        z = policy.get_first_position()
        slice_data, meta = self.scan(volume, z_position=z)
        self.scanned_slices.append((slice_data, meta))
        policy.update(z)

        while len(self.scanned_slices) <= scan_budget:
            # Generate samples
            slices = torch.stack([s for s, _ in self.scanned_slices])
            metas = torch.stack([m for _, m in self.scanned_slices])
            
            slices = slices.unsqueeze(0).to(device)
            metas = metas.unsqueeze(0).to(device)

            with torch.no_grad():
                samples = self.model.inference(
                    slices, metas,
                    num_samples=self.num_samples,
                    temperature=self.temperature
                )
            
            # Calculate and store dice score
            mean_dice_scores = self.calculate_dice(samples.cpu(), volume_onehot.cpu())
            self.dice_history.append(mean_dice_scores)
            print(
                f"Scan {len(self.scanned_slices)} at z={z}, "
                f"LV: {mean_dice_scores[1]:.3f}, "
                f"MYO: {mean_dice_scores[2]:.3f}, "
                f"RV: {mean_dice_scores[3]:.3f}, "
                f"Avg: {np.mean([mean_dice_scores[1], mean_dice_scores[2], mean_dice_scores[3]]):.3f}"
            )
            # Get next position from policy
            z = policy.get_next_position(samples)
            
            
            # Take new scan
            slice_data, meta = self.scan(volume, z)
            self.scanned_slices.append((slice_data, meta))
            policy.update(z)
        
        return self.dice_history
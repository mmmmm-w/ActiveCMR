import torch
import numpy as np
from active_cmr.utils import extract_slice_and_meta, label2onehot, onehot2label
from active_cmr.policy import ScanPolicy
import monai.metrics as mm
import torch.nn.functional as F

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
    
    def scan_long_axis(self, volume):
        """
        Take a long axis scan (2-chamber view)
        
        Args:
            volume: [D, H, W] ground truth volume
        Returns:
            tuple: (slice_tensor, meta_tensor)
        """
        # Calculate center point (x,y,z)
        center = np.array([
            self.volume_size[0]//2,  # z (center of volume)
            self.volume_size[1]//2,  # x (H/2)
            self.volume_size[2]//2   # y (W/2)
        ])
        
        # Normal vector for long axis view [0, 1, 0] (2-chamber view)
        normal = np.array([0, -1, 0])
        
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
        """Calculate Dice score for each class (1–3) across samples using MONAI"""
        pred_onehots = []

        for sample in samples:
            pred_onehot = label2onehot(onehot2label(sample)).to(bool)  # [C, D, H, W]
            pred_onehots.append(pred_onehot)

        # Stack predictions into [B, C, D, H, W]
        pred_batch = torch.stack(pred_onehots, dim=0)
        gt_batch = ground_truth.unsqueeze(0).expand(len(samples), -1, -1, -1, -1)  # repeat GT for each sample

        # Compute Dice: [B, C]
        dice_scores = mm.compute_dice(y_pred=pred_batch, y=gt_batch, include_background=False)

        mean_dice_scores = {1: torch.mean(dice_scores[:, 0]), 2: torch.mean(dice_scores[:, 1]), 3: torch.mean(dice_scores[:, 2])}
        return mean_dice_scores
    
    def clear_history(self):
        self.scanned_slices = []
        self.dice_history = []
    
    def process_single_scan(self, volume, policy, z, log=False):
        """
        Process a single scan iteration
        
        Args:
            volume: ground truth volume
            policy: scan policy instance
            z: z position to scan
            log: whether to print progress
            
        Returns:
            tuple: (samples, mean_dice_scores, next_z, uncertainty_map)
        """
        device = next(self.model.parameters()).device
        volume_onehot = label2onehot(volume)
        
        # Take scan at current position
        slice_data, meta = self.scan(volume, z_position=z)
        self.scanned_slices.append((slice_data, meta))
        policy.update(z)

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
        
        # Calculate dice score
        mean_dice_scores = self.calculate_dice(samples.to(device), volume_onehot.to(device))
        self.dice_history.append(mean_dice_scores)
        
        if log:
            print(
                f"Scan {len(self.scanned_slices)} at z={z}, "
                f"LV: {mean_dice_scores[1]:.3f}, "
                f"MYO: {mean_dice_scores[2]:.3f}, "
                f"RV: {mean_dice_scores[3]:.3f}, "
                f"Avg: {np.mean([mean_dice_scores[1].cpu(), mean_dice_scores[2].cpu(), mean_dice_scores[3].cpu()]):.3f}"
            )
            
        # Get uncertainty map and next position from policy
        uncertainty_map = policy.calculate_uncertainty(samples)
        next_z = policy.get_next_position(samples)
        
        return samples, mean_dice_scores, next_z, uncertainty_map

    def run_inference(self, volume, policy_class: ScanPolicy, scan_budget: int, long_axis=False, log=False):
        """
        Run inference pipeline
        
        Args:
            volume: ground truth volume
            policy_class: scan policy class
            scan_budget: number of scans to take
            long_axis: whether to take a long axis scan before other scans
            log: whether to print progress
        """
        self.clear_history()
        policy = policy_class(self.volume_size, scan_budget)
        
        # Take long axis scan if requested
        if long_axis:
            if log:
                print("Scan 1 long axis")
            slice_data, meta = self.scan_long_axis(volume)
            self.scanned_slices.append((slice_data, meta))
        
        # First short axis scan
        z = policy.get_first_position()
        samples, _, z, _ = self.process_single_scan(volume, policy, z, log)

        # Continue with remaining scans
        while len(self.scanned_slices) < scan_budget:
            print(len(self.scanned_slices))
            samples, _, z, _ = self.process_single_scan(volume, policy, z, log)
        
        return torch.mean(samples, dim=0), self.dice_history, self.scanned_slices[:-1]
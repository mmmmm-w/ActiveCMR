import torch
import numpy as np
from active_cmr.utils import extract_slice_and_meta, label2onehot
import matplotlib.pyplot as plt

class ScanPolicy:
    """Base class for scan policies"""
    def __init__(self, volume_size):
        self.volume_size = volume_size
        self.scanned_positions = []

    def get_first_position(self):
        """Get the first scan position"""
        return self.volume_size[0] // 2

    def get_next_position(self, samples=None):
        """Get next scan position based on policy"""
        raise NotImplementedError
    
    def update(self, z_position):
        """Update policy state with new scan position"""
        self.scanned_positions.append(z_position)

class RandomPolicy(ScanPolicy):
    def __init__(self, volume_size):
        super().__init__(volume_size)
        self.available_z = set(range(volume_size[0]))

    def get_next_position(self, samples=None):
        # Remove already scanned positions
        for z in self.scanned_positions:
            self.available_z.discard(z)
            
        if not self.available_z:
            print("Warning: All z positions have been sampled!")
            return np.random.randint(0, self.volume_size[0])
            
        return np.random.choice(list(self.available_z))
    
class UncertaintyPolicy(ScanPolicy):
    def __init__(self, volume_size):
        super().__init__(volume_size)

    def calculate_uncertainty(self, samples):
        """Calculate uncertainty from samples"""
        probs = torch.softmax(samples, dim=1)
        mean_probs = torch.mean(probs, dim=0)
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=0)
        
        # Calculate distance penalty
        distance_penalty = torch.ones_like(entropy)
        for z in self.scanned_positions:
            z_dist = torch.arange(self.volume_size[0], device=entropy.device)
            z_penalty = torch.exp((2*(z_dist - z) / (self.volume_size[0]))**2)
            distance_penalty *= z_penalty.view(-1, 1, 1)
        
        entropy_norm = (entropy - entropy.min()) / (entropy.max() - entropy.min() + 1e-10)
        return entropy_norm * distance_penalty

    def get_next_position(self, samples):
        uncertainty = self.calculate_uncertainty(samples)
        z_uncertainty = torch.mean(uncertainty, dim=(1,2))
        return torch.argmax(z_uncertainty).item()

class SequentialPolicy(ScanPolicy):
    """Scan slices sequentially from top to bottom"""
    def __init__(self, volume_size):
        super().__init__(volume_size)
        self.current_z = 0

    def get_next_position(self, samples):
        self.current_z = (self.current_z + self.volume_size[0]//10) % self.volume_size[0]
        return self.current_z
    
class AlternatingPolicy(ScanPolicy):
    """Alternates between top and bottom half of the volume"""
    def __init__(self, volume_size):
        super().__init__(volume_size)
        self.top_half = set(range(0, volume_size[0]//2))
        self.bottom_half = set(range(volume_size[0]//2, volume_size[0]))
        self.use_top = True

    def get_next_position(self, samples):
        if self.use_top:
            z_set = self.top_half
        else:
            z_set = self.bottom_half
            
        # Remove scanned positions
        z_set = z_set - set(self.scanned_positions)
        
        if not z_set:  # If current half is exhausted, switch to other half
            self.use_top = not self.use_top
            return self.get_next_position(samples)
            
        z = np.random.choice(list(z_set))
        self.use_top = not self.use_top
        return z
    
class MaxGradientPolicy(ScanPolicy):
    """Scan areas with highest gradient in generated samples"""
    def __init__(self, volume_size):
        super().__init__(volume_size)

    def get_next_position(self, samples):    
        # Calculate spatial gradients
        gradients = torch.abs(torch.diff(samples.mean(0), dim=0))  # [3, D-1, H, W]
        grad_score = gradients.mean(dim=(0,2,3))  # [D-1]
        
        # Avoid scanning near previous positions
        for z in self.scanned_positions:
            if z > 0:
                grad_score[z-1] *= 0.5
            if z < len(grad_score):
                grad_score[z] *= 0.5
                
        return torch.argmax(grad_score).item()

class InferencePipeline:
    def __init__(self, model, policy_class=UncertaintyPolicy, volume_size=(64,128,128), slice_size=(128,128), 
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
        
        # Initialize policy
        self.policy = policy_class(volume_size)
        
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
        print(f"Class 1 (LV) Dice: {mean_dice_scores[1]:.3f}")
        print(f"Class 2 (Myo) Dice: {mean_dice_scores[2]:.3f}")
        print(f"Class 3 (RV) Dice: {mean_dice_scores[3]:.3f}")
        
        return np.mean([mean_dice_scores[c] for c in range(1, 4)])
    
    def run_inference(self, volume, scan_budget):
        """Run inference pipeline"""
        device = next(self.model.parameters()).device
        volume_onehot = label2onehot(volume)
        
        # First scan
        z = self.policy.get_first_position()
        slice_data, meta = self.scan(volume, z_position=z)
        self.scanned_slices.append((slice_data, meta))
        self.policy.update(z)

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
            print(f"Scanning at z={z}")
            dice = self.calculate_dice(samples.cpu(), volume_onehot.cpu())
            self.dice_history.append(dice)
            print(f"Scan {len(self.scanned_slices)}, Dice Score: {dice:.3f}")
            
            # Get next position from policy
            z = self.policy.get_next_position(samples)
            
            
            # Take new scan
            slice_data, meta = self.scan(volume, z)
            self.scanned_slices.append((slice_data, meta))
            self.policy.update(z)
        
        return self.dice_history
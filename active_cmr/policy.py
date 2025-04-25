import torch
import numpy as np

class ScanPolicy:
    """Base class for scan policies"""
    def __init__(self, volume_size, scan_budget):
        self.volume_size = volume_size
        self.scanned_positions = []
        self.scan_budget = scan_budget

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
    def __init__(self, volume_size, scan_budget):
        super().__init__(volume_size, scan_budget)
        self.available_z = set(range(volume_size[0]))

    def get_first_position(self):
        return np.random.randint(0, self.volume_size[0])

    def get_next_position(self, samples=None):
        # Remove already scanned positions
        for z in self.scanned_positions:
            self.available_z.discard(z)
            
        if not self.available_z:
            print("Warning: All z positions have been sampled!")
            return np.random.randint(0, self.volume_size[0])
            
        return np.random.choice(list(self.available_z))
    
class UncertaintyPolicy(ScanPolicy):
    def __init__(self, volume_size, scan_budget):
        super().__init__(volume_size, scan_budget)

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
    def __init__(self, volume_size, scan_budget):
        super().__init__(volume_size, scan_budget)
        self.current_z = 0

    def get_first_position(self):
        self.current_z = self.volume_size[0]//self.scan_budget
        return self.current_z

    def get_next_position(self, samples):
        self.current_z = (self.current_z + self.volume_size[0]//self.scan_budget) % self.volume_size[0]
        return self.current_z
    
class AlternatingPolicy(ScanPolicy):
    """Alternates between top and bottom half of the volume"""
    def __init__(self, volume_size, scan_budget):
        super().__init__(volume_size, scan_budget)
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
    def __init__(self, volume_size, scan_budget):
        super().__init__(volume_size, scan_budget)

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

class AnatomicalPolicy(ScanPolicy):
    """
    Policy based on typical cardiac anatomy:
    - Focuses on key anatomical landmarks
    - Samples more densely around valve planes and apex
    - Ensures coverage of all cardiac chambers
    """
    def __init__(self, volume_size, scan_budget):
        super().__init__(volume_size, scan_budget)
        # Approximate positions for cardiac landmarks (in normalized z-coordinates)
        self.landmarks = {
            'base': 0.8,    # near the valve plane
            'mid': 0.5,     # mid-ventricle
            'apex': 0.2     # apex of the heart
        }
        self.current_focus = 'mid'
        
    def get_next_position(self, samples=None):
        D = self.volume_size[0]
        
        # First scan mid-ventricle
        if not self.scanned_positions:
            z = int(self.landmarks['mid'] * D)
            return z
            
        # Then alternate between base and apex
        if self.current_focus == 'mid':
            self.current_focus = 'base'
        elif self.current_focus == 'base':
            self.current_focus = 'apex'
        else:
            self.current_focus = 'mid'
            
        target_z = int(self.landmarks[self.current_focus] * D)
        
        # Avoid exact repeats
        if target_z in self.scanned_positions:
            target_z += np.random.randint(-5, 5)
            target_z = np.clip(target_z, 0, D-1)
            
        return target_z

class GradientPhysicsPolicy(ScanPolicy):
    """
    Policy based on physical continuity of cardiac structures:
    - Analyzes structural gradients
    - Ensures smooth transitions between scanned regions
    - Focuses on areas of high anatomical change
    """
    def __init__(self, volume_size, scan_budget):
        super().__init__(volume_size, scan_budget)
        
    def get_next_position(self, samples):
        if samples is None:
            return self.volume_size[0] // 2
            
        # Calculate structural continuity score
        mean_pred = samples.mean(0)  # [4, D, H, W]
        
        # Compute both first and second order gradients
        first_grad = torch.abs(torch.diff(mean_pred, dim=1))  # [4, D-1, H, W]
        second_grad = torch.abs(torch.diff(first_grad, dim=1))  # [4, D-2, H, W]
        
        # Combine gradients for scoring
        grad_score = (first_grad.mean(dim=(0,2,3))[:-1] + 
                     second_grad.mean(dim=(0,2,3))) # [D-2]
        
        # Add continuity penalty - prefer positions between existing scans
        continuity_bonus = torch.zeros_like(grad_score)
        for i in range(len(self.scanned_positions) - 1):
            z1, z2 = sorted(self.scanned_positions[i:i+2])
            if z2 - z1 > 1:  # if there's a gap
                mid_point = (z1 + z2) // 2
                if mid_point < len(continuity_bonus):
                    continuity_bonus[mid_point] += 0.5
                    
        # Combine scores
        final_score = grad_score + continuity_bonus
        
        # Mask out already scanned positions
        for z in self.scanned_positions:
            if z < len(final_score):
                final_score[z] = -float('inf')
                
        return torch.argmax(final_score).item()

class SampleVariancePolicy(ScanPolicy):
    """Policy that measures uncertainty as variance across different samples"""
    def __init__(self, volume_size, scan_budget):
        super().__init__(volume_size, scan_budget)
        self.neighborhood_size = 3  # Size of neighborhood to penalize around scanned positions

    def calculate_uncertainty(self, samples):
        """
        Calculate uncertainty based on variance across samples
        Args:
            samples: [num_samples, 4, D, H, W] model predictions
        """
        # Convert to probabilities
        probs = torch.softmax(samples, dim=1)  # [num_samples, 4, D, H, W]
        
        # Calculate variance across samples for each class and position
        variance = torch.var(probs, dim=0)  # [4, D, H, W]
        
        # Sum variance across classes to get total uncertainty
        total_variance = torch.sum(variance, dim=0)  # [D, H, W]
        
        # Calculate distance penalty with neighborhood consideration
        distance_penalty = torch.ones_like(total_variance)
        for z in self.scanned_positions:
            # Create stronger penalty for immediate neighborhood
            z_dist = torch.arange(self.volume_size[0], device=total_variance.device)
            
            # Hard penalty for immediate neighbors
            neighbor_mask = torch.abs(z_dist - z) <= self.neighborhood_size
            distance_penalty[neighbor_mask] *= 0.1  # Strongly discourage scanning nearby
            
            # Exponential penalty for other positions
            z_penalty = torch.exp((2*(z_dist - z) / (self.volume_size[0]))**2)
            distance_penalty *= z_penalty.view(-1, 1, 1)
        
        # Additional penalty for regions with too many scans
        scan_density = torch.zeros(self.volume_size[0], device=total_variance.device)
        for z in self.scanned_positions:
            # Add gaussian influence for each scan
            z_indices = torch.arange(self.volume_size[0], device=total_variance.device)
            scan_density += torch.exp(-(z_indices - z)**2 / (2 * self.neighborhood_size**2))
        
        # Apply scan density penalty
        density_penalty = 1.0 / (1.0 + scan_density).view(-1, 1, 1)
        
        # Normalize variance and apply penalties
        variance_norm = (total_variance - total_variance.min()) / (total_variance.max() - total_variance.min() + 1e-10)
        return variance_norm * distance_penalty * density_penalty

    def get_next_position(self, samples):
        if samples is None:
            return self.volume_size[0] // 2
            
        uncertainty = self.calculate_uncertainty(samples)
        z_uncertainty = torch.mean(uncertainty, dim=(1,2))  # Average over H,W
        
        # Set uncertainty to -inf for already scanned positions
        for z in self.scanned_positions:
            z_uncertainty[z] = float('-inf')
            
        # Ensure minimum spacing between scans
        for z in self.scanned_positions:
            start = max(0, z - self.neighborhood_size)
            end = min(self.volume_size[0], z + self.neighborhood_size + 1)
            z_uncertainty[start:end] = float('-inf')
        
        return torch.argmax(z_uncertainty).item()
    
class HybridPolicy(ScanPolicy):
    """
    Hybrid policy that combines sequential and uncertainty-based scanning:
    - First half of budget: Sequential scanning for even coverage
    - Second half: Uncertainty-based scanning for refinement
    """
    def __init__(self, volume_size, scan_budget):
        super().__init__(volume_size, scan_budget)
        self.sequential_steps = scan_budget // 2  # First half uses sequential
        self.neighborhood_size = 3
        
    def get_first_position(self):
        """Start with first sequential position"""
        return self.volume_size[0] // (self.sequential_steps + 1)
    
    def get_sequential_position(self):
        """Get next position using sequential strategy"""
        step_size = self.volume_size[0] // (self.sequential_steps + 1)
        next_z = (len(self.scanned_positions) + 1) * step_size
        return next_z
    
    def calculate_uncertainty(self, samples):
        """Calculate uncertainty based on sample variance"""
        # Convert to probabilities
        probs = torch.softmax(samples, dim=1)  # [num_samples, 4, D, H, W]
        
        # Calculate variance across samples
        variance = torch.var(probs, dim=0)  # [4, D, H, W]
        total_variance = torch.sum(variance, dim=0)  # [D, H, W]
        
        # Calculate distance penalty
        distance_penalty = torch.ones_like(total_variance)
        for z in self.scanned_positions:
            z_dist = torch.arange(self.volume_size[0], device=total_variance.device)
            
            # Hard penalty for immediate neighbors
            neighbor_mask = torch.abs(z_dist - z) <= self.neighborhood_size
            distance_penalty[neighbor_mask] *= 0.1
            
            # Exponential penalty for other positions
            z_penalty = torch.exp(((z_dist - z) / (self.volume_size[0]))**2)
            distance_penalty *= z_penalty.view(-1, 1, 1)
        
        # Normalize variance and apply penalty
        variance_norm = (total_variance - total_variance.min()) / (total_variance.max() - total_variance.min() + 1e-10)
        return variance_norm * distance_penalty

    def get_uncertainty_position(self, samples):
        """Get next position using uncertainty strategy"""
        uncertainty = self.calculate_uncertainty(samples)
        z_uncertainty = torch.mean(uncertainty, dim=(1,2))  # Average over H,W
        
        # Mask out already scanned positions and their neighborhoods
        for z in self.scanned_positions:
            start = max(0, z - self.neighborhood_size)
            end = min(self.volume_size[0], z + self.neighborhood_size + 1)
            z_uncertainty[start:end] = float('-inf')
        
        return torch.argmax(z_uncertainty).item()

    def get_next_position(self, samples):
        """
        Choose strategy based on current scan count:
        - First half: Sequential
        - Second half: Uncertainty-based
        """
        current_scan_count = len(self.scanned_positions)
        
        if current_scan_count < self.sequential_steps:
            # Use sequential strategy for first half
            return self.get_sequential_position()
        else:
            # Use uncertainty-based strategy for second half
            return self.get_uncertainty_position(samples)
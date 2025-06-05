import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from .utils import center_crop, label2onehot, get_label_center, crop_around_center, extract_slice_and_meta


class CardiacSliceDataset(Dataset):
    """
    Dataset class for loading cardiac segmentation volumes and extracting 2D slices.
    
    Args:
        root_dir (str): Path to the dataset root directory
        state (str): One of ['HR_ED', 'HR_ES', 'LR_ED', 'LR_ES']
        volume_size (tuple): Desired volume size after cropping (x, y, z)
        num_slices (int): Number of 2D slices to extract per volume
        direction (str): Slice extraction strategy - 'axial' or 'random'
        slice_size (tuple): Size of extracted 2D slices (default: (128, 128))
    """
    def __init__(self, root_dir, state, volume_size, num_slices, 
                 direction='axial', slice_size=(128, 128)):
        assert state in ['HR_ED', 'HR_ES', 'LR_ED', 'LR_ES'], f"Invalid state: {state}"
        assert direction in ['axial', 'random'], f"Invalid direction: {direction}"
        
        self.root_dir = root_dir
        self.state = state
        self.volume_size = volume_size
        self.num_slices = num_slices
        self.direction = direction
        self.slice_size = slice_size
        
        # Get list of all subject directories
        self.subject_dirs = sorted([
            os.path.join(root_dir, name)
            for name in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, name))
        ])

    def _load_volume(self, subject_path):
        """Load and preprocess a single volume."""
        # Load nifti file
        path = os.path.join(subject_path, f"{self.state}.nii.gz")
        img = nib.load(path)
        # Load as (X,Y,Z) and transpose to (Z,X,Y)
        volume = torch.from_numpy(img.get_fdata().astype(np.float32))
        volume = volume.permute(2, 0, 1)  # (X,Y,Z) -> (Z,X,Y)
        
        # Get center of the labeled regions
        center = get_label_center(volume)
        
        # Crop volume around the center
        volume = crop_around_center(volume, self.volume_size, center)

        # Find the bounds of non-zero labels along z-axis (first dimension now)
        mask = volume > 0
        z_indices = torch.where(mask)[0]  # first dimension is Z now
        z_min, z_max = z_indices.min(), z_indices.max()
        
        return volume, center, (z_min, z_max)

    def _get_slice_params(self, volume_shape, num_slices, z_bounds=None):
        """
        Generate slice parameters based on direction strategy.
        
        Args:
            volume_shape: Shape of the volume
            num_slices: Number of slices to generate
            z_bounds: Tuple of (z_min, z_max) for valid z-coordinates
        """
        if self.direction == 'axial':
            # For axial slices, use fixed normal vector [1, 0, 0]
            normals = torch.zeros(num_slices, 3, dtype=torch.float32)
            normals[:, 0] = 1.0
            
            # Generate random z-coordinates within valid bounds
            if z_bounds is not None:
                z_min, z_max = z_bounds
                z_coords = torch.randint(int(z_min), int(z_max) + 1, (num_slices,), dtype=torch.float32)
            else:
                z_coords = torch.randint(0, volume_shape[0], (num_slices,), dtype=torch.float32)
            
            centers = torch.zeros(num_slices, 3, dtype=torch.float32)
            centers[:, 0] = z_coords
            centers[:, 1] = volume_shape[1] // 2
            centers[:, 2] = volume_shape[2] // 2
            
        else:  # random direction
            raise NotImplementedError("Random direction not implemented yet")
        
        return centers, normals

    def __len__(self):
        return len(self.subject_dirs)

    def __getitem__(self, idx):
        subject_path = self.subject_dirs[idx]
        
        # Load and preprocess volume
        volume, volume_center, z_bounds = self._load_volume(subject_path)
        
        # Generate slice parameters
        centers, normals = self._get_slice_params(volume.shape, self.num_slices, z_bounds)
        
        # Extract slices and collect metadata
        slices = []
        metas = []
        
        for center, normal in zip(centers, normals):
            slice_2d, meta = extract_slice_and_meta(
                volume,
                center=center.numpy(),
                normal=normal.numpy(),
                slice_shape=self.slice_size
            )
            
            # Convert slice to one-hot encoding
            slice_2d = label2onehot(torch.from_numpy(slice_2d))
            slices.append(slice_2d)
            
            # Collect position metadata
            pos_meta = torch.tensor([
                *meta['center'],  # x, y, z
                meta['theta'],    # azimuth angle
                meta['phi']       # elevation angle
            ])
            metas.append(pos_meta)
        
        # Stack all slices and metadata
        slices = torch.stack(slices)  # shape: [num_slices, C, H, W]
        metas = torch.stack(metas)    # shape: [num_slices, 5]
        
        # Convert volume to one-hot
        volume = label2onehot(volume)
        
        sample = {
            "volume": volume,          # shape: [C, X, Y, Z]
            "slices": slices,          # shape: [num_slices, C, H, W]
            "meta": metas,             # shape: [num_slices, 5]
        }
        
        return sample

def test_cardiac_slice_dataset(dataset_root):
    """Test the CardiacSliceDataset implementation."""
    dataset = CardiacSliceDataset(
        root_dir=dataset_root,
        state="HR_ED",
        volume_size=(64, 128, 128),
        num_slices=8,
        direction="axial",
        slice_size=(128, 128)
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test first sample
    sample = dataset[0]
    
    # Check shapes
    print(f"Volume shape: {sample['volume'].shape}")
    print(f"Slices shape: {sample['slices'].shape}")
    print(f"Meta shape: {sample['meta'].shape}")
    
    # Verify one-hot encoding
    assert torch.allclose(sample['volume'].sum(dim=0), torch.ones_like(sample['volume'][0]))
    assert torch.allclose(sample['slices'].sum(dim=1), torch.ones_like(sample['slices'][:,0,:,:]))
    
    # Check meta ranges
    meta = sample['meta']
    assert meta[..., :3].min() >= 0, "Negative coordinates found"
    assert (meta[..., 3] >= -180).all() and (meta[..., 3] <= 180).all(), "Invalid theta angle"
    assert (meta[..., 4] >= 0).all() and (meta[..., 4] <= 180).all(), "Invalid phi angle"
    
    print("✅ All tests passed!")

if __name__ == "__main__":
    # Replace this with the path to your dataset root directory
    dataset_root = "Dataset"
    
    if not Path(dataset_root).exists():
        raise FileNotFoundError(f"{dataset_root} not found. Please set the correct path.")
    
    test_cardiac_slice_dataset(dataset_root)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from ActiveCMR.utils import center_crop, label2onehot

class CardiacReconstructionDataset(Dataset):
    def __init__(self, root_dir):
        """
        Args:
            root_dir (str): Path to the root directory containing patient folders (0, 1, 2, ...)
            transform (callable, optional): Optional transform to apply on a sample
        """
        self.root_dir = root_dir
        self.subject_dirs = sorted([
            os.path.join(root_dir, name)
            for name in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, name))
        ])

    def __len__(self):
        return len(self.subject_dirs)

    def __getitem__(self, idx):
        subject_path = self.subject_dirs[idx]
        
        def load_nifti(path):
            img = nib.load(path)
            data = img.get_fdata().astype(np.float32)
            return data

        # Load all four volumes
        # LR_ED = load_nifti(os.path.join(subject_path, "LR_ED.nii.gz"))
        # LR_ES = load_nifti(os.path.join(subject_path, "LR_ES.nii.gz"))
        HR_ED = load_nifti(os.path.join(subject_path, "HR_ED.nii.gz"))
        HR_ES = load_nifti(os.path.join(subject_path, "HR_ES.nii.gz"))

        # LR_ED = center_crop(torch.from_numpy(LR_ED), (128, 128, 12))
        # LR_ES = center_crop(torch.from_numpy(LR_ES), (128, 128, 12))
        HR_ED = center_crop(torch.from_numpy(HR_ED), (128, 128, 64))
        HR_ES = center_crop(torch.from_numpy(HR_ES), (128, 128, 64))

        # LR_ED = label2onehot(LR_ED).permute(0,3,1,2)
        # LR_ES = label2onehot(LR_ES).permute(0,3,1,2)
        HR_ED = label2onehot(HR_ED).permute(0,3,1,2)
        HR_ES = label2onehot(HR_ES).permute(0,3,1,2)

        sample = {
            # "LR_ED": LR_ED,
            # "LR_ES": LR_ES,
            "HR_ED": HR_ED,
            "HR_ES": HR_ES,
        }

        return sample
    
def test_dataset_loading(dataset_root):
    print(f"📁 Testing dataset at: {dataset_root}")
    
    dataset = CardiacReconstructionDataset(dataset_root)
    print(f"Found {len(dataset)} samples.")

    assert len(dataset) > 0, "No samples found in dataset!"

    sample = dataset[0]

    expected_shapes = {
        "HR_ED": (4, 64, 128, 128),
        "HR_ES": (4, 64, 128, 128),
        # "LR_ED": (4, 12, 128, 128),
        # "LR_ES": (4, 12, 128, 128),
    }

    # for key in ["LR_ED", "LR_ES", "HR_ED", "HR_ES"]:
    for key in ["HR_ED", "HR_ES"]:
        tensor = sample[key]
        print(f"{key} shape: {tensor.shape}, dtype: {tensor.dtype}")

        # Check type
        assert isinstance(tensor, torch.Tensor), f"{key} is not a tensor!"

        # Check shape
        expected_shape = expected_shapes[key]
        assert tensor.shape == expected_shape, f"{key} has wrong shape! Expected {expected_shape}"

        # Check one-hot encoding validity
        voxel_sum = tensor.sum(dim=0)  # sum over channel axis
        max_deviation = torch.abs(voxel_sum - 1.0).max().item()
        print(f"{key} max deviation from one-hot: {max_deviation:.4e}")
        assert max_deviation < 1e-3, f"{key} is not one-hot valid!"

    print("✅ All dataset tests passed!")

if __name__ == "__main__":
    # Replace this with the path to your dataset root directory
    dataset_root = "ActiveCMR/Dataset"
    
    if not Path(dataset_root).exists():
        raise FileNotFoundError(f"{dataset_root} not found. Please set the correct path.")
    
    test_dataset_loading(dataset_root)
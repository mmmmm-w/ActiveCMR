import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
from active_cmr.model import GenVAE3D_conditional
from active_cmr.dataset import CardiacSliceDataset
from active_cmr.pipeline import InferencePipeline, RandomInferencePipeline, UncertaintyPolicy, RandomPolicy
import numpy as np
from active_cmr.utils import onehot2label

if __name__ == "__main__":
    # Load model
    z_dim = 64
    beta = 0.001
    scan_budget = 5
    num_samples = 16
    temperature = 0.1
    checkpoint_path = f"checkpoints/cvae/z{z_dim}_beta{beta}/best_cvae_z{z_dim}_beta{beta}.pth"

    model = GenVAE3D_conditional(
        img_size=128,
        depth=64,
        z_dim=z_dim,
        cond_emb_dim=128,
        n_heads=4
    )

    # Load trained weights
    model.load_state_dict(torch.load(checkpoint_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    pipeline = InferencePipeline(model=model, policy_class=UncertaintyPolicy, volume_size=(64,128,128), num_samples=num_samples, temperature=temperature)
    random_pipeline = InferencePipeline(model=model, policy_class=RandomPolicy, volume_size=(64,128,128), num_samples=num_samples, temperature=temperature)

    dataset = CardiacSliceDataset(root_dir="Dataset", 
                                state="HR_ED",   
                                volume_size=(64, 128, 128),
                                num_slices=1,
                                direction="axial")

    #pick a random sample from the dataset
    index = np.random.randint(0, len(dataset))
    print(f"Scanning sample {index}")
    random_sample = dataset[index]
    ground_truth_volume = onehot2label(random_sample['volume']) #[64, 128, 128]

    print("Running active learning pipeline")
    pipeline.run_inference(ground_truth_volume, scan_budget=scan_budget)

    print("Running random sampling pipeline")
    random_pipeline.run_inference(ground_truth_volume, scan_budget=scan_budget)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
from active_cmr.model import GenVAE3D_conditional
from active_cmr.dataset import CardiacSliceDataset
from active_cmr.pipeline import InferencePipeline
from active_cmr.policy import *
import numpy as np
from active_cmr.utils import onehot2label


#sample 1295 is a good sample for testing

if __name__ == "__main__":
    # Load model
    z_dim = 64
    beta = 0.001
    scan_budget = 8
    num_samples = 16
    temperature = 0.5
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

    dataset = CardiacSliceDataset(root_dir="Dataset", 
                                state="HR_ED",   
                                volume_size=(64, 128, 128),
                                num_slices=1,
                                direction="axial")

    #pick a random sample from the dataset
    index = np.random.randint(0, len(dataset))
    index = 1295
    print(f"Scanning sample {index}")
    random_sample = dataset[index]
    ground_truth_volume = onehot2label(random_sample['volume']) #[64, 128, 128]
    pipeline = InferencePipeline(model=model, volume_size=(64,128,128), num_samples=num_samples, temperature=temperature)

    # print("Running active learning pipeline")
    # pipeline.run_inference(ground_truth_volume, policy_class=SampleVariancePolicy, scan_budget=scan_budget, log=True)
    # print("#"*70)

    # print("Running sequential sampling pipeline")
    # pipeline.run_inference(ground_truth_volume, policy_class=SequentialPolicy, scan_budget=scan_budget, log=True)
    # print("#"*70)

    print("Running hybrid sampling pipeline")
    pipeline.run_inference(ground_truth_volume, policy_class=HybridPolicy, scan_budget=scan_budget, log=True)
    print("#"*70)
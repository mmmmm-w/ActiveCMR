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
import json
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

def run_experiment(model, dataset, policies, scan_budget, num_samples, temperature, num_subjects=None):
    """
    Run experiment across multiple subjects and policies
    
    Args:
        model: trained model
        dataset: CardiacSliceDataset
        policies: dict of policy_name: policy_class
        scan_budget: number of scans allowed
        num_samples: number of samples to generate
        temperature: sampling temperature
        num_subjects: number of subjects to test (None for all)
    """
    pipeline = InferencePipeline(
        model=model, 
        volume_size=(64,128,128), 
        num_samples=num_samples, 
        temperature=temperature
    )
    
    # Initialize results dictionary
    results = {
        'metadata': {
            'scan_budget': int(scan_budget),
            'num_samples': int(num_samples),
            'temperature': float(temperature),
            'datetime': datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        },
        'policies': {},
    }
    
    # Number of subjects to process
    if num_subjects is None:
        num_subjects = len(dataset)
    
    test_indices = np.random.choice(len(dataset), num_subjects, replace=False).tolist()
    results['metadata']['test_indices'] = test_indices
    results['metadata']['num_subjects'] = len(test_indices)
    
    # For each policy
    for policy_name, policy_class in policies.items():
        print(f"\nTesting {policy_name} policy...")
        policy_results = []
        
        # For each subject
        for idx in tqdm(test_indices, desc=f"{policy_name}"):
            sample = dataset[idx]
            ground_truth_volume = onehot2label(sample['volume'])
            
            # Run inference
            sample, dice_history, scanned_slices = pipeline.run_inference(
                ground_truth_volume, 
                policy_class=policy_class, 
                scan_budget=scan_budget
            )
            
            # Convert dice history to serializable format
            dice_history_serializable = [
                {k: float(v) for k, v in d.items()}
                for d in dice_history
            ]
            
            # Store results for this subject
            policy_results.append({
                'subject_idx': int(idx),
                'dice_history': dice_history_serializable
            })
        
        # Calculate statistics for this policy
        dice_histories = np.array([[np.mean([d[1], d[2], d[3]]) for d in subj['dice_history']] 
                                 for subj in policy_results])
        
        mean_dice = np.mean(dice_histories, axis=0)
        std_dice = np.std(dice_histories, axis=0)
        
        # Store per-class statistics
        class_stats = {c: {} for c in [1, 2, 3]}  # LV, MYO, RV
        for c in [1, 2, 3]:
            class_histories = np.array([[d[c] for d in subj['dice_history']] 
                                     for subj in policy_results])
            class_stats[c]['mean'] = class_histories.mean(axis=0).tolist()
            class_stats[c]['std'] = class_histories.std(axis=0).tolist()
        
        results['policies'][policy_name] = {
            'mean_dice': mean_dice.tolist(),
            'std_dice': std_dice.tolist(),
            'class_stats': class_stats,
            'individual_results': policy_results
        }
    
    return results

def plot_results(results, save_path=None):
    """Plot comparison of different policies"""
    # Plot overall Dice scores
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    for policy_name, policy_results in results['policies'].items():
        mean_dice = policy_results['mean_dice']
        std_dice = policy_results['std_dice']
        x = range(1, len(mean_dice) + 1)
        
        plt.plot(x, mean_dice, label=policy_name, marker='o')
        plt.fill_between(x, 
                        np.array(mean_dice) - np.array(std_dice),
                        np.array(mean_dice) + np.array(std_dice),
                        alpha=0.2)
    
    plt.xlabel('Number of Scans')
    plt.ylabel('Average Dice Score')
    plt.title('Overall Performance')
    plt.legend()
    plt.grid(True)
    
    # Plot per-class performance for best policy
    plt.subplot(1, 2, 2)
    best_policy = max(results['policies'].items(), 
                     key=lambda x: x[1]['mean_dice'][-1])  
    class_names = ['LV', 'MYO', 'RV']
    
    for c, name in zip([1, 2, 3], class_names):
        mean = best_policy[1]['class_stats'][c]['mean']
        std = best_policy[1]['class_stats'][c]['std']
        x = range(1, len(mean) + 1)
        
        plt.plot(x, mean, label=name, marker='o')
        plt.fill_between(x, 
                        np.array(mean) - np.array(std),
                        np.array(mean) + np.array(std),
                        alpha=0.2)
    
    plt.xlabel('Number of Scans')
    plt.ylabel('Dice Score')
    plt.title(f'Per-class Performance ({best_policy[0]})')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    # Model configuration
    z_dim = 64
    beta = 0.001
    scan_budget = 8
    num_samples = 16
    temperature = 0.5
    num_subjects = 50

    # Define policies to test
    policies = {
        'Sample Variance': SampleVariancePolicy,
        'Sequential': SequentialPolicy,
        'Hybrid': HybridPolicy
    }

    results_dir = "results"

    # Load model
    checkpoint_path = f"checkpoints/cvae/z{z_dim}_beta{beta}/best_cvae_z{z_dim}_beta{beta}.pth"
    model = GenVAE3D_conditional(
        img_size=128,
        depth=64,
        z_dim=z_dim,
        cond_emb_dim=128,
        n_heads=4
    )
    model.load_state_dict(torch.load(checkpoint_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Load dataset
    dataset = CardiacSliceDataset(
        root_dir="Dataset", 
        state="HR_ED",   
        volume_size=(64, 128, 128),
        num_slices=1,
        direction="axial"
    )

    # Run experiment
    results = run_experiment(
        model=model,
        dataset=dataset,
        policies=policies,
        scan_budget=scan_budget,
        num_samples=num_samples,
        temperature=temperature,
        num_subjects=num_subjects  # Set to a number for testing, None for full dataset
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save JSON results
    results_path = os.path.join(results_dir, f"policy_comparison_{timestamp}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")
    
    # Plot and save figures
    plot_path = os.path.join(results_dir, f"policy_comparison_{timestamp}.png")
    plot_results(results, save_path=plot_path)
    print(f"Plot saved to {plot_path}")

    
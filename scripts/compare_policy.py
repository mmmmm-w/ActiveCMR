import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

from active_cmr.model import GenVAE3D_conditional
from active_cmr.dataset import CardiacSliceDataset
from active_cmr.pipeline import InferencePipeline
from active_cmr.policy import *
from active_cmr.utils import onehot2label, label2onehot
import monai.metrics as mm

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
    test_indices = [1118,884,134,1066,414,589,45,628,986,403,133,84,312,720,411,1100,851,53,912,433,129,1024,715,566,364,1243,730,748,765,28,450,185,1217,1309,447,426,558,162,933,1164,875,434,753,365,459,532,64,418,38,108]
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
            pred, dice_history, _ = pipeline.run_inference(
                ground_truth_volume, 
                policy_class=policy_class, 
                scan_budget=scan_budget
            )
            
            # --- NEW: Compute final metrics ---
            # Prepare one-hot tensors for metrics
            pred_onehot = label2onehot(onehot2label(pred)).to(bool).unsqueeze(0)
            gt_onehot = sample['volume'].to(pred_onehot.device, dtype=torch.bool).unsqueeze(0)
            
            # Compute Dice
            dice_scores = mm.compute_dice(
                y_pred=pred_onehot,
                y=gt_onehot,
                include_background=False,
            )
            # Compute HD95
            hd_per_class = mm.compute_hausdorff_distance(
                y_pred=pred_onehot,
                y=gt_onehot,
                include_background=False,
                directed=False,
                distance_metric='euclidean',
                spacing=(2.,1.2,1.2),
                percentile=99.9
            )
            # Compute ASSD
            assd = mm.compute_average_surface_distance(
                y_pred=pred_onehot,
                y=gt_onehot,
                include_background=False,
                symmetric=True,
                distance_metric='euclidean',
                spacing=(2.,1.2,1.2)
            )
            # Store per-class and mean metrics
            dice_list = dice_scores[0].cpu().numpy().tolist()
            hd_list = hd_per_class[0].cpu().numpy().tolist()
            assd_list = assd[0].cpu().numpy().tolist()
            mean_dice = float(np.mean(dice_list))
            mean_hd = float(np.mean(hd_list))
            mean_assd = float(np.mean(assd_list))
            # --- END NEW ---

            # Convert dice history to serializable format
            dice_history_serializable = [
                {k: float(v) for k, v in d.items()}
                for d in dice_history
            ]
            
            # Store results for this subject
            policy_results.append({
                'subject_idx': int(idx),
                'dice_history': dice_history_serializable,
                # --- Store final metrics ---
                'final_dice': dice_list,
                'final_hd': hd_list,
                'final_assd': assd_list,
                'mean_dice': mean_dice,
                'mean_hd': mean_hd,
                'mean_assd': mean_assd,
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
        
        # --- Aggregate final metrics across subjects ---
        final_dices = np.array([subj['final_dice'] for subj in policy_results])
        final_hds = np.array([subj['final_hd'] for subj in policy_results])
        final_assds = np.array([subj['final_assd'] for subj in policy_results])

        mean_dices = final_dices.mean(axis=0).tolist()
        std_dices = final_dices.std(axis=0).tolist()
        mean_hds = final_hds.mean(axis=0).tolist()
        std_hds = final_hds.std(axis=0).tolist()
        mean_assds = final_assds.mean(axis=0).tolist()
        std_assds = final_assds.std(axis=0).tolist()

        mean_dice_overall = float(final_dices.mean())
        std_dice_overall = float(final_dices.std())
        mean_hd_overall = float(final_hds.mean())
        std_hd_overall = float(final_hds.std())
        mean_assd_overall = float(final_assds.mean())
        std_assd_overall = float(final_assds.std())

        results['policies'][policy_name] = {
            'mean_dice': mean_dice.tolist(),
            'std_dice': std_dice.tolist(),
            'class_stats': class_stats,
            # --- Store aggregated final metrics ---
            'final_dice_per_class_mean': mean_dices,
            'final_dice_per_class_std': std_dices,
            'final_hd_per_class_mean': mean_hds,
            'final_hd_per_class_std': std_hds,
            'final_assd_per_class_mean': mean_assds,
            'final_assd_per_class_std': std_assds,
            'final_dice_mean': mean_dice_overall,
            'final_dice_std': std_dice_overall,
            'final_hd_mean': mean_hd_overall,
            'final_hd_std': std_hd_overall,
            'final_assd_mean': mean_assd_overall,
            'final_assd_std': std_assd_overall,
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

def print_summary_chart(results, save_path=None):
    """
    Print a summary table of mean and std for Dice, HD, ASSD for each class and policy.
    Optionally save to CSV if save_path is provided.
    """
    class_names = ['LV', 'MYO', 'RV']
    metrics = ['Dice', 'HD', 'ASSD']
    rows = []
    for policy_name, policy_results in results['policies'].items():
        for i, class_name in enumerate(class_names):
            row = {
                'Policy': policy_name,
                'Class': class_name,
                'Dice Mean': f"{policy_results['final_dice_per_class_mean'][i]:.3f}",
                'Dice Std': f"{policy_results['final_dice_per_class_std'][i]:.3f}",
                'HD Mean': f"{policy_results['final_hd_per_class_mean'][i]:.3f}",
                'HD Std': f"{policy_results['final_hd_per_class_std'][i]:.3f}",
                'ASSD Mean': f"{policy_results['final_assd_per_class_mean'][i]:.3f}",
                'ASSD Std': f"{policy_results['final_assd_per_class_std'][i]:.3f}",
            }
            rows.append(row)
        # Add overall mean/std as a separate row
        row = {
            'Policy': policy_name,
            'Class': 'Mean',
            'Dice Mean': f"{policy_results['final_dice_mean']:.3f}",
            'Dice Std': f"{policy_results['final_dice_std']:.3f}",
            'HD Mean': f"{policy_results['final_hd_mean']:.3f}",
            'HD Std': f"{policy_results['final_hd_std']:.3f}",
            'ASSD Mean': f"{policy_results['final_assd_mean']:.3f}",
            'ASSD Std': f"{policy_results['final_assd_std']:.3f}",
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    print("\n=== Final Metrics Summary ===")
    print(df.to_string(index=False))
    # Optionally, save to CSV
    if save_path is not None:
        df.to_csv(save_path, index=False)
        print(f"\nSaved summary to {save_path}")

if __name__ == "__main__":
    # Model configuration
    z_dim = 128
    beta = 0.0001
    scan_budget = 5
    num_samples = 16
    temperature = 1.0
    num_subjects = 50

    # Define policies to test
    policies = {
        'Sample Variance': SampleVariancePolicy,
        'Sequential': SequentialPolicy,
        'Hybrid': HybridPolicy
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/compare_policy/{timestamp}"

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
    os.makedirs(results_dir, exist_ok=True)
    
    # Save JSON results
    results_path = os.path.join(results_dir, f"policy_comparison.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")
    
    # Plot and save figures
    plot_path = os.path.join(results_dir, f"policy_comparison.png")
    plot_results(results, save_path=plot_path)
    print(f"Plot saved to {plot_path}")

    # --- Print and save summary chart with save_path ---
    summary_csv_path = os.path.join(results_dir, f"policy_comparison_summary.csv")
    print_summary_chart(results, save_path=summary_csv_path)
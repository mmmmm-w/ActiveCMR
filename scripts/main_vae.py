import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import os
import json
from train_vae import train_with_config
from active_cmr.utils import plot_loss_curves

def run_experiments():
    # Base configuration
    base_config = {
        'root_dir': "Dataset",
        'batch_size': 16,
        'learning_rate': 0.0001,
        'epochs': 200,
        'validation_interval': 10,
        'depth': 64,
        'checkpoint_dir': "checkpoints"
    }
    
    # Hyperparameters to test
    z_dims = [64]  #[8, 16, 32, 64, 128]
    betas = [0.001]  #[0, 0.1, 0.01, 0.001]
    
    # Results storage
    results = []
    
    # Create results directory
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Run experiments
    for z_dim in z_dims:
        for beta in betas:
            print(f"\nStarting experiment with z_dim={z_dim}, beta={beta}")
            
            # Update config
            config = base_config.copy()
            config['z_dim'] = z_dim
            config['beta'] = beta
            
            # Train and test
            history, test_loss, test_recon, test_kl = train_with_config(config)
            
            # Store results
            result = {
                'z_dim': z_dim,
                'beta': beta,
                'test_loss': test_loss,
                'test_recon': test_recon,
                'test_kl': test_kl,
                'best_val_loss': min(history['val_loss'])
            }
            results.append(result)
            
            # Plot loss curves
            history_file = f"{config['checkpoint_dir']}/loss_history_z{z_dim}_d{config['depth']}_beta{beta}.json"
            plot_loss_curves(history_file, results_dir)
    
    # Save results summary
    with open(f"{results_dir}/experiment_results.json", 'w') as f:
        json.dump(results, f, indent=4)
    
    # Print results summary
    print("\nExperiment Results Summary:")
    print("z_dim\tbeta\ttest_loss\ttest_recon\ttest_kl\tbest_val_loss")
    print("-" * 70)
    for result in results:
        print(f"{result['z_dim']}\t{result['beta']}\t{result['test_loss']:.4f}\t{result['test_recon']:.4f}\t{result['test_kl']:.4f}\t{result['best_val_loss']:.4f}")
    
    # Find best configuration
    best_result = min(results, key=lambda x: x['test_loss'])
    print(f"\nBest configuration:")
    print(f"z_dim: {best_result['z_dim']}")
    print(f"beta: {best_result['beta']}")
    print(f"Test Loss: {best_result['test_loss']:.4f}")
    print(f"Test Recon Loss: {best_result['test_recon']:.4f}")
    print(f"Test KL Loss: {best_result['test_kl']:.4f}")

if __name__ == "__main__":
    run_experiments() 
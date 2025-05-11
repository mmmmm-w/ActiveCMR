import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import os
import json
from train_cvae import train_with_config
from active_cmr.utils import plot_loss_curves

def run_experiments():
    # Base configuration
    base_config = {
        'root_dir': "Dataset",
        'batch_size': 16,
        'learning_rate': 0.0001,
        'epochs': 250,
        'validation_interval': 10,
        'checkpoint_dir': "checkpoints/cvae",
        'slice_align_weight': 0.0,
    }
    
    # Hyperparameters to test
    z_dims = [16, 32, 64, 128]  # Different latent space dimensions
    betas = [0.0001, 0.001, 0.01]  # Different KL loss weights
    
    # Results storage
    results = []
    
    # Create results directory
    results_dir = "results/cvae"
    os.makedirs(results_dir, exist_ok=True)
    
    # Run experiments
    for z_dim in z_dims:
        for beta in betas:
            print(f"\nStarting experiment with z_dim={z_dim}, beta={beta}")
            
            # Update config
            config = base_config.copy()
            config['z_dim'] = z_dim
            config['beta'] = beta
            
            # Create specific checkpoint directory for this experiment
            config['checkpoint_dir'] = os.path.join(base_config['checkpoint_dir'], f"z{z_dim}_beta{beta}")
            os.makedirs(config['checkpoint_dir'], exist_ok=True)
            
            # Train model
            history = train_with_config(config)
            
            # Store results
            result = {
                'z_dim': z_dim,
                'beta': beta,
                'best_val_loss': min(history['val_loss']),
                'final_train_loss': history['train_loss'][-1],
                'final_train_recon': history['train_recon'][-1],
                'final_train_kl': history['train_kl'][-1],
                'final_val_loss': history['val_loss'][-1],
                'final_val_recon': history['val_recon'][-1],
                'final_val_kl': history['val_kl'][-1]
            }
            results.append(result)
            # Plot loss curves
            history_file = os.path.join(config['checkpoint_dir'], f"loss_history_z{z_dim}_beta{beta}.json")
            plot_loss_curves(history_file, results_dir)
    
    # Save results summary
    with open(os.path.join(results_dir, "experiment_results.json"), 'w') as f:
        json.dump(results, f, indent=4)
    
    # Print results summary
    print("\nExperiment Results Summary:")
    print("z_dim\tbeta\ttrain_loss\tval_loss\tbest_val_loss")
    print("-" * 70)
    for result in results:
        print(f"{result['z_dim']}\t{result['beta']:.4f}\t{result['final_train_loss']:.4f}\t"
              f"{result['final_val_loss']:.4f}\t{result['best_val_loss']:.4f}")
    
    # Find best configuration
    best_result = min(results, key=lambda x: x['best_val_loss'])
    print(f"\nBest configuration:")
    print(f"z_dim: {best_result['z_dim']}")
    print(f"beta: {best_result['beta']}")
    print(f"Best Validation Loss: {best_result['best_val_loss']:.4f}")
    print(f"Final Training Loss: {best_result['final_train_loss']:.4f}")
    print(f"Final Validation Loss: {best_result['final_val_loss']:.4f}")

if __name__ == "__main__":
    run_experiments() 
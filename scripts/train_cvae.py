import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import json
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from tqdm import tqdm

from active_cmr.dataset import CardiacSliceDataset
from active_cmr.model import GenVAE3D_conditional
from active_cmr.utils import plot_loss_curves

def vae_loss(recon_logits, target, mu, logvar, beta=0.001):
    """
    Args:
        recon_logits: [B, 4, D, H, W]
        target: class indices [B, D, H, W]
    """
    recon_loss = F.cross_entropy(recon_logits, target, reduction='mean')
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / target.shape[0]
    total_loss = recon_loss + beta * kl

    return total_loss, recon_loss, kl

def train(model, train_loader, val_loader, optimizer, device, beta=1.0, epochs=200, 
         validation_interval=10, checkpoint_dir="./", z_dim=64):
    model.train()
    best_val_loss = float('inf')
    
    # Initialize loss history
    history = {
        'train_loss': [],
        'train_recon': [],
        'train_kl': [],
        'val_loss': [],
        'val_recon': [],
        'val_kl': [],
        'epochs': [],
        'config': {
            'z_dim': z_dim,
            'beta': beta,
            'learning_rate': optimizer.param_groups[0]['lr']
        }
    }
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        recon_loss_total = 0.0
        kl_loss_total = 0.0
        total_samples = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            volume = batch['volume'].to(device)
            slices = batch['slices'].to(device)  # [B, num_slices, C, H, W]
            meta = batch['meta'].to(device)      # [B, num_slices, 5]s
            
            optimizer.zero_grad()
            recon, mu, logvar = model(volume, slices, meta)
            target = torch.argmax(volume, dim=1)  # [B, D, H, W]
            loss, recon_loss, kl_loss = vae_loss(
                recon, target, mu, logvar, beta
            )
            
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * slices.size(0)
            recon_loss_total += recon_loss.item() * slices.size(0)
            kl_loss_total += kl_loss.item() * slices.size(0)
            total_samples += slices.size(0)

        avg_loss = epoch_loss / total_samples
        avg_recon = recon_loss_total / total_samples
        avg_kl = kl_loss_total / total_samples
        
        # Record training losses
        history['train_loss'].append(avg_loss)
        history['train_recon'].append(avg_recon)
        history['train_kl'].append(avg_kl)
        history['epochs'].append(epoch + 1)
        
        print(f"[Epoch {epoch+1}] Avg Loss: {avg_loss:.4f} | Avg Recon: {avg_recon:.4f} | Avg KL: {avg_kl:.4f}")
        
        # Validation step
        if (epoch + 1) % validation_interval == 0:
            model.eval()
            val_loss = 0.0
            val_recon_loss = 0.0
            val_kl_loss = 0.0
            total_val_samples = 0

            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    volume = batch['volume'].to(device)
                    slices = batch['slices'].to(device)
                    meta = batch['meta'].to(device)
                    
                    B, N, C, H, W = slices.shape
                    
                    recon, mu, logvar = model(volume, slices, meta)
                    target = torch.argmax(volume, dim=1)  # [B, D, H, W]
                    loss, recon_loss, kl_loss = vae_loss(
                        recon, target, mu, logvar, beta
                    )

                    val_loss += loss.item() * slices.size(0)
                    val_recon_loss += recon_loss.item() * slices.size(0)
                    val_kl_loss += kl_loss.item() * slices.size(0)
                    total_val_samples += slices.size(0)

            avg_val_loss = val_loss / total_val_samples
            avg_val_recon = val_recon_loss / total_val_samples
            avg_val_kl = val_kl_loss / total_val_samples
            
            # Record validation losses
            history['val_loss'].append(avg_val_loss)
            history['val_recon'].append(avg_val_recon)
            history['val_kl'].append(avg_val_kl)
            
            print(f"[Epoch {epoch+1}] Val Loss: {avg_val_loss:.4f} | Val Recon: {avg_val_recon:.4f} | Val KL: {avg_val_kl:.4f}")

            # Save the best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), f"{checkpoint_dir}/best_cvae_z{z_dim}_beta{beta}.pth")
                print(f"Best model saved with validation loss: {best_val_loss:.4f}")

    # Save final model and loss history
    torch.save(model.state_dict(), f"{checkpoint_dir}/cvae_z{z_dim}_beta{beta}.pth")
    with open(f"{checkpoint_dir}/loss_history_z{z_dim}_beta{beta}.json", 'w') as f:
        json.dump(history, f)
    
    return history

def test(model, test_loader, device, beta=1.0):
    """
    Test function to evaluate model performance on test set
    
    Args:
        model: The VAE model
        test_loader: DataLoader for test data
        device: Device to run the model on
        beta: Beta parameter for VAE loss
    
    Returns:
        dict: Dictionary containing test metrics
    """
    model.eval()
    test_loss = 0.0
    test_recon_loss = 0.0
    test_kl_loss = 0.0
    total_test_samples = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            volume = batch['volume'].to(device)
            slices = batch['slices'].to(device)
            meta = batch['meta'].to(device)
            
            recon, mu, logvar = model(volume, slices, meta)
            target = torch.argmax(volume, dim=1)
            loss, recon_loss, kl_loss = vae_loss(
                recon, target, mu, logvar, beta
            )

            test_loss += loss.item() * slices.size(0)
            test_recon_loss += recon_loss.item() * slices.size(0)
            test_kl_loss += kl_loss.item() * slices.size(0)
            total_test_samples += slices.size(0)

    avg_test_loss = test_loss / total_test_samples
    avg_test_recon = test_recon_loss / total_test_samples
    avg_test_kl = test_kl_loss / total_test_samples

    test_results = {
        'test_loss': avg_test_loss,
        'test_recon': avg_test_recon,
        'test_kl': avg_test_kl,
    }
    
    print(f"Test Results:")
    print(f"Loss: {avg_test_loss:.4f} | Recon: {avg_test_recon:.4f} | KL: {avg_test_kl:.4f}")
    
    return test_results

def train_with_config(config):
    """Train the model with given configuration."""
    # Extract config
    root_dir = config['root_dir']
    batch_size = config['batch_size']
    beta = config['beta']
    learning_rate = config['learning_rate']
    epochs = config['epochs']
    validation_interval = config['validation_interval']
    z_dim = config['z_dim']
    checkpoint_dir = config['checkpoint_dir']
    slice_type = config['slice_type']
    long_axis_prob = config['long_axis_prob']
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device} with config: z_dim={z_dim}, beta={beta}")
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Dataset & DataLoader
    print("Loading datasets...")
    train_dataset = CardiacSliceDataset(
        root_dir=os.path.join(root_dir, "train"),
        state="HR_ED",
        volume_size=(64, 128, 128),
        num_slices=8,
        direction=slice_type,  # axial or both
        long_axis_prob=long_axis_prob  # 50% chance of including long axis view
    )
    
    val_dataset = CardiacSliceDataset(
        root_dir=os.path.join(root_dir, "val"),
        state="HR_ED",
        volume_size=(64, 128, 128),
        num_slices=8,
        direction=slice_type,
        long_axis_prob=long_axis_prob  # 50% chance of including long axis view
    )
    
    test_dataset = CardiacSliceDataset(
        root_dir=os.path.join(root_dir, "test"),
        state="HR_ED",
        volume_size=(64, 128, 128),
        num_slices=8,
        direction=slice_type,
        long_axis_prob=long_axis_prob  # 50% chance of including long axis view
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=16, 
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=16, 
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=16, 
        drop_last=False
    )

    print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    
    # Create model
    model = GenVAE3D_conditional(
        img_size=128,
        z_dim=z_dim,
        cond_emb_dim=128,
        n_heads=4
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train
    history = train(
        model, train_loader, val_loader, optimizer, device,
        beta=beta, epochs=epochs, validation_interval=validation_interval,
        checkpoint_dir=checkpoint_dir, z_dim=z_dim
    )
    
    # Load the best model for testing
    best_model_path = f"{checkpoint_dir}/best_cvae_z{z_dim}_beta{beta}.pth"
    print(f"\nLoading best model from {best_model_path}")
    model.load_state_dict(torch.load(best_model_path))
    
    # Evaluate best model on test set
    print("\nEvaluating best model on test set...")
    test_results = test(model, test_loader, device, beta=beta)
    
    # Save test results
    history['test_results'] = test_results
    with open(f"{checkpoint_dir}/loss_history_z{z_dim}_beta{beta}.json", 'w') as f:
        json.dump(history, f)
    
    return history

if __name__ == "__main__":
    # Default config
    config = {
        'root_dir': "Dataset",
        'batch_size': 16,
        'beta': 0.001,
        'learning_rate': 0.0001,
        'epochs': 400,
        'validation_interval': 10,
        'z_dim': 128,
        "slice_type": "both",
        "long_axis_prob": 0.5,
        'checkpoint_dir': "checkpoints/cvae_both"
    }
    
    history = train_with_config(config)
    plot_loss_curves(
        os.path.join(config['checkpoint_dir'], f"loss_history_z{config['z_dim']}_beta{config['beta']}.json"),
        save_dir=config['checkpoint_dir']
    ) 
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

def vae_loss(recon_logits, target, mu, logvar, beta=0.001, 
             cond_slices=None, meta=None, slice_align_weight=0.0):
    """
    Args:
        recon_logits: [B, 4, D, H, W]
        target: class indices [B, D, H, W]
        cond_slices: [B, N, C, H, W] (condition slices)
        meta: [B, N, 5] (meta info, meta[:,:,0] is z position)
        slice_align_weight: float, weight for slice alignment loss
    """
    recon_loss = F.cross_entropy(recon_logits, target, reduction='mean')
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / target.shape[0]
    total_loss = recon_loss + beta * kl

    align_loss = torch.tensor(0.0, device=recon_logits.device)
    if cond_slices is not None and meta is not None and slice_align_weight > 0.0:
        B, N, C, H, W = cond_slices.shape
        z_indices = meta[:, :, 0].long()  # [B, N]
        # Prepare indices for advanced indexing
        batch_idx = torch.arange(B, device=recon_logits.device).view(-1, 1).expand(B, N)  # [B, N]
        # Flatten everything
        flat_batch = batch_idx.reshape(-1)      # [B*N]
        flat_z = z_indices.reshape(-1)          # [B*N]
        # Get predicted logits at the correct z for each (b, n)
        # recon_logits: [B, C, D, H, W]
        pred_slices = recon_logits[flat_batch, :, flat_z, :, :]  # [B*N, C, H, W]
        # Get condition slices as class indices
        cond_slices_flat = cond_slices.reshape(-1, C, H, W)      # [B*N, C, H, W]
        if C > 1:
            cond_class = cond_slices_flat.argmax(dim=1)           # [B*N, H, W]
        else:
            cond_class = cond_slices_flat.squeeze(1)              # [B*N, H, W]
        # Cross-entropy expects [N, C, H, W] and [N, H, W]
        align_loss = F.cross_entropy(pred_slices, cond_class, reduction='mean')
        total_loss = total_loss + slice_align_weight * align_loss

    return total_loss, recon_loss, kl, align_loss

def train(model, train_loader, val_loader, optimizer, device, beta=1.0, epochs=200, 
         validation_interval=10, checkpoint_dir="./", z_dim=64, slice_align_weight=0.0):
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
        'train_align': [],
        'val_align': [],
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
        align_loss_total = 0.0
        total_samples = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            volume = batch['volume'].to(device)
            slices = batch['slices'].to(device)  # [B, num_slices, C, H, W]
            meta = batch['meta'].to(device)      # [B, num_slices, 5]s
            
            optimizer.zero_grad()
            recon, mu, logvar = model(volume, slices, meta)
            target = torch.argmax(volume, dim=1)  # [B, D, H, W]
            loss, recon_loss, kl_loss, align_loss = vae_loss(
                recon, target, mu, logvar, beta,
                cond_slices=slices, meta=meta, slice_align_weight=slice_align_weight
            )
            
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * slices.size(0)
            recon_loss_total += recon_loss.item() * slices.size(0)
            kl_loss_total += kl_loss.item() * slices.size(0)
            align_loss_total += align_loss.item() * slices.size(0)
            total_samples += slices.size(0)

        avg_loss = epoch_loss / total_samples
        avg_recon = recon_loss_total / total_samples
        avg_kl = kl_loss_total / total_samples
        avg_align = align_loss_total / total_samples
        
        # Record training losses
        history['train_loss'].append(avg_loss)
        history['train_recon'].append(avg_recon)
        history['train_kl'].append(avg_kl)
        history['train_align'].append(avg_align)
        history['epochs'].append(epoch + 1)
        
        print(f"[Epoch {epoch+1}] Avg Loss: {avg_loss:.4f} | Avg Recon: {avg_recon:.4f} | Avg KL: {avg_kl:.4f} | Avg Align: {avg_align:.4f}")
        
        # Validation step
        if (epoch + 1) % validation_interval == 0:
            model.eval()
            val_loss = 0.0
            val_recon_loss = 0.0
            val_kl_loss = 0.0
            val_align_loss = 0.0
            total_val_samples = 0

            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    volume = batch['volume'].to(device)
                    slices = batch['slices'].to(device)
                    meta = batch['meta'].to(device)
                    
                    B, N, C, H, W = slices.shape
                    
                    recon, mu, logvar = model(volume, slices, meta)
                    target = torch.argmax(volume, dim=1)  # [B, D, H, W]
                    loss, recon_loss, kl_loss, align_loss = vae_loss(
                        recon, target, mu, logvar, beta,
                        cond_slices=slices, meta=meta, slice_align_weight=slice_align_weight
                    )

                    val_loss += loss.item() * slices.size(0)
                    val_recon_loss += recon_loss.item() * slices.size(0)
                    val_kl_loss += kl_loss.item() * slices.size(0)
                    val_align_loss += align_loss.item() * slices.size(0)
                    total_val_samples += slices.size(0)

            avg_val_loss = val_loss / total_val_samples
            avg_val_recon = val_recon_loss / total_val_samples
            avg_val_kl = val_kl_loss / total_val_samples
            avg_val_align = val_align_loss / total_val_samples
            
            # Record validation losses
            history['val_loss'].append(avg_val_loss)
            history['val_recon'].append(avg_val_recon)
            history['val_kl'].append(avg_val_kl)
            history['val_align'].append(avg_val_align)
            
            print(f"[Epoch {epoch+1}] Val Loss: {avg_val_loss:.4f} | Val Recon: {avg_val_recon:.4f} | Val KL: {avg_val_kl:.4f} | Val Align: {avg_val_align:.4f}")

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

def test(model, test_loader, device, beta=1.0, slice_align_weight=0.0):
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
    test_align_loss = 0.0
    total_test_samples = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            volume = batch['volume'].to(device)
            slices = batch['slices'].to(device)
            meta = batch['meta'].to(device)
            
            recon, mu, logvar = model(volume, slices, meta)
            target = torch.argmax(volume, dim=1)
            loss, recon_loss, kl_loss, align_loss = vae_loss(
                recon, target, mu, logvar, beta,
                cond_slices=slices, meta=meta, slice_align_weight=slice_align_weight
            )

            test_loss += loss.item() * slices.size(0)
            test_recon_loss += recon_loss.item() * slices.size(0)
            test_kl_loss += kl_loss.item() * slices.size(0)
            test_align_loss += align_loss.item() * slices.size(0)
            total_test_samples += slices.size(0)

    avg_test_loss = test_loss / total_test_samples
    avg_test_recon = test_recon_loss / total_test_samples
    avg_test_kl = test_kl_loss / total_test_samples
    avg_test_align = test_align_loss / total_test_samples

    test_results = {
        'test_loss': avg_test_loss,
        'test_recon': avg_test_recon,
        'test_kl': avg_test_kl,
        'test_align': avg_test_align
    }
    
    print(f"Test Results:")
    print(f"Loss: {avg_test_loss:.4f} | Recon: {avg_test_recon:.4f} | KL: {avg_test_kl:.4f} | Align: {avg_test_align:.4f}")
    
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
    slice_align_weight = config['slice_align_weight']
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device} with config: z_dim={z_dim}, beta={beta}")
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Dataset & DataLoader
    print("Loading dataset...")
    dataset = CardiacSliceDataset(
        root_dir=root_dir,
        state="HR_ED",
        volume_size=(64, 128, 128),
        num_slices=8,
        direction="axial"
    )
    
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

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

    print(f"Train size: {train_size}, Validation size: {val_size}, Test size: {test_size}")
    
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
        checkpoint_dir=checkpoint_dir, z_dim=z_dim, slice_align_weight=slice_align_weight
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
        'epochs': 300,
        'validation_interval': 10,
        'z_dim': 128,
        'checkpoint_dir': "checkpoints/cvae",
        'slice_align_weight': 0.0
    }
    
    history = train_with_config(config)
    plot_loss_curves(
        os.path.join(config['checkpoint_dir'], f"loss_history_z{config['z_dim']}_beta{config['beta']}.json"),
        save_dir=config['checkpoint_dir']
    ) 
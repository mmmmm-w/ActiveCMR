import torch
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
import os
import json

# === Load your model ===
from model import GenVAE3D  # adjust path accordingly
from dataset import CardiacReconstructionDataset  # your dataset

# === Loss Function ===
def vae_loss(recon_logits, target, mu, logvar, beta=0.001):
    """
    Args:
        recon_logits: [B, 4, D, H, W]
        target: class indices [B, D, H, W]
    """
    recon_loss = F.cross_entropy(recon_logits, target, reduction='mean')
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / target.shape[0]
    return recon_loss + beta * kl, recon_loss, kl

# === Training Loop ===
def train(model, train_loader, val_loader, optimizer, device, beta=1.0, epochs=50, validation_interval=10, checkpoint_dir="./", z_dim=64, depth=64):
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
            'depth': depth,
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
            x = batch["HR_ED"].to(device)  # [B, 4, D, H, W]
            batch_size = x.size(0)
            optimizer.zero_grad()
            recon_logits, mu, logvar = model(x)
            target = torch.argmax(x, dim=1)  # [B, D, H, W]
            loss, recon, kl = vae_loss(recon_logits, target, mu, logvar, beta)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_size
            recon_loss_total += recon.item() * batch_size
            kl_loss_total += kl.item() * batch_size
            total_samples += batch_size

        avg_loss = epoch_loss / total_samples
        avg_recon = recon_loss_total / total_samples
        avg_kl = kl_loss_total / total_samples
        
        # Record training losses
        history['train_loss'].append(avg_loss)
        history['train_recon'].append(avg_recon)
        history['train_kl'].append(avg_kl)
        history['epochs'].append(epoch + 1)
        
        print(f"[Epoch {epoch+1}] Avg Loss: {avg_loss:.4f} | Avg Recon: {avg_recon:.4f} | Avg KL: {avg_kl:.4f}")
        
        # Validation step every n epochs
        if (epoch + 1) % validation_interval == 0:
            model.eval()
            val_loss = 0.0
            val_recon_loss = 0.0
            val_kl_loss = 0.0
            total_val_samples = 0

            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    x = batch["HR_ED"].to(device)  # [B, 4, D, H, W]
                    batch_size = x.size(0)
                    recon_logits, mu, logvar = model(x)
                    target = torch.argmax(x, dim=1)  # [B, D, H, W]
                    loss, recon, kl = vae_loss(recon_logits, target, mu, logvar, beta)

                    val_loss += loss.item() * batch_size
                    val_recon_loss += recon.item() * batch_size
                    val_kl_loss += kl.item() * batch_size
                    total_val_samples += batch_size

            avg_val_loss = val_loss / total_val_samples
            avg_val_recon = val_recon_loss / total_val_samples
            avg_val_kl = val_kl_loss / total_val_samples
            
            # Record validation losses
            history['val_loss'].append(avg_val_loss)
            history['val_recon'].append(avg_val_recon)
            history['val_kl'].append(avg_val_kl)
            
            print(f"[Epoch {epoch+1}] Avg Val Loss: {avg_val_loss:.4f} | Avg Val Recon: {avg_val_recon:.4f} | Avg Val KL: {avg_val_kl:.4f}")

            # Save the best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), f"{checkpoint_dir}/best_z{z_dim}_d{depth}_beta{beta}.pth")
                print(f"Best model saved with validation loss: {best_val_loss:.4f}")

    # Save final model and loss history
    torch.save(model.state_dict(), f"{checkpoint_dir}/beta_vae_z{z_dim}_d{depth}_beta{beta}.pth")
    with open(f"{checkpoint_dir}/loss_history_z{z_dim}_d{depth}_beta{beta}.json", 'w') as f:
        json.dump(history, f)
    print(f"Training complete. Final model and loss history saved to {checkpoint_dir}/")
    return history

def test(model, test_loader, device, beta):
    model.eval()
    test_loss = 0.0
    test_recon_loss = 0.0
    test_kl_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            x = batch["HR_ED"].to(device)  # [B, 4, D, H, W]
            batch_size = x.size(0)
            recon_logits, mu, logvar = model(x)
            target = torch.argmax(x, dim=1)  # [B, D, H, W]
            loss, recon, kl = vae_loss(recon_logits, target, mu, logvar, beta)

            test_loss += loss.item() * batch_size
            test_recon_loss += recon.item() * batch_size
            test_kl_loss += kl.item() * batch_size
            total_samples += batch_size

    avg_test_loss = test_loss / total_samples
    avg_test_recon = test_recon_loss / total_samples
    avg_test_kl = test_kl_loss / total_samples
    print(f"Avg Test Loss: {avg_test_loss:.4f} | Avg Test Recon: {avg_test_recon:.4f} | Avg Test KL: {avg_test_kl:.4f}")
    return avg_test_loss, avg_test_recon, avg_test_kl

def train_with_config(config):
    """
    Train the model with given configuration.
    
    Args:
        config (dict): Dictionary containing training configuration
    """
    # Extract config
    root_dir = config['root_dir']
    batch_size = config['batch_size']
    beta = config['beta']
    learning_rate = config['learning_rate']
    epochs = config['epochs']
    validation_interval = config['validation_interval']
    z_dim = config['z_dim']
    depth = config['depth']
    checkpoint_dir = config['checkpoint_dir']
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device} with config: z_dim={z_dim}, beta={beta}")
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Dataset & DataLoader
    print("Loading dataset...")
    dataset = CardiacReconstructionDataset(root_dir)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16, drop_last=False)

    print(f"Train size: {train_size}, Validation size: {val_size}, Test size: {test_size}")

    # Model & Optimizer
    model = GenVAE3D(z_dim=z_dim, depth=depth).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train
    history = train(model, train_loader, val_loader, optimizer, device, beta=beta, 
                   epochs=epochs, validation_interval=validation_interval, 
                   checkpoint_dir=checkpoint_dir, z_dim=z_dim, depth=depth)

    # Load the best model for testing
    model.load_state_dict(torch.load(f"{checkpoint_dir}/best_z{z_dim}_d{depth}_beta{beta}.pth"))
    print("Testing the best model on the test dataset...")
    test_loss, test_recon, test_kl = test(model, test_loader, device, beta)
    
    return history, test_loss, test_recon, test_kl

if __name__ == "__main__":
    # Default config
    config = {
        'root_dir': "ActiveCMR/Dataset",
        'batch_size': 16,
        'beta': 0.001,
        'learning_rate': 0.0001,
        'epochs': 150,
        'validation_interval': 10,
        'z_dim': 64,
        'depth': 64,
        'checkpoint_dir': "ActiveCMR/checkpoints"
    }
    
    train_with_config(config)
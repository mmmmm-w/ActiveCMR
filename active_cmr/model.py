import torch
from torch import nn

# Flatten layer
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

# UnFlatten layer
class UnFlatten(nn.Module):
    def __init__(self, C, D, H, W):
        super(UnFlatten, self).__init__()
        self.C, self.D, self.H, self.W = C, D, H, W

    def forward(self, input):
        return input.view(input.size(0), self.C, self.D, self.H, self.W)

class ConditionEncoder(nn.Module):
    def __init__(self, label_map_channels=4, emb_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(label_map_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        # self.fc_meta = nn.Linear(5, 64)
        # self.fc_out = nn.Linear(64 * 8 * 8 + 64, emb_dim)
        self.fc_meta = nn.Sequential(nn.Linear(5, 128), nn.ReLU(), nn.Linear(128, 64))
        self.fc_out = nn.Sequential(nn.Linear(64 * 8 * 8 + 64, 128), nn.ReLU(), nn.Linear(128, emb_dim))

    def forward(self, label_maps, metas):
        """
        label_maps: [B, M, C, H, W]
        metas: [B, M, 5]
        Returns: [B, M, emb_dim]
        """
        B, M, C, H, W = label_maps.shape
        label_maps = label_maps.view(B * M, C, H, W)
        metas = metas.view(B * M, 5)

        img_feat = self.conv(label_maps).flatten(1)               # [B*M, 64*8*8]
        meta_feat = self.fc_meta(metas)                           # [B*M, 64]
        cond = self.fc_out(torch.cat([img_feat, meta_feat], -1))  # [B*M, emb_dim]
        return cond.view(B, M, -1)                                # [B, M, emb_dim]

class ConditionAttentionAggregator(nn.Module):
    def __init__(self, emb_dim=128, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=n_heads, batch_first=True)

    def forward(self, cond_embs):
        """
        cond_embs: [B, M, emb_dim]
        Returns: [B, emb_dim]
        """
        attn_out, _ = self.attn(cond_embs, cond_embs, cond_embs)
        return attn_out.mean(dim=1)  # [B, emb_dim]

class ConditionTransformerAggregator(nn.Module):
    def __init__(self, emb_dim=128, n_heads=4, hidden_dim=512, num_layers=2, dropout=0.1):
        super().__init__()
        # Build a Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        # Stack multiple layers
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Final layer norm with a residual connection
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, cond_embs):
        """
        Args:
            cond_embs (torch.Tensor): [B, M, emb_dim] sequence of condition embeddings
        Returns:
            torch.Tensor: [B, emb_dim] aggregated embedding
        """
        # Pass through Transformer
        x = self.transformer(cond_embs)              # [B, M, emb_dim]
        # Residual connection and normalization
        x = self.norm(x + cond_embs)                 # [B, M, emb_dim]
        # Pool over the M dimension
        out = x.mean(dim=1)                          # [B, emb_dim]
        return out

class GenVAE3D_conditional(nn.Module):
    """ cardiac segmentation VAE3D for GenScan """
    def __init__(self, img_size=128, z_dim=64, nf=4, depth=64, cond_emb_dim=128, n_heads=4):
        super(GenVAE3D_conditional, self).__init__()
        # bg - LV - MYO - RV  64x128x128
        # input 4 x 64 x n x n
        self.conv1 = nn.Conv3d(4, nf, kernel_size=4, stride=2, padding=1)
        # size nf x 64/2 x n/2 x n/2
        self.conv2 = nn.Conv3d(nf, nf*2, kernel_size=4, stride=2, padding=1)
        # size nf*2 x 64/4 x n/4 x n/4
        self.conv3 = nn.Conv3d(nf*2, nf*4, kernel_size=4, stride=2, padding=1)
        # size nf*4 x 64/8 x n/8 x n/8
        self.conv4 = nn.Conv3d(nf*4, nf*8, kernel_size=4, stride=2, padding=1)
        # size nf*8 x 64/16 x n/16*n/16

        h_dim = int(nf*8 * depth/16 * img_size/16 * img_size/16)

        self.fc11 = nn.Linear(h_dim, z_dim)
        self.fc12 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(z_dim, h_dim)

        self.deconv1 = nn.ConvTranspose3d(nf*8, nf*4, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose3d(nf*4, nf*2, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose3d(nf*2, nf, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose3d(nf, 4, kernel_size=4, stride=2, padding=1)

        self.encoder = nn.Sequential(
            self.conv1,
            nn.BatchNorm3d(nf),
            nn.LeakyReLU(0.2),
            self.conv2,
            nn.BatchNorm3d(nf*2),
            nn.LeakyReLU(0.2),
            self.conv3,
            nn.BatchNorm3d(nf*4),
            nn.LeakyReLU(0.2),
            self.conv4,
            nn.BatchNorm3d(nf*8),
            nn.ReLU(0.2),
            Flatten()
        )

        self.decoder = nn.Sequential(
            UnFlatten(C=int(nf*8), D=int(depth/16), H=int(img_size/16), W=int(img_size/16)),
            self.deconv1,
            nn.BatchNorm3d(nf*4),
            nn.LeakyReLU(0.2),
            self.deconv2,
            nn.BatchNorm3d(nf*2),
            nn.LeakyReLU(0.2),
            self.deconv3,
            nn.BatchNorm3d(nf),
            nn.LeakyReLU(0.2),
            self.deconv4
            # nn.Softmax(dim=1)
        )

        self.condition_encoder = ConditionEncoder(label_map_channels=4, emb_dim=cond_emb_dim)
        self.condition_attn = ConditionAttentionAggregator(emb_dim=cond_emb_dim, n_heads=n_heads)
        self.fc_cond = nn.Linear(cond_emb_dim, z_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def bottleneck(self, h):
        mu, logvar = self.fc11(h), self.fc12(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        h = self.fc2(z)
        x = self.decoder(h)
        return x

    def get_condition_vector(self, label_maps, metas):
        cond_embs = self.condition_encoder(label_maps, metas)     # [B, M, D]
        cond_vec = self.condition_attn(cond_embs)                 # [B, D]
        return cond_vec

    def forward(self, x, label_maps, metas):
        z, mu, logvar = self.encode(x)                            # [B, z_dim]
        cond_vec = self.get_condition_vector(label_maps, metas)   # [B, D]
        cond_proj = self.fc_cond(cond_vec)                        # [B, z_dim]
        z = z + cond_proj                                         # inject condition into latent
        out = self.decode(z)
        return out, mu, logvar

    def inference(self, label_maps, metas, num_samples=1, temperature=1.0):
        """Generate samples conditioned on the given label maps and metadata.
        
        Args:
            label_maps (torch.Tensor): Conditioning label maps of shape [B, M, C, H, W]
            metas (torch.Tensor): Conditioning metadata of shape [B, M, 5]
            num_samples (int): Number of samples to generate per input condition
            temperature (float): Sampling temperature (higher = more random)
        
        Returns:
            torch.Tensor: Generated samples of shape [B*num_samples, 4, D, H, W]
        """
        B = label_maps.shape[0]
        device = next(self.parameters()).device
        
        # Get condition embedding
        with torch.no_grad():
            cond_vec = self.get_condition_vector(label_maps, metas)    # [B, D]
            cond_proj = self.fc_cond(cond_vec)                        # [B, z_dim]
            
            # Repeat for multiple samples
            cond_proj = cond_proj.repeat_interleave(num_samples, dim=0)  # [B*num_samples, z_dim]
            
            # Sample from prior with temperature
            z = torch.randn(B * num_samples, self.fc11.out_features, device=device) * temperature
            
            # Add condition projection
            z = z + cond_proj
            
            # Decode
            samples = self.decode(z)
        
        return samples

def test_conditional_genvae3d():
    print("🔧 Testing Conditional GenVAE3D...")
    
    # Initialize model
    model = GenVAE3D_conditional(
        img_size=128, 
        depth=64, 
        z_dim=64,
        cond_emb_dim=128,
        n_heads=4
    ).eval()

    # Create dummy inputs
    batch_size = 2
    num_conditions = 3  # Number of conditioning samples per batch
    
    # Input volume: [B, C, D, H, W]
    x = torch.randn(batch_size, 4, 64, 128, 128)
    
    # Conditioning inputs
    label_maps = torch.randn(batch_size, num_conditions, 4, 128, 128)  # [B, M, C, H, W]
    metas = torch.randn(batch_size, num_conditions, 5)                 # [B, M, 5]

    print("🚀 Testing forward pass...")
    with torch.no_grad():
        # Test forward pass
        recon, mu, logvar = model(x, label_maps, metas)
        
        # Test shape assertions
        assert recon.shape == x.shape, f"Output shape mismatch! Got {recon.shape}, expected {x.shape}"
        assert mu.shape == logvar.shape == (batch_size, 64), f"Latent shape mismatch! Got {mu.shape}, expected (batch_size, 64)"
        
        print("✅ Forward pass successful!")
        print(f"✅ Output shape: {recon.shape}")
        print(f"✅ Latent shape: {mu.shape}")

    print("🚀 Testing inference...")
    with torch.no_grad():
        # Test inference with multiple samples
        num_samples = 4
        samples = model.inference(label_maps, metas, num_samples=num_samples, temperature=0.8)
        
        # Test shape assertions
        expected_shape = (batch_size * num_samples, 4, 64, 128, 128)
        assert samples.shape == expected_shape, \
            f"Inference output shape mismatch! Got {samples.shape}, expected {expected_shape}"
        
        print("✅ Inference successful!")
        print(f"✅ Generated samples shape: {samples.shape}")

    print("✅ All tests passed!")

if __name__ == "__main__":
    test_conditional_genvae3d()
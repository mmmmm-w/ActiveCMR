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

# beta-VAE 3D for cardiac segmentation
class GenVAE3D(nn.Module):
    """ cardiac segmentation VAE3D for GenScan """
    def __init__(self, img_size=128, z_dim=64, nf=4, depth=64):
        super(GenVAE3D, self).__init__()
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

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar


def test_genvae3d_forward():
    print("🔧 Initializing GenVAE3D...")
    model = GenVAE3D(img_size=128, depth=64, z_dim=64).eval()  # inference mode

    # Create dummy input: batch size 2, 4-channel one-hot, 64x128x128
    x = torch.randn(2, 4, 64, 128, 128)

    print("🚀 Running forward pass...")
    with torch.no_grad():
        recon, mu, logvar = model(x)

    # === SHAPE CHECKS ===
    assert recon.shape == x.shape, f"Output shape mismatch! Got {recon.shape}, expected {x.shape}"
    assert mu.shape == logvar.shape == (2, 8), "Latent vector shape mismatch"

    print(f"✅ Output shape: {recon.shape}")
    print(f"✅ Latent shape: {mu.shape}, {logvar.shape}")
    print("✅ Forward pass successful!")

if __name__ == "__main__":
    test_genvae3d_forward()
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEncoder(nn.Module):
    """Stride-2 conv blocks. Maps (B, C, H, W) to a flat feature vector."""

    def __init__(self, in_channels, img_h, img_w, filters=(32, 64, 128)):
        super().__init__()
        layers = []
        ch_in = in_channels
        for ch_out in filters:
            layers += [
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(),
            ]
            ch_in = ch_out
        self.conv = nn.Sequential(*layers)
        with torch.no_grad():
            self.feat_shape = tuple(self.conv(torch.zeros(1, in_channels, img_h, img_w)).shape[1:])
        self.flat_dim = int(np.prod(self.feat_shape))

    def forward(self, x):
        return self.conv(x).flatten(start_dim=1)


class ConvDecoder(nn.Module):
    """Mirror of ConvEncoder. Maps a latent vector back to (B, C, H, W) in [0, 1]."""

    def __init__(self, z_dim, out_channels, img_h, img_w, feat_shape, filters=(32, 64, 128)):
        super().__init__()
        self.img_h, self.img_w = img_h, img_w
        self.feat_shape = feat_shape
        self.proj = nn.Linear(z_dim, int(np.prod(feat_shape)))
        rev = list(reversed(filters))
        layers = []
        for ch_in, ch_out in zip(rev, rev[1:]):
            layers += [
                nn.ConvTranspose2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(),
            ]
        layers += [
            nn.ConvTranspose2d(rev[-1], out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        ]
        self.deconv = nn.Sequential(*layers)

    def forward(self, z):
        h = self.proj(z).view(-1, *self.feat_shape)
        return self.deconv(h)[:, :, :self.img_h, :self.img_w]


def reparameterize(mu, log_var):
    return mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)


def reconstruction_loss(x_recon, x):
    """Binary cross-entropy summed over pixels, averaged over images."""
    n = x.shape[0]
    return F.binary_cross_entropy(x_recon.reshape(n, -1), x.reshape(n, -1), reduction="sum") / n


def kl_loss(mu, log_var):
    """KL(N(mu, sigma^2) || N(0, 1)) summed over latent dims, averaged over vectors."""
    mu, log_var = mu.reshape(-1, mu.shape[-1]), log_var.reshape(-1, log_var.shape[-1])
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / mu.shape[0]


class ConvVAE(nn.Module):
    def __init__(self, z_dim, in_channels, img_h, img_w, filters=(32, 64, 128)):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = ConvEncoder(in_channels, img_h, img_w, filters)
        self.fc_mu = nn.Linear(self.encoder.flat_dim, z_dim)
        self.fc_log_var = nn.Linear(self.encoder.flat_dim, z_dim)
        self.decoder = ConvDecoder(z_dim, in_channels, img_h, img_w, self.encoder.feat_shape, filters)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_log_var(h)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        return self.decode(reparameterize(mu, log_var)), mu, log_var

    def compute_loss(self, x, beta=1.0):
        x_recon, mu, log_var = self(x)
        recon = reconstruction_loss(x_recon, x)
        kl = kl_loss(mu, log_var)
        return {"total": recon + beta * kl, "recon": recon, "kl": kl}

    @torch.no_grad()
    def extract_latents(self, images, device, batch_size=256):
        """Return mu for every scalogram in chronological order."""
        self.eval()
        out = []
        for start in range(0, len(images), batch_size):
            x = torch.as_tensor(images[start:start + batch_size]).to(device)
            out.append(self.encode(x)[0].cpu().numpy())
        return np.concatenate(out)

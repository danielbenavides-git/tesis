import numpy as np
import torch
import torch.nn as nn

from .conv_vae import ConvDecoder, ConvEncoder, kl_loss, reconstruction_loss, reparameterize


def smoothness_loss(mu, eps=1e-4):
    """Squared day-to-day change of the latent mean relative to its spread in the batch, averaged over dims.
    Equals about 1 - lag-1 autocorrelation per dim (0 = perfectly smooth, 1 = no persistence), so it does
    not drop when the latent space shrinks."""
    diff = mu[:, 1:] - mu[:, :-1]
    var = mu.reshape(-1, mu.shape[-1]).var(dim=0)
    return (diff.pow(2).mean(dim=(0, 1)) / (2 * var + eps)).mean()


class TemporalVAE(nn.Module):
    """ConvVAE with a causal recurrent layer over consecutive days.

    Each day is encoded by the CNN. A one-direction LSTM (or GRU) reads the day features in order.
    The latent distribution of day t comes from the day's own features concatenated with the
    recurrent state, so it only uses days up to t. The decoder rebuilds each day separately.
    """

    def __init__(self, z_dim, in_channels, img_h, img_w, filters=(32, 64, 128),
                 proj_dim=128, rnn_hidden=64, rnn_type="lstm"):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = ConvEncoder(in_channels, img_h, img_w, filters)
        self.proj = nn.Sequential(nn.Linear(self.encoder.flat_dim, proj_dim), nn.ReLU())
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[rnn_type.lower()]
        self.rnn = rnn_cls(input_size=proj_dim, hidden_size=rnn_hidden, num_layers=1, batch_first=True)
        self.fc_mu = nn.Linear(proj_dim + rnn_hidden, z_dim)
        self.fc_log_var = nn.Linear(proj_dim + rnn_hidden, z_dim)
        self.decoder = ConvDecoder(z_dim, in_channels, img_h, img_w, self.encoder.feat_shape, filters)

    def day_features(self, x):
        """(B, L, C, H, W) -> (B, L, proj_dim)"""
        b, l = x.shape[:2]
        h = self.proj(self.encoder(x.reshape(b * l, *x.shape[2:])))
        return h.view(b, l, -1)

    def latent_params(self, feats):
        h, _ = self.rnn(feats)
        fused = torch.cat([feats, h], dim=-1)
        return self.fc_mu(fused), self.fc_log_var(fused)

    def forward(self, x):
        b, l = x.shape[:2]
        mu, log_var = self.latent_params(self.day_features(x))
        z = reparameterize(mu, log_var)
        x_recon = self.decoder(z.reshape(b * l, -1)).view_as(x)
        return x_recon, mu, log_var

    def compute_loss(self, x, beta=1.0, lam=1.0):
        x_recon, mu, log_var = self(x)
        n_days = x.shape[0] * x.shape[1]
        recon = reconstruction_loss(x_recon.reshape(n_days, *x.shape[2:]), x.reshape(n_days, *x.shape[2:]))
        kl = kl_loss(mu, log_var)
        smooth = smoothness_loss(mu)
        return {"total": recon + beta * kl + lam * smooth, "recon": recon, "kl": kl, "smooth": smooth}

    @torch.no_grad()
    def extract_latents(self, images, device, seq_len=30, batch_size=256):
        """Return mu for every day, using only that day and the seq_len - 1 days before it."""
        self.eval()
        feats = []
        for start in range(0, len(images), batch_size):
            x = torch.as_tensor(images[start:start + batch_size]).to(device)
            feats.append(self.proj(self.encoder(x)))
        feats = torch.cat(feats)
        n = len(feats)
        seq_len = min(seq_len, n)

        h_prefix, _ = self.rnn(feats[:seq_len].unsqueeze(0))
        states = [h_prefix[0, :seq_len - 1]]
        windows = feats.unfold(0, seq_len, 1).permute(0, 2, 1)
        for start in range(0, len(windows), batch_size):
            h, _ = self.rnn(windows[start:start + batch_size])
            states.append(h[:, -1])
        states = torch.cat(states)

        mu = self.fc_mu(torch.cat([feats, states], dim=-1))
        return mu.cpu().numpy()

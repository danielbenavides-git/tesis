"""Check, without training, whether the way the Temporal VAE latents are extracted changes the regimes.

In training, the smoothness term compares consecutive days inside one LSTM pass over a 30-day sequence.
extract_latents gives each day its own pass over the 30-day window that ends on that day, so two consecutive days
come from two different passes. This script loads the trained weights of a run and extracts the latents both ways:
- window: the current extract_latents (each day sees itself and the 29 days before it)
- continuous: one causal pass over the whole series (each day sees every earlier day), so consecutive days share
  the pass, as in training
Each set of latents is clustered like the run (k-means on the latents plus the daily mean and std, fitted on the
training days, regimes numbered by mean price). If the continuous pass switches much less, the smoothness learned
in training is partly lost at extraction.

Run it from the repository root, with a run whose artifacts/model.pt is on this computer:
    python check_extraction.py temporal_vae/kmeans/vae_model/001_real_price_log_kmeans_k2 --k 4
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path.cwd()))
from src import experiments as exp
from src.clustering import fit_clusters, order_regimes, summary_metrics, with_level
from src.data import chronological_split, load_context, load_scalograms
from src.models.temporal_vae import TemporalVAE
from src.runs import nino_episodes, nino_share_in_top

parser = argparse.ArgumentParser()
parser.add_argument("run_id")
parser.add_argument("--k", type=int, default=4)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trained = exp.load_trained(args.run_id, device)
config = trained["config"]
images, metadata = load_scalograms(Path("data/processed/scalograms") / config["scalogram_set"])
context = load_context(Path("data/processed"), metadata)
train_idx, _, _ = chronological_split(len(images), config["train_ratio"], config["val_ratio"])

model = TemporalVAE(config["z_dim"], *images.shape[1:], config["filters"], config["proj_dim"],
                    config["rnn_hidden"], config["rnn_type"]).to(device)
model.load_state_dict(trained["state_dict"])
model.eval()

window = model.extract_latents(images, device, seq_len=config["seq_len"])
print(f"window latents equal the saved latents.npy: {np.allclose(window, trained['latents'], atol=1e-4)}")
with torch.no_grad():
    feats = torch.cat([model.proj(model.encoder(torch.as_tensor(images[i:i + 256]).to(device)))
                       for i in range(0, len(images), 256)])
    h, _ = model.rnn(feats.unsqueeze(0))
    continuous = model.fc_mu(torch.cat([feats, h[0]], dim=-1)).cpu().numpy()

price = context["precio_real_cop_kwh"].to_numpy()
rows = []
for name, latents in (("window (current)", window), ("continuous", continuous)):
    features = with_level(latents, metadata, train_idx)
    labels = order_regimes(fit_clusters(features, train_idx, args.k, "kmeans", config["seed"])[0], price)
    m = summary_metrics(features, labels, seed=config["seed"])
    ctx = context.assign(regime=labels)
    rows.append({"extraction": name, "switches_per_year": round(m["switches_per_year"], 1),
                 "mean_run_days": round(m["mean_run_days"], 1),
                 "top_share_pct": round(100 * (labels == args.k - 1).mean(), 1),
                 **{f"nino_{e}_in_top_pct": round(nino_share_in_top(ctx, years), 1) for e, years in nino_episodes.items()}})
print(pd.DataFrame(rows).to_string(index=False))

import random

import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _run_epoch(model, loader, device, loss_kwargs, optimizer=None):
    model.train(optimizer is not None)
    sums, n = {}, 0
    with torch.set_grad_enabled(optimizer is not None):
        for x, _ in loader:
            x = x.to(device)
            losses = model.compute_loss(x, **loss_kwargs)
            if optimizer is not None:
                optimizer.zero_grad()
                losses["total"].backward()
                optimizer.step()
            for key, value in losses.items():
                sums[key] = sums.get(key, 0.0) + value.item()
            n += 1
    return {key: value / n for key, value in sums.items()}


def train(model, train_loader, val_loader, device, epochs=30, lr=1e-3, beta=1.0, kl_warmup_epochs=0,
          extra_loss_kwargs=None, print_every=5):
    """Train any model exposing compute_loss(x, beta=..., **extra). KL weight ramps linearly over kl_warmup_epochs."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    extra_loss_kwargs = extra_loss_kwargs or {}
    history = []
    for epoch in range(1, epochs + 1):
        beta_epoch = beta * min(1.0, epoch / kl_warmup_epochs) if kl_warmup_epochs > 0 else beta
        loss_kwargs = {"beta": beta_epoch, **extra_loss_kwargs}
        train_losses = _run_epoch(model, train_loader, device, loss_kwargs, optimizer)
        val_losses = _run_epoch(model, val_loader, device, loss_kwargs)
        history.append({"epoch": epoch, "beta": beta_epoch,
                        **{f"train_{k}": v for k, v in train_losses.items()},
                        **{f"val_{k}": v for k, v in val_losses.items()}})
        if epoch == 1 or epoch % print_every == 0 or epoch == epochs:
            parts = "  ".join(f"{k}={v:.2f}" for k, v in train_losses.items())
            print(f"Epoch {epoch:3d}/{epochs}  train: {parts}  val total={val_losses['total']:.2f}")
    return history

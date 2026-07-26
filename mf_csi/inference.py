"""Autoregressive 1-step MeanFlow inference for CSI prediction (Algorithm 4),
with the informative (mu-centered) prior.

One-step generation of the next CSI frame:

    Z, mu = encoder(history)                 # mu ~= E[Y | history] (point estimate)
    H1    ~ N(mu, seed_std^2)                # informative prior (mu=0 -> classic prior)
    u     = generator(H1, Z, r=0, t=1)       # average velocity over [0, 1]
    Y_hat = H1 - u                           # displacement (t-r)*u = 1*u

Because the source is centered on mu, a single 1-NFE draw already lies near the
conditional mean -- good NMSE -- while seed_std keeps a genuine, calibrated
spread across draws (the distributional / uncertainty story).

For the AR rollout, the predicted frame is appended to the history and the
encoder is re-run, extending the horizon frame by frame.
"""

from __future__ import annotations

from typing import Optional, Tuple
import torch


@torch.no_grad()
def predict_next_frame(generator, z: torch.Tensor, mu: Optional[torch.Tensor] = None,
                       seed_std: float = 1.0, step_noise_std: float = 0.0,
                       num_samples: int = 1) -> torch.Tensor:
    """One-step prediction of the next CSI frame from latent Z. Returns [B, 2, Nt, Nc].

    mu (if given) centers the noise seed H^1 ~ N(mu, seed_std^2) -- the informative
    prior. num_samples > 1 averages that many 1-NFE draws to approximate the
    conditional mean (MMSE); with an informative prior num_samples=1 is already
    near-mean. Cost is num_samples network evals (still << a 20-step DDIM sampler)."""
    B, _, Nt, Nc = z.shape
    device = z.device
    r = torch.zeros(B, device=device)
    t = torch.ones(B, device=device)
    acc = torch.zeros(B, 2, Nt, Nc, device=device)
    for _ in range(max(1, num_samples)):
        h1 = torch.randn(B, 2, Nt, Nc, device=device) * seed_std    # noise seed
        if mu is not None:
            h1 = h1 + mu                                            # center on point estimate
        acc = acc + (h1 - generator(h1, z, r, t))                   # 1-NFE endpoint
    h0 = acc / max(1, num_samples)
    if step_noise_std > 0:
        h0 = h0 + torch.randn_like(h0) * step_noise_std
    return h0


@torch.no_grad()
def autoregressive_predict(encoder, generator, past: torch.Tensor, num_future: int,
                           seed_std: float = 1.0, step_noise_std: float = 0.0,
                           num_samples: int = 1) -> torch.Tensor:
    """Roll out `num_future` frames autoregressively.

    past: [B, Np, 2, Nt, Nc] -> returns [B, num_future, 2, Nt, Nc].
    The history window grows as predictions are appended (Eq. 23). The encoder's
    point estimate mu centers each frame's source (informative prior); num_samples
    averages that many 1-NFE draws per frame (conditional-mean estimate)."""
    was_training = (encoder.training, generator.training)
    encoder.eval(); generator.eval()
    history = past
    preds = []
    for _ in range(num_future):
        z, mu = encoder(history, return_mu=True)
        nxt = predict_next_frame(generator, z, mu, seed_std, step_noise_std, num_samples)
        preds.append(nxt)
        history = torch.cat([history, nxt.unsqueeze(1)], dim=1)
    encoder.train(was_training[0]); generator.train(was_training[1])
    return torch.stack(preds, dim=1)


@torch.no_grad()
def ar_convlstm_predict(model, past: torch.Tensor, num_future: int) -> torch.Tensor:
    """Autoregressive rollout of a next-frame ConvLSTM -> [B, num_future, 2, Nt, Nc]."""
    was = model.training
    model.eval()
    history = past
    preds = []
    for _ in range(num_future):
        nxt = model(history)
        preds.append(nxt)
        history = torch.cat([history, nxt.unsqueeze(1)], dim=1)
    model.train(was)
    return torch.stack(preds, dim=1)


@torch.no_grad()
def mu_only_predict(encoder, past: torch.Tensor, num_future: int) -> torch.Tensor:
    """Autoregressive rollout using ONLY the encoder's point estimate mu (no flow).

    This is the regression baseline embedded inside the informative-prior model:
    at each step the prediction IS mu(Z), appended to the history. Comparing this
    to the full-flow autoregressive_predict isolates how much of the NMSE comes
    from the point estimate vs. the generative flow. Returns [B, num_future, 2, Nt, Nc].
    """
    was = encoder.training
    encoder.eval()
    history = past
    preds = []
    for _ in range(num_future):
        _, mu = encoder(history, return_mu=True)
        if mu is None:
            raise ValueError("encoder has no mu head (predict_mu=False); mu-only ablation N/A")
        preds.append(mu)
        history = torch.cat([history, mu.unsqueeze(1)], dim=1)
    encoder.train(was)
    return torch.stack(preds, dim=1)


def nmse(pred: torch.Tensor, true: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalized MSE for CSI sequences.

    pred, true: [B, Nf, 2, Nt, Nc]. Returns (per_step [Nf], overall scalar), both
    linear (not dB). per_step[n] = E_B[ ||H_n - H_hat_n||^2 / ||H_n||^2 ];
    overall = mean over steps (the paper's "average NMSE across prediction steps").
    """
    err = (pred - true).pow(2).flatten(2).sum(dim=-1)            # [B, Nf]
    power = true.pow(2).flatten(2).sum(dim=-1).clamp_min(1e-12)  # [B, Nf]
    per_step = (err / power).mean(dim=0)                         # [Nf]
    overall = per_step.mean()
    return per_step, overall


def nmse_db(value: torch.Tensor) -> torch.Tensor:
    return 10.0 * torch.log10(value.clamp_min(1e-12))

import math
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Utilities
# ---------------------------


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def best_perm_bruteforce(cost: torch.Tensor):
    """
    cost: (K_true, K_hat) tensor
    returns perm p of length K_true mapping true i -> hat p[i]
    """
    Kt, Kh = cost.shape
    assert Kt == Kh, "bruteforce matcher assumes K_true == K_hat"
    K = Kt
    best = None
    best_val = float("inf")
    for p in itertools.permutations(range(K)):
        v = cost[torch.arange(K), torch.tensor(p)].sum().item()
        if v < best_val:
            best_val = v
            best = p
    return torch.tensor(best, dtype=torch.long), best_val


def orthonormal_basis(W: torch.Tensor, eps=1e-8):
    """
    W: (d, r)
    returns Q: (d, r) orthonormal basis of colspace(W) via QR
    """
    # If rank deficient, QR still works but some cols might be unstable.
    Q, _ = torch.linalg.qr(W, mode="reduced")
    return Q


def subspace_similarity(A_true: torch.Tensor, A_hat: torch.Tensor):
    """
    Compare colspaces of A_true, A_hat via principal angles.
    Returns mean cos(angle) across r dims.
    """
    Qt = orthonormal_basis(A_true)
    Qh = orthonormal_basis(A_hat)
    # singular values of Qt^T Qh are cosines of principal angles
    s = torch.linalg.svdvals(Qt.T @ Qh).clamp(0, 1)
    return s.mean().item()


# ---------------------------
# Ground-truth generator
# ---------------------------


@torch.no_grad()
def sample_mixture_pcca(
    N: int,
    K: int,
    dq: int,
    dk: int,
    r: int = None,
    min_rank: int = None,
    max_rank: int = None,
    snr: float = 3.0,
    device="cpu",
    seed=0,
):
    """
    Generate N samples from:
      c ~ Cat(pi)
      z ~ N(0, I_max_rank)
      q = mu_q[c] + A[c] z + eps_q, eps_q ~ N(0, diag(psi_q[c]))
      k = mu_k[c] + B[c] z + eps_k, eps_k ~ N(0, diag(psi_k[c]))
    Each component has an effective latent rank sampled uniformly from
    [min_rank, max_rank]. If min_rank/max_rank are not provided, falls back
    to the legacy fixed-rank behavior with r.
    snr controls factor scale vs noise scale.
    """
    set_seed(seed)

    if min_rank is None and max_rank is None:
        if r is None:
            raise ValueError("Provide `r` or both `min_rank` and `max_rank`.")
        min_rank = int(r)
        max_rank = int(r)
    elif (min_rank is None) != (max_rank is None):
        raise ValueError("`min_rank` and `max_rank` must be provided together.")
    elif r is not None:
        raise ValueError(
            "Provide either `r` (fixed rank) or `min_rank`/`max_rank` (rank range), not both."
        )

    min_rank = int(min_rank)
    max_rank = int(max_rank)
    if min_rank <= 0 or max_rank <= 0:
        raise ValueError("`min_rank` and `max_rank` must be > 0.")
    if min_rank > max_rank:
        raise ValueError("`min_rank` must be <= `max_rank`.")
    if max_rank > min(dq, dk):
        raise ValueError("`max_rank` must be <= min(dq, dk).")

    latent_dim = max_rank
    component_ranks = torch.randint(
        low=min_rank,
        high=max_rank + 1,
        size=(K,),
        device=device,
    )

    # mixture weights
    alpha = torch.ones(K) * 1.0
    pi = torch.distributions.Dirichlet(alpha).sample().to(device)

    # means
    mu_q = torch.randn(K, dq, device=device) * 0.7
    mu_k = torch.randn(K, dk, device=device) * 0.7

    # loadings (random directions + per-factor scales)
    # Make per-component "coupling spectra" somewhat distinct.
    A = torch.randn(K, dq, latent_dim, device=device)
    B = torch.randn(K, dk, latent_dim, device=device)

    # normalize columns
    A = A / (A.norm(dim=1, keepdim=True).clamp_min(1e-6))
    B = B / (B.norm(dim=1, keepdim=True).clamp_min(1e-6))

    # scales: shape (K, latent_dim)
    # larger snr => larger factors relative to noise
    base = torch.linspace(1.0, 0.4, latent_dim, device=device)[None, :].repeat(K, 1)
    scales = snr * base * (0.7 + 0.6 * torch.rand(K, latent_dim, device=device))
    A = A * scales[:, None, :]
    B = B * scales[:, None, :]

    # Zero out trailing columns so each component has its own effective rank.
    rank_mask = (
        torch.arange(latent_dim, device=device)[None, :] < component_ranks[:, None]
    ).to(A.dtype)
    A = A * rank_mask[:, None, :]
    B = B * rank_mask[:, None, :]

    # diagonal noise (per component, per dimension)
    # make noise small-ish relative to factor energy
    psi_q = (0.4 / snr) * (0.8 + 0.4 * torch.rand(K, dq, device=device))
    psi_k = (0.4 / snr) * (0.8 + 0.4 * torch.rand(K, dk, device=device))

    # sample component ids
    cat = torch.distributions.Categorical(pi)
    c = cat.sample((N,)).to(device)  # (N,)

    # sample z and noise
    z = torch.randn(N, latent_dim, device=device)
    Aq = A[c]  # (N, dq, latent_dim)
    Bk = B[c]  # (N, dk, latent_dim)
    q = mu_q[c] + torch.einsum("nr,ndr->nd", z, Aq)
    k = mu_k[c] + torch.einsum("nr,ndr->nd", z, Bk)

    q = q + torch.randn_like(q) * psi_q[c].sqrt()
    k = k + torch.randn_like(k) * psi_k[c].sqrt()

    params = dict(
        pi=pi,
        mu_q=mu_q,
        mu_k=mu_k,
        A=A,
        B=B,
        psi_q=psi_q,
        psi_k=psi_k,
        component_ranks=component_ranks,
        min_rank=min_rank,
        max_rank=max_rank,
    )
    return q, k, c, params


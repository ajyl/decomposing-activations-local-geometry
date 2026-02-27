from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass


class QKMFA(nn.Module):
    def __init__(
        self,
        q_centroids: torch.Tensor,  # (n_components, d_head) initial mu_k
        k_centroids: torch.Tensor,  # (n_components, d_head) initial mu_k
        *,
        rank: int,
        psi_init: float = 1.0,  # initial diagonal unique variance
        psi_per_component: bool = False,  # True => Psi_k per component; False => shared Psi
        scale_init: float = 1.0,  # initial loading scales s_{k,j}
        eps_floor: float = 1e-5,  # numerical floor for positivity / norms
    ):
        super().__init__()
        if q_centroids.ndim != 2:
            raise ValueError("centroids must have shape (n_components, d_head)")
        if k_centroids.ndim != 2:
            raise ValueError("centroids must have shape (n_components, d_head)")

        assert (
            q_centroids.shape == k_centroids.shape
        ), "q_centroids and k_centroids must have the same shape"

        n_components, d_head = q_centroids.shape
        if not (1 <= rank <= d_head):
            raise ValueError("rank must be in [1, d_head]")

        self.n_components, self.d_head, self.rank = n_components, d_head, rank
        self._two_pi_logD = self.d_head * math.log(2.0 * math.pi)
        self._eps = float(eps_floor)

        # Means  (n_components, d_head)
        self.mu_q = nn.Parameter(q_centroids.clone())
        self.mu_k = nn.Parameter(k_centroids.clone())

        # Loadings W_k parameterized as direction * scale
        self.W_q = nn.Parameter(
            torch.randn(n_components, d_head, self.rank, dtype=q_centroids.dtype)
            / math.sqrt(d_head)
        )  # (n_components, d_head, rank)
        self.W_k = nn.Parameter(
            torch.randn(n_components, d_head, self.rank, dtype=k_centroids.dtype)
            / math.sqrt(d_head)
        )  # (n_components, d_head, rank)

        rho_s0 = math.log(math.exp(float(scale_init)) - 1.0)
        self.scale_rho = nn.Parameter(
            torch.full((n_components, self.rank), rho_s0, dtype=q_centroids.dtype)
        )  # (n_components, rank)

        # Diagonal unique variances Psi
        psi_shape = (n_components, d_head) if psi_per_component else (d_head,)
        rho0 = math.log(math.exp(float(psi_init)) - 1.0)
        self.psi_q_rho = nn.Parameter(
            torch.full(psi_shape, rho0, dtype=q_centroids.dtype)
        )
        self.psi_k_rho = nn.Parameter(
            torch.full(psi_shape, rho0, dtype=q_centroids.dtype)
        )
        self.psi_per_component = bool(psi_per_component)

        # Mixture weights (n_components,)
        self.pi_logits = nn.Parameter(
            torch.zeros(n_components, dtype=q_centroids.dtype)
        )

        eye = torch.eye(self.rank, dtype=q_centroids.dtype)
        self.register_buffer(
            "_rot_T", eye.repeat(n_components, 1, 1)
        )  # (n_components, rank, rank)
        self.register_buffer(
            "_rot_inv_Tt", eye.repeat(n_components, 1, 1)
        )  # (n_components, rank, rank)

        self._rotation_on: bool = False

    def _psi(self) -> torch.Tensor:
        psi = F.softplus(self.psi_rho) + self._eps
        if psi.ndim == 1:
            psi = psi[None, :].expand(self.n_components, self.d_head)
        return psi  # (n_components, d_head)

    def _dir_hat(self) -> torch.Tensor:
        d = self.dir_raw
        n = d.norm(dim=1, keepdim=True).clamp_min(self._eps)  # (n_components, 1, rank)
        return d / n

    def _scale(self) -> torch.Tensor:
        return F.softplus(self.scale_rho)

    def _W(self) -> torch.Tensor:
        d_hat = self._dir_hat()  # (n_components, d_head, rank)
        s = self._scale()  # (n_components, rank)
        return d_hat * s[:, None, :]  # (n_components, d_head, rank)

    def _W_rotated(self, W: torch.Tensor) -> torch.Tensor:
        # L = A @ inv(T.T)
        return torch.einsum("kdq,kqp->kdp", W, self._rot_inv_Tt)

    def _maybe_rotate_scores(self, Ez: torch.Tensor, Sz: torch.Tensor):
        if not self._rotation_on:
            return Ez, Sz
        T = self._rot_T  # (n_components,rank,rank)

        # z_rot = z @ T
        Ez_rot = torch.einsum("bkq,kqp->bkp", Ez, T)
        Tt = T.transpose(1, 2)
        Sz_rot = torch.matmul(Tt, torch.matmul(Sz, T))
        return Ez_rot, Sz_rot

    @property
    def W(self) -> torch.Tensor:
        W = self._W()
        return self._W_rotated(W) if self._rotation_on else W

    def _core(self, x: torch.Tensor) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Args:
            x: (B, d_head)
        Returns:
            ll, Ez, Sz, L, v, psi
        """
        B, d_head = x.shape
        if d_head != self.d_head:
            raise ValueError(f"expected input dim {self.d_head}, got {d_head}")

        psi = self._psi()  # (n_components, d_head)
        psi_inv = 1.0 / psi  # (n_components, d_head)
        W = self._W()  # (n_components, d_head, rank)  (unrotated)

        A = W * psi_inv[:, :, None].sqrt()  # (n_components, d_head, rank)
        M = torch.einsum("kdi,kdj->kij", A, A)  # (n_components, rank, rank)
        Iq = torch.eye(self.rank, dtype=W.dtype, device=W.device)
        M = M + Iq[None, :, :]
        L = torch.linalg.cholesky(M)  # (n_components, rank, rank)

        xT_Pinv_x = torch.einsum("bd,kd->bk", x * x, psi_inv)  # (B, n_components)
        xT_Pinv_mu = torch.einsum(
            "bd,kd->bk", x, psi_inv * self.mu
        )  # (B, n_components)
        muT_Pinv_mu = (self.mu * self.mu * psi_inv).sum(dim=-1)  # (n_components,)
        xPsiInvx = (
            xT_Pinv_x - 2.0 * xT_Pinv_mu + muT_Pinv_mu[None, :]
        )  # (B, n_components)

        PinvW = psi_inv[:, :, None] * W  # (n_components, d_head, rank)
        WT_Pinv_x = torch.einsum("bd,kdq->bkq", x, PinvW)  # (B, n_components, rank)
        WT_Pinv_mu = torch.einsum("kd,kdq->kq", self.mu, PinvW)  # (n_components, rank)
        v = WT_Pinv_x - WT_Pinv_mu[None, :, :]  # (B, n_components, rank)

        v_perm = v.permute(1, 2, 0)  # (n_components, rank, B)
        Ez_perm = torch.cholesky_solve(
            v_perm, L, upper=False
        )  # (n_components, rank, B)
        Ez = Ez_perm.permute(2, 0, 1)  # (B, n_components, rank)

        Iq_expand = Iq.expand(self.n_components, self.rank, self.rank).clone()
        Sz = torch.cholesky_solve(
            Iq_expand, L, upper=False
        )  # (n_components, rank, rank)

        logdet_Psi = torch.log(psi).sum(dim=-1)  # (n_components,)
        logdet_M = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(
            -1
        )  # (n_components,)
        logdet_C = logdet_Psi + logdet_M  # (n_components,)

        vMinvv = (v * Ez).sum(dim=-1)  # (B, n_components)
        quad = xPsiInvx - vMinvv  # (B, n_components)

        ll = -0.5 * (
            self.d_head * math.log(2.0 * math.pi) + logdet_C[None, :] + quad
        )  # (B, n_components)
        return ll, Ez, Sz, L, v, psi

    def responsibilities(self, x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        ll, *_ = self._core(x)
        log_pi = F.log_softmax(self.pi_logits, dim=0)[None, :]
        return F.softmax((ll + log_pi) / float(tau), dim=1)

    def log_prob_components(self, x: torch.Tensor) -> torch.Tensor:
        ll, *_ = self._core(x)
        return ll

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        ll, *_ = self._core(x)
        log_pi = F.log_softmax(self.pi_logits, dim=0)  # (n_components,)
        return torch.logsumexp(ll + log_pi[None, :], dim=1)

    def nll(self, x: torch.Tensor) -> torch.Tensor:
        return (-self.log_prob(x)).mean()

    def component_posterior(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _ll, Ez, Sz, *_ = self._core(x)
        Ez, Sz = self._maybe_rotate_scores(Ez, Sz)
        return Ez, Sz

    def reconstruct(
        self, x: torch.Tensor, *, use_mixture_mean: bool = True
    ) -> torch.Tensor:
        ll, Ez, _Sz, _L, _v, _psi = self._core(x)
        # Use rotated view if enabled
        W_eff = self.W
        if self._rotation_on:
            Ez, _ = self._maybe_rotate_scores(Ez, _Sz)
        comp = self.mu[None, :, :] + torch.einsum(
            "kdq,bkq->bkd", W_eff, Ez
        )  # (B,n_components,d_head)
        if not use_mixture_mean:
            return comp
        log_pi = F.log_softmax(self.pi_logits, dim=0)[None, :]
        alpha = F.softmax(ll + log_pi, dim=1)  # (B,n_components)
        return torch.einsum("bk,bkd->bd", alpha, comp)  # (B,d_head)

    def forward(self, x):
        return self.nll(x)


def save_mfa(model: MFA, path: str, *, extra: Optional[Dict[str, Any]] = None) -> None:
    """
    Save an MFA model to disk.
    """
    meta = {
        "n_components": model.n_components,
        "d_head": model.d_head,
        "rank": model.rank,
        "psi_per_component": model.psi_per_component,
        "eps_floor": model._eps,
        "dtype": str(model.mu.dtype),
        "version": 1,
        "rotation_on": bool(getattr(model, "_rotation_on", False)),
    }
    if extra:
        meta["extra"] = extra

    torch.save(
        {
            "state_dict": model.state_dict(),  # includes rotation buffers if present
            "meta": meta,
        },
        path,
    )


def load_mfa(
    path: str,
    *,
    map_location: Optional[str | torch.device] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    strict: bool = True,
) -> MFA:
    ckpt = torch.load(path, map_location=map_location)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state: Dict[str, torch.Tensor] = ckpt["state_dict"]
        meta: Dict[str, Any] = ckpt.get("meta", {}) or {}
    else:
        state = ckpt
        meta = {}

    # Infer shapes
    mu = state["mu"]  # (n_components, d_head)
    dir_raw = state["dir_raw"]  # (n_components, d_head, rank)
    n_components, d_head = mu.shape
    rank = dir_raw.shape[-1]

    psi_rho = state["psi_rho"]  # (n_components, d_head) or (d_head,)
    psi_per_component = bool(
        meta.get(
            "psi_per_component", psi_rho.ndim == 2 and psi_rho.shape[0] == n_components
        )
    )
    eps_floor = float(meta.get("eps_floor", 1e-8))

    centroids = torch.zeros(n_components, d_head, dtype=mu.dtype)
    model = MFA(
        centroids=centroids,
        rank=rank,
        psi_per_component=psi_per_component,
        eps_floor=eps_floor,
    )

    if "_rot_T" not in state or "_rot_inv_Tt" not in state:
        eye = torch.eye(rank, dtype=mu.dtype)
        state.setdefault("_rot_T", eye.repeat(n_components, 1, 1))
        state.setdefault("_rot_inv_Tt", eye.repeat(n_components, 1, 1))

    # Load weights/buffers
    model.load_state_dict(state, strict=strict)

    model._rotation_on = bool(meta.get("rotation_on", False))

    if device is not None:
        model = model.to(device)
    if dtype is not None:
        model = model.to(dtype=dtype)

    return model


@dataclass
class EncodedBatch:
    """
    Encoded representation of a batch against an MFA dictionary.
    """

    coeffs: torch.Tensor  # (B, n_components*(1+rank))
    alpha: torch.Tensor  # (B, n_components) responsibilities
    z: (
        torch.Tensor
    )  # (B, n_components, rank) posterior means z_k aligned with dictionary
    dictionary: (
        torch.Tensor
    )  # (d_head, n_components*(1+rank))  atoms: [mu_k | W_k columns] over k
    recon: torch.Tensor  # (B, d_head) coeffs @ dictionary.T
    index_map: List[Tuple[int, Optional[int]]]


class MFAEncoderDecoder:
    """
    Encoder/decoder for MFA

    """

    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def _current_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        W = self.model.W if hasattr(self.model, "W") else self.model._W()
        mu = self.model.mu
        return W, mu

    @torch.no_grad()
    def build_dictionary(
        self,
    ) -> Tuple[torch.Tensor, List[Tuple[int, Optional[int]]], Optional[torch.Tensor]]:
        W, mu = (
            self._current_params()
        )  # (n_components,d_head,rank), (n_components,d_head)
        n_components, d_head, rank = W.shape
        device, dtype = W.device, W.dtype

        cols = []
        index_map: List[Tuple[int, Optional[int]]] = []
        for k in range(n_components):
            cols.append(mu[k].reshape(d_head, 1))
            index_map.append((k, None))
            cols.append(W[k])
            index_map.extend((k, j) for j in range(rank))

        Dmat = torch.cat(cols, dim=1).to(device=device, dtype=dtype)
        return Dmat, index_map, None

    @torch.no_grad()
    def encode(self, x: torch.Tensor, *, tau: float = 1.0) -> EncodedBatch:
        """
        Encode a batch x into coefficients on the shared dictionary.
        """
        B, d_head = x.shape
        if d_head != self.model.d_head:
            raise ValueError(f"expected input dim {self.model.d_head}, got {d_head}")

        # Responsibilities and posterior means
        alpha = self.model.responsibilities(x, tau=tau)  # (B, n_components)
        Ez, _Sz = self.model.component_posterior(x)  # (B, n_components, rank)

        # Build dictionary
        Dmat, index_map, _ = self.build_dictionary()  # (d_head, n_components*(1+rank))

        # assemble coefficient blocks
        blocks = []
        for k in range(self.model.n_components):
            ak = alpha[:, k : k + 1]  # (B,1)
            zk = Ez[:, k, :]  # (B,rank)
            blocks.append(torch.cat([ak, ak * zk], dim=1))  # (B,1+rank)
        coeffs = torch.cat(blocks, dim=1).to(Dmat.dtype)  # (B, n_components*(1+rank))

        # Decode via single matmul
        recon = (coeffs @ Dmat.T).to(x.dtype)  # (B, d_head)

        return EncodedBatch(
            coeffs=coeffs,
            alpha=alpha,
            z=Ez,
            dictionary=Dmat,
            recon=recon,
            index_map=index_map,
        )

    @torch.no_grad()
    def decode(self, coeffs: torch.Tensor) -> torch.Tensor:
        """
        Decode coefficient matrix back to R^d_head using the current dictionary.
        """
        Dmat, _imap, _ = self.build_dictionary()
        return (coeffs.to(Dmat.dtype) @ Dmat.T).to(Dmat.dtype)

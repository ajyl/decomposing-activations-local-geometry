from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass


class MFA(nn.Module):
    def __init__(
        self,
        centroids: torch.Tensor,  # (K, D) initial mu_k
        *,
        rank: int,  # q
        psi_init: float = 1.0,  # initial diagonal unique variance
        psi_per_component: bool = False,  # True => Psi_k per component; False => shared Psi
        scale_init: float = 1.0,  # initial loading scales s_{k,j}
        eps_floor: float = 1e-5,  # numerical floor for positivity / norms
    ):
        super().__init__()
        if centroids.ndim != 2:
            raise ValueError("centroids must have shape (K, D)")
        K, D = centroids.shape
        if not (1 <= rank <= D):
            raise ValueError("rank must be in [1, D]")

        self.K, self.D, self.q = K, D, int(rank)
        self._two_pi_logD = self.D * math.log(2.0 * math.pi)
        self._eps = float(eps_floor)

        # Means  (K, D)
        self.mu = nn.Parameter(centroids.clone())

        # Loadings W_k parameterized as direction * scale
        self.dir_raw = nn.Parameter(
            torch.randn(K, D, self.q, dtype=centroids.dtype) / math.sqrt(D)
        )  # (K, D, q)
        rho_s0 = math.log(math.exp(float(scale_init)) - 1.0)
        self.scale_rho = nn.Parameter(
            torch.full((K, self.q), rho_s0, dtype=centroids.dtype)
        )  # (K, q)

        # Diagonal unique variances Psi
        psi_shape = (K, D) if psi_per_component else (D,)
        rho0 = math.log(math.exp(float(psi_init)) - 1.0)
        self.psi_rho = nn.Parameter(torch.full(psi_shape, rho0, dtype=centroids.dtype))
        self.psi_per_component = bool(psi_per_component)

        # Mixture weights (K,)
        self.pi_logits = nn.Parameter(torch.zeros(K, dtype=centroids.dtype))

        eye = torch.eye(self.q, dtype=centroids.dtype)
        self.register_buffer("_rot_T", eye.repeat(K, 1, 1))  # (K,q,q)
        self.register_buffer("_rot_inv_Tt", eye.repeat(K, 1, 1))  # (K,q,q)
        self._rotation_on: bool = False
        self._rotation_kind: Optional[str] = None  # 'oblimin' or None
        self._rotation_params: dict = {}

    def _psi(self) -> torch.Tensor:
        psi = F.softplus(self.psi_rho) + self._eps
        if psi.ndim == 1:
            psi = psi[None, :].expand(self.K, self.D)
        return psi  # (K, D)

    def _dir_hat(self) -> torch.Tensor:
        d = self.dir_raw
        n = d.norm(dim=1, keepdim=True).clamp_min(self._eps)  # (K, 1, q)
        return d / n

    def _scale(self) -> torch.Tensor:
        return F.softplus(self.scale_rho)

    def _W(self) -> torch.Tensor:
        d_hat = self._dir_hat()  # (K, D, q)
        s = self._scale()  # (K, q)
        return d_hat * s[:, None, :]  # (K, D, q)

    def _W_rotated(self, W: torch.Tensor) -> torch.Tensor:
        # L = A @ inv(T.T)
        return torch.einsum("kdq,kqp->kdp", W, self._rot_inv_Tt)

    def _maybe_rotate_scores(self, Ez: torch.Tensor, Sz: torch.Tensor):
        if not self._rotation_on:
            return Ez, Sz
        T = self._rot_T  # (K,q,q)

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
            x: (B, D)
        Returns:
            ll, Ez, Sz, L, v, psi
        """
        B, D = x.shape
        if D != self.D:
            raise ValueError(f"expected input dim {self.D}, got {D}")

        psi = self._psi()  # (K, D)
        psi_inv = 1.0 / psi  # (K, D)
        W = self._W()  # (K, D, q)  (unrotated)

        A = W * psi_inv[:, :, None].sqrt()  # (K, D, q)
        M = torch.einsum("kdi,kdj->kij", A, A)  # (K, q, q)
        Iq = torch.eye(self.q, dtype=W.dtype, device=W.device)
        M = M + Iq[None, :, :]
        L = torch.linalg.cholesky(M)  # (K, q, q)

        xT_Pinv_x = torch.einsum("bd,kd->bk", x * x, psi_inv)  # (B, K)
        xT_Pinv_mu = torch.einsum("bd,kd->bk", x, psi_inv * self.mu)  # (B, K)
        muT_Pinv_mu = (self.mu * self.mu * psi_inv).sum(dim=-1)  # (K,)
        xPsiInvx = xT_Pinv_x - 2.0 * xT_Pinv_mu + muT_Pinv_mu[None, :]  # (B, K)

        PinvW = psi_inv[:, :, None] * W  # (K, D, q)
        WT_Pinv_x = torch.einsum("bd,kdq->bkq", x, PinvW)  # (B, K, q)
        WT_Pinv_mu = torch.einsum("kd,kdq->kq", self.mu, PinvW)  # (K, q)
        v = WT_Pinv_x - WT_Pinv_mu[None, :, :]  # (B, K, q)

        v_perm = v.permute(1, 2, 0)  # (K, q, B)
        Ez_perm = torch.cholesky_solve(v_perm, L, upper=False)  # (K, q, B)
        Ez = Ez_perm.permute(2, 0, 1)  # (B, K, q)

        Iq_expand = Iq.expand(self.K, self.q, self.q).clone()
        Sz = torch.cholesky_solve(Iq_expand, L, upper=False)  # (K, q, q)

        logdet_Psi = torch.log(psi).sum(dim=-1)  # (K,)
        logdet_M = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)  # (K,)
        logdet_C = logdet_Psi + logdet_M  # (K,)

        vMinvv = (v * Ez).sum(dim=-1)  # (B, K)
        quad = xPsiInvx - vMinvv  # (B, K)

        ll = -0.5 * (
            self.D * math.log(2.0 * math.pi) + logdet_C[None, :] + quad
        )  # (B, K)
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
        log_pi = F.log_softmax(self.pi_logits, dim=0)  # (K,)
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
        comp = self.mu[None, :, :] + torch.einsum("kdq,bkq->bkd", W_eff, Ez)  # (B,K,D)
        if not use_mixture_mean:
            return comp
        log_pi = F.log_softmax(self.pi_logits, dim=0)[None, :]
        alpha = F.softmax(ll + log_pi, dim=1)  # (B,K)
        return torch.einsum("bk,bkd->bd", alpha, comp)  # (B,D)

    def forward(self, x):
        return self.nll(x)


def save_mfa(model: MFA, path: str, *, extra: Optional[Dict[str, Any]] = None) -> None:
    """
    Save an MFA model to disk.
    """
    meta = {
        "K": model.K,
        "D": model.D,
        "q": model.q,
        "psi_per_component": model.psi_per_component,
        "eps_floor": model._eps,
        "dtype": str(model.mu.dtype),
        "version": 1,
        "rotation_on": bool(getattr(model, "_rotation_on", False)),
        "rotation_kind": getattr(model, "_rotation_kind", None),
        "rotation_params": getattr(model, "_rotation_params", {}),
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
    mu = state["mu"]  # (K, D)
    dir_raw = state["dir_raw"]  # (K, D, q)
    K, D = mu.shape
    q = dir_raw.shape[-1]

    psi_rho = state["psi_rho"]  # (K, D) or (D,)
    psi_per_component = bool(
        meta.get("psi_per_component", psi_rho.ndim == 2 and psi_rho.shape[0] == K)
    )
    eps_floor = float(meta.get("eps_floor", 1e-8))

    centroids = torch.zeros(K, D, dtype=mu.dtype)
    model = MFA(
        centroids=centroids,
        rank=q,
        psi_per_component=psi_per_component,
        eps_floor=eps_floor,
    )

    if "_rot_T" not in state or "_rot_inv_Tt" not in state:
        eye = torch.eye(q, dtype=mu.dtype)
        state.setdefault("_rot_T", eye.repeat(K, 1, 1))
        state.setdefault("_rot_inv_Tt", eye.repeat(K, 1, 1))

    # Load weights/buffers
    model.load_state_dict(state, strict=strict)

    model._rotation_on = bool(meta.get("rotation_on", False))
    model._rotation_kind = meta.get("rotation_kind", None)
    model._rotation_params = meta.get("rotation_params", {})

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

    coeffs: torch.Tensor  # (B, K*(1+q))
    alpha: torch.Tensor  # (B, K) responsibilities
    z: torch.Tensor  # (B, K, q) posterior means z_k aligned with dictionary
    dictionary: torch.Tensor  # (D, K*(1+q))  atoms: [mu_k | W_k columns] over k
    recon: torch.Tensor  # (B, D) coeffs @ dictionary.T
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
        W, mu = self._current_params()  # (K,D,q), (K,D)
        K, D, q = W.shape
        device, dtype = W.device, W.dtype

        cols = []
        index_map: List[Tuple[int, Optional[int]]] = []
        for k in range(K):
            cols.append(mu[k].reshape(D, 1))
            index_map.append((k, None))
            cols.append(W[k])
            index_map.extend((k, j) for j in range(q))

        Dmat = torch.cat(cols, dim=1).to(device=device, dtype=dtype)
        return Dmat, index_map, None

    @torch.no_grad()
    def encode(self, x: torch.Tensor, *, tau: float = 1.0) -> EncodedBatch:
        """
        Encode a batch x into coefficients on the shared dictionary.
        """
        B, D = x.shape
        if D != self.model.D:
            raise ValueError(f"expected input dim {self.model.D}, got {D}")

        # Responsibilities and posterior means
        alpha = self.model.responsibilities(x, tau=tau)  # (B, K)
        Ez, _Sz = self.model.component_posterior(x)  # (B, K, q)

        # Build dictionary
        Dmat, index_map, _ = self.build_dictionary()  # (D, K*(1+q))

        # assemble coefficient blocks
        blocks = []
        for k in range(self.model.K):
            ak = alpha[:, k : k + 1]  # (B,1)
            zk = Ez[:, k, :]  # (B,q)
            blocks.append(torch.cat([ak, ak * zk], dim=1))  # (B,1+q)
        coeffs = torch.cat(blocks, dim=1).to(Dmat.dtype)  # (B, K*(1+q))

        # Decode via single matmul
        recon = (coeffs @ Dmat.T).to(x.dtype)  # (B, D)

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
        Decode coefficient matrix back to R^D using the current dictionary.
        """
        Dmat, _imap, _ = self.build_dictionary()
        return (coeffs.to(Dmat.dtype) @ Dmat.T).to(Dmat.dtype)

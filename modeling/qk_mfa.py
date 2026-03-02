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
        self.A_param = nn.Parameter(
            torch.randn(n_components, d_head, self.rank, dtype=q_centroids.dtype)
            / math.sqrt(d_head)
        )  # (n_components, d_head, rank)
        self.B_param = nn.Parameter(
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

    def _psi_q(self) -> torch.Tensor:
        psi = F.softplus(self.psi_q_rho) + self._eps
        if psi.ndim == 1:
            psi = psi[None, :].expand(self.n_components, self.d_head)
        return psi  # (n_components, d_head)

    def _psi_k(self) -> torch.Tensor:
        psi = F.softplus(self.psi_k_rho) + self._eps
        if psi.ndim == 1:
            psi = psi[None, :].expand(self.n_components, self.d_head)
        return psi  # (n_components, d_head)

    def _W_hat(self, weights, dim) -> torch.Tensor:
        n = weights.norm(dim=1, keepdim=True).clamp_min(
            self._eps
        )  # (n_components, 1, rank)
        return weights / n

    def _scale(self) -> torch.Tensor:
        return F.softplus(self.scale_rho)

    def _A_B(self):
        s = self._scale()
        A = self._W_hat(self.A_param, self.d_head) * s[:, None, :]
        B = self._W_hat(self.B_param, self.d_head) * s[:, None, :]
        return A, B

    def _W_rotated(self, W: torch.Tensor) -> torch.Tensor:
        # L = A @ inv(T.T)
        return torch.einsum("kdq,kqp->kdp", W, self._rot_inv_Tt)

    # def _maybe_rotate_scores(self, Ez: torch.Tensor, Sz: torch.Tensor):
    #    if not self._rotation_on:
    #        return Ez, Sz
    #    T = self._rot_T  # (n_components,rank,rank)

    #    # z_rot = z @ T
    #    Ez_rot = torch.einsum("bkq,kqp->bkp", Ez, T)
    #    Tt = T.transpose(1, 2)
    #    Sz_rot = torch.matmul(Tt, torch.matmul(Sz, T))
    #    return Ez_rot, Sz_rot

    @property
    def A_B(self) -> torch.Tensor:
        A, B = self._A_B()
        if self._rotation_on:
            A = self._W_rotated(A)
            B = self._W_rotated(B)
        return A, B

    def _core(self, _q, k):
        """
        Args:
            q: (B, d_head)
            k: (B, d_head)
        Returns:
            ll: (B, n_components),
            Ez: (B, n_components, ranks)
            Sz: (n_components, ranks, ranks)
            L: (n_components, ranks, ranks)
            v: (B, n_components, ranks)
            psi_q: (n_components, d_head)
            psi_k: (n_components, d_head)
        """

        batch_size, d_head = _q.shape
        if d_head != self.d_head:
            raise ValueError(f"expected input dim {self.d_head}, got {d_head}")

        psi_q = self._psi_q()  # (n_components, d_head)
        psi_k = self._psi_k()  # (n_components, d_head)

        psi_q_inv = 1.0 / psi_q  # (n_components, d_head)
        psi_k_inv = 1.0 / psi_k  # (n_components, d_head)

        # A: [n_components, d_head, rank]
        # B: [n_components, d_head, rank]
        A, B = self._A_B()

        _q = _q.to(A.device)
        k = k.to(B.device)

        # Build M = I + A^T \Psi_q^-1 A + B^T \Psi_k^-1 B
        I_r = torch.eye(self.rank, dtype=_q.dtype, device=_q.device)

        # A * psi^-1/2, which will be "squared" later to construct A^T \Psi^-1 A
        Aq = A * psi_q_inv[:, :, None].sqrt()  # (n_components, d_head, rank)
        Bk = B * psi_k_inv[:, :, None].sqrt()  # (n_components, d_head, rank)

        M = (
            torch.einsum("kdi,kdj->kij", Aq, Aq)
            + torch.einsum("kdi,kdj->kij", Bk, Bk)
            + I_r[None, :, :]
        )

        # Quadratic base term: (q-mu_q)^T Psi_q^-1 (q-mu_q) + (k-mu_k)^T Psi_k^-1 (k-mu_k)
        qT_Pinv_q = torch.einsum("bd,kd->bk", _q * _q, psi_q_inv)  # (B, n_components)
        qT_Pinv_mu = torch.einsum("bd,kd->bk", _q, psi_q_inv * self.mu_q)
        muT_Pinv_mu_q = (self.mu_q * self.mu_q * psi_q_inv).sum(
            dim=-1
        )  # (n_components,)
        quad_q = (
            qT_Pinv_q - 2.0 * qT_Pinv_mu + muT_Pinv_mu_q[None, :]
        )  # (B, n_components)

        kT_Pinv_k = torch.einsum("bd,kd->bk", k * k, psi_k_inv)  # (B, n_components)
        kT_Pinv_mu = torch.einsum("bd,kd->bk", k, psi_k_inv * self.mu_k)
        muT_Pinv_mu_k = (self.mu_k * self.mu_k * psi_k_inv).sum(
            dim=-1
        )  # (n_components,)
        quad_k = (
            kT_Pinv_k - 2.0 * kT_Pinv_mu + muT_Pinv_mu_k[None, :]
        )  # (B, n_components)

        # v = A^T Psi_q^-1 (q-mu_q) + B^T Psi_k^-1 (k-mu_k)
        PinvA = psi_q_inv[:, :, None] * A  # (n_components, d_head, rank)
        PinvB = psi_k_inv[:, :, None] * B  # (n_components, d_head, rank)
        AT_Pinv_q = torch.einsum("bd,kdi->bki", _q, PinvA)  # (B, n_components, rank)
        AT_Pinv_mu = torch.einsum(
            "kd,kdi->ki", self.mu_q, PinvA
        )  # (n_components, rank)
        BT_Pinv_k = torch.einsum("bd,kdi->bki", k, PinvB)  # (B, n_components, rank)
        BT_Pinv_mu = torch.einsum(
            "kd,kdi->ki", self.mu_k, PinvB
        )  # (n_components, rank)
        v = (
            AT_Pinv_q - AT_Pinv_mu[None, :, :] + BT_Pinv_k - BT_Pinv_mu[None, :, :]
        )  # (B, n_components, rank)

        L = torch.linalg.cholesky(M)  # (n_components, rank, rank)
        v_perm = v.permute(1, 2, 0)  # (n_components, rank, B)
        Ez_perm = torch.cholesky_solve(
            v_perm, L, upper=False
        )  # (n_components, rank, B)
        Ez = Ez_perm.permute(2, 0, 1)  # (B, n_components, rank)

        I_r_expand = I_r.expand(self.n_components, self.rank, self.rank).clone()
        Sz = torch.cholesky_solve(
            I_r_expand, L, upper=False
        )  # (n_components, rank, rank)

        log_det_Psi = torch.log(psi_q).sum(dim=-1) + torch.log(psi_k).sum(
            dim=-1
        )  # (n_components,)
        log_det_M = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(
            -1
        )  # (n_components,)
        logdet_C = log_det_Psi + log_det_M  # (n_components,)

        v_Minv_v = (v * Ez).sum(dim=-1)  # (B, n_components)
        quad = (quad_q + quad_k) - v_Minv_v  # (B, n_components)

        ll = -0.5 * (
            self.d_head * math.log(2.0 * math.pi) + logdet_C[None, :] + quad
        )  # (B, n_components)
        return ll, Ez, Sz, L, v, psi_q, psi_k

    def responsibilities(
        self, _q: torch.Tensor, k: torch.Tensor, tau: float = 1.0
    ) -> torch.Tensor:
        ll, *_ = self._core(_q, k)
        log_pi = F.log_softmax(self.pi_logits, dim=0)[None, :]
        return F.softmax((ll + log_pi) / float(tau), dim=1)

    def log_prob(self, _q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        ll, *_ = self._core(_q, k)
        log_pi = F.log_softmax(self.pi_logits, dim=0)  # (n_components,)
        return torch.logsumexp(ll + log_pi[None, :], dim=1)

    def nll(self, _q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        return (-self.log_prob(_q, k)).mean()

    def component_posterior(
        self, _q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _ll, Ez, Sz, *_ = self._core(_q, k)
        # Ez, Sz = self._maybe_rotate_scores(Ez, Sz)
        return Ez, Sz

    def reconstruct(
        self, _q: torch.Tensor, k: torch.Tensor, *, use_mixture_mean: bool = True
    ) -> torch.Tensor:
        ll, Ez, _Sz, _L, _v, _psi_q, _psi_k = self._core(_q, k)
        A_eff, B_eff = self.A_B
        # if self._rotation_on:
        #    Ez, _ = self._maybe_rotate_scores(Ez, _Sz)
        comp_q = self.mu_q[None, :, :] + torch.einsum(
            "kdr,bkr->bkd", A_eff, Ez
        )  # (B, n_components, d_head)
        comp_k = self.mu_k[None, :, :] + torch.einsum("kdr,bkr->bkd", B_eff, Ez)
        if not use_mixture_mean:
            return comp_q, comp_k

        log_pi = F.log_softmax(self.pi_logits, dim=0)[None, :]
        alpha = F.softmax(ll + log_pi, dim=1)  # (B,n_components)

        q_hat = torch.einsum("bk,bkd->bd", alpha, comp_q)  # (B,d_head)
        k_hat = torch.einsum("bk,bkd->bd", alpha, comp_k)
        return q_hat, k_hat

    def forward(self, _q, k):
        return self.nll(_q, k)


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

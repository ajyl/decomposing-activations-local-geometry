import math
import torch
from torch.utils.data import DataLoader
from typing import Optional
import torch.nn.functional as F

# optional weighted sampling for better coverage -- not used by default

class WeightedReservoirSampler:
    def __init__(self, m: int, weights: Optional[torch.Tensor] = None, device=None, dtype=None):
        self.m = m
        self.w = weights.to(device) if (weights is not None and device is not None) else weights
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype  = dtype or torch.float32 

    @torch.no_grad()
    def sample(self, loader: DataLoader) -> torch.Tensor:
        pool, keys = None, None
        filled = 0

        for x, tok in loader:
            x = x.to(self.device, dtype=self.dtype)
            if isinstance(tok, torch.Tensor):
                tok = tok.to(self.device)

            if pool is None:
                D = x.size(1)
                pool = torch.empty((self.m, D), device=self.device, dtype=self.dtype)
                keys = torch.full((self.m,), -float("inf"), device=self.device, dtype=torch.float32)

            u = torch.rand(x.size(0), device=self.device, dtype=torch.float32)
            if self.w is None:
                k = u
            else:
                w_i = self.w[tok].float()
                k = u.pow(1.0 / torch.clamp(w_i, min=1e-12))

            if filled < self.m:
                take = min(self.m - filled, x.size(0))
                pool[filled:filled+take] = x[:take]
                keys[filled:filled+take] = k[:take]
                filled += take
                if filled < self.m:
                    continue
                x = x[take:]; k = k[take:]
                if x.numel() == 0:
                    continue

            if x.numel() == 0:
                continue
            min_key = keys.min()
            mask_cand = k > min_key
            if not mask_cand.any():
                continue

            k_cand = k[mask_cand]
            x_cand = x[mask_cand]

            r = min(k_cand.numel(), self.m)
            topk_vals, topk_idx_local = torch.topk(k_cand, k=r, largest=True)
            x_rep = x_cand[topk_idx_local]

            res_vals, res_idx = torch.topk(keys, k=r, largest=False)
            pool[res_idx] = x_rep
            keys[res_idx] = topk_vals

        return pool


@torch.no_grad()
def make_orthonormal_projector(D: int, d: int, device=None, dtype=torch.float32, seed: Optional[int]=None) -> torch.Tensor:
    if d > D:
        raise ValueError("proj_dim (d) must be ≤ D")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)
        A = torch.randn(D, d, device=device, dtype=torch.float32, generator=gen)
    else:
        A = torch.randn(D, d, device=device, dtype=torch.float32)
    Q, R = torch.linalg.qr(A, mode="reduced")
    signs = torch.sign(torch.diag(R))
    signs[signs == 0] = 1.0
    Q = Q * signs
    return Q.to(dtype=dtype)


def _normed(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=1)

def _pairwise_scores_cosine(Xb, Cb):
    return Xb @ Cb.T

def _pairwise_dist2_euclidean(Xb, Cb):
    x2 = (Xb * Xb).sum(dim=1, keepdim=True) # (bx,1)
    c2 = (Cb * Cb).sum(dim=1, keepdim=True).T # (1,bc)
    return x2 + c2 - 2.0 * (Xb @ Cb.T)

class KMeansTorch:
    def __init__(
        self,
        k: int,
        metric: str = "euclidean",
        n_iter: int = 20,
        restarts: int = 2,
        tol: float = 1e-4,
        seed: Optional[int] = None,
        device=None,
        dtype: torch.dtype = torch.float32,
        block_x: int = 8192,
        block_c: int = 8192,
    ):
        if metric not in {"euclidean", "cosine"}:
            raise ValueError("metric must be 'euclidean' or 'cosine'")
        self.k, self.metric  = k, metric
        self.n_iter          = n_iter
        self.restarts        = restarts
        self.tol             = tol
        self.seed            = seed
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype  = dtype
        self.block_x = block_x
        self.block_c = block_c

        self.centroids: Optional[torch.Tensor] = None
        self.inertia_: Optional[float]         = None
        self.n_iter_run_: Optional[int]        = None

    @torch.no_grad()
    def _assign_streamed(self, X: torch.Tensor, C: torch.Tensor):
        N = X.size(0); k = C.size(0)
        labels = torch.empty(N, device=self.device, dtype=torch.long)

        if self.metric == "cosine":
            X = _normed(X)
            C = _normed(C)

        for s in range(0, N, self.block_x):
            xb = X[s:s+self.block_x]
            bx = xb.size(0)

            if self.metric == "cosine":
                best_val = torch.full((bx,), -float("inf"), device=self.device, dtype=self.dtype)
            else:
                best_val = torch.full((bx,), float("inf"), device=self.device, dtype=X.dtype)

            best_idx = torch.full((bx,), -1, device=self.device, dtype=torch.long)

            for t in range(0, k, self.block_c):
                cb = C[t:t+self.block_c]
                if self.metric == "cosine":
                    scores = _pairwise_scores_cosine(xb, cb).to(self.dtype)
                    vals, idxs = scores.max(dim=1)
                    better = vals > best_val
                    best_val[better] = vals[better]
                    best_idx[better] = (idxs[better] + t)
                else:
                    d2 = _pairwise_dist2_euclidean(xb, cb)
                    vals, idxs = d2.min(dim=1)
                    better = vals < best_val
                    best_val[better] = vals[better]
                    best_idx[better] = (idxs[better] + t)

                if self.metric == "cosine":
                    del scores
                else:
                    del d2

            labels[s:s+bx] = best_idx

        return labels

    @torch.no_grad()
    def _kpp_streamed(self, X: torch.Tensor) -> torch.Tensor:
        N = X.size(0)
        X = X.to(self.device, dtype=self.dtype if self.metric == "cosine" else torch.float32)
        if self.metric == "cosine":
            X = _normed(X)

        idx = torch.randint(N, (1,), device=self.device)
        C = X[idx].clone()

        if self.metric == "cosine":
            def dblock(xb, cb): return 1.0 - (xb @ cb.T).to(torch.float32).squeeze(1).clamp(-1,1)
        else:
            def dblock(xb, cb): return _pairwise_dist2_euclidean(xb, cb).squeeze(1)

        d2 = torch.empty(N, device=self.device, dtype=torch.float32)
        for s in range(0, N, self.block_x):
            xb = X[s:s+self.block_x]
            d2[s:s+xb.size(0)] = dblock(xb, C[:1])

        for _ in range(1, self.k):
            prob = (d2 / d2.sum()).clamp_min(0)
            nxt = torch.multinomial(prob, 1)
            C = torch.cat([C, X[nxt]], dim=0)

            for s in range(0, N, self.block_x):
                xb = X[s:s+self.block_x]
                d_new = dblock(xb, C[-1:].contiguous())
                cur = d2[s:s+xb.size(0)]
                d2[s:s+xb.size(0)] = torch.minimum(cur, d_new)

        if self.metric == "cosine":
            C = _normed(C)
        else:
            C = C.to(torch.float32)
        return C

    @torch.no_grad()
    def fit(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(self.device, dtype=self.dtype if self.metric == "cosine" else torch.float32)

        best_I, best_C = math.inf, None
        for r in range(self.restarts):
            if self.seed is not None:
                torch.manual_seed(self.seed + r)
                torch.cuda.manual_seed_all(self.seed + r)

            C = self._kpp_streamed(X)

            for i in range(self.n_iter):
                prev_C = C.clone()

                lbl = self._assign_streamed(X, C)

                k = C.size(0)
                sums = torch.zeros_like(C, dtype=X.dtype)
                counts = torch.zeros(k, device=self.device, dtype=torch.float32)

                for s in range(0, X.size(0), self.block_x):
                    xb = X[s:s+self.block_x]
                    l  = lbl[s:s+xb.size(0)]
                    sums.index_add_(0, l, xb)
                    counts += torch.bincount(l, minlength=k).to(counts)

                nonzero = counts > 0
                newC = C.clone()
                newC[nonzero] = sums[nonzero] / counts[nonzero].unsqueeze(1).to(sums.dtype)
                if self.metric == "cosine":
                    newC[nonzero] = _normed(newC[nonzero])

                empty = (~nonzero).nonzero(as_tuple=True)[0]
                if empty.numel():
                    rnd = torch.randint(X.size(0), (empty.numel(),), device=self.device)
                    newC[empty] = X[rnd]
                    if self.metric == "cosine":
                        newC[empty] = _normed(newC[empty])

                C = newC
                if (C - prev_C).norm(dim=1).max() < self.tol:
                    self.n_iter_run_ = i + 1
                    break

            if self.metric == "euclidean":
                obj = 0.0
                for s in range(0, X.size(0), self.block_x):
                    xb = X[s:s+self.block_x]
                    best = torch.full((xb.size(0),), float("inf"), device=self.device)
                    for t in range(0, C.size(0), self.block_c):
                        cb = C[t:t+self.block_c]
                        d2 = _pairwise_dist2_euclidean(xb, cb)
                        best = torch.minimum(best, d2.min(dim=1).values)
                    obj += best.sum().item()
            else:
                obj = 0.0
                for s in range(0, X.size(0), self.block_x):
                    xb = _normed(X[s:s+self.block_x])
                    best = torch.full((xb.size(0),), -float("inf"), device=self.device, dtype=self.dtype)
                    for t in range(0, C.size(0), self.block_c):
                        cb = _normed(C[t:t+self.block_c])
                        sc = (xb @ cb.T).to(self.dtype)
                        best = torch.maximum(best, sc.max(dim=1).values)
                    obj += (1.0 - best.float()).sum().item()

            if obj < best_I:
                best_I = obj
                best_C = C.clone()
                if not hasattr(self, "n_iter_run_"):
                    self.n_iter_run_ = self.n_iter

        self.centroids = best_C
        self.inertia_  = best_I
        return best_C

@torch.no_grad()
def _assign_projected_streamed(x_proj: torch.Tensor, C_proj: torch.Tensor, metric: str, block_c: int = 8192):
    B = x_proj.size(0)
    k = C_proj.size(0)
    device = x_proj.device

    if metric == "cosine":
        x_proj = F.normalize(x_proj, p=2, dim=1)
        C_proj = F.normalize(C_proj, p=2, dim=1)
        best = torch.full((B,), -float("inf"), device=device, dtype=x_proj.dtype)
    else:
        best = torch.full((B,), float("inf"), device=device, dtype=torch.float32)

    best_idx = torch.full((B,), -1, device=device, dtype=torch.long)

    for t in range(0, k, block_c):
        cb = C_proj[t:t+block_c]
        if metric == "cosine":
            sc = x_proj @ cb.T
            vals, idxs = sc.max(dim=1)
            better = vals > best
            best[better] = vals[better]
            best_idx[better] = idxs[better] + t
        else:
            x2 = (x_proj * x_proj).sum(dim=1, keepdim=True) 
            c2 = (cb * cb).sum(dim=1, keepdim=True).T # (1,bc)
            d2 = x2 + c2 - 2.0 * (x_proj @ cb.T)
            vals, idxs = d2.min(dim=1)
            better = vals < best
            best[better] = vals[better]
            best_idx[better] = idxs[better] + t

    return best_idx


@torch.no_grad()
def lloyd_refine_projected(
    loader,
    C_full: torch.Tensor,
    R: torch.Tensor,
    *,
    metric: str = "euclidean",
    max_epochs: int = 5,
    tol: float = 1e-4,
    x_dtype: torch.dtype = torch.float32,
    block_c: int = 8192,
):
    device = C_full.device
    k, D = C_full.shape
    d = R.shape[1]

    C_full = C_full.to(device=device, dtype=torch.float32)
    C_proj = (C_full @ R).to(dtype=x_dtype)

    if metric == "cosine":
        C_full = F.normalize(C_full, p=2, dim=1)
        C_proj = F.normalize(C_proj, p=2, dim=1)

    for _ in range(max_epochs):
        prev = C_full.clone()

        sums = torch.zeros((k, D), device=device, dtype=torch.float32)
        counts = torch.zeros((k,), device=device, dtype=torch.float32)

        for x, _ in loader:
            x_full = x.to(device=device, dtype=torch.float32)
            x_proj = (x_full @ R).to(dtype=x_dtype)

            lbl = _assign_projected_streamed(x_proj, C_proj, metric, block_c=block_c)

            sums.index_add_(0, lbl, x_full)
            counts += torch.bincount(lbl, minlength=k).to(counts)

        nonzero = counts > 0
        C_new = C_full.clone()
        C_new[nonzero] = sums[nonzero] / counts[nonzero].unsqueeze(1)

        if metric == "cosine":
            C_new[nonzero] = F.normalize(C_new[nonzero], p=2, dim=1)

        empty = (~nonzero).nonzero(as_tuple=True)[0]
        if empty.numel():
            rnd = torch.randint(0, k, (empty.numel(),), device=device)
            C_new[empty] = C_full[rnd]

        delta = (C_new - C_full).norm(dim=1).max()
        C_full = C_new

        C_proj = (C_full @ R).to(dtype=x_dtype)
        if metric == "cosine":
            C_proj = F.normalize(C_proj, p=2, dim=1)

        if delta < tol:
            break

    return C_full


class ReservoirKMeans:
    def __init__(
        self,
        n_clusters: int,
        pool_size: int,
        vocab_size: int,
        smoothing: float = 1.0,
        power: float = 1.0,
        kmeans_iters: int = 50,
        kmeans_restarts: int = 10,
        tol: float = 1e-4,
        seed: Optional[int] = None,
        device=None,
        *,
        metric: str = "euclidean",
        proj_dim: Optional[int] = None,
        proj_dtype: torch.dtype = torch.float32,
    ):
        if pool_size < n_clusters:
            raise ValueError("pool_size must be ≥ n_clusters")
        self.k              = n_clusters
        self.pool_size      = pool_size
        self.vocab_size     = vocab_size
        self.smoothing      = smoothing
        self.power          = power
        self.kmeans_iters   = kmeans_iters
        self.kmeans_restarts= kmeans_restarts
        self.tol            = tol
        self.seed           = seed
        self.metric         = metric
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.proj_dim   = proj_dim
        self.proj_dtype = proj_dtype
        self.R: Optional[torch.Tensor] = None  # (D, d) if used

    @torch.no_grad()
    def _compute_weights(self, token_loader: DataLoader) -> torch.Tensor:
        counts = torch.zeros(self.vocab_size, dtype=torch.long)
        for batch in token_loader:
            tokens = batch[0] if isinstance(batch, (list, tuple)) else batch
            counts += torch.bincount(tokens.view(-1).cpu(), minlength=self.vocab_size)
        freq = counts.float()
        inv  = 1.0 / ((freq + self.smoothing) ** self.power)
        return (inv / inv.mean()).to(self.device)

    @torch.no_grad()
    def _lloyd_epochs(
        self,
        loader: DataLoader,
        centroids: torch.Tensor,
        max_epochs: int = 20,
        tol: float = 1e-4,
        metric: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        device=None,
        block_x: int = 8192,
        block_c: int = 8192,
    ) -> torch.Tensor:
        metric = metric or self.metric
        device = device or centroids.device
        C = centroids.to(device, dtype=dtype if metric == "cosine" else torch.float32)

        if metric == "cosine":
            C = F.normalize(C, p=2, dim=1)

        for _ in range(max_epochs):
            sums   = torch.zeros_like(C, dtype=torch.float32)
            counts = torch.zeros(C.size(0), device=device, dtype=torch.float32)

            for x, _ in loader:
                xb = x.to(device, dtype=dtype if metric == "cosine" else torch.float32)
                if metric == "cosine":
                    xb = F.normalize(xb, p=2, dim=1)
                    best = torch.full((xb.size(0),), -float("inf"), device=device, dtype=dtype)
                else:
                    best = torch.full((xb.size(0),), float("inf"), device=device, dtype=torch.float32)
                best_idx = torch.full((xb.size(0),), -1, device=device, dtype=torch.long)

                for t in range(0, C.size(0), block_c):
                    cb = C[t:t+block_c]
                    if metric == "cosine":
                        sc = (xb @ cb.T).to(d_ := dtype)
                        vals, idxs = sc.max(dim=1)
                        better = vals > best
                        best[better] = vals[better]
                        best_idx[better] = (idxs[better] + t)
                    else:
                        d2 = _pairwise_dist2_euclidean(xb, cb)
                        vals, idxs = d2.min(dim=1)
                        better = vals < best
                        best[better] = vals[better]
                        best_idx[better] = (idxs[better] + t)

                sums.index_add_(0, best_idx, xb.to(torch.float32))
                counts += torch.bincount(best_idx, minlength=C.size(0)).to(counts)

            nonzero = counts > 0
            newC = C.clone()
            newC[nonzero] = (sums[nonzero] / counts[nonzero].unsqueeze(1))
            if metric == "cosine":
                newC[nonzero] = F.normalize(newC[nonzero], p=2, dim=1)

            empty = (~nonzero).nonzero(as_tuple=True)[0]
            if empty.numel():
                rnd = torch.randint(sums.size(0), (empty.numel(),), device=device)
                newC[empty] = C[rnd]
                if metric == "cosine":
                    newC[empty] = F.normalize(newC[empty], p=2, dim=1)

            delta = (newC - C).norm(dim=1).max()
            C = newC
            if delta < tol:
                break

        return C

    @torch.no_grad()
    def fit(
        self,
        activation_loader: DataLoader,
        token_loader: Optional[DataLoader]=None,
        refine_epochs: int = 25,
    ):
        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

        if token_loader:
            weights  = self._compute_weights(token_loader)
        else:
            weights = None
        sampler  = WeightedReservoirSampler(self.pool_size, weights=weights, device=self.device)
        pool     = sampler.sample(activation_loader)

        D = pool.size(1)

        use_proj = (self.proj_dim is not None) and (0 < self.proj_dim < D)
        if use_proj:
            d = int(self.proj_dim)
            self.R = make_orthonormal_projector(D, d, device=self.device, dtype=self.proj_dtype, seed=self.seed)
            pool_sketch = pool @ self.R                        # (pool_size, d)
            km = KMeansTorch(
                self.k,
                metric     = self.metric,
                n_iter     = self.kmeans_iters,
                restarts   = self.kmeans_restarts,
                tol        = self.tol,
                seed       = self.seed,
                device     = self.device,
                dtype      = self.proj_dtype,
            )
            C_proj = km.fit(pool_sketch)
            centroids = (C_proj @ self.R.T).to(torch.float32)
        else:
            km = KMeansTorch(
                self.k,
                metric     = self.metric,
                n_iter     = self.kmeans_iters,
                restarts   = self.kmeans_restarts,
                tol        = self.tol,
                seed       = self.seed,
                device     = self.device,
                dtype      = torch.float32 if self.metric == "euclidean" else torch.float32,
            )
            centroids = km.fit(pool)

        if self.R is not None:
            centroids = lloyd_refine_projected(
                activation_loader,
                centroids.to(self.device),
                self.R,
                metric=self.metric,
                max_epochs=refine_epochs,
                tol=self.tol,
            )
        else:
            centroids = self._lloyd_epochs(
                activation_loader,
                centroids,
                max_epochs = refine_epochs,
                tol        = self.tol,
                metric     = self.metric,
            )

        return centroids



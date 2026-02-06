from __future__ import annotations

from typing import List, Optional, Union
import torch
from transformer_lens import HookedTransformer, utils


class MFASteerer:
    _SITES = {
        "resid_post":  lambda L: f"blocks.{L}.hook_resid_post",
        "mlp_act":     lambda L: f"blocks.{L}.mlp.hook_post",
        "mlp_out":     lambda L: f"blocks.{L}.hook_mlp_out",
    }

    def __init__(self, model: HookedTransformer, mfa, intervention_type: str = "resid_post"):
        if intervention_type not in self._SITES:
            raise ValueError(f"Unsupported intervention_type: {intervention_type}")
        self.model = model
        self.mfa = mfa
        self.intervention_type = intervention_type

    def _site(self, layer: int) -> str:
        return self._SITES[self.intervention_type](layer)

    def _flatten(self, value: torch.Tensor):
        orig_shape = value.shape
        if value.ndim == 3:
            B, P, D = value.shape
            flat = value.reshape(B * P, D)
            return flat, orig_shape
        elif value.ndim == 2:
            return value, orig_shape
        else:
            return None, orig_shape

    def _get_W(self) -> torch.Tensor:
        if hasattr(self.mfa, "W"):
            return self.mfa.W
        if hasattr(self.mfa, "Lambda"):
            return self.mfa.Lambda
        if hasattr(self.mfa, "loadings"):
            return self.mfa.loadings
        raise AttributeError("MFA loadings not found: expected .W, .Lambda, or .loadings")

    def _normalize_z(self, z: Union[torch.Tensor, list], device, dtype) -> torch.Tensor:
        if not isinstance(z, torch.Tensor):
            z = torch.tensor(z)
        return z.to(device=device, dtype=dtype)

    def _responsibilities(self, flat: torch.Tensor) -> torch.Tensor:
        r = self.mfa.responsibilities(flat)  # (N, K)
        if r.device != flat.device:
            r = r.to(flat.device)
        if r.dtype != torch.float32:
            r = r.float()
        return r

    def _hook_mean(self, alpha: float, k: Optional[int]):
        mfa = self.mfa
        mu = mfa.mu  # (K, D)

        @torch.inference_mode()
        def hook_fn(value: torch.Tensor, hook) -> torch.Tensor:
            flat, orig_shape = self._flatten(value)
            if flat is None:
                return value

            device, dtype = flat.device, flat.dtype
            mu_local = mu.to(device=device, dtype=dtype)

            if k is None:
                r = self._responsibilities(flat)                    # fp32
                target = (r @ mu_local.float()).to(dtype=dtype)     # (N, D)
            else:
                if not (0 <= k < mu_local.shape[0]):
                    raise ValueError(f"k must be in [0,{mu_local.shape[0]-1}] or None")
                target = mu_local[k].expand_as(flat)

            out = (1.0 - alpha) * flat + alpha * target
            return out.reshape(orig_shape)

        return hook_fn
    
    def _hook_latent_two_stage(
        self,
        alpha_centroid: float,
        z: Union[torch.Tensor, list],
        k: Optional[int],
    ):
        mfa = self.mfa
        mu = mfa.mu
        W = self._get_W()

        @torch.inference_mode()
        def hook_fn(value: torch.Tensor, hook) -> torch.Tensor:
            flat, orig_shape = self._flatten(value)
            if flat is None:
                return value

            device, dtype = flat.device, flat.dtype
            mu_local = mu.to(device=device, dtype=dtype)
            W_local  = W.to(device=device, dtype=dtype)
            z_local  = self._normalize_z(z, device, dtype)

            K, Dm = mu_local.shape
            Kw, Dw, q = W_local.shape
            if Kw != K or Dw != Dm:
                raise ValueError(f"Expected W shape (K,D,q) matching mu (K,D); got W={W_local.shape}, mu={mu_local.shape}")

            N = flat.shape[0]

            if k is not None:
                if not (0 <= k < K):
                    raise ValueError(f"k must be in [0,{K-1}] or None")

                # Step 1: centroid pull
                centroid = mu_local[k].unsqueeze(0).expand(N, Dm)
                x1 = flat + alpha_centroid * (centroid - flat)

                # Step 2: within move (NOT scaled by alpha_centroid)
                if z_local.ndim == 1:
                    delta = (W_local[k] @ z_local).view(1, Dm).expand(N, Dm)
                elif z_local.ndim == 2 and z_local.shape == (N, q):
                    delta = z_local @ W_local[k].T
                else:
                    raise ValueError(f"When k is specified, z must be (q,) or (N,q) with (N,q)=({N},{q}). Got {tuple(z_local.shape)}")

                out = x1 + delta
                return out.reshape(orig_shape)

            r = self._responsibilities(flat)  # (N, K) in fp32

            centroid = (r @ mu_local.float()).to(dtype=dtype)   # (N,D)
            x1 = flat + alpha_centroid * (centroid - flat)

            if z_local.ndim == 1:
                # delta_n = (sum_k r_nk W_k) z
                W_eff = torch.einsum("nk,kdq->ndq", r, W_local.float())      # (N,D,q) fp32
                delta = torch.einsum("ndq,q->nd", W_eff, z_local.float())    # (N,D) fp32
                delta = delta.to(dtype=dtype)
            elif z_local.ndim == 2 and z_local.shape == (N, q):
                W_eff = torch.einsum("nk,kdq->ndq", r, W_local.float())      # (N,D,q)
                delta = torch.einsum("ndq,nq->nd", W_eff, z_local.float())   # (N,D)
                delta = delta.to(dtype=dtype)
            elif z_local.ndim == 2 and z_local.shape == (K, q):
                # per-component z_k: delta_n = sum_k r_nk (W_k z_k)
                Wz_k = torch.einsum("kdq,kq->kd", W_local.float(), z_local.float())  # (K,D)
                delta = (r @ Wz_k).to(dtype=dtype)                                   # (N,D)
            else:
                raise ValueError(
                    f"When k=None, z must be (q,), (N,q)=({N},{q}), or (K,q)=({K},{q}). Got {tuple(z_local.shape)}"
                )

            out = x1 + delta
            return out.reshape(orig_shape)

        return hook_fn

    @torch.inference_mode()
    def intervene(
        self,
        prompt_or_tokens: Union[str, torch.Tensor],
        layers: List[int],
        alpha: float,
        k: Optional[int] = None,
    ) -> torch.Tensor:
        tokens = self.model.to_tokens(prompt_or_tokens) if isinstance(prompt_or_tokens, str) else prompt_or_tokens
        hook = self._hook_mean(alpha=alpha, k=k)
        hooks = [(self._site(L), hook) for L in layers]
        return self.model.run_with_hooks(tokens, fwd_hooks=hooks)

    @torch.inference_mode()
    def generate(
        self,
        prompt_or_tokens: Union[str, torch.Tensor],
        layers: List[int],
        alpha: float,
        k: Optional[int] = None,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
    ) -> str:
        tokens = self.model.to_tokens(prompt_or_tokens) if isinstance(prompt_or_tokens, str) else prompt_or_tokens
        hook = self._hook_mean(alpha=alpha, k=k)
        hooks = [(self._site(L), hook) for L in layers]
        out_tokens = self.model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            fwd_hooks=hooks,
        )
        return self.model.to_string(out_tokens[0])

    @torch.inference_mode()
    def intervene_to_latent_two_stage(
        self,
        prompt_or_tokens: Union[str, torch.Tensor],
        layers: List[int],
        alpha_centroid: float,
        z: Union[torch.Tensor, list],
        k: Optional[int] = None,
    ) -> torch.Tensor:
        tokens = self.model.to_tokens(prompt_or_tokens) if isinstance(prompt_or_tokens, str) else prompt_or_tokens
        hook = self._hook_latent_two_stage(alpha_centroid=alpha_centroid, z=z, k=k)
        hooks = [(self._site(L), hook) for L in layers]
        return self.model.run_with_hooks(tokens, fwd_hooks=hooks)

    @torch.inference_mode()
    def generate_to_latent_two_stage(
        self,
        prompt_or_tokens: Union[str, torch.Tensor],
        layers: List[int],
        alpha_centroid: float,
        z: Union[torch.Tensor, list],
        k: Optional[int] = None,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
    ) -> str:
        tokens = self.model.to_tokens(prompt_or_tokens) if isinstance(prompt_or_tokens, str) else prompt_or_tokens
        hook = self._hook_latent_two_stage(alpha_centroid=alpha_centroid, z=z, k=k)
        hooks = [(self._site(L), hook) for L in layers]
        out_tokens = self.model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            fwd_hooks=hooks,
        )
        return self.model.to_string(out_tokens[0])

    @torch.inference_mode()
    def generate_to_latent_two_stage_sampling(
        self,
        prompt: str,
        layers: List[int],
        alpha_centroid: float,
        z: Union[torch.Tensor, list],
        k: Optional[int] = None,
        *,
        max_new_tokens: int = 50,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: float = 1.0,
        freq_penalty: float = 0.0,
        m: int = 1,
        use_past_kv_cache: bool = True,
    ) -> List[str]:
        device = self.model.cfg.device
        tokens = self.model.to_tokens(prompt).to(device) # [1, T]
        tokens = tokens.repeat(m, 1) # [m, T]

        past_kv_cache = None
        if use_past_kv_cache:
            from transformer_lens import HookedTransformerKeyValueCache
            past_kv_cache = HookedTransformerKeyValueCache.init_cache(self.model.cfg, device, m)

        hook = self._hook_latent_two_stage(alpha_centroid=alpha_centroid, z=z, k=k)
        fwd_hooks = [(self._site(L), hook) for L in layers]

        for i in range(max_new_tokens):
            if use_past_kv_cache:
                if i == 0:
                    logits = self.model.run_with_hooks(tokens, fwd_hooks=fwd_hooks, past_kv_cache=past_kv_cache)
                else:
                    logits = self.model.run_with_hooks(tokens[:, -1:], fwd_hooks=fwd_hooks, past_kv_cache=past_kv_cache)
            else:
                logits = self.model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)

            last_logits = logits[:, -1, :] # [m, d_vocab]
            next_tok = utils.sample_logits(
                last_logits,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                freq_penalty=freq_penalty,
                tokens=tokens,
            ).unsqueeze(1)  # [m, 1]
            tokens = torch.cat([tokens, next_tok], dim=1)

        return [self.model.to_string(tokens[i]) for i in range(m)]

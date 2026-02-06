import torch
import torch.nn.functional as F
import heapq
from typing import Callable, Dict, List, Tuple, Any, Optional, Union, Literal
from collections import defaultdict


@torch.no_grad()
def get_top_strings_per_concept(
    model,
    loader,
    tok2str: Callable[[Any], str],
    *,
    topk: int = 50,
    device: Optional[torch.device] = None,
    return_scores: bool = False,
    score: Literal["posterior", "likelihood"] = "posterior",
    aggregate: Literal["occurrence", "max", "sum"] = "occurrence",
) -> Dict[int, List[Union[str, Tuple[str, float]]]]:
    """
    Works with the provided MFA class.

    score:
      - "posterior": α_k(h) = p(k | h)  (via MFA.responsibilities)
      - "likelihood": ll_k(h) = log p(h | k)  (via MFA.log_prob_components)

    aggregate:
      - "occurrence": keep top individual (token occurrence, can repeat)
      - "max":        de-dup by token string; keep max score per token
      - "sum":        de-dup by token string; sum scores across occurrences
                      (for posterior, sums α; for likelihood, sums ll)
    """
    was_training = model.training
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    # Per-concept heaps for "occurrence"
    heaps: Dict[int, List[Tuple[float, int, str]]] = {}
    # Per-concept maps for aggregation ("max"/"sum")
    if aggregate == "sum":
        agg_maps: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    else:
        agg_maps = defaultdict(dict)  # type: ignore[assignment]

    counter = 0
    K_seen: Optional[int] = None

    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            h, toks = batch[0], batch[1]
        else:
            raise ValueError("Loader must yield (activations, tokens).")
        h = h.to(device, non_blocking=True)

        # --- Compute per-(B,K) score matrices for ranking and values ---
        if score == "posterior":
            # α = p(k|h) in (0,1); use α for both ranking and aggregation
            alpha = model.responsibilities(h)  # (B, K)
            scores_rank = alpha
            scores_value = alpha
        else:
            # ll_k(h) = log p(h | k)
            ll = model.log_prob_components(h)  # (B, K)
            scores_rank = ll
            scores_value = ll

        B, K = scores_rank.shape
        if K_seen is None:
            K_seen = K
            if aggregate == "occurrence":
                for k in range(K):
                    heaps[k] = []

        # convenience accessor for tokens
        def get_tok_i(toks, i):
            if isinstance(toks, torch.Tensor):
                return toks[i]
            elif isinstance(toks, (list, tuple)):
                return toks[i]
            else:
                return (toks, i)

        # Work on CPU for Python data structures
        rank_cpu = scores_rank.detach().cpu()
        value_cpu = scores_value.detach().cpu()

        for i in range(B):
            s = tok2str(get_tok_i(toks, i))
            row_r = rank_cpu[i]   # (K,)
            row_v = value_cpu[i]  # (K,)

            # Skip if NaNs/Infs in ranking row
            if not torch.isfinite(row_r).all():
                continue

            # Iterate over concepts
            # (convert to python float once per (i,k))
            for k in range(K):
                key = float(row_r[k])   # ranking key
                val = float(row_v[k])   # value used for aggregation/return

                if aggregate == "occurrence":
                    hp = heaps[k]
                    if len(hp) < topk:
                        heapq.heappush(hp, (key, counter, s))
                    else:
                        if key > hp[0][0]:
                            heapq.heapreplace(hp, (key, counter, s))
                    counter += 1
                else:
                    # aggregate by token string
                    if aggregate == "max":
                        prev = agg_maps[k].get(s, float("-inf"))
                        if val > prev:
                            agg_maps[k][s] = val
                    else:  # "sum"
                        agg_maps[k][s] += val

    result: Dict[int, List[Union[str, Tuple[str, float]]]] = {}

    if aggregate == "occurrence":
        for k, hp in heaps.items():
            # sort by ranking key descending
            items = sorted(hp, key=lambda t: t[0], reverse=True)
            if return_scores:
                if score == "posterior":
                    # return α for readability
                    out = [(s, float(scores)) for (scores, _, s) in items]
                else:
                    out = [(s, float(scores)) for (scores, _, s) in items]
                result[k] = out
            else:
                result[k] = [s for (_sc, _c, s) in items]
    else:
        for k, d in agg_maps.items():
            items = list(d.items())  # [(str, agg_score)]
            items.sort(key=lambda kv: kv[1], reverse=True)
            items = items[:topk]
            result[k] = items if return_scores else [s for s, _ in items]

    if was_training:
        model.train()

    return result

@torch.no_grad()
def get_top_indices_per_concept(
    model,
    loader,
    *,
    topk: int = 50,
    device: Optional[torch.device] = None,
    return_scores: bool = False,
    score: Literal["posterior", "likelihood"] = "posterior",
) -> Dict[int, List[Union[int, Tuple[int, float]]]]:
    """
    Like get_top_strings_per_concept, but returns global sample indices
    (the index of the example in the loader's iteration order).

    Returns:
      Dict[k] -> list of indices (or (index, score) if return_scores=True),
                 sorted by descending score, length <= topk
    """
    was_training = model.training
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    # Per-concept min-heaps of (score_key, tie_breaker, global_idx)
    heaps: Dict[int, List[Tuple[float, int, int]]] = {}

    global_idx = 0
    K_seen: Optional[int] = None
    tie_breaker = 0  # strictly increases to stabilize heap ordering

    for batch in loader:
        # Expect (activations, tokens) or (activations, ...)
        if isinstance(batch, (list, tuple)) and len(batch) >= 1:
            h = batch[0]
        else:
            raise ValueError("Loader must yield (activations, ...).")

        h = h.to(device, non_blocking=True)

        # --- Scores per (B, K) ---
        if score == "posterior":
            S = model.responsibilities(h)         # (B, K), α in (0,1)
        else:
            S = model.log_prob_components(h)      # (B, K), log p(h|k)

        B, K = S.shape
        if K_seen is None:
            K_seen = K
            for k in range(K):
                heaps[k] = []

        S_cpu = S.detach().cpu()

        for i in range(B):
            row = S_cpu[i]  # (K,)
            if not torch.isfinite(row).all():
                global_idx += 1
                continue

            for k in range(K):
                key = float(row[k])
                hp = heaps[k]
                if len(hp) < topk:
                    heapq.heappush(hp, (key, tie_breaker, global_idx))
                else:
                    if key > hp[0][0]:
                        heapq.heapreplace(hp, (key, tie_breaker, global_idx))
                tie_breaker += 1
            global_idx += 1

    # Build result dict
    result: Dict[int, List[Union[int, Tuple[int, float]]]] = {}
    for k, hp in heaps.items():
        # sort by score descending
        items = sorted(hp, key=lambda t: t[0], reverse=True)
        if return_scores:
            result[k] = [(idx, float(sc)) for (sc, _tb, idx) in items]
        else:
            result[k] = [idx for (sc, _tb, idx) in items]

    if was_training:
        model.train()

    return result

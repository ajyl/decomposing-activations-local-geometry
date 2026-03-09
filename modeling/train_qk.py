import torch
import torch.nn.functional as F
from modeling.mfa import save_mfa
from tqdm import tqdm
import itertools
import math


def _build_tau_schedule(total_steps: int, tau_schedule: dict | None):
    cfg = dict(tau_schedule or {})
    kind = str(cfg.get("type", "linear")).lower()
    tau_start = float(cfg.get("tau_start", 1.0))
    tau_end = float(cfg.get("tau_end", 1.0))
    warmup_steps = int(cfg.get("warmup_steps", 0))
    anneal_steps_raw = cfg.get("anneal_steps", None)

    if total_steps <= 0:
        raise ValueError("`total_steps` must be > 0 for tau scheduling.")
    if tau_start <= 0.0 or tau_end <= 0.0:
        raise ValueError("`tau_start` and `tau_end` must both be > 0.")
    if warmup_steps < 0:
        raise ValueError("`warmup_steps` must be >= 0.")
    if kind not in {"linear", "cosine", "exp"}:
        raise ValueError("`tau_schedule.type` must be one of: linear, cosine, exp.")

    if anneal_steps_raw is None:
        anneal_steps = max(1, total_steps - warmup_steps)
    else:
        anneal_steps = int(anneal_steps_raw)
        if anneal_steps <= 0:
            raise ValueError("`tau_schedule.anneal_steps` must be > 0 when provided.")

    def get_tau(step: int) -> float:
        s = max(0, int(step))
        if s < warmup_steps:
            return tau_start

        p = (s - warmup_steps) / float(max(1, anneal_steps))
        p = min(max(p, 0.0), 1.0)

        if kind == "linear":
            return tau_start + (tau_end - tau_start) * p
        if kind == "cosine":
            return tau_end + (tau_start - tau_end) * 0.5 * (1.0 + math.cos(math.pi * p))
        # kind == "exp"
        return tau_start * ((tau_end / tau_start) ** p)

    schedule_meta = {
        "type": kind,
        "tau_start": tau_start,
        "tau_end": tau_end,
        "warmup_steps": warmup_steps,
        "anneal_steps": anneal_steps,
        "total_steps": int(total_steps),
    }
    return get_tau, schedule_meta


def _batch_qk_and_labels(batch):
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        _q = batch[0]
        _k = batch[1]
        labels = batch[2] if len(batch) >= 3 else None
        return _q, _k, labels
    raise ValueError("Loader batches must provide at least q and k tensors.")


def _best_perm_bruteforce(cost: torch.Tensor):
    """
    cost: (K_true, K_hat) tensor
    returns perm p of length K_true mapping true i -> hat p[i]
    """
    Kt, Kh = cost.shape
    if Kt != Kh:
        raise ValueError("Permutation matching requires K_true == K_hat.")
    K = Kt
    best = None
    best_val = float("inf")
    for p in itertools.permutations(range(K)):
        v = cost[torch.arange(K), torch.tensor(p, device=cost.device)].sum().item()
        if v < best_val:
            best_val = v
            best = p
    return torch.tensor(best, dtype=torch.long, device=cost.device), best_val


def _subspace_similarity(A_true: torch.Tensor, A_hat: torch.Tensor):
    Qt, _ = torch.linalg.qr(A_true, mode="reduced")
    Qh, _ = torch.linalg.qr(A_hat, mode="reduced")
    s = torch.linalg.svdvals(Qt.T @ Qh).clamp(0, 1)
    return s.mean().item()


def _effective_rank(svals: torch.Tensor, frac: float = 0.99) -> int:
    if svals.numel() == 0:
        return 0
    energy = svals.square()
    total = energy.sum()
    if total <= 0:
        return 0
    threshold = float(frac) * total
    cumulative = torch.cumsum(energy, dim=0)
    return int(torch.searchsorted(cumulative, threshold, right=False).item()) + 1


def _cpu_state_dict(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _fmt_optional_idx(value):
    return f"{value:02d}" if value is not None else "NA"


def _fmt_optional_float(value, width: int = 8):
    if value is None:
        return "N/A".rjust(width)
    return f"{float(value):>{width}.6f}"


def _fmt_optional_rank(value, width: int = 3):
    if value is None:
        return "N/A".rjust(width)
    return f"{int(value):>{width}d}"


def _build_pi_rank_rows(
    gt_pi_list,
    trained_pi_list,
    gt_ranks_list,
    a_eff_rank99,
    b_eff_rank99,
):
    k_true = len(gt_pi_list)
    k_hat = len(trained_pi_list)
    gt_pi_order = sorted(range(k_true), key=lambda i: gt_pi_list[i])
    tr_pi_order_all = sorted(range(k_hat), key=lambda i: trained_pi_list[i])
    if k_hat > k_true > 0:
        top_trained = sorted(
            sorted(range(k_hat), key=lambda i: trained_pi_list[i], reverse=True)[:k_true],
            key=lambda i: trained_pi_list[i],
        )
        top_trained_set = set(top_trained)
        extra_trained = [i for i in tr_pi_order_all if i not in top_trained_set]
        tr_pi_order = top_trained + extra_trained
    else:
        tr_pi_order = tr_pi_order_all

    rows = []
    for i in range(max(k_true, k_hat)):
        gt_idx = gt_pi_order[i] if i < k_true else None
        tr_idx = tr_pi_order[i] if i < k_hat else None
        gt_pi = gt_pi_list[gt_idx] if gt_idx is not None else None
        tr_pi = trained_pi_list[tr_idx] if tr_idx is not None else None
        gt_rank = (
            gt_ranks_list[gt_idx]
            if gt_idx is not None and gt_ranks_list is not None
            else None
        )
        a_eff = int(a_eff_rank99[tr_idx]) if tr_idx is not None else None
        b_eff = int(b_eff_rank99[tr_idx]) if tr_idx is not None else None

        rows.append(
            " ".join(
                [
                    f"i={i:02d}",
                    f"gt_idx={_fmt_optional_idx(gt_idx)}",
                    f"tr_idx={_fmt_optional_idx(tr_idx)}",
                    f"gt_pi={_fmt_optional_float(gt_pi)}",
                    f"tr_pi={_fmt_optional_float(tr_pi)}",
                    "|",
                    f"gt_rank={_fmt_optional_rank(gt_rank)}",
                    f"A_eff99={_fmt_optional_rank(a_eff)}",
                    f"B_eff99={_fmt_optional_rank(b_eff)}",
                ]
            )
        )

    return rows


@torch.no_grad()
def _eval_nll(model, loader, device, *, eval_targets=None, tau: float = 1.0):
    model.eval()
    tot_nll, tot_q_mse, tot_k_mse, tot_n = 0.0, 0.0, 0.0, 0
    pred_labels = []
    true_labels = []
    for batch in loader:
        _q, _k, labels = _batch_qk_and_labels(batch)
        _q = _q.view(_q.size(0), -1).to(device)
        _k = _k.view(_k.size(0), -1).to(device)
        nll = model.nll(_q, _k, tau=tau)  # mean over batch
        q_hat, k_hat = model.reconstruct(_q, _k, use_mixture_mean=True)
        B = _q.size(0)
        tot_nll += nll.item() * B
        tot_q_mse += ((_q - q_hat) ** 2).mean().item() * B
        tot_k_mse += ((_k - k_hat) ** 2).mean().item() * B
        tot_n += B
        if labels is not None:
            pred_labels.append(
                model.responsibilities(_q, _k, tau=tau).argmax(dim=1).detach().cpu()
            )
            true_labels.append(labels.detach().view(-1).cpu())

    out = {
        "nll": (tot_nll / tot_n) if tot_n > 0 else float("nan"),
        "q_mse": (tot_q_mse / tot_n) if tot_n > 0 else float("nan"),
        "k_mse": (tot_k_mse / tot_n) if tot_n > 0 else float("nan"),
    }
    A_hat_eval, B_hat_eval = model.A_B
    A_hat_eval = A_hat_eval.detach()
    B_hat_eval = B_hat_eval.detach()
    a_eff_rank99 = [
        _effective_rank(torch.linalg.svdvals(A_hat_eval[k]), frac=0.99)
        for k in range(A_hat_eval.size(0))
    ]
    b_eff_rank99 = [
        _effective_rank(torch.linalg.svdvals(B_hat_eval[k]), frac=0.99)
        for k in range(B_hat_eval.size(0))
    ]
    out["A_eff_rank99"] = [int(v) for v in a_eff_rank99]
    out["B_eff_rank99"] = [int(v) for v in b_eff_rank99]
    out["A_eff_rank99_mean"] = float(sum(a_eff_rank99) / max(1, len(a_eff_rank99)))
    out["B_eff_rank99_mean"] = float(sum(b_eff_rank99) / max(1, len(b_eff_rank99)))

    # if pred_labels and true_labels:
    #    pred = torch.cat(pred_labels, dim=0)
    #    truth = torch.cat(true_labels, dim=0).long()
    #    K_hat = int(model.n_components)
    #    K_true = int(truth.max().item()) + 1 if truth.numel() else 0
    #    if K_true == K_hat and K_true > 0 and K_true <= 8:
    #        confusion = torch.zeros(K_true, K_hat, dtype=torch.float32)
    #        for t, p in zip(truth, pred):
    #            if 0 <= int(t) < K_true and 0 <= int(p) < K_hat:
    #                confusion[int(t), int(p)] += 1.0
    #        cost = -confusion
    #        perm, _ = _best_perm_bruteforce(cost)
    #        invperm = torch.empty_like(perm)
    #        invperm[perm] = torch.arange(K_true, device=perm.device)
    #        pred_true_space = invperm[pred.to(invperm.device)].cpu()
    #        out["cluster_acc"] = (pred_true_space == truth).float().mean().item()
    #        out["perm"] = [int(x) for x in perm.cpu().tolist()]
    #    else:
    #        out["cluster_acc"] = float("nan")

    if eval_targets and "toy_params" in eval_targets:
        tp = eval_targets["toy_params"]
        mu_q_true = tp["mu_q"].to(device)
        mu_k_true = tp["mu_k"].to(device)
        A_true = tp["A"].to(device)
        B_true = tp["B"].to(device)
        pi_true = tp.get("pi")
        if pi_true is not None:
            pi_true = pi_true.to(device).float()
        pi_hat = torch.softmax(model.pi_logits.detach().float(), dim=0)
        trained_pi_list = [float(x) for x in pi_hat.cpu().tolist()]
        out["trained_pi"] = trained_pi_list
        gt_pi_list = None
        if pi_true is not None:
            gt_pi_list = [float(x) for x in pi_true.cpu().tolist()]
            out["gt_pi"] = gt_pi_list
        gt_component_ranks = tp.get("component_ranks")
        gt_ranks_list = None
        if gt_component_ranks is not None:
            gt_component_ranks = gt_component_ranks.to(device).long()
            gt_ranks_list = [int(x) for x in gt_component_ranks.cpu().tolist()]
            out["gt_component_ranks"] = gt_ranks_list

        mu_q_hat = model.mu_q.detach()
        mu_k_hat = model.mu_k.detach()
        A_hat, B_hat = model.A_B
        A_hat = A_hat.detach()
        B_hat = B_hat.detach()

        K_true = int(mu_q_true.size(0))
        K_hat = int(mu_q_hat.size(0))
        out["K_true"] = K_true
        out["K_hat"] = K_hat
        if gt_pi_list is not None and (K_true > 0 or K_hat > 0):
            out["pi_indexed_rank_rows"] = _build_pi_rank_rows(
                gt_pi_list,
                trained_pi_list,
                gt_ranks_list,
                a_eff_rank99,
                b_eff_rank99,
            )
        if K_true == K_hat and K_true > 0 and K_true <= 8:
            center_cost = torch.cdist(mu_q_true, mu_q_hat) + torch.cdist(
                mu_k_true, mu_k_hat
            )
            perm, _ = _best_perm_bruteforce(center_cost)

            mu_q_hat = mu_q_hat[perm]
            mu_k_hat = mu_k_hat[perm]
            A_hat = A_hat[perm]
            B_hat = B_hat[perm]

            out["mean_err_q"] = (mu_q_hat - mu_q_true).norm(dim=1).mean().item()
            out["mean_err_k"] = (mu_k_hat - mu_k_true).norm(dim=1).mean().item()

            if gt_component_ranks is not None:
                a_eff_rank99_aligned = [
                    _effective_rank(torch.linalg.svdvals(A_hat[k]), frac=0.99)
                    for k in range(K_true)
                ]
                b_eff_rank99_aligned = [
                    _effective_rank(torch.linalg.svdvals(B_hat[k]), frac=0.99)
                    for k in range(K_true)
                ]
                a_eff_rank99_aligned_int = [int(v) for v in a_eff_rank99_aligned]
                out["A_eff_rank99_aligned"] = a_eff_rank99_aligned_int
                b_eff_rank99_aligned_int = [int(v) for v in b_eff_rank99_aligned]
                out["B_eff_rank99_aligned"] = b_eff_rank99_aligned_int

        else:
            out["mean_err_q"] = float("nan")
            out["mean_err_k"] = float("nan")
            if gt_component_ranks is not None and K_true == K_hat and K_true > 0:
                a_eff_rank99_int = [int(v) for v in a_eff_rank99]
                out["A_eff_rank99_aligned"] = a_eff_rank99_int
                b_eff_rank99_int = [int(v) for v in b_eff_rank99]
                out["B_eff_rank99_aligned"] = b_eff_rank99_int

    return out


def train_nll(
    model,
    loader,
    *,
    val_loader=None,
    eval_loader=None,
    epochs=5,
    lr=1e-3,
    grad_clip=None,
    save_path=None,
    save_func=None,
    log_interval=100,
    steps_per_epoch=None,
    eval_targets=None,
    tau_schedule=None,
):
    """
    Train with NLL, keep the best (lowest) NLL model.
    Works with loaders
    """
    device = next(model.parameters()).device
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_steps_per_epoch = int(steps_per_epoch) if steps_per_epoch is not None else len(loader)
    total_steps = int(epochs) * int(train_steps_per_epoch)
    tau_at_step, tau_schedule_meta = _build_tau_schedule(total_steps, tau_schedule)
    global_step = 0

    best_metric = float("inf")
    best_state = _cpu_state_dict(model)
    best_epoch = 0

    for ep in range(1, epochs + 1):
        model.train()
        total_nll, total_n = 0.0, 0
        last_tau = tau_at_step(global_step)

        iterable = enumerate(loader, 1)
        pbar = tqdm(iterable, total=steps_per_epoch)

        for batch_idx, batch in pbar:
            _q = batch[0]  # [B, d_head]
            _k = batch[1]  # [B, d_head]
            opt.zero_grad(set_to_none=True)
            tau = tau_at_step(global_step)
            last_tau = tau
            nll = model.nll(_q, _k, tau=tau)  # mean over batch
            nll.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            opt.step()

            B = _q.size(0)
            total_nll += float(nll.item()) * B
            total_n += B

            if (batch_idx % log_interval) == 0:
                avg_so_far = total_nll / max(1, total_n)
                pbar.set_description(
                    f"Epoch {ep:02d} | Step {batch_idx:06d} Train NLL={avg_so_far:.6f} tau={tau:.4f}"
                )

            global_step += 1
            if steps_per_epoch is not None and batch_idx >= steps_per_epoch:
                break

            # free ASAP
            del _q, _k, nll

        if total_n == 0:
            avg_train_nll = float("nan")
        else:
            avg_train_nll = total_nll / total_n

        if ep % 1 == 0:
            if val_loader is not None:
                eval_dl = eval_loader if eval_loader is not None else val_loader
                val_metrics = _eval_nll(
                    model, eval_dl, device, eval_targets=eval_targets, tau=last_tau
                )
                val_nll = val_metrics["nll"]
                select_metric = val_nll
            else:
                val_nll = float("nan")
                val_metrics = {"nll": val_nll, "q_mse": float("nan"), "k_mse": float("nan")}
                select_metric = avg_train_nll

            improved = (
                (select_metric < best_metric)
                if not (torch.isnan(torch.tensor(select_metric)))
                else False
            )
            if improved:
                best_metric = select_metric
                best_state = _cpu_state_dict(model)
                best_epoch = ep
                if save_path and save_func:
                    save_func(model, save_path)

            print(
                f"[epoch {ep:02d}] "
                f"train NLL={avg_train_nll:.6f}  "
                f"tau={last_tau:.6f} "
                f"val NLL={val_nll:.6f} "
                f"val qMSE={val_metrics['q_mse']:.6f} "
                f"val kMSE={val_metrics['k_mse']:.6f} "
                f"{'** best **' if improved else ''}"
            )
            if "cluster_acc" in val_metrics:
                print(f"           val cluster_acc={val_metrics['cluster_acc']:.6f}")
            if "mean_err_q" in val_metrics:
                print(
                    "           "
                    f"mean_err_q={val_metrics['mean_err_q']:.6f} "
                    f"mean_err_k={val_metrics['mean_err_k']:.6f} "
                    f"A_eff_rank99_mean={val_metrics['A_eff_rank99_mean']:.6f} "
                    f"B_eff_rank99_mean={val_metrics['B_eff_rank99_mean']:.6f} "
                )
        if "pi_indexed_rank_rows" in val_metrics:
            print(
                "           pi-sorted comparison (gt and trained each sorted ascending by their own pi)"
            )
            for row in val_metrics["pi_indexed_rank_rows"]:
                print(f"           {row}")
        if "extra_trained_pi_rows" in val_metrics:
            print("           extra trained components (no GT counterpart)")
            for row in val_metrics["extra_trained_pi_rows"]:
                print(f"           {row}")
        if "extra_gt_pi_rows" in val_metrics:
            print("           extra GT components (no trained counterpart)")
            for row in val_metrics["extra_gt_pi_rows"]:
                print(f"           {row}")

    model.load_state_dict(best_state)
    print(
        f"Restored best model from epoch {best_epoch:02d} with metric={best_metric:.6f}"
    )

    metrics = dict(best_epoch=best_epoch, best_metric=best_metric)
    metrics["tau_schedule"] = tau_schedule_meta
    metrics["tau_final"] = float(tau_at_step(global_step))
    if val_loader is not None:
        eval_dl = eval_loader if eval_loader is not None else val_loader
        metrics["val"] = _eval_nll(
            model,
            eval_dl,
            device,
            eval_targets=eval_targets,
            tau=float(tau_at_step(global_step)),
        )
    return metrics

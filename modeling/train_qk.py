import torch
import torch.nn.functional as F
from modeling.mfa import save_mfa
from tqdm import tqdm
import itertools


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


@torch.no_grad()
def _eval_nll(model, loader, device, *, eval_targets=None):
    model.eval()
    tot_nll, tot_q_mse, tot_k_mse, tot_n = 0.0, 0.0, 0.0, 0
    pred_labels = []
    true_labels = []
    for batch in loader:
        _q, _k, labels = _batch_qk_and_labels(batch)
        _q = _q.view(_q.size(0), -1).to(device)
        _k = _k.view(_k.size(0), -1).to(device)
        nll = model.nll(_q, _k)  # mean over batch
        q_hat, k_hat = model.reconstruct(_q, _k, use_mixture_mean=True)
        B = _q.size(0)
        tot_nll += nll.item() * B
        tot_q_mse += ((_q - q_hat) ** 2).mean().item() * B
        tot_k_mse += ((_k - k_hat) ** 2).mean().item() * B
        tot_n += B
        if labels is not None:
            pred_labels.append(
                model.responsibilities(_q, _k).argmax(dim=1).detach().cpu()
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

        mu_q_hat = model.mu_q.detach()
        mu_k_hat = model.mu_k.detach()
        A_hat, B_hat = model.A_B
        A_hat = A_hat.detach()
        B_hat = B_hat.detach()

        K_true = int(mu_q_true.size(0))
        K_hat = int(mu_q_hat.size(0))
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

            # coup_err = []
            # coup_corr = []
            # sub_a = []
            # sub_b = []
            # for c in range(K_true):
            #    C_true = A_true[c] @ B_true[c].T
            #    C_hat = A_hat[c] @ B_hat[c].T
            #    coup_err.append((C_hat - C_true).norm().item())
            #    coup_corr.append(
            #        F.cosine_similarity(C_true.flatten(), C_hat.flatten(), dim=0).item()
            #    )
            #    #sub_a.append(_subspace_similarity(A_true[c], A_hat[c]))
            #    #sub_b.append(_subspace_similarity(B_true[c], B_hat[c]))

            # out["coup_err_mean"] = float(sum(coup_err) / K_true)
            # out["coup_corr_mean"] = float(sum(coup_corr) / K_true)
            # out["subA"] = float(sum(sub_a) / K_true)
            # out["subB"] = float(sum(sub_b) / K_true)
        else:
            out["mean_err_q"] = float("nan")
            out["mean_err_k"] = float("nan")
            # out["coup_err_mean"] = float("nan")
            # out["coup_corr_mean"] = float("nan")
            # out["subA"] = float("nan")
            # out["subB"] = float("nan")

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
):
    """
    Train with NLL, keep the best (lowest) NLL model.
    Works with loaders
    """
    device = next(model.parameters()).device
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_metric = float("inf")
    best_state = _cpu_state_dict(model)
    best_epoch = 0

    for ep in range(1, epochs + 1):
        model.train()
        total_nll, total_n = 0.0, 0

        iterable = enumerate(loader, 1)
        pbar = tqdm(iterable, total=steps_per_epoch)

        for batch_idx, batch in pbar:
            _q = batch[0]  # [B, d_head]
            _k = batch[1]  # [B, d_head]
            opt.zero_grad(set_to_none=True)
            nll = model.nll(_q, _k)  # mean over batch
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
                    f"Epoch {ep:02d} | Step {batch_idx:06d} Train NLL={avg_so_far:.6f}"
                )

            if steps_per_epoch is not None and batch_idx >= steps_per_epoch:
                break

            # free ASAP
            del _q, _k, nll

        if total_n == 0:
            avg_train_nll = float("nan")
        else:
            avg_train_nll = total_nll / total_n

        if val_loader is not None:
            eval_dl = eval_loader if eval_loader is not None else val_loader
            val_metrics = _eval_nll(model, eval_dl, device, eval_targets=eval_targets)
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
                # f"coup_err={val_metrics['coup_err_mean']:.6f} "
                # f"coup_corr={val_metrics['coup_corr_mean']:.6f} "
                # f"subA={val_metrics['subA']:.6f} "
                # f"subB={val_metrics['subB']:.6f}"
            )

    model.load_state_dict(best_state)
    print(
        f"Restored best model from epoch {best_epoch:02d} with metric={best_metric:.6f}"
    )

    metrics = dict(best_epoch=best_epoch, best_metric=best_metric)
    if val_loader is not None:
        eval_dl = eval_loader if eval_loader is not None else val_loader
        metrics["val"] = _eval_nll(model, eval_dl, device, eval_targets=eval_targets)
    return metrics

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, List, Tuple

import torch
import yaml
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split

from data_utils.concept_dataset import SupervisedConceptDataset
from dgp.mixture import sample_mixture_pcca
from initializations.projected_knn_qk import ReservoirKMeans
from llm_utils.qk_generator import QKGenerator
from modeling.qk_mfa import QKMFA
from modeling.train_qk import train_nll


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_heads_field(raw: Any) -> List[Tuple[int, int]]:
    if isinstance(raw, str):
        out: List[Tuple[int, int]] = []
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            layer_s, head_s = item.split(":")
            out.append((int(layer_s), int(head_s)))
        if not out:
            raise ValueError("`data.heads` must not be empty.")
        return out

    if isinstance(raw, list):
        out = []
        for item in raw:
            if (
                isinstance(item, (list, tuple))
                and len(item) == 2
                and all(isinstance(v, int) for v in item)
            ):
                out.append((item[0], item[1]))
            else:
                raise ValueError(
                    "`data.heads` list entries must be two-int lists, e.g. [[16, 1]]."
                )
        if not out:
            raise ValueError("`data.heads` must not be empty.")
        return out

    raise ValueError("`data.heads` must be a string ('16:1,17:3') or a list.")


def make_run_dir(base_dir: Path, run_name: str | None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = run_name if run_name else f"qk_run_{stamp}"
    run_dir = base_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError("YAML config must parse to a mapping/object.")
    return raw


def validate_config(cfg: dict) -> None:
    required_top = ["data", "split", "knn", "mfa", "output"]
    missing = [k for k in required_top if k not in cfg]
    if missing:
        raise ValueError(f"Missing top-level config sections: {missing}")

    if not (0.0 < float(cfg["split"]["val_frac"]) < 1.0):
        raise ValueError("`split.val_frac` must be in (0, 1).")
    if not (0.0 < float(cfg["knn"]["pool_frac"]) <= 1.0):
        raise ValueError("`knn.pool_frac` must be in (0, 1].")
    if int(cfg["data"]["max_pairs"]) <= 0:
        raise ValueError("`data.max_pairs` must be > 0.")
    if int(cfg["knn"]["num_centroids"]) <= 0:
        raise ValueError("`knn.num_centroids` must be > 0.")
    if "num_samples" in cfg["knn"] and cfg["knn"]["num_samples"] is not None:
        if int(cfg["knn"]["num_samples"]) <= 0:
            raise ValueError("`knn.num_samples` must be > 0 when provided.")
    if int(cfg["mfa"]["rank"]) <= 0:
        raise ValueError("`mfa.rank` must be > 0.")
    if int(cfg["mfa"]["epochs"]) <= 0:
        raise ValueError("`mfa.epochs` must be > 0.")

    data_source = str(cfg["data"].get("source", "llm"))
    if data_source not in {"llm", "toy_mixture"}:
        raise ValueError("`data.source` must be either 'llm' or 'toy_mixture'.")

    if data_source == "toy_mixture":
        toy = cfg["data"].get("toy_mixture")
        if not isinstance(toy, dict):
            raise ValueError(
                "`data.toy_mixture` must be provided for source='toy_mixture'."
            )
        required_toy = ["num_samples", "num_components", "dq", "dk", "rank", "snr"]
        toy_missing = [k for k in required_toy if k not in toy]
        if toy_missing:
            raise ValueError(f"Missing toy mixture config fields: {toy_missing}")
        if int(toy["num_samples"]) <= 0:
            raise ValueError("`data.toy_mixture.num_samples` must be > 0.")
        if int(toy["num_components"]) <= 0:
            raise ValueError("`data.toy_mixture.num_components` must be > 0.")
        if int(toy["dq"]) <= 0 or int(toy["dk"]) <= 0:
            raise ValueError(
                "`data.toy_mixture.dq` and `data.toy_mixture.dk` must be > 0."
            )
        if int(toy["rank"]) <= 0:
            raise ValueError("`data.toy_mixture.rank` must be > 0.")
        if float(toy["snr"]) <= 0.0:
            raise ValueError("`data.toy_mixture.snr` must be > 0.")
        if int(toy["dq"]) != int(toy["dk"]):
            raise ValueError(
                "`data.toy_mixture.dq` must equal `data.toy_mixture.dk` "
                "because QKMFA currently requires q and k to have the same dimension."
            )


def get_default_config() -> dict:
    return {
        "seed": 42,
        "data": {
            "source": "llm",
            "data_path": "./data/supervised.json",
            "model_name": "meta-llama/Llama-3.1-8B-Instruct",
            "heads": [[16, 1]],
            "data_device": "auto",
            "model_device": "auto",
            "max_pairs": 250000,
            "qk_batch_size": 16,
            "top_k": 8,
            "attn_min": 0.01,
            "toy_mixture": {
                "num_samples": 50000,
                "num_components": 8,
                "dq": 128,
                "dk": 128,
                "rank": 8,
                "snr": 3.0,
            },
        },
        "split": {
            "train_batch_size": 128,
            "val_frac": 0.1,
        },
        "knn": {
            "pool_frac": 0.2,
            "num_centroids": 500,
            "num_samples": None,
            "proj_dim": 32,
            "metric": "euclidean",
            "kmeans_iters": 50,
            "kmeans_restarts": 10,
            "tol": 1e-4,
            "refine_epochs": 25,
        },
        "mfa": {
            "rank": 10,
            "epochs": 30,
            "lr": 1e-3,
            "grad_clip": None,
        },
        "output": {
            "output_dir": "./runs",
            "run_name": None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train one QK-MFA run from YAML config."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional override for output.run_name in config.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional override for output.output_dir in config.",
    )
    args = parser.parse_args()

    cfg = deep_merge(get_default_config(), load_yaml(Path(args.config)))
    if args.run_name is not None:
        cfg["output"]["run_name"] = args.run_name
    if args.output_dir is not None:
        cfg["output"]["output_dir"] = args.output_dir
    validate_config(cfg)

    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data_cfg = cfg["data"]
    split_cfg = cfg["split"]
    knn_cfg = cfg["knn"]
    mfa_cfg = cfg["mfa"]
    output_cfg = cfg["output"]

    data_device = resolve_device(str(data_cfg["data_device"]))
    model_device = resolve_device(str(data_cfg["model_device"]))
    data_source = str(data_cfg.get("source", "llm"))
    heads = parse_heads_field(data_cfg["heads"]) if data_source == "llm" else []

    run_dir = make_run_dir(
        Path(str(output_cfg["output_dir"])),
        output_cfg["run_name"],
    )
    print(f"[run] Output directory: {run_dir}")

    resolved_cfg = deep_merge(
        cfg,
        {
            "data": {
                "data_device": data_device,
                "model_device": model_device,
                "source": data_source,
            },
            "data_resolved": {"heads": heads},
        },
    )
    (run_dir / "config.json").write_text(json.dumps(resolved_cfg, indent=2) + "\n")

    toy_params = None
    c_all = None
    if data_source == "llm":
        dataset_obj = SupervisedConceptDataset(str(data_cfg["data_path"]))
        if len(dataset_obj) == 0:
            raise ValueError(f"No examples found at {data_cfg['data_path']}")

        qk_generator = QKGenerator(
            str(data_cfg["model_name"]),
            model_device=model_device,
            data_device=data_device,
        )

        print("[run] Generating query/key vectors...")
        query_vecs, key_vecs, _ = qk_generator.generate_query_key_vecs(
            dataset_obj,
            heads,
            batch_size=int(data_cfg["qk_batch_size"]),
            top_k=int(data_cfg["top_k"]),
            attn_min=float(data_cfg["attn_min"]),
        )
        max_pairs = int(data_cfg["max_pairs"])
        q_all = query_vecs[0][:max_pairs]
        k_all = key_vecs[0][:max_pairs]
        vocab_size = qk_generator.model.config.vocab_size
    else:
        toy_cfg = data_cfg["toy_mixture"]
        toy_device = model_device
        print("[run] Generating toy mixture query/key vectors...")
        q_all, k_all, c_all, toy_params = sample_mixture_pcca(
            N=int(toy_cfg["num_samples"]),
            K=int(toy_cfg["num_components"]),
            dq=int(toy_cfg["dq"]),
            dk=int(toy_cfg["dk"]),
            r=int(toy_cfg["rank"]),
            snr=float(toy_cfg["snr"]),
            device=toy_device,
            seed=seed,
        )
        max_pairs = int(data_cfg["max_pairs"])
        q_all = q_all[:max_pairs]
        k_all = k_all[:max_pairs]
        c_all = c_all[:max_pairs]
        torch.save(c_all.detach().cpu(), run_dir / "toy_labels.pt")
        vocab_size = int(toy_cfg["num_components"])

    # Keep dataset tensors on CPU for DataLoader compatibility.
    # QKMFA will move batches to the model device internally.
    q_all = q_all.detach().cpu()
    k_all = k_all.detach().cpu()

    if q_all.size(0) == 0:
        if data_source == "llm":
            raise ValueError(
                "No query/key pairs after filtering. Try lowering `data.attn_min`."
            )
        raise ValueError(
            "No toy query/key pairs were generated; check `data.toy_mixture` settings."
        )

    full_ds = TensorDataset(q_all, k_all)
    val_size = max(1, int(len(full_ds) * float(split_cfg["val_frac"])))
    train_size = len(full_ds) - val_size
    if train_size <= 0:
        raise ValueError("Not enough samples for train/validation split.")

    train_ds, val_ds = random_split(
        full_ds,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_batch_size = int(split_cfg["train_batch_size"])
    pin_memory = model_device == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=train_batch_size,
        shuffle=True,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )
    val_metrics_loader = val_loader
    if c_all is not None:
        c_all = c_all.detach().cpu()
        full_ds_labeled = TensorDataset(q_all, k_all, c_all)
        val_ds_labeled = Subset(full_ds_labeled, val_ds.indices)
        val_metrics_loader = DataLoader(
            val_ds_labeled,
            batch_size=train_batch_size,
            shuffle=False,
            pin_memory=pin_memory,
        )

    knn_num_samples_raw = knn_cfg.get("num_samples")
    if knn_num_samples_raw is None:
        knn_ds = train_ds
    else:
        knn_num_samples = min(int(knn_num_samples_raw), len(train_ds))
        knn_indices = torch.randperm(len(train_ds), generator=torch.Generator().manual_seed(seed))[
            :knn_num_samples
        ]
        knn_ds = Subset(train_ds, knn_indices.tolist())

    knn_loader = DataLoader(
        knn_ds,
        batch_size=train_batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )

    pool_size = max(1, round(len(knn_loader.dataset) * float(knn_cfg["pool_frac"])))
    num_centroids = min(int(knn_cfg["num_centroids"]), pool_size)
    if num_centroids < 1:
        raise ValueError(
            "No centroids can be initialized with current dataset/pool size."
        )

    print(
        f"[run] Initializing centroids with knn_samples={len(knn_loader.dataset)}, "
        f"pool_size={pool_size}, num_centroids={num_centroids}..."
    )
    knn_q = ReservoirKMeans(
        n_clusters=num_centroids,
        pool_size=pool_size,
        query_or_key="query",
        vocab_size=vocab_size,
        seed=seed,
        device=model_device,
        metric=str(knn_cfg["metric"]),
        proj_dim=int(knn_cfg["proj_dim"]),
        kmeans_iters=int(knn_cfg["kmeans_iters"]),
        kmeans_restarts=int(knn_cfg["kmeans_restarts"]),
        tol=float(knn_cfg["tol"]),
    )
    q_centroids = knn_q.fit(knn_loader, refine_epochs=int(knn_cfg["refine_epochs"]))

    knn_k = ReservoirKMeans(
        n_clusters=num_centroids,
        pool_size=pool_size,
        query_or_key="key",
        vocab_size=vocab_size,
        seed=seed,
        device=model_device,
        metric=str(knn_cfg["metric"]),
        proj_dim=int(knn_cfg["proj_dim"]),
        kmeans_iters=int(knn_cfg["kmeans_iters"]),
        kmeans_restarts=int(knn_cfg["kmeans_restarts"]),
        tol=float(knn_cfg["tol"]),
    )
    k_centroids = knn_k.fit(knn_loader, refine_epochs=int(knn_cfg["refine_epochs"]))

    model = QKMFA(
        q_centroids=q_centroids,
        k_centroids=k_centroids,
        rank=int(mfa_cfg["rank"]),
        lambda_nuc=float(mfa_cfg["lambda_nuc"]),
    ).to(model_device)

    print("[run] Training...")
    metrics = train_nll(
        model,
        train_loader,
        val_loader=val_loader,
        epochs=int(mfa_cfg["epochs"]),
        lr=float(mfa_cfg["lr"]),
        grad_clip=mfa_cfg["grad_clip"],
        eval_loader=val_metrics_loader,
        eval_targets={"toy_params": toy_params} if toy_params is not None else None,
    )

    torch.save(
        {"q_centroids": q_centroids, "k_centroids": k_centroids},
        run_dir / "centroids.pt",
    )
    if toy_params is not None:
        torch.save(
            {k: v.detach().cpu() for k, v in toy_params.items()},
            run_dir / "toy_params.pt",
        )
    torch.save(model.state_dict(), run_dir / "model_state.pt")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    print("[run] Done.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

"""Entry point CLI para entrenar v2_pinn end-to-end (headless cluster-friendly).

Pipeline completo en un solo proceso:

    1. `data.load_dataset(h5_path)` → metadata.
    2. `data.split_indices(N, ...)` → train/val/test deterministas.
    3. `data.compute_train_stats(...)` → x_scaler + b_mean/std + pts_mean/std.
    4. `data.make_loader(...)` × 2 (train, val).
    5. `model.LitPINN(...)` con stats inyectados como buffers (viajan en ckpt).
    6. `train.train(...)` → fit con EarlyStopping + ModelCheckpoint + CometLogger.
    7. `metrics.evaluate(...)` × 3 (train/val/test) → RMSE + R² en mT.
    8. `torch.save(aux.pt)` con scalers + splits + métricas.

Uso:
    python -m Models.v2_pinn \\
        --h5 /path/v1_xy100_z225_step10_n5000.h5 \\
        --epochs 200 --hidden 256 256 256 256 \\
        --balance-grads --manual-clip 1.0 \\
        --out-dir /path/data/modelos --run-tag v2_pinn_5k_balanced

Diseñado para correrse desde un .sbatch en el cluster UC. La notebook
`Python/v2_pinn.ipynb` consume las mismas funciones, no este `__main__`.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch

from . import data
from .model import LitPINN, count_params
from .train import train as pinn_train
from .metrics import evaluate, report


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m Models.v2_pinn",
        description="Entrena v2_pinn (PINN per-componente) sobre un dataset HDF5.",
    )
    # Datos
    p.add_argument("--h5", required=True, type=Path,
                   help="Ruta al HDF5 generado por generate_dataset.jl.")
    p.add_argument("--val-frac",  type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed",      type=int,   default=42)

    # Modelo
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 256, 256, 256],
                   help="Tamaños de las capas ocultas (lista). Ej: --hidden 256 256 256 256")
    p.add_argument("--activation", choices=["silu", "tanh", "gelu"], default="silu")

    # Losses
    p.add_argument("--lambda-div", type=float, default=1e-2)
    p.add_argument("--lambda-rot", type=float, default=1e-2)
    p.add_argument("--lambda-tv",  type=float, default=1e-4)
    p.add_argument("--balance-grads", action="store_true",
                   help="Reescala grads de div/rot/tv para igualar norma de data.")
    p.add_argument("--balance-eps", type=float, default=1e-8)

    # Optimización
    p.add_argument("--epochs",       type=int,   default=200)
    p.add_argument("--patience",     type=int,   default=20)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip",    type=float, default=1.0,
                   help="Gradient clip. En modo balanceado se aplica vía manual-clip.")

    # Batching
    p.add_argument("--points-per-sample", type=int, default=4096,
                   help="K — puntos al azar por step (1 sample por step).")

    # I/O y runtime
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Directorio para ckpt + aux.pt. Default: <repo>/data/modelos.")
    p.add_argument("--run-tag", type=str, default=None,
                   help="Identificador del run. Default: v2_pinn_<h5_stem>.")
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers. Default 0 — el dataset se carga "
                        "entero a RAM en PointDataset, no hay I/O que paralelizar.")
    p.add_argument("--pin-memory",  action="store_true",
                   help="DataLoader pin_memory. Recomendado en cluster con CUDA "
                        "(DMA async CPU→GPU).")
    p.add_argument("--accelerator", default="auto", choices=["auto", "gpu", "cpu"])
    p.add_argument("--devices",     default="auto",
                   help="auto | <N> | índices CSV (ej. '0,1'). Lightning lo parsea.")
    p.add_argument("--no-deterministic", action="store_true",
                   help="Desactiva modo determinístico para más velocidad.")

    return p.parse_args(argv)


def _resolve_devices(devices_arg):
    if devices_arg == "auto":
        return "auto"
    if "," in devices_arg:
        return [int(x) for x in devices_arg.split(",")]
    try:
        return int(devices_arg)
    except ValueError:
        return devices_arg


def main(argv=None):
    args = _parse_args(argv)

    # --- Setup ---
    pl.seed_everything(args.seed, workers=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    h5_path = args.h5.resolve()
    if not h5_path.exists():
        print(f"ERROR: HDF5 no existe: {h5_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = (args.out_dir or h5_path.parents[1] / "modelos").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_tag = args.run_tag or f"v2_pinn_{h5_path.stem}"

    print(f"[v2_pinn] device={device}  h5={h5_path}  out_dir={out_dir}  run_tag={run_tag}")
    print(f"[v2_pinn] hparams: hidden={args.hidden}  act={args.activation}  "
          f"lr={args.lr}  wd={args.weight_decay}  K={args.points_per_sample}  "
          f"balance={args.balance_grads}")

    # --- Data ---
    t0 = time.time()
    ds = data.load_dataset(h5_path)
    splits = data.split_indices(ds["N"], val_frac=args.val_frac,
                                test_frac=args.test_frac, seed=args.seed)
    stats  = data.compute_train_stats(h5_path, splits["train"], chunk=64)
    x_scaler = stats["x_scaler"]
    b_mean, b_std = stats["b_mean"], stats["b_std"]
    pts_mean, pts_std = stats["pts_mean"], stats["pts_std"]
    print(f"[v2_pinn] N={ds['N']}  I={ds['I']}  J={ds['J']}  "
          f"train={splits['n_train']} val={splits['n_val']} test={splits['n_test']}")
    print(f"[v2_pinn] b_mean={b_mean.tolist()}  b_std={b_std.tolist()}")
    print(f"[v2_pinn] pts_mean={pts_mean.tolist()}  pts_std={pts_std.tolist()}")
    print(f"[v2_pinn] stats listos en {time.time() - t0:.1f}s")

    loader_kwargs = dict(
        points_per_sample=args.points_per_sample,
        x_scaler=x_scaler,
        b_mean=b_mean, b_std=b_std,
        pts_mean=pts_mean, pts_std=pts_std,
        num_workers=args.num_workers, pin_memory=args.pin_memory,
    )
    loader_tr = data.make_loader(h5_path, splits["train"], shuffle=True,
                                 seed=args.seed, **loader_kwargs)
    loader_va = data.make_loader(h5_path, splits["val"],   shuffle=False,
                                 seed=args.seed + 1, **loader_kwargs)

    # --- Model ---
    lit_model = LitPINN(
        n_sensors=ds["I"], hidden_layers=list(args.hidden),
        activation=args.activation,
        lr=args.lr, weight_decay=args.weight_decay,
        lambda_div=args.lambda_div, lambda_rot=args.lambda_rot, lambda_tv=args.lambda_tv,
        balance_grads=args.balance_grads, balance_eps=args.balance_eps,
        manual_clip_val=(args.grad_clip if args.balance_grads else None),
        b_mean=tuple(b_mean.tolist()), b_std=tuple(b_std.tolist()),
        pts_mean=tuple(pts_mean.tolist()), pts_std=tuple(pts_std.tolist()),
    )
    print(f"[v2_pinn] params entrenables: {count_params(lit_model):,}")

    # --- Train ---
    trainer = pinn_train(
        lit_model, loader_tr, loader_va,
        n_epochs=args.epochs, patience=args.patience,
        ckpt_dir=out_dir, run_tag=run_tag,
        accelerator=args.accelerator,
        devices=_resolve_devices(args.devices),
        deterministic=(not args.no_deterministic),
        gradient_clip_val=(None if args.balance_grads else args.grad_clip),
    )
    best_ckpt_path = trainer.checkpoint_callback.best_model_path
    print(f"[v2_pinn] best ckpt: {best_ckpt_path}")
    lit_model = LitPINN.load_from_checkpoint(best_ckpt_path)

    # --- Evaluate ---
    eval_kwargs = dict(
        x_scaler=x_scaler, b_mean=b_mean, b_std=b_std,
        pts_mean=pts_mean, pts_std=pts_std, device=device,
        rmse_per_component=True,
    )
    m_tr = evaluate(lit_model, h5_path, splits["train"], **eval_kwargs)
    m_va = evaluate(lit_model, h5_path, splits["val"],   **eval_kwargs)
    m_te = evaluate(lit_model, h5_path, splits["test"],  **eval_kwargs)
    report("train", m_tr); report("val", m_va); report("test", m_te)

    # --- Persist scalers + metadata ---
    aux_path = out_dir / f"{run_tag}_aux.pt"
    torch.save({
        "x_scaler": x_scaler,
        "x_mean":   x_scaler.mean_.astype(np.float32),
        "x_scale":  x_scaler.scale_.astype(np.float32),
        "b_mean":   b_mean.astype(np.float32),
        "b_std":    b_std.astype(np.float32),
        "pts_mean": pts_mean.astype(np.float32),
        "pts_std":  pts_std.astype(np.float32),
        "splits":   splits,
        "metrics":  {"train": m_tr, "val": m_va, "test": m_te},
        "hparams":  vars(args) | {"hidden_layers": list(args.hidden)},
        "h5_path":  str(h5_path),
        "ckpt_path": best_ckpt_path,
    }, aux_path)
    print(f"[v2_pinn] aux: {aux_path}")

    # JSON resumido para parsing rápido en el cluster (logs).
    summary_path = out_dir / f"{run_tag}_summary.json"
    summary_path.write_text(json.dumps({
        "run_tag": run_tag,
        "h5_path": str(h5_path),
        "ckpt_path": str(best_ckpt_path),
        "metrics": {"train": m_tr, "val": m_va, "test": m_te},
    }, indent=2))
    print(f"[v2_pinn] summary: {summary_path}")


if __name__ == "__main__":
    main()

"""Script CLI para entrenar v3_pwnn (point-wise By-only) en cluster o local.

Equivalente a `train_v2_1.py` pero con:
- Input/output reducidos a By solamente.
- Dataset pre-cargado a GPU (`--data-device cuda`) — elimina el bottleneck
  de host→device transfer per step que limitaba a v2.1.

Uso típico:

    python python/train_v3.py \\
        --h5 data/datasets/v1_xy100_z225_step10_n5000.h5 \\
        --epochs 100 \\
        --comet-project pmdkernel \\
        --run-tag v3_pwnn_n5000

Requiere `COMET_API_KEY` y `COMET_WORKSPACE` en env (o `~/.comet.config`).
"""
import argparse
import json
import sys
import time
from pathlib import Path

# Comet ML antes de torch para enganchar auto-logging hooks.
import comet_ml  # noqa: F401

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import torch
import pytorch_lightning as pl

from Models.v3_pwnn import data
from Models.v3_pwnn.model import LitFCNN, count_params
from Models.v3_pwnn.train import train as fit_model
from Models.v3_pwnn.metrics import evaluate, report


def build_parser():
    p = argparse.ArgumentParser(description="Train v3_pwnn (point-wise, By-only) on a dataset.")
    p.add_argument("--h5", type=Path, required=True)
    p.add_argument("--ckpt-dir", type=Path, default=None)
    p.add_argument("--run-tag",  type=str,  default=None)

    p.add_argument("--val-frac",  type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed",      type=int,   default=42)

    p.add_argument("--hidden-layers", type=int, nargs="+",
                   default=[256, 256, 256, 256])
    p.add_argument("--activation", type=str, default="silu",
                   choices=["silu", "gelu", "relu", "tanh"])

    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--weight-decay",  type=float, default=1e-5)
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--patience",      type=int,   default=20)
    p.add_argument("--grad-clip",     type=float, default=None)

    p.add_argument("--points-per-sample", type=int, default=4096)
    p.add_argument("--data-device", type=str, default=None,
                   choices=["cpu", "cuda"],
                   help="Device donde pre-cargar el dataset. Default: cuda si "
                        "está disponible, si no cpu. 'cuda' elimina el "
                        "host→GPU transfer per step.")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--pin-memory",  action="store_true")

    p.add_argument("--comet-project", type=str, default="pmdkernel")
    p.add_argument("--log-every-n",   type=int, default=50)

    p.add_argument("--matmul-precision", type=str, default="high",
                   choices=["highest", "high", "medium"])
    p.add_argument("--quick", action="store_true")
    return p


def print_banner(title):
    print(f"\n=== {title} ===", flush=True)


def print_env():
    print_banner("ENV")
    print(f"python   : {sys.version.split()[0]}")
    print(f"torch    : {torch.__version__}")
    print(f"lightning: {pl.__version__}")
    cuda = torch.cuda.is_available()
    print(f"cuda     : {cuda}")
    if cuda:
        print(f"device   : {torch.cuda.get_device_name(0)}")
        print(f"capabil. : {torch.cuda.get_device_capability(0)}")


def print_hparams(args):
    print_banner("HPARAMS")
    for k, v in sorted(vars(args).items()):
        print(f"{k:20s}: {v}")


def print_dataset_info(ds):
    print_banner("DATASET")
    print(f"H5 path        : {ds['h5_path']}")
    print(f"N samples      : {ds['N']}")
    print(f"Sensores I     : {ds['I']}  (input I + 3 = {ds['I'] + 3})")
    print(f"Grilla J       : {ds['Nx']} x {ds['Ny']} x {ds['Nz']} = {ds['J']}")
    print(f"Rango x        : [{ds['grid_x'].min():.0f}, {ds['grid_x'].max():.0f}] mm")
    print(f"Rango y        : [{ds['grid_y'].min():.0f}, {ds['grid_y'].max():.0f}] mm")
    print(f"Rango z        : [{ds['grid_z'].min():.0f}, {ds['grid_z'].max():.0f}] mm")
    print(f"Perturb        : kind={ds['attrs'].get('kind')}, sigma_deg={ds['attrs'].get('sigma_deg')}")


def save_aux(aux_path, *, x_scaler, b_mean, b_std, pts_mean, pts_std,
             splits, metrics, hparams, h5_path, ckpt_path, run_tag, comet_url):
    """Persiste lo que Lightning no guarda en el `.ckpt`: scaler, splits,
    métricas, URL del run de Comet."""
    torch.save({
        "x_scaler":    x_scaler,
        "x_mean":      x_scaler.mean_.astype(np.float32),
        "x_scale":     x_scaler.scale_.astype(np.float32),
        "b_mean":      float(b_mean),
        "b_std":       float(b_std),
        "pts_mean":    np.asarray(pts_mean, dtype=np.float32),
        "pts_std":     np.asarray(pts_std,  dtype=np.float32),
        "splits":      splits,
        "metrics":     metrics,
        "hparams":     hparams,
        "h5_path":     str(h5_path),
        "ckpt_path":   str(ckpt_path),
        "run_tag":     run_tag,
        "comet_url":   comet_url,
    }, aux_path)


def main():
    args = build_parser().parse_args()

    if args.quick:
        args.epochs = 3
        args.patience = 2
        print("[quick mode] epochs=3, patience=2")

    if args.run_tag is None:
        args.run_tag = f"v3_pwnn_{args.h5.stem}"
    if args.ckpt_dir is None:
        args.ckpt_dir = SCRIPT_DIR / "Models" / "v3_pwnn" / "logs" / args.run_tag
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Default: cuda si está disponible, si no cpu.
    if args.data_device is None:
        args.data_device = "cuda" if torch.cuda.is_available() else "cpu"

    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision(args.matmul_precision)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print_env()
    print_hparams(args)

    print_banner("LOAD DATASET METADATA")
    ds = data.load_dataset_metadata(args.h5)
    print_dataset_info(ds)

    print_banner("SPLIT + SCALERS")
    splits = data.split_indices(ds["N"], val_frac=args.val_frac,
                                 test_frac=args.test_frac, seed=args.seed)
    print(f"train = {splits['n_train']}   val = {splits['n_val']}   test = {splits['n_test']}")

    t0 = time.time()
    stats = data.compute_train_stats(args.h5, splits["train"], chunk=64)
    print(f"compute_train_stats took {time.time() - t0:.1f}s")
    x_scaler = stats["x_scaler"]
    b_mean   = stats["b_mean"]
    b_std    = stats["b_std"]
    pts_mean = stats["pts_mean"]
    pts_std  = stats["pts_std"]
    print(f"b_mean   = {b_mean:.6e}")
    print(f"b_std    = {b_std:.6e}")
    print(f"pts_mean = {pts_mean}")
    print(f"pts_std  = {pts_std}")

    print_banner("DATALOADERS")
    print(f"data_device = {args.data_device}")
    loader_kwargs = dict(points_per_sample=args.points_per_sample,
                         x_scaler=x_scaler,
                         b_mean=b_mean, b_std=b_std,
                         pts_mean=pts_mean, pts_std=pts_std,
                         data_device=args.data_device,
                         num_workers=args.num_workers, pin_memory=args.pin_memory)
    t0 = time.time()
    loader_tr = data.make_loader(args.h5, splits["train"], shuffle=True,  seed=args.seed,     **loader_kwargs)
    loader_va = data.make_loader(args.h5, splits["val"],   shuffle=False, seed=args.seed + 1, **loader_kwargs)
    print(f"loaders ready in {time.time() - t0:.1f}s (pre-carga a {args.data_device})")

    print_banner("MODEL")
    lit_model = LitFCNN(
        n_sensors=ds["I"], hidden_layers=args.hidden_layers,
        activation=args.activation,
        lr=args.lr, weight_decay=args.weight_decay,
        b_mean=b_mean, b_std=b_std,
        pts_mean=tuple(pts_mean.tolist()), pts_std=tuple(pts_std.tolist()),
    )
    print(f"params (trainable): {count_params(lit_model):,}")

    print_banner("TRAIN")
    t_train = time.time()
    trainer = fit_model(
        lit_model, loader_tr, loader_va,
        n_epochs=args.epochs, patience=args.patience,
        ckpt_dir=args.ckpt_dir, run_tag=args.run_tag,
        gradient_clip_val=args.grad_clip,
        comet_project=args.comet_project,
        log_every_n_steps=args.log_every_n,
    )
    print(f"train wallclock: {time.time() - t_train:.1f}s ({(time.time() - t_train)/60:.1f} min)")
    best_ckpt_path = trainer.checkpoint_callback.best_model_path
    comet_url = (trainer.logger.experiment.url
                 if hasattr(trainer.logger, "experiment") else None)
    print(f"best ckpt    : {best_ckpt_path}")
    print(f"comet run url: {comet_url}")

    print_banner("EVAL")
    lit_model = LitFCNN.load_from_checkpoint(best_ckpt_path)
    eval_kwargs = dict(x_scaler=x_scaler,
                       b_mean=b_mean, b_std=b_std,
                       pts_mean=pts_mean, pts_std=pts_std,
                       device=device)
    m_tr = evaluate(lit_model, args.h5, splits["train"], **eval_kwargs)
    m_va = evaluate(lit_model, args.h5, splits["val"],   **eval_kwargs)
    m_te = evaluate(lit_model, args.h5, splits["test"],  **eval_kwargs)
    report("train", m_tr)
    report("val",   m_va)
    report("test",  m_te)

    print_banner("SAVE AUX")
    aux_path = args.ckpt_dir / f"{args.run_tag}_aux.pt"
    save_aux(
        aux_path,
        x_scaler=x_scaler,
        b_mean=b_mean, b_std=b_std,
        pts_mean=pts_mean, pts_std=pts_std,
        splits=splits,
        metrics={"train": m_tr, "val": m_va, "test": m_te},
        hparams=vars(args).copy(),
        h5_path=args.h5,
        ckpt_path=best_ckpt_path,
        run_tag=args.run_tag,
        comet_url=comet_url,
    )
    print(f"aux saved at: {aux_path}")

    summary = {
        "run_tag": args.run_tag,
        "ckpt":    str(best_ckpt_path),
        "aux":     str(aux_path),
        "comet":   comet_url,
        "rmse_mt": {"train": m_tr["rmse_mt"], "val": m_va["rmse_mt"], "test": m_te["rmse_mt"]},
        "r2":      {"train": m_tr["r2"],      "val": m_va["r2"],      "test": m_te["r2"]},
    }
    print_banner("SUMMARY")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

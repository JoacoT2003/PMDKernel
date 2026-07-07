"""Script CLI para entrenar v4_deepsets (DeepSets + relative coords, By-only).

Comparación apples-to-apples con v3_pwnn — mismo dataset, mismo split, mismo
target (By), mismo loss (MSE). Lo único que cambia es la arquitectura.

Uso típico:

    python python/train_v4.py \\
        --h5 data/datasets/v1_xy100_z225_step10_n5000.h5 \\
        --epochs 100 \\
        --run-tag v4_deepsets_n5000
"""
import argparse
import json
import sys
import time
from pathlib import Path

import comet_ml  # noqa: F401

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import torch
import pytorch_lightning as pl

from Models.v4_deepsets import data
from Models.v4_deepsets.model import LitDeepSet, count_params
from Models.v4_deepsets.train import train as fit_model
from Models.v4_deepsets.metrics import evaluate, report


def build_parser():
    p = argparse.ArgumentParser(description="Train v4_deepsets (DeepSets, By-only).")
    p.add_argument("--h5", type=Path, required=True)
    p.add_argument("--ckpt-dir", type=Path, default=None)
    p.add_argument("--run-tag",  type=str,  default=None)

    p.add_argument("--val-frac",  type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed",      type=int,   default=42)

    # DeepSets arch
    p.add_argument("--encoder-hidden", type=int, nargs="+", default=[64, 64])
    p.add_argument("--decoder-hidden", type=int, nargs="+", default=[128, 64])
    p.add_argument("--embed-dim",      type=int, default=128)
    p.add_argument("--activation",     type=str, default="relu",
                   choices=["silu", "gelu", "relu", "tanh"])

    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--weight-decay",  type=float, default=1e-5)
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--patience",      type=int,   default=20)
    p.add_argument("--grad-clip",     type=float, default=None)

    p.add_argument("--points-per-sample", type=int, default=4096)
    p.add_argument("--batch-size",        type=int, default=32)
    p.add_argument("--precision",         type=str, default="16-mixed",
                   choices=["32-true", "16-mixed", "bf16-mixed"])
    p.add_argument("--data-device",       type=str, default=None,
                   choices=["cpu", "cuda"])
    p.add_argument("--num-workers",       type=int, default=0)
    p.add_argument("--pin-memory",        action="store_true")

    p.add_argument("--comet-project", type=str, default="pmdkernel")
    p.add_argument("--log-every-n",   type=int, default=50)
    p.add_argument("--no-progress",   action="store_true",
                   help="Disable Lightning progress bar (recommended for Colab).")

    p.add_argument("--matmul-precision", type=str, default="high",
                   choices=["highest", "high", "medium"])
    p.add_argument("--quick", action="store_true")
    return p


def print_banner(t):
    print(f"\n=== {t} ===", flush=True)


def main():
    args = build_parser().parse_args()

    if args.quick:
        args.epochs = 3
        args.patience = 2
        print("[quick mode] epochs=3, patience=2")

    if args.run_tag is None:
        args.run_tag = f"v4_deepsets_{args.h5.stem}"
    if args.ckpt_dir is None:
        args.ckpt_dir = SCRIPT_DIR / "Models" / "v4_deepsets" / "logs" / args.run_tag
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    if args.data_device is None:
        args.data_device = "cuda" if torch.cuda.is_available() else "cpu"

    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision(args.matmul_precision)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print_banner("DATASET")
    ds = data.load_dataset_metadata(args.h5)
    print(f"N={ds['N']}  I={ds['I']}  J={ds['J']}  ({ds['Nx']}x{ds['Ny']}x{ds['Nz']})")

    if args.points_per_sample > ds["J"]:
        print(f"[auto] points-per-sample {args.points_per_sample} > J={ds['J']}, clamping to J")
        args.points_per_sample = ds["J"]

    print_banner("HPARAMS")
    for k, v in sorted(vars(args).items()):
        print(f"{k:20s}: {v}")

    print_banner("SPLIT + STATS")
    splits = data.split_indices(ds["N"], val_frac=args.val_frac,
                                 test_frac=args.test_frac, seed=args.seed)
    print(f"train={splits['n_train']}  val={splits['n_val']}  test={splits['n_test']}")

    t0 = time.time()
    stats = data.compute_train_stats(args.h5, splits["train"], chunk=64, seed=args.seed)
    print(f"compute_train_stats: {time.time() - t0:.1f}s")
    b_mean, b_std         = stats["b_mean"], stats["b_std"]
    feat_mean, feat_std   = stats["feat_mean"], stats["feat_std"]
    sensor_xyz            = stats["sensor_xyz"]
    print(f"b_mean    = {b_mean:.6e}    b_std    = {b_std:.6e}")
    print(f"feat_mean = {feat_mean}")
    print(f"feat_std  = {feat_std}")

    print_banner("LOADERS")
    print(f"data_device = {args.data_device}")
    loader_kwargs = dict(points_per_sample=args.points_per_sample,
                         batch_size=args.batch_size,
                         b_mean=b_mean, b_std=b_std,
                         data_device=args.data_device,
                         num_workers=args.num_workers, pin_memory=args.pin_memory)
    t0 = time.time()
    loader_tr = data.make_loader(args.h5, splits["train"], shuffle=True,  seed=args.seed,     **loader_kwargs)
    loader_va = data.make_loader(args.h5, splits["val"],   shuffle=False, seed=args.seed + 1, **loader_kwargs)
    print(f"loaders ready: {time.time() - t0:.1f}s")

    print_banner("MODEL")
    lit_model = LitDeepSet(
        n_sensors=ds["I"], sensor_xyz=sensor_xyz,
        encoder_hidden=args.encoder_hidden, embed_dim=args.embed_dim,
        decoder_hidden=args.decoder_hidden, activation=args.activation,
        lr=args.lr, weight_decay=args.weight_decay,
        b_mean=b_mean, b_std=b_std,
        feat_mean=tuple(feat_mean.tolist()), feat_std=tuple(feat_std.tolist()),
    )
    print(f"params (trainable): {count_params(lit_model):,}")

    print_banner("TRAIN")
    t_train = time.time()
    trainer = fit_model(
        lit_model, loader_tr, loader_va,
        n_epochs=args.epochs, patience=args.patience,
        ckpt_dir=args.ckpt_dir, run_tag=args.run_tag,
        gradient_clip_val=args.grad_clip,
        precision=args.precision,
        comet_project=args.comet_project,
        log_every_n_steps=args.log_every_n,
        enable_progress_bar=not args.no_progress,
    )
    elapsed = time.time() - t_train
    print(f"train wallclock: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    best_ckpt_path = trainer.checkpoint_callback.best_model_path
    comet_url = (trainer.logger.experiment.url
                 if hasattr(trainer.logger, "experiment") else None)
    print(f"best ckpt    : {best_ckpt_path}")
    print(f"comet url    : {comet_url}")

    print_banner("EVAL")
    lit_model = LitDeepSet.load_from_checkpoint(best_ckpt_path, weights_only=False)
    m_tr = evaluate(lit_model, args.h5, splits["train"], device)
    m_va = evaluate(lit_model, args.h5, splits["val"],   device)
    m_te = evaluate(lit_model, args.h5, splits["test"],  device)
    report("train", m_tr)
    report("val",   m_va)
    report("test",  m_te)

    print_banner("SAVE AUX")
    aux_path = args.ckpt_dir / f"{args.run_tag}_aux.pt"
    torch.save({
        "b_mean":     float(b_mean),
        "b_std":      float(b_std),
        "feat_mean":  np.asarray(feat_mean, dtype=np.float32),
        "feat_std":   np.asarray(feat_std,  dtype=np.float32),
        "sensor_xyz": np.asarray(sensor_xyz, dtype=np.float32),
        "splits":     splits,
        "metrics":    {"train": m_tr, "val": m_va, "test": m_te},
        "hparams":    vars(args).copy(),
        "h5_path":    str(args.h5),
        "ckpt_path":  str(best_ckpt_path),
        "run_tag":    args.run_tag,
        "comet_url":  comet_url,
    }, aux_path)
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

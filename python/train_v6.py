"""Script CLI para entrenar v6_deepxde (PI-DeepONet + Maxwell losses).

Reemplazo de v5 (DeepSets, ~10 min/época). DeepONet CartesianProd corre el
branch 1×/muestra y el trunk 1×/punto → ~10× más rápido por step (medido).

Uso típico:

    python python/train_v6.py \\
        --h5 data/datasets/v1_xyz100_step10_n5000.h5 \\
        --epochs 200 \\
        --lambda-div 1e-3 --lambda-curl 1e-3 \\
        --run-tag v6_deeponet_n5000
"""
import argparse
import json
import shutil
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

from Models.v6_deepxde import data
from Models.v6_deepxde.model import LitDeepONetPINN, count_params
from Models.v6_deepxde.train import train as fit_model
from Models.v6_deepxde.metrics import evaluate, report


def build_parser():
    p = argparse.ArgumentParser(description="Train v6_deepxde (PI-DeepONet + ∇·B/∇×B).")
    p.add_argument("--h5", type=Path, required=True)
    p.add_argument("--ckpt-dir", type=Path, default=None)
    p.add_argument("--run-tag",  type=str,  default=None)
    p.add_argument("--resume",   type=str,  default=None,
                   help="'auto' usa <write-dir>/last.ckpt si existe, o pasá una ruta a un .ckpt.")
    p.add_argument("--local-ckpt-dir", type=Path, default=None,
                   help="Disco local rápido donde Lightning escribe los .ckpt; se espejan a --ckpt-dir.")
    p.add_argument("--ckpt-every-n-steps", type=int, default=0,
                   help="Si >0, snapshotea last.ckpt cada N steps (checkpoint intra-época).")

    p.add_argument("--val-frac",  type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed",      type=int,   default=42)

    p.add_argument("--branch-hidden", type=int, nargs="+", default=[128, 128])
    p.add_argument("--trunk-hidden",  type=int, nargs="+", default=[128, 128])
    p.add_argument("--latent-dim",    type=int, default=128)
    p.add_argument("--activation",    type=str, default="tanh",
                   choices=["silu", "gelu", "relu", "tanh"])

    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--patience",     type=int,   default=20)
    p.add_argument("--grad-clip",    type=float, default=None)

    p.add_argument("--lambda-div",  type=float, default=1e-3)
    p.add_argument("--lambda-curl", type=float, default=1e-3)

    p.add_argument("--batch-size",  type=int, default=64,
                   help="Muestras (configs de imanes) por step. CartesianProd: cada step usa la grilla completa compartida.")
    p.add_argument("--data-device", type=str, default=None, choices=["cpu", "cuda"])
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--pin-memory",  action="store_true")

    p.add_argument("--comet-project", type=str, default="pmdkernel")
    p.add_argument("--log-every-n",   type=int, default=50)
    p.add_argument("--no-progress",   action="store_true")
    p.add_argument("--deterministic", action="store_true",
                   help="Modo determinista (más lento; con forward-mode jvp puede dar warnings).")

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
        args.run_tag = f"v6_deeponet_{args.h5.stem}"
    if args.ckpt_dir is None:
        args.ckpt_dir = SCRIPT_DIR / "Models" / "v6_deepxde" / "logs" / args.run_tag
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    write_dir = args.local_ckpt_dir if args.local_ckpt_dir is not None else args.ckpt_dir
    write_dir.mkdir(parents=True, exist_ok=True)
    mirror_to_drive = args.local_ckpt_dir is not None and args.local_ckpt_dir != args.ckpt_dir
    if mirror_to_drive:
        restored = 0
        for f in args.ckpt_dir.glob("*.ckpt"):
            shutil.copy2(f, write_dir / f.name)
            restored += 1
        print(f"[mirror] write_dir={write_dir}  espejo={args.ckpt_dir}  ({restored} .ckpt restaurados)")

    resume_ckpt = None
    if args.resume == "auto":
        cand = write_dir / "last.ckpt"
        if cand.exists():
            resume_ckpt = str(cand)
            print(f"[resume] reanudando desde {resume_ckpt}")
        else:
            print(f"[resume] no hay last.ckpt en {write_dir}, empiezo de cero")
    elif args.resume:
        if not Path(args.resume).exists():
            sys.exit(f"[resume] el checkpoint no existe: {args.resume}")
        resume_ckpt = args.resume
        print(f"[resume] reanudando desde {resume_ckpt}")

    if args.data_device is None:
        args.data_device = "cuda" if torch.cuda.is_available() else "cpu"

    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision(args.matmul_precision)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print_banner("DATASET")
    ds = data.load_dataset_metadata(args.h5)
    print(f"N={ds['N']}  I={ds['I']}  J={ds['J']}  ({ds['Nx']}x{ds['Ny']}x{ds['Nz']})")

    print_banner("HPARAMS")
    for k, v in sorted(vars(args).items()):
        print(f"{k:20s}: {v}")

    print_banner("SPLIT + STATS")
    splits = data.split_indices(ds["N"], val_frac=args.val_frac,
                                test_frac=args.test_frac, seed=args.seed)
    print(f"train={splits['n_train']}  val={splits['n_val']}  test={splits['n_test']}")

    t0 = time.time()
    stats = data.compute_train_stats(args.h5, splits["train"], chunk=64)
    print(f"compute_train_stats: {time.time() - t0:.1f}s")

    b_mean, b_std         = stats["b_mean"], stats["b_std"]
    branch_mean, branch_std = stats["branch_mean"], stats["branch_std"]
    coord_mean, coord_std = stats["coord_mean"], stats["coord_std"]
    print(f"b_mean     = {b_mean}")
    print(f"b_std      = {b_std}")
    print(f"coord_mean = {coord_mean}")
    print(f"coord_std  = {coord_std}")

    print_banner("LOADERS")
    print(f"data_device = {args.data_device}")
    loader_kwargs = dict(batch_size=args.batch_size,
                         b_mean=b_mean, b_std=b_std,
                         branch_mean=branch_mean, branch_std=branch_std,
                         data_device=args.data_device,
                         num_workers=args.num_workers, pin_memory=args.pin_memory)
    t0 = time.time()
    loader_tr = data.make_loader(args.h5, splits["train"], shuffle=True,  **loader_kwargs)
    loader_va = data.make_loader(args.h5, splits["val"],   shuffle=False, **loader_kwargs)
    print(f"loaders ready: {time.time() - t0:.1f}s")

    print_banner("MODEL")
    lit_model = LitDeepONetPINN(
        n_sensors=ds["I"], grid_xyz=ds["R_grid_xyz"],
        branch_hidden=args.branch_hidden, trunk_hidden=args.trunk_hidden,
        latent_dim=args.latent_dim, activation=args.activation,
        lr=args.lr, weight_decay=args.weight_decay,
        b_mean=tuple(b_mean.tolist()), b_std=tuple(b_std.tolist()),
        branch_mean=tuple(branch_mean.tolist()), branch_std=tuple(branch_std.tolist()),
        coord_mean=tuple(coord_mean.tolist()), coord_std=tuple(coord_std.tolist()),
        lambda_div=args.lambda_div, lambda_curl=args.lambda_curl,
    )
    print(f"params (trainable): {count_params(lit_model):,}")

    print_banner("TRAIN")
    t_train = time.time()
    trainer = fit_model(
        lit_model, loader_tr, loader_va,
        n_epochs=args.epochs, patience=args.patience,
        ckpt_dir=args.ckpt_dir, run_tag=args.run_tag,
        deterministic=args.deterministic,
        gradient_clip_val=args.grad_clip,
        comet_project=args.comet_project,
        log_every_n_steps=args.log_every_n,
        enable_progress_bar=not args.no_progress,
        resume_ckpt=resume_ckpt,
        local_ckpt_dir=args.local_ckpt_dir,
        ckpt_every_n_steps=args.ckpt_every_n_steps,
    )
    elapsed = time.time() - t_train
    print(f"train wallclock: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    best_ckpt_path = trainer.checkpoint_callback.best_model_path
    comet_url = (trainer.logger.experiment.url
                 if hasattr(trainer.logger, "experiment") else None)
    print(f"best ckpt    : {best_ckpt_path}")
    print(f"comet url    : {comet_url}")

    print_banner("EVAL")
    lit_model = LitDeepONetPINN.load_from_checkpoint(best_ckpt_path, weights_only=False)
    m_tr = evaluate(lit_model, args.h5, splits["train"], device, rmse_per_component=True)
    m_va = evaluate(lit_model, args.h5, splits["val"],   device, rmse_per_component=True)
    m_te = evaluate(lit_model, args.h5, splits["test"],  device, rmse_per_component=True)
    report("train", m_tr)
    report("val",   m_va)
    report("test",  m_te)

    print_banner("SAVE AUX")
    aux_path = write_dir / f"{args.run_tag}_aux.pt"
    final_ckpt_path = (str(args.ckpt_dir / Path(best_ckpt_path).name)
                       if (mirror_to_drive and best_ckpt_path) else str(best_ckpt_path))
    torch.save({
        "b_mean":      np.asarray(b_mean,   dtype=np.float32),
        "b_std":       np.asarray(b_std,    dtype=np.float32),
        "branch_mean": np.asarray(branch_mean, dtype=np.float32),
        "branch_std":  np.asarray(branch_std,  dtype=np.float32),
        "coord_mean":  np.asarray(coord_mean, dtype=np.float32),
        "coord_std":   np.asarray(coord_std,  dtype=np.float32),
        "splits":      splits,
        "metrics":     {"train": m_tr, "val": m_va, "test": m_te},
        "hparams":     vars(args).copy(),
        "h5_path":     str(args.h5),
        "ckpt_path":   final_ckpt_path,
        "run_tag":     args.run_tag,
        "comet_url":   comet_url,
    }, aux_path)
    print(f"aux saved at: {aux_path}")

    if mirror_to_drive:
        for f in list(write_dir.glob("*.ckpt")) + [aux_path]:
            shutil.copy2(f, args.ckpt_dir / f.name)
        aux_path = args.ckpt_dir / aux_path.name
        print(f"espejado a Drive: {args.ckpt_dir}")

    summary = {
        "run_tag": args.run_tag,
        "ckpt":    final_ckpt_path,
        "aux":     str(aux_path),
        "comet":   comet_url,
        "rmse_mt": {"train": m_tr["rmse_mt"], "val": m_va["rmse_mt"], "test": m_te["rmse_mt"]},
        "r2":      {"train": m_tr["r2"],      "val": m_va["r2"],      "test": m_te["r2"]},
    }
    print_banner("SUMMARY")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

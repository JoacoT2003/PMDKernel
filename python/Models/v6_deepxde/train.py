"""Wrapper de pl.Trainer.fit para v6_deepxde (PI-DeepONet + Maxwell).

Reusa la infra de checkpointing de Colab de v5: `local_ckpt_dir` (disco local
rápido/atómico) + `_DriveMirror` que espeja los `.ckpt` al `ckpt_dir` (Drive)
con `shutil.copy2` (escritura plana que el FUSE sí maneja). Con
`ckpt_every_n_steps>0` snapshotea `last.ckpt` intra-época.

Como v5, este modelo usa autograd dentro del step (física forward-mode), así que
el default queda en fp32 (precision='16-mixed' con second-order grad es frágil).
"""
import shutil
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CometLogger


class _DriveMirror(Callback):
    """Espeja los `.ckpt` de `src` (local) a `dst` (p.ej. Google Drive)."""

    def __init__(self, src, dst, every_n_steps=0):
        self.src = Path(src)
        self.dst = Path(dst)
        self.every_n_steps = int(every_n_steps or 0)

    def _sync(self):
        try:
            self.dst.mkdir(parents=True, exist_ok=True)
            for f in self.src.glob("*.ckpt"):
                shutil.copy2(f, self.dst / f.name)
        except Exception as e:
            print(f"[mirror] aviso: no pude espejar a {self.dst}: {e}", flush=True)

    def on_validation_end(self, trainer, pl_module):
        self._sync()

    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        n = self.every_n_steps
        if n and trainer.global_step > 0 and trainer.global_step % n == 0:
            self._sync()


def train(lit_model, loader_tr, loader_va, *,
          n_epochs, patience, ckpt_dir, run_tag,
          accelerator="auto", devices="auto",
          deterministic=True,
          gradient_clip_val=None,
          comet_project="pmdkernel",
          log_every_n_steps=50,
          enable_progress_bar=True,
          resume_ckpt=None,
          local_ckpt_dir=None,
          ckpt_every_n_steps=0):
    write_dir = Path(local_ckpt_dir) if local_ckpt_dir is not None else Path(ckpt_dir)
    mirror = local_ckpt_dir is not None and Path(ckpt_dir) != write_dir
    every = int(ckpt_every_n_steps or 0)

    early_stop = EarlyStopping(monitor="val_loss", patience=patience, mode="min")
    callbacks = [early_stop]

    if every > 0:
        callbacks.append(ModelCheckpoint(
            dirpath=str(write_dir), filename=run_tag,
            monitor="val_loss", mode="min", save_top_k=1, save_last=False,
        ))
        callbacks.append(ModelCheckpoint(
            dirpath=str(write_dir), save_top_k=0, save_last=True,
            every_n_train_steps=every,
        ))
    else:
        callbacks.append(ModelCheckpoint(
            dirpath=str(write_dir), filename=run_tag,
            monitor="val_loss", mode="min", save_top_k=1, save_last=True,
        ))

    if mirror:
        callbacks.append(_DriveMirror(write_dir, ckpt_dir, every_n_steps=every))

    logger = CometLogger(project=comet_project, name=run_tag)
    logger.experiment.add_tags(["v6_deepxde", "pi_deeponet", "physics", "div_curl"])

    trainer = pl.Trainer(
        max_epochs=n_epochs,
        callbacks=callbacks,
        logger=logger,
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=log_every_n_steps,
        deterministic=deterministic,
        gradient_clip_val=gradient_clip_val,
        enable_progress_bar=enable_progress_bar,
    )
    trainer.fit(lit_model, loader_tr, loader_va, ckpt_path=resume_ckpt, weights_only=False)
    return trainer

"""Wrapper de pl.Trainer.fit para v5_deepsets_pinn.

Importante: este modelo usa autograd dentro del step, por lo que NO se puede
correr con `precision='16-mixed'` directamente sin cuidado extra (los second-
order grad pueden fallar en fp16). Default queda en fp32.

Si `train_div / train_curl` se quedan en ~1e-18 desde epoch 0 (el bug que
mostró v2), significa que la red está ignorando `query_xyz` o que los
gradientes están saturando. Diagnóstico rápido: hacer un forward manual con
dos query_xyz distintos y verificar que `B_pred` cambia.

Checkpointing resiliente (Colab): escribir `.ckpt` directo sobre el mount FUSE
de Google Drive es poco confiable — `os.replace()` falla al sobrescribir y
quedan copias `last-v1.ckpt`. Por eso `train()` acepta `local_ckpt_dir`:
Lightning escribe ahí (disco local, rápido y atómico) y el callback
`_DriveMirror` copia los `.ckpt` al `ckpt_dir` espejo (Drive) con
`shutil.copy2` (escritura plana, que el FUSE sí maneja). Con
`ckpt_every_n_steps>0` además snapshotea `last.ckpt` intra-época para no perder
una época entera si Colab se desconecta.
"""
import shutil
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CometLogger


class _DriveMirror(Callback):
    """Espeja los `.ckpt` de `src` (local) a `dst` (p.ej. Google Drive).

    `shutil.copy2` es una escritura plana (truncar + escribir) que el FUSE de
    Drive sí maneja, a diferencia del temp + rename de Lightning. Un hipo de
    Drive solo emite un aviso, nunca tira el training.
    """

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
    # Lightning escribe en write_dir. Si hay local_ckpt_dir es disco local
    # (rápido/atómico) y _DriveMirror copia los .ckpt al ckpt_dir espejo.
    write_dir = Path(local_ckpt_dir) if local_ckpt_dir is not None else Path(ckpt_dir)
    mirror = local_ckpt_dir is not None and Path(ckpt_dir) != write_dir
    every = int(ckpt_every_n_steps or 0)

    early_stop = EarlyStopping(monitor="val_loss", patience=patience, mode="min")
    callbacks = [early_stop]

    if every > 0:
        # Dos ModelCheckpoint: el mejor por val_loss (al final de cada época) y
        # un last.ckpt frecuente para reanudar. Van separados porque combinar
        # monitor + every_n_train_steps en uno solo desactiva el guardado por
        # época del mejor. checkpoint_callback (el primero) = ckpt_best.
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
    logger.experiment.add_tags(["v5_deepsets_pinn", "physics", "div_curl"])

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
    # ckpt_path restaura pesos + optimizer + epoch/step + estado de los callbacks
    # (EarlyStopping y el best_model_score de ModelCheckpoint). None = de cero.
    # weights_only=False: nuestros ckpt guardan sensor_xyz (numpy) en los hparams,
    # y torch>=2.6 rechaza eso con el default weights_only=True. El ckpt es propio.
    trainer.fit(lit_model, loader_tr, loader_va, ckpt_path=resume_ckpt, weights_only=False)
    return trainer

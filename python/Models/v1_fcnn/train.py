"""Wrapper de `pytorch_lightning.Trainer.fit` para v1_fcnn.

EarlyStopping + ModelCheckpoint + CometLogger. Mismo patrón que v2_pinn — la
única diferencia funcional es que v1 no necesita `gradient_clip_val` por
default (no hay autograd sobre coords que pueda explotar).

Requiere `COMET_API_KEY` y `COMET_WORKSPACE` en el entorno (o `~/.comet.config`).
"""
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CometLogger


def train(lit_model, loader_tr, loader_va, *,
          n_epochs, patience, ckpt_dir, run_tag,
          accelerator="auto", devices="auto",
          deterministic=True,
          gradient_clip_val=None,
          comet_project="pmdkernel"):
    """Entrena `lit_model` con early stopping sobre `val_loss`.

    Returns trainer fitted. `trainer.checkpoint_callback.best_model_path`
    apunta al `.ckpt` con mejor val_loss.
    """
    early_stop = EarlyStopping(monitor="val_loss", patience=patience, mode="min")
    ckpt = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename=run_tag,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
    )
    logger = CometLogger(project=comet_project, name=run_tag)
    logger.experiment.add_tags(["v1_fcnn"])

    trainer = pl.Trainer(
        max_epochs=n_epochs,
        callbacks=[early_stop, ckpt],
        logger=logger,
        accelerator=accelerator,
        devices=devices,
        enable_progress_bar=True,
        log_every_n_steps=1,
        deterministic=deterministic,
        gradient_clip_val=gradient_clip_val,
    )
    trainer.fit(lit_model, loader_tr, loader_va)
    return trainer

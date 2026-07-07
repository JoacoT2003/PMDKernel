"""Wrapper de `pytorch_lightning.Trainer.fit` para v3_pwnn.

EarlyStopping + ModelCheckpoint + CometLogger. Sin physics losses → flow
Lightning estándar. `log_every_n_steps` por default a 50 (vs 1 en v2.1)
para reducir overhead de Comet — las curvas siguen siendo legibles.

Requiere `~/.comet.config` con `api_key` y `workspace`.
"""
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CometLogger


def train(lit_model, loader_tr, loader_va, *,
          n_epochs, patience, ckpt_dir, run_tag,
          accelerator="auto", devices="auto",
          deterministic=True,
          gradient_clip_val=None,
          comet_project="pmdkernel",
          log_every_n_steps=50):
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
    logger.experiment.add_tags(["v3_pwnn", "mse_only", "by_only"])

    trainer = pl.Trainer(
        max_epochs=n_epochs,
        callbacks=[early_stop, ckpt],
        logger=logger,
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=log_every_n_steps,
        deterministic=deterministic,
        gradient_clip_val=gradient_clip_val,
    )
    trainer.fit(lit_model, loader_tr, loader_va)
    return trainer

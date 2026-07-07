"""Wrapper de `pytorch_lightning.Trainer.fit` para v2_pinn.

EarlyStopping + ModelCheckpoint + CometLogger. En modo `balance_grads=True`,
`LitPINN` desactiva `automatic_optimization` y maneja zero/backward/clip/step
manualmente — el caller debe pasar `gradient_clip_val=None` en ese modo (lo
prohíbe Lightning) y configurar `manual_clip_val` al construir el `LitPINN`.

Requiere `~/.comet.config` con `api_key` y `workspace`. Las curvas viven en
Comet — la notebook las recupera via `comet_ml.API`, no hay CSV local.
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
    apunta al `.ckpt` con mejor val_loss. `trainer.logger.experiment.url`
    da la URL del run en Comet.
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
    logger.experiment.add_tags(["v2_pinn"])

    trainer = pl.Trainer(
        max_epochs=n_epochs,
        callbacks=[early_stop, ckpt],
        logger=logger,
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=1,
        deterministic=deterministic,
        gradient_clip_val=gradient_clip_val,
    )
    trainer.fit(lit_model, loader_tr, loader_va)
    return trainer

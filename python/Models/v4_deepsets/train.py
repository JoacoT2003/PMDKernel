"""Wrapper de pl.Trainer.fit para v4_deepsets."""
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CometLogger


def train(lit_model, loader_tr, loader_va, *,
          n_epochs, patience, ckpt_dir, run_tag,
          accelerator="auto", devices="auto",
          deterministic=True,
          gradient_clip_val=None,
          precision="16-mixed",
          comet_project="pmdkernel",
          log_every_n_steps=50,
          enable_progress_bar=True):
    early_stop = EarlyStopping(monitor="val_loss", patience=patience, mode="min")
    ckpt = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename=run_tag,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    logger = CometLogger(project=comet_project, name=run_tag)
    logger.experiment.add_tags(["v4_deepsets", "mse_only", "by_only"])

    trainer = pl.Trainer(
        max_epochs=n_epochs,
        callbacks=[early_stop, ckpt],
        logger=logger,
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=log_every_n_steps,
        deterministic=deterministic,
        gradient_clip_val=gradient_clip_val,
        precision=precision,
        enable_progress_bar=enable_progress_bar,
    )
    trainer.fit(lit_model, loader_tr, loader_va)
    return trainer

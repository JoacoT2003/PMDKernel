"""v4_deepsets — DeepSets per-sensor con coords relativas, MSE sobre By.

Arquitectura:

    sensores_by (K, I), query_xyz (K, 3) raw mm
        │
        ▼  build_features (interno)
    feat = concat(dx, dy, dz, r, By_sens)  shape (K, I, 5) raw
        │
        ▼  (feat - feat_mean) / feat_std   ← buffer (5,)
    feat_norm shape (K, I, 5)
        │
        ▼  φ (SensorEncoder, shared per-sensor MLP)
    h shape (K, I, embed_dim)
        │
        ▼  mean over I (aggregation)
    h_agg shape (K, embed_dim)
        │
        ▼  ρ (Decoder MLP)
    By_pred_norm shape (K, 1)
        │
        ▼  · b_std + b_mean   (en `predict_mT`)
    By_pred_mT shape (K, 1)

Propiedades:
- Invariante al orden de sensores (la agregación `mean` lo garantiza).
- Coords relativas (`dx, dy, dz, r`) — fuerzan a la red a usar info espacial.
  Esto ataca el bug de v2 donde la red ignoraba el `xyz` absoluto.
- Sin physics losses (eso es v5). Comparación apples-to-apples con v3 sobre
  RMSE de By.
"""
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.regression import MeanSquaredError

_ACTIVATIONS = {"silu": nn.SiLU, "tanh": nn.Tanh, "gelu": nn.GELU, "relu": nn.ReLU}


def _build_mlp(n_in, hidden_layers, n_out, activation):
    if activation not in _ACTIVATIONS:
        raise ValueError(f"activation debe ser uno de {list(_ACTIVATIONS)}; dado {activation}")
    act_cls = _ACTIVATIONS[activation]
    layers, prev = [], n_in
    for h in hidden_layers:
        layers += [nn.Linear(prev, h), act_cls()]
        prev = h
    layers.append(nn.Linear(prev, n_out))
    return nn.Sequential(*layers)


class DeepSetModel(nn.Module):
    """φ → mean → ρ. Sin physics losses, sin output 3D."""

    def __init__(self, *,
                 n_sensor_features=5,
                 encoder_hidden=(64, 64), embed_dim=128,
                 decoder_hidden=(128, 64), n_out=1,
                 activation="relu"):
        super().__init__()
        self.phi = _build_mlp(n_sensor_features, list(encoder_hidden), embed_dim, activation)
        self.rho = _build_mlp(embed_dim, list(decoder_hidden), n_out, activation)

    def forward(self, features_norm):
        """features_norm: (K, I, 5) ya normalizado.
        Returns: (K, n_out)."""
        h     = self.phi(features_norm)      # (K, I, embed)
        h_agg = h.mean(dim=1)                # (K, embed)        ← aggregation
        return self.rho(h_agg)               # (K, n_out)


class LitDeepSet(pl.LightningModule):
    """Wrapper Lightning. MSE sobre By normalizado."""

    def __init__(self, n_sensors, sensor_xyz, *,
                 encoder_hidden=(64, 64), embed_dim=128,
                 decoder_hidden=(128, 64), activation="relu",
                 lr=1e-3, weight_decay=1e-5,
                 b_mean=0.0, b_std=1.0,
                 feat_mean=(0.0, 0.0, 0.0, 0.0, 0.0),
                 feat_std=(1.0, 1.0, 1.0, 1.0, 1.0)):
        super().__init__()
        self.save_hyperparameters()
        self.net = DeepSetModel(
            n_sensor_features=5,
            encoder_hidden=tuple(encoder_hidden), embed_dim=embed_dim,
            decoder_hidden=tuple(decoder_hidden), n_out=1,
            activation=activation,
        )
        self.train_mse = MeanSquaredError()
        self.val_mse   = MeanSquaredError()

        # Sensor positions fijas (geometría del hardware). Se inyectan en el
        # forward para computar (dx, dy, dz, r). Buffer → viaja en el .ckpt.
        sensor_xyz_t = torch.as_tensor(sensor_xyz, dtype=torch.float32)
        if sensor_xyz_t.shape != (n_sensors, 3):
            raise ValueError(f"sensor_xyz debe tener shape ({n_sensors}, 3), "
                             f"recibido {tuple(sensor_xyz_t.shape)}")
        self.register_buffer("sensor_xyz", sensor_xyz_t)                                 # (I, 3)

        self.register_buffer("b_mean",    torch.tensor(float(b_mean), dtype=torch.float32))
        self.register_buffer("b_std",     torch.tensor(float(b_std),  dtype=torch.float32))
        self.register_buffer("feat_mean", torch.as_tensor(feat_mean, dtype=torch.float32))  # (5,)
        self.register_buffer("feat_std",  torch.as_tensor(feat_std,  dtype=torch.float32))  # (5,)

    def _build_features(self, sensors_by, query_xyz):
        """sensors_by: (K, I) raw mT;  query_xyz: (K, 3) raw mm.
        Returns: (K, I, 5) normalizado por feat_mean/std."""
        # diff[k, i] = sensor_xyz[i] - query_xyz[k]
        diff = self.sensor_xyz.unsqueeze(0) - query_xyz.unsqueeze(1)      # (K, I, 3)
        r    = torch.linalg.norm(diff, dim=-1, keepdim=True)              # (K, I, 1)
        by   = sensors_by.unsqueeze(-1)                                    # (K, I, 1)
        feats = torch.cat([diff, r, by], dim=-1)                           # (K, I, 5)
        return (feats - self.feat_mean) / self.feat_std

    def forward(self, sensors_by, query_xyz):
        features_norm = self._build_features(sensors_by, query_xyz)
        return self.net(features_norm)                                     # (K, 1) normalizado

    def predict_mT(self, sensors_by, query_xyz):
        by_norm = self(sensors_by, query_xyz)
        return by_norm * self.b_std + self.b_mean

    def _step(self, batch, *, stage):
        sensors_by, query_xyz, by_target_norm = batch                      # (K, I), (K, 3), (K, 1)
        by_pred = self(sensors_by, query_xyz)
        loss = F.mse_loss(by_pred, by_target_norm)

        mse_metric = self.train_mse if stage == "train" else self.val_mse
        mse_metric(by_pred.detach(), by_target_norm)

        prog = (stage == "train")
        self.log(f"{stage}_loss",     loss,       on_step=False, on_epoch=True, prog_bar=prog)
        self.log(f"{stage}_mse_norm", mse_metric, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, stage="val")

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )


def count_params(model):
    return sum(p.numel() for p in model.parameters())

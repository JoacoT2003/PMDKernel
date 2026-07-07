"""FCNN v1 + LightningModule con MSE.

Red plana sin codificación espacial: el input son los valores de campo en los
sensores aplanados `(I·3,)` y el output es el campo en TODA la grilla aplanado
`(J·3,)`. Una sola pasada por sample — no hay autograd sobre coords ni físicas.

Activación libre (ReLU/GELU/SiLU/Tanh): no se requiere segunda derivada porque
no hay losses derivativas. ReLU es default razonable para regresión plana.

`b_scale` se persiste como buffer del modelo → viaja en el `.ckpt` y la
inferencia puede devolver mT físicas sin metadatos externos.
"""
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.regression import MeanSquaredError

_ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU, "tanh": nn.Tanh}


class FCNN(nn.Module):
    """FCNN simple: input (n_in) → hidden_layers → output (n_out).

    Última capa lineal sin activación (regresión de campo en mT/b_scale, sin
    activación de salida).
    """

    def __init__(self, n_in, n_out, hidden_layers, *,
                 activation="relu", dropout=0.0):
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"activation debe ser uno de {list(_ACTIVATIONS)}; dado {activation}")
        act_cls = _ACTIVATIONS[activation]

        layers, prev = [], n_in
        for h in hidden_layers:
            layers += [nn.Linear(prev, h), act_cls()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LitFCNN(pl.LightningModule):
    """Wrapper Lightning de FCNN v1 con MSE."""

    def __init__(self, n_in, n_out, hidden_layers, *,
                 activation="relu", dropout=0.0,
                 lr=1e-3, weight_decay=1e-5,
                 b_scale=1.0):
        super().__init__()
        self.save_hyperparameters()
        self.net       = FCNN(n_in, n_out, hidden_layers,
                              activation=activation, dropout=dropout)
        self.train_mse = MeanSquaredError()
        self.val_mse   = MeanSquaredError()
        self.register_buffer("b_scale", torch.tensor(float(b_scale)))

    def forward(self, sensors):
        return self.net(sensors)

    def _step(self, batch, *, stage):
        sensors, b_target = batch       # (B, n_in), (B, n_out)
        b_pred = self(sensors)
        loss = F.mse_loss(b_pred, b_target)

        mse_metric = self.train_mse if stage == "train" else self.val_mse
        mse_metric(b_pred.detach(), b_target)

        # Loss en espacio normalizado (B / b_scale). Para mT físicas, ver
        # `metrics.evaluate(...)` que multiplica por b_scale afuera.
        # `on_step=False` para que el CSV tenga columnas limpias `train_loss`
        # en vez de duplicarse en `train_loss_step` y `train_loss_epoch`.
        self.log(f"{stage}_loss",     loss,       on_step=False, on_epoch=True, prog_bar=True)
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

    def predict_mT(self, sensors_norm):
        """Forward + denormalización a mT físicas."""
        with torch.no_grad():
            return self.b_scale * self.net(sensors_norm)


def count_params(model):
    return sum(p.numel() for p in model.parameters())

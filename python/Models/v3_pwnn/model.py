"""v3_pwnn — FCNN puntual sobre By, MSE puro.

Variante simplificada de `v2_1_pinn`:
- Input  : `(I + 3)` = By de los I sensores + xyz del punto query.
- Output : `(1,)`    = By en el punto query (no las 3 componentes).
- Misma arquitectura per-punto, mismo loss (MSE), sin physics losses.

## Por qué solo By?

El target final del análisis (inhomogeneidad de B0) es sobre `|By|`
específicamente, ya que en el OSI² B0 apunta sobre -Y (ver memoria
`project_b0_direction`). Modelar Bx/Bz como output extra:
- agrega parámetros que la red tiene que aprender pero no usás aguas abajo,
- diluye el gradient de By en el MSE 3-componente,
- requiere sensores 3-axiales que físicamente no necesitás.

v3 reduce el modelo a la mínima superficie que respeta el objetivo.

`b_mean`, `b_std` (escalares) y `pts_mean`, `pts_std` (3-vectores) se
persisten como buffers para desnormalizar a mT físicas en `metrics.evaluate`.
"""
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.regression import MeanSquaredError

_ACTIVATIONS = {"silu": nn.SiLU, "tanh": nn.Tanh, "gelu": nn.GELU, "relu": nn.ReLU}


class FCNN(nn.Module):
    """FCNN simple: input `(n_in,)` → hidden_layers → output `(1,)`.

    `n_in = I + 3` (By de I sensores + xyz). Última capa lineal sin activación.
    La salida es By normalizado: `By_norm = (By_real - b_mean) / b_std`.
    """

    def __init__(self, n_in, hidden_layers, *, activation="silu"):
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"activation debe ser uno de {list(_ACTIVATIONS)}; dado {activation}")
        act_cls = _ACTIVATIONS[activation]

        layers, prev = [], n_in
        for h in hidden_layers:
            layers += [nn.Linear(prev, h), act_cls()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LitFCNN(pl.LightningModule):
    """Wrapper Lightning con MSE puro sobre By normalizado.

    Stats escalares para By (`b_mean`, `b_std`) y vector (3,) para coords
    (`pts_mean`, `pts_std`) viajan en el `.ckpt` como buffers, así
    `predict_mT` y `metrics.evaluate` desnormalizan sin metadatos externos.
    """

    def __init__(self, n_sensors, hidden_layers, *,
                 activation="silu",
                 lr=1e-3, weight_decay=1e-5,
                 b_mean=0.0, b_std=1.0,
                 pts_mean=(0.0, 0.0, 0.0), pts_std=(1.0, 1.0, 1.0)):
        super().__init__()
        self.save_hyperparameters()
        n_in = n_sensors + 3
        self.net       = FCNN(n_in, hidden_layers, activation=activation)
        self.train_mse = MeanSquaredError()
        self.val_mse   = MeanSquaredError()

        self.register_buffer("b_mean",   torch.tensor(float(b_mean), dtype=torch.float32))
        self.register_buffer("b_std",    torch.tensor(float(b_std),  dtype=torch.float32))
        self.register_buffer("pts_mean", torch.tensor(pts_mean,      dtype=torch.float32))
        self.register_buffer("pts_std",  torch.tensor(pts_std,       dtype=torch.float32))

    def forward(self, sensors, points_norm):
        """sensors: (K, I), points_norm: (K, 3) → output: (K, 1) normalizado."""
        x = torch.cat([sensors, points_norm], dim=-1)
        return self.net(x)

    def predict_mT(self, sensors, points_norm):
        """Forward + denormalización a mT físicas: `By_real = By_norm·b_std + b_mean`."""
        by_norm = self(sensors, points_norm)
        return by_norm * self.b_std + self.b_mean

    def _step(self, batch, *, stage):
        sensors, points_norm, by_target_norm = batch     # (K, I), (K, 3), (K, 1)
        by_pred = self(sensors, points_norm)
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

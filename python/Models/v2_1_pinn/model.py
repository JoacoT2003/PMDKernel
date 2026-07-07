"""v2.1 — red de campo condicional con SOLO MSE (sin physics losses).

Variante simplificada de `v2_pinn`:
- Misma arquitectura per-punto: `f(B_sens, x, y, z) → B(x, y, z)`.
- Mismo data pipeline (per-componente normalization).
- **Loss única: MSE** sobre B normalizado. Sin ∇·B, ∇×B, TV.
- Sin autograd sobre coords → 3-4× más rápido por step que v2 full.

## Por qué?

El v2 con `lambda_div, lambda_rot, lambda_tv` mostró `train_div_step ≈ 1e-18`
desde epoch 0 (ruido fp32). Hipótesis: la red ignora `points_norm` y solo usa
`sensors`, dando trivialmente `∂B/∂x ≈ 0` --> divergencia trivialmente 0 →
losses físicas inactivas.

v2.1 elimina las losses físicas para verificar:
1. Si la arquitectura per-punto siquiera aprende a usar `points_norm`.
2. Cuál es el RMSE alcanzable solo con MSE (baseline para comparar contra v2).
3. Si v2.1 ≈ v2 en RMSE, las physics están aportando 0 → confirmamos hipótesis.
4. Si v2.1 << v2 en RMSE, las physics SÍ ayudan y hay que debuggear el bug
   numérico del v2.

`b_mean`, `b_std`, `pts_mean`, `pts_std` se persisten como buffers para
desnormalizar predicciones a mT físicas en `metrics.evaluate`.
"""
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.regression import MeanSquaredError

_ACTIVATIONS = {"silu": nn.SiLU, "tanh": nn.Tanh, "gelu": nn.GELU, "relu": nn.ReLU}


class PINN(nn.Module):
    """FCNN simple: input (n_in) → hidden_layers → output (3).

    `n_in = I*3 + 3` (sensores + xyz). Última capa lineal sin activación.
    La salida es B normalizado: `B_norm = (B_real - b_mean) / b_std`.

    Sin physics losses, las activaciones admiten ReLU/GELU también — pero SiLU
    sigue siendo buen default por simetría con v2.
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
        layers.append(nn.Linear(prev, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LitPINN(pl.LightningModule):
    """Wrapper Lightning con SOLO MSE — sin losses físicas, sin balance_grads.

    Stats per-componente (`b_mean`, `b_std`, `pts_mean`, `pts_std`) se
    persisten como buffers — viajan en el `.ckpt` para que `predict_mT` y
    `metrics.evaluate` puedan desnormalizar sin metadatos externos.
    """

    def __init__(self, n_sensors, hidden_layers, *,
                 activation="silu",
                 lr=1e-3, weight_decay=1e-5,
                 b_mean=(0.0, 0.0, 0.0), b_std=(1.0, 1.0, 1.0),
                 pts_mean=(0.0, 0.0, 0.0), pts_std=(1.0, 1.0, 1.0)):
        super().__init__()
        self.save_hyperparameters()
        n_in = n_sensors * 3 + 3
        self.net       = PINN(n_in, hidden_layers, activation=activation)
        self.train_mse = MeanSquaredError()
        self.val_mse   = MeanSquaredError()

        self.register_buffer("b_mean",   torch.tensor(b_mean,   dtype=torch.float32))
        self.register_buffer("b_std",    torch.tensor(b_std,    dtype=torch.float32))
        self.register_buffer("pts_mean", torch.tensor(pts_mean, dtype=torch.float32))
        self.register_buffer("pts_std",  torch.tensor(pts_std,  dtype=torch.float32))

    # -- Forward / inference ---------------------------------------------

    def forward(self, sensors, points_norm):
        """sensors: (K, I·3) ya normalizado por `x_scaler`.
        points_norm: (K, 3) ya normalizado por `(pts - mean)/std`."""
        x = torch.cat([sensors, points_norm], dim=-1)
        return self.net(x)

    def predict_mT(self, sensors, points_norm):
        """Forward + denormalización a mT físicas: `B_real = B_norm·b_std + b_mean`."""
        b_norm = self(sensors, points_norm)
        return b_norm * self.b_std + self.b_mean

    # -- Steps -----------------------------------------------------------

    def _step(self, batch, *, stage):
        sensors, points_norm, b_target_norm = batch     # (K, I·3), (K, 3), (K, 3)
        b_pred = self(sensors, points_norm)
        loss = F.mse_loss(b_pred, b_target_norm)

        mse_metric = self.train_mse if stage == "train" else self.val_mse
        mse_metric(b_pred.detach(), b_target_norm)

        # `on_step=False` para columnas limpias en CSV/Comet
        # (`train_loss` en vez de `train_loss_step` + `train_loss_epoch`).
        prog = (stage == "train")
        self.log(f"{stage}_loss",     loss,       on_step=False, on_epoch=True, prog_bar=prog)
        self.log(f"{stage}_mse_norm", mse_metric, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        # No hace falta `torch.enable_grad()` acá: v2.1 no usa autograd
        # sobre coords (no hay physics losses).
        return self._step(batch, stage="val")

    # -- Optim ------------------------------------------------------------

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )


def count_params(model):
    return sum(p.numel() for p in model.parameters())

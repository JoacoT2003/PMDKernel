"""v5_deepsets_pinn — DeepSets + Physics-Informed Losses (∇·B = 0, ∇×B = 0).

Arquitectura: idéntica a v4 hasta el output, pero:
- Output: `(K, 3)` = `(Bx, By, Bz)` para poder imponer Maxwell.
- Loss total: `loss = MSE_data + λ_div · div² + λ_curl · |curl|²`.
- Las derivadas se computan vía `torch.autograd.grad` sobre `query_xyz` con
  `requires_grad_(True)` activado en el `_step`.

## Unidades de las physics losses

Detalle importante: el modelo predice `B_norm = (B - b_mean) / b_std`, con
`b_std` por componente. La cantidad físicamente correcta es:

    ∇·B_real = Σᵢ b_std[i] · ∂B_norm[i] / ∂xᵢ_mm    [mT/mm = T/m]

NO `Σᵢ ∂B_norm[i] / ∂xᵢ` (sin peso). El factor `b_std[i]` per-componente es
crítico porque las 3 componentes pueden tener escalas muy distintas (Bx, Bz
~mT vs By ~50 mT en este Halbach).

La div/curl en mT/mm es numéricamente igual a T/m, así que no hace falta
conversion mm→m (las unidades se cancelan: 1e-3 mT/T · 1e3 mm/m = 1).

## Tip de implementación

Lightning desactiva grad en `validation_step`. Para que el forward con
autograd funcione tanto en train como en val, envolvemos el step entero en
`torch.enable_grad()` y clonamos `query_xyz` antes de `requires_grad_()`
para no contaminar el tensor original del dataset.
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


class DeepSetModelPINN(nn.Module):
    def __init__(self, *,
                 n_sensor_features=5,
                 encoder_hidden=(64, 64), embed_dim=128,
                 decoder_hidden=(128, 64),
                 activation="silu"):
        super().__init__()
        self.phi = _build_mlp(n_sensor_features, list(encoder_hidden), embed_dim, activation)
        self.rho = _build_mlp(embed_dim, list(decoder_hidden), 3, activation)   # ← n_out = 3

    def forward(self, features_norm):
        h     = self.phi(features_norm)
        h_agg = h.mean(dim=1)
        return self.rho(h_agg)                                                   # (K, 3)


class LitDeepSetPINN(pl.LightningModule):
    """DeepSets + (∇·B = 0) + (∇×B = 0). Output 3-componente."""

    def __init__(self, n_sensors, sensor_xyz, *,
                 encoder_hidden=(64, 64), embed_dim=128,
                 decoder_hidden=(128, 64), activation="silu",
                 lr=1e-3, weight_decay=1e-5,
                 b_mean=(0.0, 0.0, 0.0), b_std=(1.0, 1.0, 1.0),
                 feat_mean=(0.0, 0.0, 0.0, 0.0, 0.0),
                 feat_std=(1.0, 1.0, 1.0, 1.0, 1.0),
                 lambda_div=1e-3,
                 lambda_curl=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.net = DeepSetModelPINN(
            n_sensor_features=5,
            encoder_hidden=tuple(encoder_hidden), embed_dim=embed_dim,
            decoder_hidden=tuple(decoder_hidden),
            activation=activation,
        )
        self.train_mse = MeanSquaredError()
        self.val_mse   = MeanSquaredError()

        sensor_xyz_t = torch.as_tensor(sensor_xyz, dtype=torch.float32)
        if sensor_xyz_t.shape != (n_sensors, 3):
            raise ValueError(f"sensor_xyz debe tener shape ({n_sensors}, 3), "
                             f"recibido {tuple(sensor_xyz_t.shape)}")
        self.register_buffer("sensor_xyz", sensor_xyz_t)                                   # (I, 3)

        self.register_buffer("b_mean",    torch.as_tensor(b_mean,   dtype=torch.float32))  # (3,)
        self.register_buffer("b_std",     torch.as_tensor(b_std,    dtype=torch.float32))  # (3,)
        self.register_buffer("feat_mean", torch.as_tensor(feat_mean, dtype=torch.float32)) # (5,)
        self.register_buffer("feat_std",  torch.as_tensor(feat_std,  dtype=torch.float32)) # (5,)

    def _build_features(self, sensors_by, query_xyz):
        diff = self.sensor_xyz.unsqueeze(0) - query_xyz.unsqueeze(1)         # (K, I, 3)
        r    = torch.linalg.norm(diff, dim=-1, keepdim=True)                 # (K, I, 1)
        by   = sensors_by.unsqueeze(-1)                                       # (K, I, 1)
        feats = torch.cat([diff, r, by], dim=-1)                              # (K, I, 5)
        return (feats - self.feat_mean) / self.feat_std

    def forward(self, sensors_by, query_xyz):
        features_norm = self._build_features(sensors_by, query_xyz)
        return self.net(features_norm)                                         # (K, 3) normalizado

    def predict_mT(self, sensors_by, query_xyz):
        return self(sensors_by, query_xyz) * self.b_std + self.b_mean          # (K, 3) mT

    def _physics_residuals(self, sensors_by, query_xyz):
        """Devuelve `(b_pred_norm, div, curl_x, curl_y, curl_z)`.

        `query_xyz` debe tener `requires_grad_(True)` antes de llamar acá.
        Las derivadas están en mT/mm = T/m (las unidades se cancelan).
        El factor `b_std[i]` per-componente se aplica porque ∇·B_real =
        Σᵢ b_std[i] · ∂B_norm[i]/∂xᵢ — el normalizado por componente importa.
        """
        b_norm = self(sensors_by, query_xyz)                                  # (K, 3) normalizado
        bx_n, by_n, bz_n = b_norm[:, 0], b_norm[:, 1], b_norm[:, 2]

        # Cada autograd.grad nos da ∂(componente_norm)/∂query_xyz → shape (K, 3)
        gx = torch.autograd.grad(bx_n.sum(), query_xyz, create_graph=True)[0]  # (K, 3)
        gy = torch.autograd.grad(by_n.sum(), query_xyz, create_graph=True)[0]
        gz = torch.autograd.grad(bz_n.sum(), query_xyz, create_graph=True)[0]

        s = self.b_std                                                         # (3,)
        # Divergencia en unidades físicas: Σ b_std[i] · ∂B_norm[i]/∂xᵢ
        div    = s[0]*gx[:, 0] + s[1]*gy[:, 1] + s[2]*gz[:, 2]                 # (K,)
        curl_x = s[2]*gz[:, 1] - s[1]*gy[:, 2]
        curl_y = s[0]*gx[:, 2] - s[2]*gz[:, 0]
        curl_z = s[1]*gy[:, 0] - s[0]*gx[:, 1]
        return b_norm, div, curl_x, curl_y, curl_z

    def _step(self, batch, *, stage):
        sensors_by, query_xyz, b_target_norm = batch                          # (K, I), (K, 3), (K, 3)

        # autograd requiere requires_grad sobre las coords del query.
        # Clonamos para no contaminar el tensor original (que vive en el
        # dataset y se reutiliza entre batches).
        with torch.enable_grad():
            query_xyz_g = query_xyz.detach().clone().requires_grad_(True)
            b_pred_norm, div, cx, cy, cz = self._physics_residuals(sensors_by, query_xyz_g)

            loss_data = F.mse_loss(b_pred_norm, b_target_norm)
            loss_div  = (div ** 2).mean()
            loss_curl = (cx ** 2 + cy ** 2 + cz ** 2).mean()
            loss = (loss_data
                    + self.hparams.lambda_div  * loss_div
                    + self.hparams.lambda_curl * loss_curl)

        mse_metric = self.train_mse if stage == "train" else self.val_mse
        mse_metric(b_pred_norm.detach(), b_target_norm)

        prog = (stage == "train")
        self.log(f"{stage}_loss",      loss,        on_step=False, on_epoch=True, prog_bar=prog)
        self.log(f"{stage}_loss_data", loss_data,   on_step=False, on_epoch=True)
        self.log(f"{stage}_loss_div",  loss_div,    on_step=False, on_epoch=True)
        self.log(f"{stage}_loss_curl", loss_curl,   on_step=False, on_epoch=True)
        self.log(f"{stage}_mse_norm",  mse_metric,  on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        # `torch.enable_grad()` ya está dentro de _step — Lightning desactiva
        # grad en val_step pero el context manager local lo re-habilita.
        return self._step(batch, stage="val")

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )


def count_params(model):
    return sum(p.numel() for p in model.parameters())

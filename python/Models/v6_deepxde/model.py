"""v6_deepxde — PI-DeepONet (DeepONetCartesianProd) + Maxwell (∇·B=0, ∇×B=0).

Reemplaza v5 (DeepSets, O(J·I) por re-encodear los I sensores en cada punto).
DeepONet evalúa branch 1×/muestra + trunk 1×/punto → O(I)+O(J). Medido ~10×
más rápido por step que v5 en la misma config (ver spike_plumbing.py).

## Autodiff: FORWARD mode (obligatorio)
`dde.grad.jacobian` en reverse mode tira `NotImplementedError` sobre el output
3D `(N,J,3)` de CartesianProd. Forward mode (torch.func.jvp) sí funciona y
devuelve `(tensor, callable)` → hay que tomar `[0]`. Ver README, hallazgo #1.

## Unidades de las physics losses (doble factor de denormalización)
La red predice `B_norm = (B - b_mean) / b_std` (per-componente) sobre un trunk
**normalizado** `x_norm = (x_mm - coord_mean) / coord_std`. La cantidad física es
`∂B_real[i]/∂x_mm[j]`, así que hay que deshacer AMBAS normalizaciones:

    ∂B_real[i]/∂x_mm[j] = (b_std[i] / coord_std[j]) · ∂B_norm[i]/∂x_norm[j]

`div = Σᵢ (b_std[i]/coord_std[i]) · g(i,i)`, curl análogo. El resultado queda en
mT/mm = T/m (sin conversión extra), igual que v5. En v5 el trunk era raw mm y el
factor `coord_std` no existía; acá normalizamos el trunk (tanh se satura con
coords ~±60 mm) y lo compensamos explícito.
"""
import os

os.environ.setdefault("DDE_BACKEND", "pytorch")

import deepxde as dde
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.regression import MeanSquaredError

dde.config.set_default_autodiff("forward")  # global; obligatorio para output 3D


def _jac(outputs, forward_call, xs, i, j):
    """jacobian en forward mode devuelve (tensor, callable) -> tomar [0]. Shape (B,J,1)."""
    out = dde.grad.jacobian((outputs, forward_call), xs, i=i, j=j)
    return out[0] if isinstance(out, tuple) else out


class LitDeepONetPINN(pl.LightningModule):
    """PI-DeepONet: branch(By sensores) ⊗ trunk(coords) → (Bx,By,Bz) + Maxwell."""

    def __init__(self, n_sensors, grid_xyz, *,
                 branch_hidden=(128, 128), trunk_hidden=(128, 128),
                 latent_dim=128, activation="tanh",
                 lr=1e-3, weight_decay=1e-5,
                 b_mean=(0.0, 0.0, 0.0), b_std=(1.0, 1.0, 1.0),
                 branch_mean=None, branch_std=None,
                 coord_mean=(0.0, 0.0, 0.0), coord_std=(1.0, 1.0, 1.0),
                 lambda_div=1e-3, lambda_curl=1e-3):
        super().__init__()
        self.save_hyperparameters()

        layers_branch = [int(n_sensors)] + list(branch_hidden) + [int(latent_dim)]
        layers_trunk  = [3] + list(trunk_hidden) + [int(latent_dim)]
        self.net = dde.nn.DeepONetCartesianProd(
            layer_sizes_branch=layers_branch,
            layer_sizes_trunk=layers_trunk,
            activation=activation,
            kernel_initializer="Glorot normal",
            num_outputs=3,
            multi_output_strategy="independent",
        )

        self.train_mse = MeanSquaredError()
        self.val_mse   = MeanSquaredError()

        grid_xyz_t = torch.as_tensor(grid_xyz, dtype=torch.float32)            # (J, 3) raw mm
        if grid_xyz_t.ndim != 2 or grid_xyz_t.shape[1] != 3:
            raise ValueError(f"grid_xyz debe ser (J, 3); recibido {tuple(grid_xyz_t.shape)}")

        if branch_mean is None:
            branch_mean = [0.0] * int(n_sensors)
        if branch_std is None:
            branch_std = [1.0] * int(n_sensors)

        self.register_buffer("b_mean",      torch.as_tensor(b_mean,   dtype=torch.float32))   # (3,)
        self.register_buffer("b_std",       torch.as_tensor(b_std,    dtype=torch.float32))   # (3,)
        self.register_buffer("coord_mean",  torch.as_tensor(coord_mean, dtype=torch.float32)) # (3,)
        self.register_buffer("coord_std",   torch.as_tensor(coord_std,  dtype=torch.float32)) # (3,)
        self.register_buffer("branch_mean", torch.as_tensor(branch_mean, dtype=torch.float32))# (I,)
        self.register_buffer("branch_std",  torch.as_tensor(branch_std,  dtype=torch.float32))# (I,)

        # trunk normalizado (fijo) — se clona+requires_grad por step para la física.
        grid_norm = (grid_xyz_t - torch.as_tensor(coord_mean, dtype=torch.float32)) \
                    / torch.as_tensor(coord_std, dtype=torch.float32)
        self.register_buffer("grid_norm", grid_norm)                           # (J, 3)

    # ---- forward / predicción --------------------------------------------
    def forward(self, branch_norm):
        """branch_norm (B, I) ya normalizado -> (B, J, 3) normalizado."""
        return self.net((branch_norm, self.grid_norm))

    def predict_mT(self, branch_norm):
        return self(branch_norm) * self.b_std + self.b_mean                    # (B, J, 3) mT

    # ---- física (forward mode) -------------------------------------------
    def _physics_residuals(self, branch_norm, trunk_norm=None):
        """Devuelve `(b_norm (B,J,3), div, cx, cy, cz)` con div/curl en (B,J).

        Las derivadas están en mT/mm = T/m, con el doble factor
        `b_std[i]/coord_std[j]` (denorm de output e input). Ver docstring del módulo.
        """
        if trunk_norm is None:
            trunk_norm = self.grid_norm.detach().clone().requires_grad_(True)
        outputs = self.net((branch_norm, trunk_norm))                          # (B, J, 3)
        forward_call = lambda t: self.net((branch_norm, t))

        def g(i, j):
            return _jac(outputs, forward_call, trunk_norm, i, j)[..., 0]        # (B, J)

        s  = self.b_std
        cs = 1.0 / self.coord_std
        div    = s[0]*cs[0]*g(0, 0) + s[1]*cs[1]*g(1, 1) + s[2]*cs[2]*g(2, 2)
        curl_x = s[2]*cs[1]*g(2, 1) - s[1]*cs[2]*g(1, 2)
        curl_y = s[0]*cs[2]*g(0, 2) - s[2]*cs[0]*g(2, 0)
        curl_z = s[1]*cs[0]*g(1, 0) - s[0]*cs[1]*g(0, 1)
        return outputs, div, curl_x, curl_y, curl_z

    def _step(self, batch, *, stage):
        branch_norm, target_norm = batch                                       # (B,I), (B,J,3)
        with torch.enable_grad():
            trunk_g = self.grid_norm.detach().clone().requires_grad_(True)
            b_pred, div, cx, cy, cz = self._physics_residuals(branch_norm, trunk_g)

            loss_data = F.mse_loss(b_pred, target_norm)
            loss_div  = (div ** 2).mean()
            loss_curl = (cx ** 2 + cy ** 2 + cz ** 2).mean()
            loss = (loss_data
                    + self.hparams.lambda_div  * loss_div
                    + self.hparams.lambda_curl * loss_curl)
        dde.grad.clear()  # evita que el cache de jvp acumule grafos entre steps

        mse_metric = self.train_mse if stage == "train" else self.val_mse
        mse_metric(b_pred.detach(), target_norm)

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
        # enable_grad() está dentro de _step; Lightning desactiva grad en val
        # pero el context manager local lo re-habilita (física necesita autograd).
        return self._step(batch, stage="val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)


def count_params(model):
    return sum(p.numel() for p in model.parameters())

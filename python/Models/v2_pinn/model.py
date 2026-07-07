"""PINN: red de campo condicional + LightningModule con losses físicas.

`f(B_sens, x, y, z) → B(x, y, z)` — función de campo continua. Las losses
físicas (∇·B, ∇×B, TV) se computan vía `torch.autograd.grad` sobre las
coordenadas de input.

Activación SiLU por default — diferenciable en todos lados, no satura, y
funciona bien con autograd para segundas derivadas. **Nunca usar ReLU**: su
segunda derivada es delta (cero casi en todos lados → físicas se rompen).

## Normalización per-componente y regla de la cadena

La red ve coords y B normalizados con `(3,)` mean/std independientes por eje.
Si imponés `∇·B_norm = 0` directamente, NO equivale a `∇·B_físico = 0` cuando
las escalas son distintas por componente. Por eso `_maxwell_loss` y `_tv_loss`
multiplican cada derivada parcial por `b_std[i] / pts_std[j]` antes de armar
divergencia/rotacional → la pérdida física se evalúa en mT/mm.

## Gradient balancing (opcional)

Con `balance_grads=True`, en cada paso se reescala dinámicamente la
contribución de cada loss física para igualar la norma del gradiente con la
de `loss_data`. Reemplaza `lambda_div / lambda_rot / lambda_tv` (que se
ignoran en ese modo). Tradeoff: 3 backwards adicionales por step.
"""
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.regression import MeanSquaredError

_ACTIVATIONS = {"silu": nn.SiLU, "tanh": nn.Tanh, "gelu": nn.GELU}


class PINN(nn.Module):
    """FCNN simple: input (n_in) → hidden_layers → output (3).

    `n_in = I*3 + 3` (sensores + xyz). Última capa lineal sin activación.
    La salida es B normalizado: `B_norm = (B_real - b_mean) / b_std`.
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


def _physics_grads(b_pred, points):
    """Computa ∇Bx, ∇By, ∇Bz wrt `points` en espacio NORMALIZADO. Cada uno (K, 3).

    `create_graph=True` para que las losses derivadas tengan grafo y sean
    diferenciables (necesario para que el optimizador propague el gradiente).
    """
    ones = torch.ones_like(b_pred[:, 0])
    gBx = torch.autograd.grad(b_pred[:, 0], points, ones, create_graph=True)[0]
    gBy = torch.autograd.grad(b_pred[:, 1], points, ones, create_graph=True)[0]
    gBz = torch.autograd.grad(b_pred[:, 2], points, ones, create_graph=True)[0]
    return gBx, gBy, gBz


def _physical_partials(gBx, gBy, gBz, b_std, pts_std):
    """Aplica regla de la cadena: ∂B_real_i/∂x_real_j = (b_std[i]/pts_std[j]) · ∂B_norm_i/∂x_norm_j.

    `b_std`, `pts_std` son tensores shape (3,) en el mismo device. Devuelve
    `(gBx_phys, gBy_phys, gBz_phys)` con shape (K, 3) cada uno — gradientes en
    unidades físicas (mT/mm).
    """
    # `pts_std` se aplica por columna (eje del input), `b_std` por fila (componente del output).
    inv_pts = 1.0 / pts_std                                  # (3,)
    gBx_phys = gBx * inv_pts * b_std[0]                      # (K, 3)
    gBy_phys = gBy * inv_pts * b_std[1]
    gBz_phys = gBz * inv_pts * b_std[2]
    return gBx_phys, gBy_phys, gBz_phys


def _div_loss(gBx_phys, gBy_phys, gBz_phys):
    """Mean((∇·B)²) en (mT/mm)²."""
    div = gBx_phys[:, 0] + gBy_phys[:, 1] + gBz_phys[:, 2]
    return (div ** 2).mean()


def _rot_loss(gBx_phys, gBy_phys, gBz_phys):
    """Mean(‖∇×B‖²) en (mT/mm)²."""
    rot_x = gBz_phys[:, 1] - gBy_phys[:, 2]
    rot_y = gBx_phys[:, 2] - gBz_phys[:, 0]
    rot_z = gBy_phys[:, 0] - gBx_phys[:, 1]
    return (rot_x ** 2 + rot_y ** 2 + rot_z ** 2).mean()


def _tv_loss(gBx_phys, gBy_phys, gBz_phys):
    """Mean(|∂B/∂x| + |∂B/∂y| + |∂B/∂z|) en mT/mm — promedio de magnitudes de gradiente."""
    return (gBx_phys.abs().sum(-1) + gBy_phys.abs().sum(-1) + gBz_phys.abs().sum(-1)).mean()


def _grad_norm(grads):
    """Norma L2 sobre la lista de tensores como si fuera un solo vector flattened."""
    return torch.sqrt(sum((g ** 2).sum() for g in grads if g is not None))


class LitPINN(pl.LightningModule):
    """Wrapper Lightning con losses físicas + (opcional) gradient balancing.

    Modo fijo (`balance_grads=False`):
        loss = data + λ_div · div + λ_rot · rot + λ_tv · tv

    Modo balanceado (`balance_grads=True`):
        - automatic_optimization desactivado.
        - Se computan grads de (data, div, rot, tv) por separado.
        - Cada loss física se reescala para igualar la norma de grads con la
          de data: `α_k = ‖g_data‖ / (‖g_k‖ + ε)`.
        - p.grad ← g_data + α_div·g_div + α_rot·g_rot + α_tv·g_tv.
        - Los λ se ignoran en este modo.

    Stats per-componente (`b_mean`, `b_std`, `pts_mean`, `pts_std`) se
    persisten como buffers — viajan en el `.ckpt` y la inferencia puede
    desnormalizar sin metadatos externos.
    """

    def __init__(self, n_sensors, hidden_layers, *,
                 activation="silu",
                 lr=1e-3, weight_decay=1e-5,
                 lambda_div=1e-2, lambda_rot=1e-2, lambda_tv=1e-4,
                 balance_grads=False, balance_eps=1e-8,
                 manual_clip_val=None,
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

        if balance_grads:
            self.automatic_optimization = False

    # -- Forward ----------------------------------------------------------

    def forward(self, sensors, points_norm):
        """sensors: (K, I·3) ya normalizado por `x_scaler`.
        points_norm: (K, 3) ya normalizado por `(pts - mean)/std`."""
        x = torch.cat([sensors, points_norm], dim=-1)
        return self.net(x)

    def predict_mT(self, sensors, points_norm):
        """Forward + denormalización a mT físicas: `B_real = B_norm·b_std + b_mean`."""
        b_norm = self(sensors, points_norm)
        return b_norm * self.b_std + self.b_mean

    # -- Losses (compartidas entre training/validation) ------------------

    def _compute_losses(self, batch):
        """Devuelve `(loss_data, loss_div, loss_rot, loss_tv, b_pred)`.

        `b_pred` se devuelve para alimentar el `MeanSquaredError` métrico.
        """
        sensors, points_norm, b_target_norm = batch     # (K, I·3), (K, 3), (K, 3)
        points_norm = points_norm.detach().requires_grad_(True)
        b_pred = self(sensors, points_norm)

        loss_data = F.mse_loss(b_pred, b_target_norm)

        gBx, gBy, gBz = _physics_grads(b_pred, points_norm)
        gBx_p, gBy_p, gBz_p = _physical_partials(gBx, gBy, gBz, self.b_std, self.pts_std)

        loss_div = _div_loss(gBx_p, gBy_p, gBz_p)
        loss_rot = _rot_loss(gBx_p, gBy_p, gBz_p)
        loss_tv  = _tv_loss (gBx_p, gBy_p, gBz_p)

        return loss_data, loss_div, loss_rot, loss_tv, b_pred

    # -- Steps -----------------------------------------------------------

    def _log_components(self, *, stage,
                        loss_total, loss_data, loss_div, loss_rot, loss_tv,
                        a_div=None, a_rot=None, a_tv=None,
                        b_pred=None, b_target=None):
        """Log unificado para train/val. `a_*` son escalas de balancing si aplica."""
        prog = (stage == "train")
        self.log(f"{stage}_loss",   loss_total, on_epoch=True, prog_bar=prog)
        self.log(f"{stage}_data",   loss_data,  on_epoch=True)
        self.log(f"{stage}_div",    loss_div,   on_epoch=True)
        self.log(f"{stage}_rot",    loss_rot,   on_epoch=True)
        self.log(f"{stage}_tv",     loss_tv,    on_epoch=True)
        if a_div is not None:
            self.log(f"{stage}_alpha_div", a_div, on_epoch=True)
            self.log(f"{stage}_alpha_rot", a_rot, on_epoch=True)
            self.log(f"{stage}_alpha_tv",  a_tv,  on_epoch=True)
        if b_pred is not None and b_target is not None:
            mse_metric = self.train_mse if stage == "train" else self.val_mse
            mse_metric(b_pred.detach(), b_target)
            self.log(f"{stage}_mse_norm", mse_metric, on_epoch=True)

    def training_step(self, batch, batch_idx):
        if self.hparams.balance_grads:
            return self._training_step_balanced(batch)
        return self._training_step_fixed(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        # Las losses físicas requieren autograd graph; Lightning desactiva
        # grad en val por default → reactivar.
        with torch.enable_grad():
            return self._training_step_fixed(batch, stage="val", train=False)

    # ---- Modo λ fijo (Lightning automático) ---------------------------

    def _training_step_fixed(self, batch, *, stage, train=True):
        loss_data, loss_div, loss_rot, loss_tv, b_pred = self._compute_losses(batch)
        loss = (loss_data
                + self.hparams.lambda_div * loss_div
                + self.hparams.lambda_rot * loss_rot
                + self.hparams.lambda_tv  * loss_tv)
        self._log_components(stage=stage,
                             loss_total=loss, loss_data=loss_data,
                             loss_div=loss_div, loss_rot=loss_rot, loss_tv=loss_tv,
                             b_pred=b_pred, b_target=batch[2])
        return loss

    # ---- Modo balanceado (manual optimization) ------------------------

    def _training_step_balanced(self, batch):
        opt = self.optimizers()
        opt.zero_grad()

        loss_data, loss_div, loss_rot, loss_tv, b_pred = self._compute_losses(batch)

        params = [p for p in self.parameters() if p.requires_grad]
        # Computar grads de cada loss por separado (retain_graph para reusar el grafo).
        g_data = torch.autograd.grad(loss_data, params, retain_graph=True, allow_unused=True)
        g_div  = torch.autograd.grad(loss_div,  params, retain_graph=True, allow_unused=True)
        g_rot  = torch.autograd.grad(loss_rot,  params, retain_graph=True, allow_unused=True)
        g_tv   = torch.autograd.grad(loss_tv,   params, retain_graph=False, allow_unused=True)

        eps = self.hparams.balance_eps
        n_data = _grad_norm(g_data)
        n_div  = _grad_norm(g_div)
        n_rot  = _grad_norm(g_rot)
        n_tv   = _grad_norm(g_tv)
        a_div  = (n_data / (n_div + eps)).detach()
        a_rot  = (n_data / (n_rot + eps)).detach()
        a_tv   = (n_data / (n_tv  + eps)).detach()

        # Asignar p.grad como combinación lineal de los gradientes individuales.
        for p, gd, gdiv, grot, gtv in zip(params, g_data, g_div, g_rot, g_tv):
            grad = _maybe_zeros_like(p, gd)
            if gdiv is not None: grad = grad + a_div * gdiv
            if grot is not None: grad = grad + a_rot * grot
            if gtv  is not None: grad = grad + a_tv  * gtv
            p.grad = grad

        # `Trainer.gradient_clip_val` no aplica en automatic_optimization=False.
        # En modo balanceado, usar `manual_clip_val` declarado en hparams.
        clip = self.hparams.manual_clip_val
        if clip is not None and clip > 0:
            torch.nn.utils.clip_grad_norm_(params, clip)

        opt.step()

        # Loss "total" reportable: combinación efectiva con escalas usadas.
        loss_total = (loss_data
                      + a_div * loss_div.detach()
                      + a_rot * loss_rot.detach()
                      + a_tv  * loss_tv.detach())
        self._log_components(stage="train",
                             loss_total=loss_total, loss_data=loss_data,
                             loss_div=loss_div, loss_rot=loss_rot, loss_tv=loss_tv,
                             a_div=a_div, a_rot=a_rot, a_tv=a_tv,
                             b_pred=b_pred, b_target=batch[2])
        return loss_total

    # -- Optim ------------------------------------------------------------

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )


def _maybe_zeros_like(p, g):
    """Si `g is None` (param no usado en ese loss), arranca con ceros del shape de `p`."""
    return torch.zeros_like(p) if g is None else g.clone()


def count_params(model):
    return sum(p.numel() for p in model.parameters())

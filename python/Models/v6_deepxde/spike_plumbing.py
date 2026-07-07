"""Spike de fontanería DeepXDE (tarea #1 del README) — valida empíricamente:

1. `DeepONetCartesianProd(num_outputs=3, multi_output_strategy="independent")`
   produce output `(N, J, 3)` con branch `(N, I)` ⊗ trunk `(J, 3)`.
2. **Forward-mode autodiff** permite `dde.grad.jacobian` sobre el output 3D
   (reverse mode tira `NotImplementedError`).
3. `div` y `curl` salen `(N, J, 1)` — per-sample y per-punto.

Correr:  python -m Models.v6_deepxde.spike_plumbing   (con DDE_BACKEND=pytorch)
No es parte del entrenamiento; es un test de humo de la API antes de armar el módulo.
"""
import os

os.environ.setdefault("DDE_BACKEND", "pytorch")

import torch
import deepxde as dde

dde.config.set_default_autodiff("forward")  # OBLIGATORIO para output 3D (ver README)


def build_net(I, P=64, width=128, depth=2):
    layers_branch = [I] + [width] * depth + [P]
    layers_trunk = [3] + [width] * depth + [P]
    return dde.nn.DeepONetCartesianProd(
        layer_sizes_branch=layers_branch,
        layer_sizes_trunk=layers_trunk,
        activation="tanh",
        kernel_initializer="Glorot normal",
        num_outputs=3,
        multi_output_strategy="independent",
    )


def _jac(outputs, forward_call, xs, i, j):
    """`dde.grad.jacobian` en forward mode devuelve (tensor, callable) -> [0]."""
    out = dde.grad.jacobian((outputs, forward_call), xs, i=i, j=j)
    return out[0] if isinstance(out, tuple) else out


def physics_residuals(net, branch, trunk, b_std):
    """div y curl en unidades físicas (Σ b_std[k]·∂B_norm[k]/∂x_j). Shapes (N,J,1)."""
    inputs = (branch, trunk)
    outputs = net(inputs)  # (N, J, 3) normalizado

    forward_call = lambda t: net((branch, t))

    # dB[i]/dx[j]
    g = lambda i, j: _jac(outputs, forward_call, trunk, i, j)
    sx, sy, sz = b_std[0], b_std[1], b_std[2]

    div = sx * g(0, 0) + sy * g(1, 1) + sz * g(2, 2)
    curl_x = sz * g(2, 1) - sy * g(1, 2)
    curl_y = sx * g(0, 2) - sz * g(2, 0)
    curl_z = sy * g(1, 0) - sx * g(0, 1)
    return outputs, div, curl_x, curl_y, curl_z


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    I, P, N, J = 179, 64, 8, 125  # 179 sensores reales, grilla 5x5x5 = 125
    print(f"device={device}  I={I}  P={P}  N={N}  J={J}")

    net = build_net(I, P).to(device)
    branch = torch.randn(N, I, device=device)
    trunk = torch.randn(J, 3, device=device, requires_grad=True)
    b_std = torch.tensor([1.0, 50.0, 1.0], device=device)  # Bx,Bz ~mT vs By ~50mT

    out, div, cx, cy, cz = physics_residuals(net, branch, trunk, b_std)

    print(f"output : {tuple(out.shape)}   (esperado (N,J,3) = ({N},{J},3))")
    print(f"div    : {tuple(div.shape)}   (esperado (N,J,1) = ({N},{J},1))")
    print(f"curl_x : {tuple(cx.shape)}")
    assert out.shape == (N, J, 3), out.shape
    assert div.shape == (N, J, 1), div.shape
    assert cx.shape == (N, J, 1)

    # sanity: el residuo depende del trunk (no es constante) -> autograd vivo
    print(f"div  range: [{div.min().item():+.3e}, {div.max().item():+.3e}]")
    print(f"curl range: [{cx.min().item():+.3e}, {cx.max().item():+.3e}]")

    # data loss dummy para confirmar que un backward total compila el grafo 2do orden
    target = torch.randn(N, J, 3, device=device)
    loss = (out - target).pow(2).mean() + div.pow(2).mean() + cx.pow(2).mean()
    loss.backward()
    nparams = sum(p.numel() for p in net.parameters())
    grad_ok = all(p.grad is not None for p in net.parameters() if p.requires_grad)
    print(f"loss={loss.item():.4f}  params={nparams:,}  backward_grads_ok={grad_ok}")
    print("OK: fontaneria DeepXDE validada (forward-mode + div/curl + backward).")


if __name__ == "__main__":
    main()

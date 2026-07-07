"""HDF5 lazy

Unidad de entrenamiento: una sample completa = `(B_sens flat, B_grid flat)`. La
red mapea `B_sens (I·3) --> B_grid (J·3)` en una sola pasada, sin coordenadas
como input. El DataLoader hace batching estándar.

- N = nº de muestras (configuraciones de perturbación)
- I = nº de sensores
- J = nº de puntos de grilla (= Nx · Ny · Nz)
"""
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


def load_dataset_metadata(h5_path):
    """Fetcha el header del HDF5: shapes, grilla, sensores, attrs.

    NO carga `B_grid`/`B_sens` (esos pueden ser GBs). Diseñado para correr una
    sola vez al inicio y tener info.
    """
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        N = int(f["B_grid"].shape[0])
        J = int(f["B_grid"].shape[1])
        I = int(f["B_sens"].shape[1])
        R_grid_xyz = f["R_grid_xyz_mm"][:].astype(np.float32)
        sens_xyz   = f["sens_xyz_mm"][:].astype(np.float32)
        grid_x     = f["grid_x"][:].astype(np.float32)
        grid_y     = f["grid_y"][:].astype(np.float32)
        grid_z     = f["grid_z"][:].astype(np.float32)
        attrs      = dict(f.attrs)

    Nx, Ny, Nz = len(grid_x), len(grid_y), len(grid_z)
    assert J == Nx * Ny * Nz, f"J={J} != Nx*Ny*Nz={Nx*Ny*Nz}"

    # El reshape (J, 3) -> (Nx, Ny, Nz, 3) asume orden ij meshgrid (x-outer,
    # z-inner). Es el contrato con `construir_R(coords=:cartesian, ...)` en
    # B0.jl y la escritura `permutedims(R_grid)` de generate_dataset.jl.
    R_grid_3d = R_grid_xyz.reshape(Nx, Ny, Nz, 3)
    gxx, gyy, gzz = np.meshgrid(grid_x, grid_y, grid_z, indexing="ij")
    assert (np.allclose(R_grid_3d[..., 0], gxx)
            and np.allclose(R_grid_3d[..., 1], gyy)
            and np.allclose(R_grid_3d[..., 2], gzz)), \
        ("Layout del HDF5 no respeta ij-meshgrid (x-outer, z-inner). "
         "Verificar que construir_R en B0.jl y la escritura en "
         "generate_dataset.jl mantengan esa convención.")

    return {
        "h5_path": str(h5_path),
        "N": N, "I": I, "J": J,
        "Nx": Nx, "Ny": Ny, "Nz": Nz,
        "grid_x": grid_x, "grid_y": grid_y, "grid_z": grid_z,
        "R_grid_xyz": R_grid_xyz, "R_grid_3d": R_grid_3d,
        "sens_xyz": sens_xyz,
        "attrs": attrs,
    }


def split_indices(N, *, val_frac, test_frac, seed):
    """Split vía `sklearn.model_selection.train_test_split`."""
    idx = np.arange(N, dtype=np.int64)
    idx_trainval, idx_test = train_test_split(
        idx, test_size=test_frac, random_state=seed,
    )
    val_size_rel = val_frac / (1.0 - test_frac)
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=val_size_rel, random_state=seed,
    )
    return {
        "train":   idx_train,
        "val":     idx_val,
        "test":    idx_test,
        "n_train": len(idx_train),
        "n_val":   len(idx_val),
        "n_test":  len(idx_test),
    }


def compute_train_stats(h5_path, train_indices, *, chunk=64):
    """Stats
    - `x_scaler` : `StandardScaler` per-feature sobre `B_sens` flatten (input).
    - `b_scale`  : RMS de `B_grid` sobre train (un escalar único para todas
                   las componentes y puntos --> preserva dirección del campo,
                   relación Bx:By:Bz, y estructura espacial).

    """
    h5_path = Path(h5_path)
    train_idx = np.asarray(train_indices, dtype=np.int64)

    x_scaler = StandardScaler()
    b_sum_sq = 0.0
    b_count  = 0

    with h5py.File(h5_path, "r") as f:
        for start in range(0, len(train_idx), chunk):
            idx = train_idx[start:start + chunk]
            idx_sorted = np.sort(idx)

            xs = f["B_sens"][idx_sorted].reshape(len(idx_sorted), -1).astype(np.float64)
            x_scaler.partial_fit(xs)

            bs = f["B_grid"][idx_sorted].astype(np.float64)   # (chunk, J, 3)
            b_sum_sq += float((bs ** 2).sum())
            b_count  += int(bs.size)

    b_scale = float(np.sqrt(b_sum_sq / max(b_count, 1)))

    return {"x_scaler": x_scaler, "b_scale": b_scale}


class SampleDataset(Dataset):
    """Dataset `(sensores_flat, B_grid_flat)` ya normalizados.

    Normalización:
      - sensores: `x_scaler.transform`
      - B target: `B / b_scale`

    Una sample completa por item; el DataLoader colaciona en batches normales.

    """

    def __init__(self, h5_path, sample_indices, *,
                 x_scaler=None, b_scale=1.0):
        self.h5_path        = str(h5_path)
        self.sample_indices = np.asarray(sample_indices, dtype=np.int64)
        self.x_scaler       = x_scaler
        self.b_scale        = float(b_scale)
        self._file          = None

    def _open(self):
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        return self._file

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, k):
        f = self._open()
        sample_i = int(self.sample_indices[k])

        sensors = f["B_sens"][sample_i].reshape(-1).astype(np.float32)
        if self.x_scaler is not None:
            sensors = self.x_scaler.transform(sensors[None, :]).reshape(-1).astype(np.float32)

        b_target = (f["B_grid"][sample_i].reshape(-1).astype(np.float32) / self.b_scale)

        return torch.from_numpy(sensors), torch.from_numpy(b_target)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        return state


def make_loader(h5_path, sample_indices, *,
                batch_size, shuffle,
                x_scaler=None, b_scale=1.0,
                num_workers=0, pin_memory=False):
    """DataLoader sobre `SampleDataset` con batching estándar."""
    ds = SampleDataset(h5_path, sample_indices,
                       x_scaler=x_scaler, b_scale=b_scale)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )


def load_full_sample(h5_path, sample_idx, *, x_scaler=None):
    """Helper para evaluación: carga `(B_sens_flat normalizado, B_grid_flat raw mT)`.

    Aplica `x_scaler` a los sensores. **NO normaliza el B_target** — devuelve
    raw mT para que `metrics.evaluate(...)` compare contra física directa.
    """
    h5_path = Path(h5_path)
    sample_idx = int(sample_idx)
    with h5py.File(h5_path, "r") as f:
        sensors = f["B_sens"][sample_idx].reshape(-1).astype(np.float32)
        if x_scaler is not None:
            sensors = x_scaler.transform(sensors[None, :]).reshape(-1).astype(np.float32)
        b_full  = f["B_grid"][sample_idx].astype(np.float32)            # (J, 3) mT raw
    return torch.from_numpy(sensors), torch.from_numpy(b_full)

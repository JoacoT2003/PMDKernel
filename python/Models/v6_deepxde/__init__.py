"""v6_deepxde — PI-DeepONet con DeepXDE (operator learning sensores→campo + Maxwell).

Reemplaza v5 (DeepSets, O(J·I)). DeepONet CartesianProd corre el branch
1×/muestra + trunk 1×/punto → ~10× más rápido por step (ver spike_plumbing.py).
Ver README.md para la nota de diseño y la API de DeepXDE fijada (1.15.0, forward-mode).
"""
from . import data, metrics, model, train  # noqa: F401

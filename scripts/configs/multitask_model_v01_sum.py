"""
Multitask Model trained on the Unified DB
- The four datasets QM9, Alchemy, OE62, and HOPV, with the unified split
  (split v2) are used.
- Two sets of molecular energies are trained: (HOMO, LUMO, gap) x (B3LYP, PBE0)
- With respect to multiask_model ("" a.k.a. "_v01"), the Set2Set output module
  is replaced with the sum output module (mind the statistics in meanstensor
  and stddevtensor).
"""
import torch
from torch.optim import Adam
import schnetpack as spk
import schnetpack.atomistic.model
from schnetpack.train import ReduceLROnPlateauHook
from schnetpack.train.metrics import MeanAbsoluteError

from maltose.loss import build_gated_mse_loss
from maltose.metrics import MultitaskMetricWrapper


properties = ['HOMO-B3LYP', 'LUMO-B3LYP', 'Gap-B3LYP', 'HOMO-PBE0', 'LUMO-PBE0', 'Gap-PBE0']
datasets = ['qm9', 'alchemy', 'oe62', 'hopv']
batch_size = 16

# For reproducibility, use precomputed statistics for mean and std
divide_by_atoms = True
meanstensor = torch.tensor([-0.3204, 0.0050, 0.3255, -0.1812, -0.0448, 0.1365])
stddevstensor = torch.tensor([0.0875, 0.0671, 0.0887, 0.1007, 0.0348, 0.0850])

# model build
means = {k: v for k, v in zip(properties, meanstensor)}
stddevs = {k: v for k, v in zip(properties, stddevstensor)}
representation = spk.SchNet(n_interactions=6)
output_modules = [
    spk.atomistic.Atomwise(
        n_in=representation.n_atom_basis,
        n_layers=2,
        n_out=1,
        mean=means[prop],
        stddev=stddevs[prop],
        aggregation_mode="sum",
        property=prop,
    )
for prop in properties]
model = schnetpack.AtomisticModel(representation, output_modules)

# build optimizer
optimizer = Adam(model.parameters(), lr=1e-4)

# hooks
metrics = [MultitaskMetricWrapper(MeanAbsoluteError(p, p)) for p in properties]
hooks = [
    ReduceLROnPlateauHook(
        optimizer,
        min_lr=0.5e-6,
        stop_after_min=True),
]

loss = build_gated_mse_loss(properties)

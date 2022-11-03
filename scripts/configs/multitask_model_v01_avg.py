"""
Multitask Model trained on the Unified DB
- The four datasets QM9, Alchemy, OE62, and HOPV, with the unified split
  (split v2) are used.
- Two sets of molecular energies are trained: (HOMO, LUMO, gap) x (B3LYP, PBE0)
- With respect to multiask_model ("" a.k.a. "_v01"), the Set2Set output module
  is replaced with the avg output module.
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
divide_by_atoms = False
meanstensor = torch.tensor([-6.34, 0.18, 6.52, -6.49, -1.58, 4.90])
stddevstensor = torch.tensor([0.68, 1.29, 1.52, 0.72, 0.90, 1.18])

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
        aggregation_mode="avg",
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

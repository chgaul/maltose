"""
Multitask Model trained on the Unified DB
- The dataset with the smallest molecules is used (QM9 only), with the unified
  split (split v2).
- Only the B3LYP targets are available: (HOMO, LUMO, gap) x (B3LYP)
- With respect to multiask_model_v08, the Set2Set output module is replaced
  with the sum output module (mind the statistics in meanstensor and
  stddevtensor).
"""
import torch
from torch.optim import Adam
import schnetpack as spk
import schnetpack.atomistic.model
from schnetpack.train import ReduceLROnPlateauHook
from schnetpack.train.metrics import MeanAbsoluteError

from maltose.loss import build_gated_mse_loss
from maltose.metrics import MultitaskMetricWrapper

properties = ['HOMO-B3LYP', 'LUMO-B3LYP', 'Gap-B3LYP']
datasets = ['qm9']
batch_size = 16

# Precomputed statistics
divide_by_atoms = True
meanstensor = torch.tensor([-0.3748, 0.0095, 0.3843])
stddevstensor = torch.tensor([0.0863, 0.0756, 0.0722])

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
    ) for prop in properties]
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

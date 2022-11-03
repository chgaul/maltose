"""
Multitask Model trained on the Unified DB
- The two datasets with the smallest molecules are used (QM9, Alchemy), with
  the unified split (split v2).
- Only the B3LYP targets are available: (HOMO, LUMO, gap) x (B3LYP)
- The network has the Set2Set + molecule-level output module with three
  outputs. Otherwise, the architecture is the same as qm9_hopv_model_v13 and
  similar to multitask_model_v01.
"""
import torch
from torch.optim import Adam
import schnetpack as spk
import schnetpack.atomistic.model
from schnetpack.train import ReduceLROnPlateauHook
from schnetpack.train.metrics import MeanAbsoluteError

from maltose.output_modules import Set2Set
from maltose.loss import build_gated_mse_loss
from maltose.metrics import MultitaskMetricWrapper


properties = ['HOMO-B3LYP', 'LUMO-B3LYP', 'Gap-B3LYP']
datasets = ['qm9', 'alchemy']
batch_size = 16

# For reproducibility, use precomputed statistics for mean and std
divide_by_atoms = False
meanstensor = torch.tensor([-6.36, 0.24, 6.59])
stddevstensor = torch.tensor([0.69, 1.25, 1.44])

# model build
representation = spk.SchNet(n_interactions=6)
output_modules = [
    Set2Set(
        n_in=representation.n_atom_basis,
        processing_steps=3,
        m_net_layers=2,
        m_net_neurons=32,
        properties=properties,
        means=meanstensor,
        stddevs=stddevstensor,
    )
]
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

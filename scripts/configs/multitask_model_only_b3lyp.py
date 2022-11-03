"""
Multitask Model trained on the Unified DB, B3LYP only
- The four datasets QM9, Alchemy, OE62, and HOPV, with the unified split
  (split v2) are used.
- One set of molecular energies is trained: (HOMO, LUMO, gap) x (B3LYP)
- The network has the Set2Set + molecule-level output module with six outputs.
  Same as multitask_model_v01.
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
datasets = ['qm9', 'alchemy', 'hopv']
batch_size = 8

# Precomputed statistics
divide_by_atoms = False
meanstensor = torch.tensor([-6.35, 0.17, 6.52])
stddevstensor = torch.tensor([0.71, 1.28, 1.51])

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

"""
Multitask Model trained on the Unified DB
- The four datasets QM9, Alchemy, OE62, and HOPV, with the unified split
  (split v2) are used.
- Two sets of molecular energies are trained: (HOMO, LUMO, gap) x (B3LYP, PBE0)
- The network has the Set2Set + molecule-level output module with six outputs.
"""
import torch
from torch.optim import Adam
import schnetpack as spk
import schnetpack.atomistic.model
from schnetpack.train import ReduceLROnPlateauHook, CSVHook
from schnetpack.train.metrics import MeanAbsoluteError

from maltose.output_modules import Set2Set
from maltose.loss import build_gated_mse_loss
from maltose.metrics import MultitaskMetricWrapper


properties = ['HOMO-B3LYP', 'LUMO-B3LYP', 'Gap-B3LYP', 'HOMO-PBE0', 'LUMO-PBE0', 'Gap-PBE0']
datasets = ['qm9', 'alchemy', 'oe62', 'hopv']
batch_size = 16

# For reproducibility, use precomputed statistics for mean and std
divide_by_atoms = False
meanstensor = torch.tensor([-6.35, 0.17, 6.52, -6.48, -1.56, 4.92])
stddevstensor = torch.tensor([0.71, 1.28, 1.51, 0.71, 0.91, 1.17])

def build_model():
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
    return schnetpack.AtomisticModel(representation, output_modules)

def build_optimizer(model):
    return Adam(model.parameters(), lr=1e-4)

def build_hooks(optimizer, log_path):
    metrics = [MultitaskMetricWrapper(MeanAbsoluteError(p, p)) for p in properties]
    return [
        CSVHook(log_path=log_path, metrics=metrics),
        ReduceLROnPlateauHook(
            optimizer,
            min_lr=0.5e-6,
            stop_after_min=True),
    ]

def build_loss():
    return build_gated_mse_loss(properties)

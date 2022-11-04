"""
Multitask Model trained on the Unified DB
- The dataset with the smallest molecules is used (QM9 only), with the unified
  split (split v2).
- Only the B3LYP targets are available: (HOMO, LUMO, gap) x (B3LYP)
- The network has the Set2Set + molecule-level output module with three
  outputs. Otherwise, the architecture is similar to multitask_model_v01.
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

properties = ['HOMO-B3LYP', 'LUMO-B3LYP', 'Gap-B3LYP']
datasets = ['qm9']
batch_size = 16

# For reproducibility, use precomputed statistics for mean and std
divide_by_atoms = False
meanstensor = torch.tensor([-6.53, 0.29, 6.82])
stddevstensor = torch.tensor([0.60, 1.28, 1.29])

# model build
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

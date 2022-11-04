"""
Multitask Model trained on the Unified DB
- The dataset with the smallest molecules is used (QM9 only), with the unified
  split (split v2).
- Only the B3LYP targets are available: (HOMO, LUMO, gap) x (B3LYP)
- With respect to multiask_model_v08, the Set2Set output module is replaced
  with the avg output module.
"""
import torch
from torch.optim import Adam
import schnetpack as spk
import schnetpack.atomistic.model
from schnetpack.train import ReduceLROnPlateauHook, CSVHook
from schnetpack.train.metrics import MeanAbsoluteError

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
        ) for prop in properties]
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

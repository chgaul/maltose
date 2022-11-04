"""
Multitask Model trained on OE62 and HOPV
- The datasets OE62, and HOPV, with the unified split (split v2) are used.
- Two sets of molecular energies are trained: (HOMO, LUMO, gap) x (B3LYP, PBE0)
- The network has the architecture SchNet(6) + Set2Set(with six outputs), like qm9_hopv_model_v13.
- Batch size is 8.
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
datasets = ['oe62', 'hopv']
batch_size = 8

# For reproducibility, use precomputed statistics for mean and std
divide_by_atoms = False
meanstensor = torch.tensor([-5.11, -2.77, 2.34, -6.49, -1.60, 4.89])
stddevstensor = torch.tensor([0.23, 0.36, 0.40, 0.72, 0.89, 1.19])

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

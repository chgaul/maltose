"""
Multitask Model trained on the Unified DB
- The four datasets QM9, Alchemy, OE62, and HOPV, with the unified split
  (split v2) are used.
- Two sets of molecular energies are trained: (HOMO, LUMO, gap) x (B3LYP, PBE0)
- The network has the Set2Set + molecule-level output module with six outputs.
"""
import argparse
import numpy as np
import logging
import torch
from torch.optim import Adam
import os
import schnetpack as spk
import schnetpack.atomistic.model
from schnetpack.data.atoms import AtomsData, ConcatAtomsData
from schnetpack.train import Trainer, CSVHook, ReduceLROnPlateauHook
from schnetpack.train.metrics import MeanAbsoluteError

from maltose.output_modules import Set2Set
from maltose.atoms import MultitaskAtomsData
from maltose.loss import build_gated_mse_loss
from maltose.metrics import MultitaskMetricWrapper

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model-base-dir", default="./models",
    help="Parent directory for the trained model.")
parser.add_argument(
    "--data-base-dir", default="./data",
    help="Base directory where the primary data is located.")
parser.add_argument(
    "--device", default="cpu",
    help="Device for running the training. For example 'cpu', 'cuda', 'cuda:2'")
args = parser.parse_args()

# basic settings
device = args.device
data_base_dir = args.data_base_dir
model_dir = os.path.join(args.model_base_dir, "multitask_model_v01")

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

if not os.path.exists(model_dir):
    print("Create model directory {}".format(model_dir))
    os.makedirs(model_dir)
else:
    print("{} exists:".format(model_dir))
    for file in os.listdir(model_dir):
        print(file)

properties = ['HOMO-B3LYP', 'LUMO-B3LYP', 'Gap-B3LYP', 'HOMO-PBE0', 'LUMO-PBE0', 'Gap-PBE0']
batch_size = 16


logging.info("Preparing the datasets")
train_sets, val_sets, test_sets = [], [], []
for dataset_name, mapping in [
    ('qm9', [
        ('HOMO-B3LYP', 'homo'), ('LUMO-B3LYP', 'lumo'), ('Gap-B3LYP', 'gap'), 
        ('HOMO-PBE0', None), ('LUMO-PBE0', None), ('Gap-PBE0', None)]),
    ('alchemy', [
        ('HOMO-B3LYP', 'homo'), ('LUMO-B3LYP', 'lumo'), ('Gap-B3LYP', 'gap'), 
        ('HOMO-PBE0', None), ('LUMO-PBE0', None), ('Gap-PBE0', None)]),
    ('oe62', [
        ('HOMO-B3LYP', None), ('LUMO-B3LYP', None), ('Gap-B3LYP', None), 
        ('HOMO-PBE0', 'homo PBE0_vacuum'), ('LUMO-PBE0', 'lumo PBE0_vacuum'), ('Gap-PBE0', 'gap PBE0_vacuum')]),
    ('hopv', [
        ('HOMO-B3LYP', 'HOMO B3LYP/def2-SVP'), ('LUMO-B3LYP', 'LUMO B3LYP/def2-SVP'), ('Gap-B3LYP', 'Gap B3LYP/def2-SVP'),
        ('HOMO-PBE0', 'HOMO PBE0/def2-SVP'), ('LUMO-PBE0', 'LUMO PBE0/def2-SVP'), ('Gap-PBE0', 'Gap PBE0/def2-SVP')])
    ]:
    dataset_file = os.path.join(data_base_dir, dataset_name, 'data.db')
    if not os.path.exists(dataset_file):
        print(dataset_file , 'does not exist!')
        print('Please run the respective scipt in scripts/primary_data/ first,')
        print('and make sure the --data-base-dir is specified correctly.')
    dataset = AtomsData(dataset_file, load_only=properties)
    split_file = os.path.join(data_base_dir, dataset_name, 'split_v2.npz')
    if not os.path.exists(split_file):
        print(split_file, 'does not exist!')
        print('Please run scripts/primary_data/unified_split.py first!')
    train, val, test = spk.data.partitioning.train_test_split(
        data=dataset, split_file=split_file)
    train_sets.append(MultitaskAtomsData(train, mapping))
    val_sets.append(MultitaskAtomsData(val, mapping))
    test_sets.append(MultitaskAtomsData(test, mapping))

train = ConcatAtomsData(train_sets)
val = ConcatAtomsData(val_sets)
test = ConcatAtomsData(test_sets)

train_loader = spk.AtomsLoader(train, batch_size=batch_size, shuffle=True)
val_loader = spk.AtomsLoader(val, batch_size=batch_size)

print('Size of training set: ', len(train))
print('Size of validation set: ', len(val))

# Measure mean and variance of the predicted properties (on a random sample)
def measure_mean_std(n_batches, batch_size):    
    loader = spk.AtomsLoader(train, batch_size=batch_size, shuffle=True)
    valids = {p: [] for p in properties}
    for i, b in enumerate(loader):
        for p in properties:
            data = np.array(b[p])
            validity = data[:, 0] > 0
            valids[p].append(data[:, 1][validity])
        print(len(valids[p]), end='\r')
        if i==n_batches: break
    for p in properties:
        arr = np.concatenate(valids[p])
        print('{:>10} ({} items): mean={:.2f}, std={:.2f}'.format(p, len(arr), np.mean(arr), np.std(arr)))
measure_mean_std(100, 100)


logging.info("Setting up the model")
# For reproducibility, use precomputed statistics for mean and std
meanstensor = torch.tensor([-6.35, 0.17, 6.52, -6.48, -1.56, 4.92])
stddevstensor = torch.tensor([0.71, 1.28, 1.51, 0.71, 0.91, 1.17])

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

# build the optimizer
optimizer = Adam(model.parameters(), lr=1e-4)

# hooks
metrics = [MultitaskMetricWrapper(MeanAbsoluteError(p, p)) for p in properties]
hooks = [
    CSVHook(log_path=model_dir, metrics=metrics),
    ReduceLROnPlateauHook(
        optimizer,
        min_lr=0.5e-6,
        stop_after_min=True),
]

loss = build_gated_mse_loss(properties)

# run training
trainer = Trainer(
    model_dir,
    model=model,
    hooks=hooks,
    loss_fn=loss,
    optimizer=optimizer,
    train_loader=train_loader,
    validation_loader=val_loader,
    keep_n_checkpoints=40,
    checkpoint_interval=10,
)
logging.info("Training")
trainer.train(device=device)

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
import os
import schnetpack as spk
import schnetpack.atomistic.model
from schnetpack.train import Trainer
from schnetpack.train.metrics import MeanAbsoluteError

from maltose.metrics import MultitaskMetricWrapper

import multitask_data

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", type=str, required=True,
    help="Name of the configuration to be trained, e.g., multitask_model_v01.")
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

import importlib

module_name = 'configs.{config_name}'.format(config_name=args.config)
print('Trying to load {}'.format(module_name))
config = importlib.import_module(module_name)


# basic settings
device = args.device
model_dir = os.path.join(args.model_base_dir, args.config)

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

if not os.path.exists(model_dir):
    print("Create model directory {}".format(model_dir))
    os.makedirs(model_dir)
else:
    print("{} exists:".format(model_dir))
    for file in os.listdir(model_dir):
        print(file)

properties = config.properties
batch_size = config.batch_size


logging.info("Preparing the datasets")
train, val, _ = multitask_data.join_multitask_data(
    args.data_base_dir,
    select_tasks=properties,
    select_datasets=config.datasets)

train_loader = spk.AtomsLoader(train, batch_size=batch_size, shuffle=True)
val_loader = spk.AtomsLoader(val, batch_size=batch_size)

print('Size of training set: ', len(train))
print('Size of validation set: ', len(val))


logging.info("Check mean and variance of training data")

# Measure mean and variance of the predicted properties (on a random sample)
def measure_mean_std(n_batches, divide_by_atoms):
    loader = spk.AtomsLoader(train, batch_size=1, shuffle=True)
    valids = {p: [] for p in properties}
    for i, b in enumerate(loader):
        if i % 10 == 0:
            print(i, '/', n_batches, end='\r')
        n_atoms = len(b['_atomic_numbers'][0])
        for p in properties:
            dt = np.array(b[p])
            validity = dt[:, 0] > 0
            if divide_by_atoms:
                valids[p].append(dt[:, 1][validity]/n_atoms)
            else:
                valids[p].append(dt[:, 1][validity])
        if i==n_batches: break
    arrs = [np.concatenate(valids[p]) for p in properties]
    return [np.mean(arr) for arr in arrs], [np.std(arr) for arr in arrs]

measured_means, measured_stds = measure_mean_std(
        10000, divide_by_atoms=config.divide_by_atoms)

# Check that the measured statistics coincide roughly with the precomputed
# statistics from the config:
reference_magnitude = np.sqrt(np.mean(np.square(measured_means)))
for p, m, c in zip(properties, measured_means, config.meanstensor):
    assert np.abs(m - float(c)) < 0.05 * reference_magnitude, \
        "{}: configured and measured mean differ: {}!={}".format(p, c, m)
reference_magnitude = np.sqrt(np.mean(np.square(measured_stds)))
for p, m, c in zip(properties, measured_stds, config.stddevstensor):
    assert np.abs(m - float(c)) < 0.1 * reference_magnitude, \
        "{}: configured and measured std differ: {}!={}".format(p, c, m)

logging.info("Setting up the model")
# hooks
metrics = [MultitaskMetricWrapper(MeanAbsoluteError(p, p)) for p in properties]

model = config.build_model()
optimizer = config.build_optimizer(model)
hooks = config.build_hooks(optimizer, log_path=model_dir)
loss_fn = config.build_loss()

logging.info("Training")
trainer = Trainer(
    model_dir,
    model=model,
    hooks=hooks,
    loss_fn=loss_fn,
    optimizer=optimizer,
    train_loader=train_loader,
    validation_loader=val_loader,
    keep_n_checkpoints=40,
    checkpoint_interval=10,
)
trainer.train(device=device)

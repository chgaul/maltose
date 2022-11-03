#!/usr/bin/env python
# coding: utf-8

# # Multitask Model trained on the Unified DB
# - The four datasets QM9, Alchemy, OE62, and HOPV, with the unified split (split v2) are used.
# - Two sets of molecular energies are trained: (HOMO, LUMO, gap) x (B3LYP, PBE0)
# - With respect to multiask_model ("" a.k.a. "_v01"), the Set2Set output module is replaced with the sum output module (mind the statistics in meanstensor and stddevtensor).

# In[ ]:


import logging
import torch
from torch.optim import Adam
import os
import schnetpack as spk
import schnetpack.atomistic.model
from schnetpack.data.atoms import AtomsData, ConcatAtomsData
from schnetpack.train import Trainer, CSVHook, ReduceLROnPlateauHook
from schnetpack.train.metrics import MeanAbsoluteError


# In[ ]:


import sys
sys.path.append('../') # add parent dir ('schnetpack_exps') dir to path
from utils import is_main
import data.paths
from schnetpack_custom.output_modules import Set2Set
from schnetpack_custom.atoms import MultitaskAtomsData
from schnetpack_custom.loss import build_gated_mse_loss
from schnetpack_custom.metrics import MultitaskMetricWrapper


# In[ ]:


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


# In[ ]:


# basic settings
model_dir = os.path.join(data.paths.model_dir, "multitask_model_v01_sum")
if not os.path.exists(model_dir):
    print("Create model directory {}".format(model_dir))
    os.makedirs(model_dir)
else:
    print("{} exists:".format(model_dir))
    for file in os.listdir(model_dir):
        print(file)


# In[ ]:


properties = ['HOMO-B3LYP', 'LUMO-B3LYP', 'Gap-B3LYP', 'HOMO-PBE0', 'LUMO-PBE0', 'Gap-PBE0']
batch_size = 16


# In[ ]:


logging.info("get the datasets")
train_sets, val_sets, test_sets = [], [], []
for DS, mapping in [
    (data.paths.QM9, [
        ('HOMO-B3LYP', 'homo'), ('LUMO-B3LYP', 'lumo'), ('Gap-B3LYP', 'gap'), 
        ('HOMO-PBE0', None), ('LUMO-PBE0', None), ('Gap-PBE0', None)]),
    (data.paths.Alchemy, [
        ('HOMO-B3LYP', 'homo'), ('LUMO-B3LYP', 'lumo'), ('Gap-B3LYP', 'gap'), 
        ('HOMO-PBE0', None), ('LUMO-PBE0', None), ('Gap-PBE0', None)]),
    (data.paths.OE62, [
        ('HOMO-B3LYP', None), ('LUMO-B3LYP', None), ('Gap-B3LYP', None), 
        ('HOMO-PBE0', 'homo PBE0_vacuum'), ('LUMO-PBE0', 'lumo PBE0_vacuum'), ('Gap-PBE0', 'gap PBE0_vacuum')]),
    (data.paths.HOPV, [
        ('HOMO-B3LYP', 'HOMO B3LYP/def2-SVP'), ('LUMO-B3LYP', 'LUMO B3LYP/def2-SVP'), ('Gap-B3LYP', 'Gap B3LYP/def2-SVP'),
        ('HOMO-PBE0', 'HOMO PBE0/def2-SVP'), ('LUMO-PBE0', 'LUMO PBE0/def2-SVP'), ('Gap-PBE0', 'Gap PBE0/def2-SVP')])
    ]:
    dataset = AtomsData(DS.db, load_only=properties)
    train, val, test = spk.data.partitioning.train_test_split(
        data=dataset, split_file=DS.split_v2)
    train_sets.append(MultitaskAtomsData(train, mapping))
    val_sets.append(MultitaskAtomsData(val, mapping))
    test_sets.append(MultitaskAtomsData(test, mapping))


# In[ ]:


train = ConcatAtomsData(train_sets)
val = ConcatAtomsData(val_sets)
test = ConcatAtomsData(test_sets)


# In[ ]:


train_loader = spk.AtomsLoader(train, batch_size=batch_size, shuffle=True)
val_loader = spk.AtomsLoader(val, batch_size=batch_size)


# In[ ]:


len(train), len(val)


# In[ ]:


# For average aggregation, leave divide_by_atoms=False.
if is_main(__name__, globals()):
    import numpy as np
    def measure_mean_std(n_batches, divide_by_atoms):
        loader = spk.AtomsLoader(train, batch_size=1, shuffle=True)
        valids = {p: [] for p in properties}
        for i, b in enumerate(loader):
            n_atoms = len(b['_atomic_numbers'][0])
            for p in properties:
                dt = np.array(b[p])
                validity = dt[:, 0] > 0
                if divide_by_atoms:
                    valids[p].append(dt[:, 1][validity]/n_atoms)
                else:
                    valids[p].append(dt[:, 1][validity])
            print(len(valids[p]), end='\r')
            if i==n_batches: break
        for p in properties:
            arr = np.concatenate(valids[p])
            print('{:>10} ({} items): mean={:.4f}, std={:.4f}'.format(p, len(arr), np.mean(arr), np.std(arr)))
    measure_mean_std(10000, divide_by_atoms=True)


# In[ ]:


# Precomputed statistics
meanstensor = torch.tensor([-0.3204, 0.0050, 0.3255, -0.1812, -0.0448, 0.1365])
stddevstensor = torch.tensor([0.0875, 0.0671, 0.0887, 0.1007, 0.0348, 0.0850])


# In[ ]:


means = {k: v for k, v in zip(properties, meanstensor)}
stddevs = {k: v for k, v in zip(properties, stddevstensor)}
# model build
logging.info("build model")
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
    )
for prop in properties]
model = schnetpack.AtomisticModel(representation, output_modules)


# In[ ]:


# build optimizer
optimizer = Adam(model.parameters(), lr=1e-4)


# In[ ]:


# hooks
logging.info("build trainer")
metrics = [MultitaskMetricWrapper(MeanAbsoluteError(p, p)) for p in properties]
hooks = [
    CSVHook(log_path=model_dir, metrics=metrics),
    ReduceLROnPlateauHook(
        optimizer,
        min_lr=0.5e-6,
        stop_after_min=True),
]


# In[ ]:


loss = build_gated_mse_loss(properties)


# In[ ]:


# run training
if is_main(__name__, globals()):
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
    logging.info("training")
    trainer.train(device="cuda")


# In[ ]:





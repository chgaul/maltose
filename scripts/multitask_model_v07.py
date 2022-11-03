#!/usr/bin/env python
# coding: utf-8

# # Multitask Model trained on the Unified DB
# - All datasets except HOPV are used (i.e., QM9, Alchemy, and OE62), with the unified split (split v2). Batch size reduced to 8 in order to run on the Quadro P4000 of workstation2.
# - B3LYP and PBE0 targets are available (but on disjoint subsets). The model trains (HOMO, LUMO, gap) x (B3LYP, PBE0)
# - The network has the Set2Set + molecule-level output module with six outputs. Otherwise, the architecture is the same multitask_model, multitask_model_v05.

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
model_dir = os.path.join(data.paths.model_dir, "multitask_model_v07")
if not os.path.exists(model_dir):
    print("Create model directory {}".format(model_dir))
    os.makedirs(model_dir)
else:
    print("{} exists:".format(model_dir))
    for file in os.listdir(model_dir):
        print(file)


# In[ ]:


properties = ['HOMO-B3LYP', 'LUMO-B3LYP', 'Gap-B3LYP', 'HOMO-PBE0', 'LUMO-PBE0', 'Gap-PBE0']
batch_size = 8


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


if is_main(__name__, globals()):
    import numpy as np
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


# In[ ]:


# Precomputed statistics
meanstensor = torch.tensor([-6.37, 0.24, 6.61, -6.61, -1.48, 5.12])
stddevstensor = torch.tensor([0.68, 1.24, 1.44, 0.68, 0.86, 1.05])


# In[ ]:


# model build
logging.info("build model")
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





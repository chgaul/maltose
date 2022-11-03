#!/usr/bin/env python
# coding: utf-8

# # Multitask Model trained on the Unified DB, B3LYP only
# - The four datasets QM9, Alchemy, OE62, and HOPV, with the unified split (split v2) are used.
# - Two sets of molecular energies are trained: (HOMO, LUMO, gap) x (B3LYP, PBE0)
# - The network has the Set2Set + molecule-level output module with six outputs. Otherwise, the architecture is the same as qm9_hopv_model_v13.

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
model_dir = os.path.join(
    data.paths.model_dir, "multitask_model_only_b3lyp")
if not os.path.exists(model_dir):
    print("Create model directory {}".format(model_dir))
    os.makedirs(model_dir)
else:
    print("{} exists:".format(model_dir))
    for file in os.listdir(model_dir):
        print(file)


# In[ ]:


properties = ['HOMO-B3LYP', 'LUMO-B3LYP', 'Gap-B3LYP']
batch_size = 8


# In[ ]:


logging.info("get the datasets")
train_sets, val_sets, test_sets = [], [], []
for DS, mapping in [
    (data.paths.QM9, [
        ('HOMO-B3LYP', 'homo'), ('LUMO-B3LYP', 'lumo'), ('Gap-B3LYP', 'gap')]),
    (data.paths.Alchemy, [
        ('HOMO-B3LYP', 'homo'), ('LUMO-B3LYP', 'lumo'), ('Gap-B3LYP', 'gap')]),
    (data.paths.HOPV, [
        ('HOMO-B3LYP', 'HOMO B3LYP/def2-SVP'), ('LUMO-B3LYP', 'LUMO B3LYP/def2-SVP'), ('Gap-B3LYP', 'Gap B3LYP/def2-SVP')])
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
meanstensor = torch.tensor([-6.35, 0.17, 6.52])
stddevstensor = torch.tensor([0.71, 1.28, 1.51])


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





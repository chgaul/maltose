import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ase.io
import torch
import schnetpack

from maltose.atoms import MultitaskAtomsData

import multitask_data
from multitask_data import DATASET_NAMES


b3lyp_tasks = ['HOMO-B3LYP', 'LUMO-B3LYP', 'Gap-B3LYP']
pbe0_tasks = ['HOMO-PBE0', 'LUMO-PBE0', 'Gap-PBE0']

default_data_base_dir = os.path.join('..', 'data')

def testset(
        dataset_name,
        select_tasks,
        data_base_dir=default_data_base_dir):
    (_, _, test), mapping = multitask_data.split_dataset(
        data_base_dir=data_base_dir,
        dataset_name=dataset_name,
        select_tasks=select_tasks)
    return MultitaskAtomsData(test, mapping, validity_column=False)

def get_available_properties(model):
    try: # for Set2Set output module
         return [p for om in model.output_modules for p in om.properties]
    except: # for Atomwise output module
        return [om.property for om in model.output_modules]

def evaluate_unified(
        model, dataset_name, n_points=None, seed=None,
        data_base_dir=default_data_base_dir, device='cpu'):
    datasets = {
        "qm9": testset('qm9', select_tasks=b3lyp_tasks, data_base_dir=data_base_dir),
        "alchemy": testset('alchemy', select_tasks=b3lyp_tasks, data_base_dir=data_base_dir),
        "oe62": testset('oe62', select_tasks=pbe0_tasks, data_base_dir=data_base_dir),
        "hopv": testset('hopv', select_tasks=b3lyp_tasks + pbe0_tasks, data_base_dir=data_base_dir),
    }
    dataset = datasets[dataset_name]
    batch_size = 10
    
    gen = torch.Generator()
    if seed:
        gen.manual_seed(seed)
    else:
        gen.seed()
    sampler = torch.utils.data.RandomSampler(dataset, replacement=False, generator=gen)
    batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size, drop_last=False)

    ret = {
        'tgt': {p: np.array([]) for p in dataset.available_properties},
        'est': {p: np.array([]) for p in get_available_properties(model=model)},
    }
    test_loader = schnetpack.data.loader.AtomsLoader(dataset, batch_sampler=batch_sampler)
    for i, b in enumerate(test_loader):
        for p in ret['tgt'].keys():
            ret['tgt'][p] = np.append(ret['tgt'][p], b[p])
        b = {k: v.to(device) for k, v in b.items()}
        est = model(b)
        for p in ret['est'].keys():
            ret['est'][p] = np.append(ret['est'][p], est[p].detach().to('cpu').numpy())
        if n_points is not None and (i+1) * batch_size >= n_points:
            break
    return ret

def compute_regular_data(
        model, n_points=100, seed=None,
        data_base_dir=default_data_base_dir, device='cpu'):
    return {
        dataset_name: evaluate_unified(
            model, dataset_name, n_points, seed=seed, device=device, data_base_dir=data_base_dir) for dataset_name in DATASET_NAMES
    }

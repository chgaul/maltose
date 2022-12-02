import os
import numpy as np
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
    dataset_tasks = {
        "qm9": b3lyp_tasks,
        "alchemy": b3lyp_tasks,
        "oe62": pbe0_tasks,
        "hopv": b3lyp_tasks + pbe0_tasks,
    }[dataset_name]
    dataset = testset(
        dataset_name, select_tasks=dataset_tasks, data_base_dir=data_base_dir)
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
        model, data_base_dir=default_data_base_dir,
        n_points=None, seed=None, device='cpu'):
    return {
        dataset_name: evaluate_unified(
            model, dataset_name,
            n_points, seed=seed,
            device=device,
            data_base_dir=data_base_dir) for dataset_name in DATASET_NAMES
    }

# Functions to evaluate the Kuzmich 2017 data, which is in xyz format
def predict_on_xyz(model, xyzfile, device='cpu'):
    mol = schnetpack.data.loader._collate_aseatoms([
        schnetpack.data.atoms.torchify_dict(
            schnetpack.data.atoms._convert_atoms(
                ase.io.read(xyzfile)
            )
        )
    ])
    mol = {k: v.to(device) for k, v in mol.items()}
    return model.forward(mol)

def evaluate_kuzmich(
        model, data_base_dir, seed=None, n_points=-1, device='cpu'):
    kuzmich_dir = os.path.join(data_base_dir, 'Kuzmich2017')
    df = pd.read_csv(os.path.join(kuzmich_dir, 'table1.csv'))
    mapping = {
        'DTDfBTTDPP2': 'DTDfBT(TDPP)2',
        '10_DBFI-MTT': 'DBFI-MTT',
    }
    ambiguous = ['M10']

    # Get valid files and establish canonical order
    files = {}
    for f in sorted(os.listdir(os.path.join(kuzmich_dir, 'xyz'))):
        if f.endswith('.xyz'):
            id = f[3:-13]
            if id in mapping:
                id = mapping[id]
            if id in ambiguous:
                print('Kuzmich2017: omitting ambiguous id: {}'.format(f[3:-13]))
                continue
            lb = f[3:-13]
            files[lb] = (f, id)
    # Sort by keys:
    fs = sorted(list(files.items()))

    # Shuffle order
    if seed is not None:
        random_state = np.random.RandomState(seed=seed)
        random_state.shuffle(fs)

    # Compute only on the desired random subset
    ret = {}
    for lb, (f, id) in fs[:n_points]:
        xyzfile = os.path.join(kuzmich_dir, 'xyz', f)
        pred = predict_on_xyz(model, xyzfile, device=device)
        est = {k: float(v) for k, v in pred.items()}
        tgt = {
            'LUMO-B3LYP': float(df[df['Acceptor’s Label']==id]['LUMO (eV)'])
        }
        ret[lb] = {
            'tgt': tgt,
            'est': est,
        }
    return ret

# Bring data into the regular format and add it to the
# target-estimates collection:
def add_kuzmich(
        tgt_est: dict,
        model,
        data_base_dir: str,
        seed: int = None,
        n_points: int = -1,
        device='cpu'):
    tgt_est_kuzmich = evaluate_kuzmich(
        model, data_base_dir=data_base_dir, seed=seed, n_points=n_points,
        device=device)
    # Drop keys:
    k_data = list(tgt_est_kuzmich.values())
    ret = {
            'tgt': {p: np.array([]) for p in k_data[0]['tgt'].keys()},
            'est': {p: np.array([]) for p in k_data[0]['est'].keys()},
        }
    for kd in k_data:
        for k in ret['tgt'].keys():
            ret['tgt'][k] = np.append(ret['tgt'][k], [kd['tgt'][k]])
        for k in ret['est'].keys():
            ret['est'][k] = np.append(ret['est'][k], [kd['est'][k]])
    tgt_est['Kuzmich2017'] = ret

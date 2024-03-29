import os.path
from typing import Optional, List
import schnetpack as spk
from schnetpack.data.atoms import AtomsData, ConcatAtomsData
from maltose.atoms import MultitaskAtomsData

hartree = 27.2114 # eV

TARGET_MAPPINGS = {
    'qm9': {
        'HOMO-B3LYP': 'homo',
        'LUMO-B3LYP': 'lumo',
        'Gap-B3LYP': 'gap',
        'HOMO-PBE0': None,
        'LUMO-PBE0': None,
        'Gap-PBE0': None,
    },
    'alchemy': {
        'HOMO-B3LYP': 'homo',
        'LUMO-B3LYP': 'lumo',
        'Gap-B3LYP': 'gap',
        'HOMO-PBE0': None,
        'LUMO-PBE0': None,
        'Gap-PBE0': None,
    },
    'oe62': {
        'HOMO-B3LYP': None,
        'LUMO-B3LYP': None,
        'Gap-B3LYP': None,
        'HOMO-PBE0': 'homo PBE0_vacuum',
        'LUMO-PBE0': 'lumo PBE0_vacuum',
        'Gap-PBE0': 'gap PBE0_vacuum',
    },
    'hopv': {
        'HOMO-B3LYP': 'HOMO B3LYP/def2-SVP',
        'LUMO-B3LYP': 'LUMO B3LYP/def2-SVP',
        'Gap-B3LYP': 'Gap B3LYP/def2-SVP',
        'HOMO-PBE0': 'HOMO PBE0/def2-SVP',
        'LUMO-PBE0': 'LUMO PBE0/def2-SVP',
        'Gap-PBE0': 'Gap PBE0/def2-SVP',
    },
}

DATASET_NAMES = list(TARGET_MAPPINGS.keys())

def split_dataset(data_base_dir, dataset_name, select_tasks=None):

    dataset_file = os.path.join(data_base_dir, dataset_name, 'data.db')
    if not os.path.exists(dataset_file):
        print(dataset_file , 'does not exist!')
        print('Please run the respective scipt in scripts/primary_data/ first,')
        print('and make sure the --data-base-dir is specified correctly.')
    dataset = AtomsData(dataset_file)

    mapping = TARGET_MAPPINGS[dataset_name]
    if select_tasks is not None:
        mapping = {a: b for a, b in mapping.items() if a in select_tasks}

    split_file = os.path.join(data_base_dir, dataset_name, 'split_v2.npz')
    if not os.path.exists(split_file):
        print(split_file, 'does not exist!')
        print('Please run scripts/primary_data/unified_split.py first!')

    return (
        spk.data.partitioning.train_test_split(
            data=dataset, split_file=split_file),
        mapping
    )


def join_multitask_data(
        data_base_dir: str,
        select_datasets: Optional[List[str]] = None,
        select_tasks: Optional[List[str]] = None):
    train_sets, val_sets, test_sets = [], [], []
    for dataset_name in TARGET_MAPPINGS.keys():
        if select_datasets is not None and not dataset_name in select_datasets:
            print('Skipping dataset', dataset_name)
            continue
        (train, val, test), mapping = split_dataset(
            data_base_dir, dataset_name, select_tasks)
        train_sets.append(MultitaskAtomsData(train, mapping))
        val_sets.append(MultitaskAtomsData(val, mapping))
        test_sets.append(MultitaskAtomsData(test, mapping))

    return [ConcatAtomsData(sets) for sets in [train_sets, val_sets, test_sets]]
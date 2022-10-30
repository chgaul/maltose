"""
Download and prepare the data from the Alchemy contest.

Prepare an ase database suitable for training a model with schnetpack. Save the
attributes 'homo', 'lumo', and 'gap'.
"""
import os
import urllib.request
import zipfile
import csv
import json
import numpy as np
import ase.io
from schnetpack import AtomsData
import rdkit.Chem


scratch_dir = os.path.join('scratch', 'Alchemy')

zip_path = os.path.join(scratch_dir, 'alchemy-v20191129.zip')
if not os.path.exists(zip_path):
    zip_url = 'https://alchemy.tencent.com/data/alchemy-v20191129.zip'
    print('Downloading {} to {} ...'.format(zip_url, zip_path), end='')
    os.makedirs(scratch_dir, exist_ok=True)
    urllib.request.urlretrieve(zip_url, zip_path)
    print(' Done!')

csv_path = os.path.join(scratch_dir, 'Alchemy-v20191129', 'final_version.csv')
if not os.path.exists(csv_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(scratch_dir)
else:
    print("The file {} exists. Assume that all other files are there and \
up-to-date, too. Skip unzipping of {}".format(csv_path, zip_path))

tgt_dir = './data/alchemy/'
os.makedirs(tgt_dir, exist_ok=True)
db_file    = os.path.join(tgt_dir, 'data.db')
split_file = os.path.join(tgt_dir, 'split.npz')
ids_file   = os.path.join(tgt_dir, 'original-ids.json')
inchi_file = os.path.join(tgt_dir, 'inchis.json')
output_files = [db_file, split_file, ids_file, inchi_file]

if os.path.exists(db_file):
    ret = input("Database file exists. Should I delete all output files [{}] \
and start over? (y/n)".format(', '.join(output_files)))
    if ret=='y':
        for file in output_files:
            try:
                os.remove(file)
            except FileNotFoundError:
                pass

if not os.path.exists(db_file):
    print("Create a new dataset. Will re-create all other output files, too.")
    new_dataset = AtomsData(
        db_file,
        available_properties=['homo', 'lumo', 'gap', 'alchemy_id'])
    fill_dataset = True
else:
    print("Open existing dataset. Will leave other files untouched, too.")
    new_dataset = AtomsData(db_file)
    with open(inchi_file, 'r', encoding='utf-8') as f:
        inchis = json.load(f)
    fill_dataset = False

with open(csv_path) as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    properties = {}
    keys = None
    for row in reader:
        if not keys:
            keys = row[1:]
            continue
        ID, *vals = row
        properties[int(ID)] = {k.replace('\n', ' '): float(v) for k, v in zip(keys, vals)}

Ha = 27.2114 # 1Ha = 27.2eV
for k, d in properties.items():
    d['homo'] = Ha * d['HOMO (Ha, energy of HOMO)']
    d['lumo'] = Ha * d['LUMO (Ha, energy of LUMO)']
    d['gap'] = d['lumo'] - d['homo'] 

from pprint import pprint
for x in properties.items():
    pprint(x[1])
    break


print('Filling dataset...')
if fill_dataset:
    inchis = {}
    for i, (k, props) in enumerate(properties.items()):
        props['alchemy_id'] = k
        for subdir in ['atom_9', 'atom_10', 'atom_11', 'atom_12']:
            path = os.path.join(
                scratch_dir, 'Alchemy-v20191129', subdir, '{}.sdf'.format(k))
            if os.path.exists(path):
                # add to dataset:
                new_dataset.add_system(
                    atoms=ase.io.read(path),
                    properties=props)
                # determine inchi and store it
                suppl = rdkit.Chem.rdmolfiles.SDMolSupplier(path)
                assert len(suppl) == 1, "Expected 1 structure in {} but got {}!".format(
                    path, len(suppl))
                rmol = next(suppl)
                inchi = rdkit.Chem.rdinchi.MolToInchi(rmol)[0]
                inchis[str(i)] = inchi
                break
        else: # Not found in any subdir:
            raise FileNotFoundError('{} not found!'.format(k))
        print('Progress: {:.1f}%'.format(100*i/len(properties)), end='\r')
    with open(inchi_file, 'w', encoding='utf-8') as f:
        json.dump(inchis, f, indent=0)
else:
    print('Do not do anything. Just use the existing dataset and inchi list.')
print('Dataset filled!')


print("For future reference, save the Alchemy IDs")
if os.path.exists(ids_file):
    with open(ids_file, 'r', encoding='utf-8') as f:
        alchemy_ids = json.load(f)
else:
    alchemy_ids = {}
    L = len(new_dataset)
    for i in range(L):
        _, d = new_dataset.get_properties(i)
        alchemy_ids[str(i)] = int(d['alchemy_id'])
        print('Progress: {:.1f}%'.format(100*i/len(new_dataset)), end='\r')
    with open(ids_file, 'w', encoding='utf-8') as f:
        json.dump(alchemy_ids, f, indent=0)


print('Get the train/valid/test split form the Alchemy Contest and convert to local idx')
def gdb_ids_from_zipfile(infile, outfile):
    zip = zipfile.ZipFile(infile)
    lst = [n.split("/") for n in zip.namelist()]
    lst = [n[-1] for n in lst if len(n)==4]
    lst = [int(n[:-4]) for n in lst if n.endswith('.sdf')]
    with open(outfile, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for n in lst:
            writer.writerow([n])

for split in ['test', 'dev', 'valid']:
    idfile = os.path.join(scratch_dir, "{}_gdb_idx.txt".format(split))
    if not os.path.exists(idfile):
        zip_path = os.path.join(scratch_dir, "{}_v20190730.zip".format(split))
        if not os.path.exists(zip_path):
            zip_url = "https://alchemy.tencent.com/data/{}_v20190730.zip".format(split)
            print('Downloading {} to {} ...'.format(zip_url, zip_path), end='')
            os.makedirs(scratch_dir, exist_ok=True)
            urllib.request.urlretrieve(zip_url, zip_path)
        gdb_ids_from_zipfile(zip_path, idfile)
        print('Created {}.'.format(idfile))
    else:
        print('{} is present'.format(idfile))

def get_idx_list(alchemy_id_file):
    index_from_alchemy_id = {v: k for k, v in alchemy_ids.items()}
    ids = []
    with open(alchemy_id_file) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            ids.append(index_from_alchemy_id[int(row[0])])
    return(sorted(ids))

if not os.path.exists(split_file):
    np.savez(file=split_file,
         train_idx=get_idx_list(os.path.join(scratch_dir, "dev_gdb_idx.txt")),
         val_idx=get_idx_list(os.path.join(scratch_dir, "valid_gdb_idx.txt")),
         test_idx=get_idx_list(os.path.join(scratch_dir, "test_gdb_idx.txt")))


print('Statistics on atom counts:')
na_list = []
for i in range(len(new_dataset)):
    amol = new_dataset[i]
    na_list.append(len(amol['_atomic_numbers']))
    print(i, end='\r')
print('Number of structures: ', len(na_list))
print('Median number of atoms: ', np.median(na_list))
print('Max number of atoms: ', np.max(na_list))

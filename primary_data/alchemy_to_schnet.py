#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os.path
import csv
import json
import numpy as np
import ase.io
from schnetpack import AtomsData


# In[ ]:


import rdkit


# # Import the Alchemy Dataset from zipped xyzs and the pdf

# In[ ]:


get_ipython().system('pwd')


# In[ ]:


# Download the molecular data (zipped xyz files)
if not os.path.exists("Alchemy/alchemy-v20191129.zip"):
    get_ipython().system('mkdir -p Alchemy')
    get_ipython().system('(cd Alchemy && wget https://alchemy.tencent.com/data/alchemy-v20191129.zip)')


# In[ ]:


get_ipython().system('(cd Alchemy && unzip "alchemy-v20191129.zip")')


# In[ ]:


basedir = '../schnetpack_exps/data/'
basename = 'alchemy'
db_file    = os.path.join(basedir, basename + '.db')
split_file = os.path.join(basedir, basename + '-split.npz')
ids_file   = os.path.join(basedir, basename + '-original-ids.json')
inchi_file = os.path.join(basedir, basename + '-inchis.json')
output_files = [db_file, split_file, ids_file, inchi_file]


# In[ ]:


if os.path.exists(db_file):
    ret = input("Should I delete all output files '{}'? (y/n)".format(
        ', '.join(output_files)))
    if ret=='y':
        for file in output_files:
            os.remove(file)
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


# In[ ]:


with open('Alchemy/Alchemy-v20191129/final_version.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    properties = {}
    keys = None
    for row in reader:
        if not keys:
            keys = row[1:]
            continue
        ID, *vals = row
        properties[int(ID)] = {k.replace('\n', ' '): float(v) for k, v in zip(keys, vals)}


# In[ ]:


Ha = 27.2114 # 1Ha = 27.2eV
for k, d in properties.items():
    d['homo'] = Ha * d['HOMO (Ha, energy of HOMO)']
    d['lumo'] = Ha * d['LUMO (Ha, energy of LUMO)']
    d['gap'] = d['lumo'] - d['homo'] 


# In[ ]:


from pprint import pprint
for x in properties.items():
    pprint(x[1])
    break


# In[ ]:


if fill_dataset:
    inchis = {}
    for i, (k, p) in enumerate(properties.items()):
        props = dict(list(p.items()) + [('alchemy_id', k)])
        for subdir in ['atom_9', 'atom_10', 'atom_11', 'atom_12']:
            path = os.path.join(
                    'Alchemy', 'Alchemy-v20191129',
                    subdir, '{}.sdf'.format(k))
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
            raise '{} not found!'.format(k)
        print('Progress: {:.1f}%'.format(100*i/len(properties)), end='\r')
    with open(inchi_file, 'w', encoding='utf-8') as f:
        json.dump(inchis, f, indent=0)
else:
    print('Do not do anything. Just use the existing dataset and inchi list.')


# ### Map local index to GDBMedChem id

# In[ ]:


if not os.path.exists(ids_file):
    alchemy_ids = {}
    L = len(new_dataset)
    for i in range(L):
        d = new_dataset[i]
        alchemy_ids[str(int(d['_idx']))] = int(str(int(d['alchemy_id'])))
        print('Progress: {:.1f}%'.format(100*i/len(new_dataset)), end='\r')
    with open(ids_file, 'w', encoding='utf-8') as f:
        json.dump(alchemy_ids, f, indent=0)
else:
    with open(ids_file, 'r', encoding='utf-8') as f:
        alchemy_ids = json.load(f)


# ### Get the train/valid/test split form the Alchemy Contest and convert to local idx

# In[ ]:


# Generate test files with the GDB ids of the train/test/validation split of the Alchemy contest
from zipfile import ZipFile
def gdb_ids_from_zipfile(infile, outfile):
    zip = ZipFile(infile)
    lst = [n.split("/") for n in zip.namelist()]
    lst = [n[-1] for n in lst if len(n)==4]
    lst = [int(n[:-4]) for n in lst if n.endswith('.sdf')]
    with open(outfile, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for n in lst:
            writer.writerow([n])

for split in ['test', 'dev', 'valid']:
    zipfile = "Alchemy/{}_v20190730.zip".format(split)
    idfile = "Alchemy/{}_gdb_idx.txt".format(split)
    if not os.path.exists(idfile):
        if not os.path.exists(zipfile):
            print("Please download:")
            print("wget https://alchemy.tencent.com/data/{}".format(zipfile))
        gdb_ids_from_zipfile(zipfile, idfile)


# In[ ]:


def get_idx_list(alchemy_id_file):
    index_from_alchemy_id = {v: k for k, v in alchemy_ids.items()}
    ids = []
    with open(alchemy_id_file) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        properties = {}
        for row in reader:
            ids.append(index_from_alchemy_id[int(row[0])])
    return(sorted(ids))


# In[ ]:


if not os.path.exists(split_file):
    np.savez(file=split_file,
         train_idx=get_idx_list("Alchemy/dev_gdb_idx.txt"),
         val_idx=get_idx_list("Alchemy/valid_gdb_idx.txt"),
         test_idx=get_idx_list("Alchemy/test_gdb_idx.txt"))


# ### Statistics on atom counts

# In[ ]:


na_list = []
for i in range(len(new_dataset)):
    amol = new_dataset[i]
    na_list.append(len(amol['_atomic_numbers']))
    print(i, end='\r')


# In[114]:


len(na_list), np.median(na_list), np.max(na_list)


# In[ ]:





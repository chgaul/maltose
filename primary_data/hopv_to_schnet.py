#!/usr/bin/env python
# coding: utf-8

# # Import the data from the Harvard Organic Photovoltaic Dataset to SchNetPack

# - Read the HOPV data into an ase db
# - provide a train/valid/test split
# - each sub-db contains multiple conformers of molecules with ~100 different SMILES.

# In[ ]:


from hopv import data as hopv_data # This will take some time


# In[ ]:


# Some statistics: count the atoms per molecule
hopv_smiles = set(hopv_data['SMILES'])
from rdkit import Chem
for i, s in enumerate(hopv_smiles):
    print(Chem.MolFromSmiles(s).GetNumAtoms(), s)
    if i>=3:
        break


# In[ ]:


XCs = ['B3LYP', 'BP86', 'PBE0', 'M06-2X']


# In[14]:


available_properties = ['{q} {xc}/def2-SVP'.format(q=q, xc=xc) for xc in XCs for q in ['HOMO', 'LUMO', 'Gap']]


# In[ ]:


# Convert hops data into an ase db plus dictionary mapping id to smiles 
from ase import Atoms
from schnetpack.data.atoms import AtomsData
import numpy as np
import os.path
import tempfile
import json
dirname = "../schnetpack_exps/data/hopv"
assert os.path.exists(dirname)
dbname = os.path.join(dirname, "main.db")
assert not os.path.exists(dbname)
ad = AtomsData(
    dbname,
    available_properties=available_properties)
smiles_dict = {}
inchi_dict = {}
for i, (smiles, inchi, dft, xyz) in enumerate(zip(
        hopv_data['SMILES'], hopv_data['InChI'], hopv_data['DFTPROPS'], hopv_data['XYZ'])):
    smiles_dict[i] = smiles
    inchi_dict[i] = inchi
    Ha = 27.2114 # eV
    properties = {
        '{q} {xc}/def2-SVP'.format(q=q, xc=xc): \
            dft['QChem {xc}/def2-SVP DFT'.format(xc=xc)][q]*Ha \
                for q in ['HOMO', 'LUMO', 'Gap'] for xc in XCs}
    xyzlist = xyz.values.tolist()
    atoms = [x[0] for x in xyzlist]
    xyz_coordinates = [x[1:] for x in xyzlist]
    atoms = Atoms(''.join(atoms), xyz_coordinates)
    ad.add_system(atoms=atoms, properties=properties)
    if i>=30 and False:
        break
with open(os.path.join(dirname, "smiles.json"), 'w') as json_file:
    json.dump(smiles_dict, json_file, sort_keys=True, indent=0)
with open(os.path.join(dirname, "inchis.json"), 'w') as json_file:
    json.dump(inchi_dict, json_file, sort_keys=True, indent=0)


# In[ ]:


assert len(ad) == len(smiles_dict)
last = len(ad) - 1
{smiles_dict[last]: ad.get_properties(last)}


# In[ ]:


print('HOPV contains {nconf} entries (i.e., conformers) of {ndist} distinct compounds'.format(
    nconf=len(hopv_data), ndist=len(set(hopv_data['InChI']))))


# The train/validation/test split should be compound disjoint.

# In[ ]:


smiles_set = set(smiles_dict.values())
len(smiles_set)


# In[ ]:


inchi_set = set(inchi_dict.values())
len(inchi_set)


# In[ ]:


for smiles in smiles_set:
    idxs = [k for k, v in smiles_dict.items() if v==smiles]
    inchis = {inchi_dict[idx] for idx in idxs}
    if len(inchis) > 1:
        print('{n} different InChIs for SMILES {smiles}:'.format(n=len(inchis), smiles=smiles))
        for inchi in inchis:
            print(inchi)
        print()


# In[ ]:


for inchi in inchi_set:
    idxs = [k for k, v in inchi_dict.items() if v==inchi]
    smiless = {smiles_dict[idx] for idx in idxs}
    if len(smiless) > 1:
        print('{n} different SMILESs for InChI {inchi}:'.format(n=len(smiless), inchi=inchi))
        for smiles in smiless:
            print(smiles)
        print()


# To be on the safe side, we make the train/test/validation split in terms of InChIs.

# In[ ]:


import random
random.seed(2021)
trainInChIs = random.sample(list(inchi_set), 200)
complementInChIs = {i for i in inchi_set if i not in trainInChIs}
testInChIs = random.sample(list(complementInChIs), 75)
validInChIs = {i for i in complementInChIs if i not in testInChIs}
[len(s) for s in [trainInChIs, testInChIs, validInChIs]]


# In[ ]:


train_idx = [k for k, v in inchi_dict.items() if v in trainInChIs]
val_idx = [k for k, v in inchi_dict.items() if v in validInChIs]
test_idx = [k for k, v in inchi_dict.items() if v in testInChIs]
(len(train_idx), len(val_idx), len(test_idx))
split_file = os.path.join(dirname, "split.npz")
assert not os.path.exists(split_file)
np.savez(file=split_file,
         train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)


# In[ ]:


from schnetpack.data.partitioning import train_test_split
train, val, test = train_test_split(
        data=ad, split_file=split_file)


# In[ ]:





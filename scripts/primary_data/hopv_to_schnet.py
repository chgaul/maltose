"""
Import the data from the Harvard Organic Photovoltaic Dataset to SchNetPack

- Read the HOPV data into an ase db
- Also dump the mapping of IDs to InChIs and SMILESs
"""
import os
import urllib.request
from maltose.primary_data.hopv import read_dataset
from ase import Atoms
from schnetpack.data.atoms import AtomsData
import json

scratch_dir = os.path.join('scratch', 'HOPV')

data_path = os.path.join(scratch_dir, 'HOPV_15_revised_2.data')
if not os.path.exists(data_path):
    data_url = 'https://figshare.com/ndownloader/files/4513735'
    print('Downloading {} to {} ...'.format(data_url, data_path), end='')
    os.makedirs(scratch_dir, exist_ok=True)
    urllib.request.urlretrieve(data_url, data_path)
    print(' Done!')

print('Reading the HOPV dataset ...', end='')
hopv_data = read_dataset(data_path)
print('Done!')

hopv_smiles = set(hopv_data['SMILES'])

XCs = ['B3LYP', 'BP86', 'PBE0', 'M06-2X']

available_properties = [
    '{q} {xc}/def2-SVP'.format(
        q=q, xc=xc) for xc in XCs for q in ['HOMO', 'LUMO', 'Gap']
]


# Convert HOPS data into an ase db plus dictionary mapping id to smiles
dirname = os.path.join("data", "hopv")
if not os.path.exists(dirname):
    os.makedirs(dirname)

dbname = os.path.join(dirname, "main.db")
assert not os.path.exists(dbname), "Target db {} exits. Please delete!".format(
    dbname)
print('Filling AtomsData ...', end='')
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
print('AtomsData completed!')
with open(os.path.join(dirname, "smiles.json"), 'w') as json_file:
    json.dump(smiles_dict, json_file, sort_keys=True, indent=0)
with open(os.path.join(dirname, "inchis.json"), 'w') as json_file:
    json.dump(inchi_dict, json_file, sort_keys=True, indent=0)

assert len(ad) == len(smiles_dict)
last = len(ad) - 1

print('HOPV contains {nconf} entries (i.e., conformers) of {ndist} distinct compounds'.format(
    nconf=len(hopv_data), ndist=len(set(hopv_data['InChI']))))

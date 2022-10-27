import os
import urllib.request
import zipfile
import shutil

scratch_dir = os.path.join('scratch', 'Kuzmich2017')

# The following dataset contains structures of 80 non-fullerene acceptors from
# the paper Kuzmich et al. (2017), DOI: 10.1039/C6EE03654F.
# The geometries are optimzed at the DFT/B3LYP/6-31G* level.
# Some properties (LUMO, LUMO+1) are provided in the paper
# (Table 1 of Kuzmich_2017_TrendsNFA), use the "Acceptor's label" to join this
# table with the zipped xyz files.
# Further, reported PCEs of acceptor donor pairs are given in the table.
# Download the supplementary material from the journal webpage:

zip_path = os.path.join(scratch_dir, 'c6ee03654f1.zip')
if not os.path.exists(zip_path):
    zip_url = "https://www.rsc.org/suppdata/c6/ee/c6ee03654f/c6ee03654f1.zip"
    print('Downloading {} to {} ...'.format(zip_url, zip_path), end='')
    os.makedirs(scratch_dir, exist_ok=True)
    urllib.request.urlretrieve(zip_url, zip_path)
    print(' Done!')

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(scratch_dir)

"""
- Read the (non-standard) xyz files from the folder Opt_Struct/.
- Replace the atomic charge in the first column with the chemical symbol.
- Write the corrected files to the folder fixed_xyz/.
"""
in_dir = os.path.join(scratch_dir, 'Opt_Structs')
out_dir = os.path.join('data', 'Kuzmich2017', 'xyz')
os.makedirs(out_dir, exist_ok=True)
for filename in os.listdir(in_dir):
    if filename.endswith('.xyz'):
        with open(os.path.join(in_dir, filename)) as in_file:
            with open(os.path.join(out_dir, filename), "w") as out_file:
                for i, l in enumerate(in_file.readlines()):
                    if i<2:
                        out_file.write(l)
                    else:
                        vals = l.split()
                        vals[0] = {
                            '1': 'H',
                            '5': 'B',
                            '6': 'C',
                            '7': 'N',
                            '8': 'O',
                            '9': 'F',
                            '14': 'Si',
                            '16': 'S',
                            '17': 'Cl',
                            '30': 'Zn',
                            '34': 'Se',
                        }[vals[0]]
                        out_file.write('\t'.join(vals) + '\n')

# Copy the annotations table to the data folder
shutil.copyfile(
    os.path.join('Kuzmich2017', 'table1.csv'),
    os.path.join('data', 'Kuzmich2017', 'table1.csv'))
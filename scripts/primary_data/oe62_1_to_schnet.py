"""
Prepare the data from the OE62 papercontest.

Prepare an ase database suitable for training a model with schnetpack. Save the
attributes 'homo', 'lumo', and 'gap'.
"""
import os
import pandas as pd
import json
from schnetpack import AtomsData
from OE62.helpers import get_level, xyz2ase


SRCDIR = os.path.join('scratch', 'OE62')
TGTDIR = os.path.join('data', 'oe62')

for file in [
        'README',
        # 'atomic_energies.ods',
        'df_62k.json',
        # 'df_31k.json', 'df_5k.json',
        'SHA512sums',
    ]:
    if not os.path.exists(os.path.join(SRCDIR, file)):
        print(file + ' does not exist in ' + SRCDIR + "!")
        print('Please run oe62_0_download.sh first!')
        exit(1)


df_62k = pd.read_json(os.path.join(SRCDIR, 'df_62k.json'), orient='split')


df_inchis = df_62k['inchi']
# Remove trailing newline:
df_inchis = df_inchis.apply(lambda istr: istr.strip('\n'))


os.makedirs(TGTDIR, exist_ok=True)
with open(os.path.join(TGTDIR, 'inchis.json'), 'w', encoding='utf-8') as f:
    json.dump(df_inchis.to_dict(), f, indent=0, sort_keys=True)


db_file    = os.path.join(TGTDIR, 'data_v2.db')
ids_file   = os.path.join(TGTDIR, 'original-ids.json')
inchi_file = os.path.join(TGTDIR, 'inchis.json')


subsets = ['PBE+vdW_vacuum', 'PBE0_vacuum']
properties = [' '.join([q, l]) for l in subsets for q in ['homo', 'lumo', 'gap']] + ['oe62_id']

output_files = [db_file, ids_file, inchi_file]
if os.path.exists(db_file):
    ret = input("Should I delete all output files '{}'? (y/n)".format(
        ', '.join(output_files)))
    if ret=='y':
        for file in output_files:
            try:
                os.remove(file)
            except OSError:
                pass
if not os.path.exists(db_file):
    print("Create a new dataset. Will (re-)create all other output files, too.")
    new_dataset = AtomsData(
        db_file,
        available_properties=properties)
    fill_dataset = True
else:
    print("Open existing dataset. Will leave other files untouched, too.")
    new_dataset = AtomsData(db_file)
    with open(inchi_file, 'r', encoding='utf-8') as f:
        inchis = json.load(f)
    fill_dataset = False


inchis = {}
original_ids = {}
# Iterate in index order, i.e., the same order as in the
# previously exported inchi file.
# Note that pandas.DataFrame.iterrows() does not guarantee
# the indexing order (it rather provides the index value,
# which may behave non-monotonically). Unfortunately, the
# AtomsData can only insert the data in natural order,
# which the resulting index increasing by one each time.
for i in range(len(df_62k)):
    row = df_62k.loc[i]
    inchi = row.inchi.strip('\n')
    inchis[str(i)] = inchi
    props = {
        'oe62_id': i,
    }
    for s in subsets:
        homo = get_level(row, level_type='HOMO', subset=s)
        lumo = get_level(row, level_type='LUMO', subset=s)
        props.update({
            'homo ' + s: homo,
            'lumo ' + s: lumo,
            'gap ' + s: lumo - homo,
        })
    new_dataset.add_system(
        atoms=xyz2ase(row.xyz_pbe_relaxed),
        properties=props)
    original_ids[str(i)] = row.refcode_csd
    print('Progress: {:.1f}%'.format(100*i/len(df_62k)), end='\r')
with open(inchi_file, 'w', encoding='utf-8') as f:
    json.dump(inchis, f, indent=0)
with open(ids_file, 'w', encoding='utf-8') as f:
    json.dump(original_ids, f, indent=0)
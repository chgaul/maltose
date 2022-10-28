"""
Prepare the data from the OE62 paper.

Prepare an ase database suitable for training a model with schnetpack. Save the
attributes 'homo', 'lumo', and 'gap'.
"""
import os
import subprocess
import pandas as pd
import json
from schnetpack import AtomsData
from maltose.primary_data.OE62.helpers import get_level, xyz2ase


download_dir = os.path.join('scratch', 'OE62')
target_dir = os.path.join('data', 'oe62')

for file in [
        'README',
        'df_62k.json',
        # 'atomic_energies.ods',
        # 'df_31k.json', 'df_5k.json',
        # 'SHA512sums',
    ]:
    download_path = os.path.join(download_dir, file)
    if not os.path.exists(download_path):
        print('Downloading {} ...'.format(file), end='')
        os.makedirs(download_dir, exist_ok=True)
        try:
            subprocess.call([
                'rsync', '-az',
                'rsync://m1507656@dataserv.ub.tum.de/m1507656/{}'.format(file),
                download_dir], env={'RSYNC_PASSWORD': 'm1507656'})
        except Exception as e:
            print('\n\nSomething went wrong with the rsync download!')
            print('Try manually to download the file {} data.'.format(file))
            print('See https://doi.org/10.14459/2019mp1507656.\n')
            raise(e)
        print(' Done!')

df_62k = pd.read_json(os.path.join(download_dir, 'df_62k.json'), orient='split')


df_inchis = df_62k['inchi']
# Remove trailing newline:
df_inchis = df_inchis.apply(lambda istr: istr.strip('\n'))


os.makedirs(target_dir, exist_ok=True)
with open(os.path.join(target_dir, 'inchis.json'), 'w', encoding='utf-8') as f:
    json.dump(df_inchis.to_dict(), f, indent=0, sort_keys=True)


db_file    = os.path.join(target_dir, 'data_v2.db')
ids_file   = os.path.join(target_dir, 'original-ids.json')
inchi_file = os.path.join(target_dir, 'inchis.json')


subsets = ['PBE+vdW_vacuum', 'PBE0_vacuum']
properties = [' '.join([q, l]) for l in subsets for q in ['homo', 'lumo', 'gap']] + ['oe62_id']

output_files = [db_file, ids_file, inchi_file]
if os.path.exists(db_file):
    ret = input("Database file {} exists. Delete all output files ({})\
 and start over? (y/n)".format(db_file, ', '.join(output_files)))
    if ret == 'y':
        for file in output_files:
            try:
                os.remove(file)
            except OSError:
                pass
    else:
        print('Exiting.')
        exit(0)
assert not os.path.exists(db_file)

print("Create a new dataset. Will (re-)create all other output files, too.")
new_dataset = AtomsData(
    db_file,
    available_properties=properties)

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

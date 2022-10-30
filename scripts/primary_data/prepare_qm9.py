import os
import json
import ase
import tempfile
import rdkit
from maltose.conversions import xyz2rdkit
from schnetpack.datasets import QM9

# How to handle paths to data properly?
base_path = 'data/qm9'
qm9_path = os.path.join(base_path, 'data.db')

# Assure that the QM9 database is downloaded via schnetpack
QM9(qm9_path)


# Convert the ase atoms of QM9 to rdkit molecules and generate InChIs
db = ase.db.connect(qm9_path)
inchis = {}
success = 0
failed = []
for idx in range(0, len(db)):
    ase_idx = idx + 1
    for row in db.select(id=int(ase_idx)):
        atoms = row.toatoms()
        
        tmpdir = tempfile.TemporaryDirectory()
        xyzfile = os.path.join(tmpdir.name, 'tmp.xyz')
        ase.io.write(xyzfile, atoms)
        try:
            try:
                rmol = xyz2rdkit(xyzfile, use_huckel=False)
                # This fails for some 18 molecules.
            except:
                print(idx, ': try with use_huckel=True...', end='')
                rmol = xyz2rdkit(xyzfile, use_huckel=True)
                # This fails only for 6 molecules out of the above 18 molecules,
                print('success!')
            inchi = rdkit.Chem.rdinchi.MolToInchi(rmol)[0]
            inchis[idx] = inchi
            success += 1
        except:
            failed.append(idx)
            print("Failed on id {}.".format(idx))
        if idx % 100 == 0:
            print("progress: {:.1f}%".format(100*idx/len(db)), end="\r")
        if success < 0:
            break
    else:
        # Continue if the inner loop wasn't broken.
        continue
    # Inner loop was broken, break the outer.
    break
print('Successful: {}; Failed: {} ({}%)'.format(
    success, len(failed), len(failed)/(success + len(failed))))

# Dump the dictionary idx -> InChI
inchi_file = os.path.join(base_path, 'inchis.json')
with open(inchi_file, 'w', encoding='utf-8') as f:
    json.dump(inchis, f, indent=0)

# Dump the list of idx where no InChI could be obtained
with open(os.path.join(base_path, 'inchi-failed.json'), 'w', encoding='utf-8') as f:
    json.dump(failed, f, indent=0)

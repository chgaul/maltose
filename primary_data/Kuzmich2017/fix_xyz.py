"""
- Read the (non-standard) xyz files from the folder Opt_Struct/.
- Replace the atomic charge in the first column with the chemical symbol.
- Write the corrected files to the folder fixed_xyz/.
"""
import os
in_dir = 'Opt_Structs'
out_dir = 'fixed_xyz'
os.makedirs(out_dir, exist_ok=True)
for filename in os.listdir('Opt_Structs'):
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

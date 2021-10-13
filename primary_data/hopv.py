import pandas as pd

def read_expprop(fp):
    """read quantum chemistry properties from file.
    Returns a dictionary."""
    quantities = [
        'DOI', 'InChIKEY', 'Construction', 'Architecture', 'Complement',
        'HOMO', 'LUMO', 'Electochemical gap', 'Optical gap', 'PCE', 'VOC',
        'JSC', 'Fill factor']
    line = fp.readline().rstrip()
    tokens = line.split(',')
    tokens_s = tokens[:6]
    tokens_f = [float(x) for x in tokens[6:]]
    return {
        k: v for k, v in zip(quantities, tokens_s + tokens_f)}

def read_qchem(fp):
    """read quantum chemistry properties from file.
    Returns a dictionary."""
    quantities = ['HOMO', 'LUMO', 'Gap', 'Scharber PCE', 'Scharber VOC', 'Scharber JSC']
    qchem = {}
    x = fp.tell()
    while True:
        line = fp.readline().rstrip()
        try:
            tokens = line.split(',')
            type = tokens[0]
            props = tokens[1:]
            if type.startswith('QChem'):
                qchem[type] = {
                    k: float(v) for k, v in zip(quantities, props)}
                x = fp.tell()
            else:
                raise (ValueError(line))
        except(ValueError):
            fp.seek(x)
            break
    return qchem

def read_xyz(fp):
    """Read xyz coordinates from file. Returns a pandas dataframe
    with columns A, X, Y, Z"""
    line = fp.readline()
    line = fp.readline()
    try:
        nb_atoms = int(line)
    except():
        raise(ValueError("the first line should contain the number of atoms (int). Got: " + line))
    xyz = pd.DataFrame(columns=['A', 'X', 'Y', 'Z'], index=range(nb_atoms))
    for i in range(nb_atoms):
        line = fp.readline().split(' ')
        xyz.iloc[i] = line[0], float(line[1]), float(line[2]), float(line[3])
    return xyz

columns=['SMILES', 'InChI', 'EXPPROPS', 'CONFORMER', 'DFTPROPS', 'XYZ']

data = pd.DataFrame(columns=columns)
i = 0
with open("Lopez_2016_HarvardOrganicPhotovoltaics/HOPV_15_revised_2.data") as fp:
    while True:
        smiles = fp.readline().rstrip()
        if not smiles:
            break
        inchi = fp.readline().rstrip()
        expdata = read_expprop(fp)
        smiles2 = fp.readline().rstrip()
        try:
            nb_conformers = int(fp.readline())
        except(ValueError):
            print(line)
            raise
        for conformer in range(nb_conformers):
            xyz = read_xyz(fp)
            qchem = read_qchem(fp)
            row = pd.DataFrame([[
                smiles, inchi, expdata, conformer + 1, qchem, xyz]], index=[i], columns=columns)
            data = pd.concat([data, row])
            i = i + 1

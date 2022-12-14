import ase.io
import xyz2mol
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolfiles import MolToXYZFile
import tempfile
import os.path

def xyz2rdkit(
        src_file: str,
        allow_charged_fragments=True,
        use_huckel=False):
    atoms, charge, xyz_coordinates = xyz2mol.read_xyz_file(src_file)
    mols = xyz2mol.xyz2mol(atoms, xyz_coordinates,
        charge=charge,
        use_graph=True,
        allow_charged_fragments=allow_charged_fragments,
        embed_chiral=True,
        use_huckel=use_huckel)
    assert len(mols) == 1
    return mols[0]

def rdkit2smiles(mol: rdkit.Chem.rdchem.Mol) -> str:
    isomeric_smiles = False
    smiles = Chem.MolToSmiles(mol, isomericSmiles=isomeric_smiles)
    # The following is called "Canonical hack" in xyz2mol
    m = Chem.MolFromSmiles(smiles)
    smiles = Chem.MolToSmiles(m, isomericSmiles=isomeric_smiles)
    return smiles

def recover_geometry(mol_in: rdkit.Chem.rdchem.Mol) -> rdkit.Chem.rdchem.Mol:
    """Create a single conformer with the default ETKDG
    algorithm with fixed random seed"""
    mol = Chem.AddHs(mol_in)
    AllChem.EmbedMolecule(mol, clearConfs=True, randomSeed=42)
    assert len(mol.GetConformers()) == 1
    return mol

def ase2rdkit(atoms: ase.atoms.Atoms) -> rdkit.Chem.rdchem.Mol:
    tmpdir = tempfile.TemporaryDirectory()
    xyzfile = os.path.join(tmpdir.name, 'file.xyz')
    ase.io.write(xyzfile, atoms, format='xyz')
    return xyz2rdkit(xyzfile)

def rdkit2ase(mol: rdkit.Chem.rdchem.Mol) -> ase.atoms.Atoms:
    tmpdir = tempfile.TemporaryDirectory()
    xyzfile = os.path.join(tmpdir.name, "file.xyz")
    MolToXYZFile(mol, xyzfile)
    return ase.io.read(xyzfile)

"""
squeeze a molecule through the SMILES bottleneck

1. transform an instance of ase.atoms.Atoms to SMILES
2. generate a new 3D geometry in rdkit
3. transform back to ase.atoms.Atoms
"""
def squeeze(atoms: ase.atoms.Atoms) -> ase.atoms.Atoms:
    mol_orig = ase2rdkit(atoms)
    smiles = rdkit2smiles(mol_orig)
    mol = Chem.MolFromSmiles(smiles)
    mol_restored = recover_geometry(mol)
    return rdkit2ase(mol_restored)

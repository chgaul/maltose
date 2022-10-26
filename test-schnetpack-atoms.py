import os
import ase
import schnetpack

sample_dir = 'sample_data'

# prepare some example data
db_file = os.path.join(sample_dir, 'test.db')
if os.path.exists(db_file):
    os.remove(db_file)
sample_db = schnetpack.AtomsData(
    db_file, available_properties=['random_property'])
for i, f in enumerate(os.listdir(sample_dir)):
    if f.endswith('.sdf'):
        path = os.path.join(sample_dir, f)
        sample_db.add_system(
            atoms=ase.io.read(path),
            properties={'random_property': float(i)})
assert len(sample_db) > 0        

# Split the dataset
prop = 'random_property'
subset0 = schnetpack.AtomsDataSubset(sample_db, [1, 3])
subset1 = schnetpack.AtomsDataSubset(sample_db, [2, 0])

# concatenate two subsets
concatenated = schnetpack.ConcatAtomsData([subset0, subset1])

# make some plausibility checks: Does the concatenated dataset deliver the expected elements?
idx = 0
assert concatenated[idx][prop] == subset0[idx][prop], """Case 1:
    Concatenated DB does not start with the same element as its
    first member!"""
assert concatenated[idx][prop] != sample_db[idx][prop], """Case 2:
    DB of concatenated subsets should not start with the same 
    element as the underlying dataset!"""
assert concatenated[idx][prop] == sample_db[subset0.indices[idx]][prop], """Case 3:
    Concatenated DB does not start with the expected element of the underlying dataset!"""
assert concatenated[len(subset0) + idx][prop] == subset1[idx][prop], """Case 4:
    Second part of concatenated DB does not start with the first
    element of its second member!"""

os.remove(db_file)

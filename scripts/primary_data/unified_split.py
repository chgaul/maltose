import os.path
import numpy as np
import json


data_path = "./data/"

def inchi_path(db_name):
    return os.path.join(data_path, db_name, 'inchis.json')
def tgt_split_path(db_name):
    return os.path.join(data_path, db_name, 'split_v2.npz')
inchi_dicts = {}
for db_name in ['qm9', 'alchemy', 'oe62', 'hopv']:
    with open(inchi_path(db_name), 'r', encoding='utf-8') as f:
        inchi_dicts[db_name] = json.load(f)


# Make a train/valid/test split based only on Chemical Formula and Connectivity Layer
L = 2

test_fraction = 0.18
val_fraction = 0.09

random_state = np.random.RandomState(seed=2022)

def inchi_up_to_layer(inchi, layer=1):
    return '/'.join(inchi.split('/')[:layer+1])

# Reserve the test set from the Alchemy Contest for testing
def idxs_per_split(ds_name, split):
    return {int(i) for i in np.load(
        os.path.join(data_path, ds_name, 'split.npz'))[split].tolist()}

def split_inchis(ds_name, split):
    with open(inchi_path(ds_name), 'r', encoding='utf-8') as f:
        idict = json.load(f)
    return {
        inchi_up_to_layer(idict[str(i)], L) for i in idxs_per_split(ds_name, split) \
        if str(i) in idict}



train_tris = set()
val_tris = set()
test_tris = set(split_inchis('alchemy', 'test_idx'))
n_alchemy_test = len(test_tris)


# Get (the largest part of) the Alchemy validation set
val_tris_orig = set(split_inchis('alchemy', 'val_idx'))
val_tris = val_tris_orig - test_tris
print('{:.1f}% of the original validation set is independent of the test set\
(in terms of disjunct truncated InChIs).'.format(100*len(val_tris)/len(val_tris_orig)))


# Assign the rest of the Alchemy Dataset
# Collect InChIs where the truncated InChIs differ (Alchemy)
truncated_inchi_to_id = {}
for k, v in inchi_dicts['alchemy'].items():
    tri = inchi_up_to_layer(v, L)
    if tri not in truncated_inchi_to_id:
        truncated_inchi_to_id[tri] = [int(k)]
    else:
        truncated_inchi_to_id[tri] += [int(k)]


local_tris = set(truncated_inchi_to_id.keys())
# Identify those tris that are already in the QM9 splits
local_train_tris = local_tris & set(train_tris)
local_val_tris = local_tris & set(val_tris)
local_test_tris = local_tris & set(test_tris)
len(local_train_tris), len(local_val_tris), len(local_test_tris)

# Split the remaining truncated InChIs randomly into train, validation, test:
remaining_tris = local_tris - local_train_tris - local_val_tris - local_test_tris
idx = random_state.permutation(len(remaining_tris))
trinchis = sorted(remaining_tris)
trinchis = [trinchis[i] for i in idx]

num_test = max(int(test_fraction * len(local_tris)) - len(local_test_tris), 0)
num_val = max(int(val_fraction * len(local_tris)) - len(local_val_tris), 0)
num_train = len(trinchis) - num_test - num_val
assert num_train > 0

local_train_tris.update(trinchis[:num_train])
local_val_tris.update(trinchis[num_train : num_train + num_val])
local_test_tris.update(trinchis[num_train + num_val :])
len(local_train_tris), len(local_val_tris), len(local_test_tris)

len(local_test_tris)/len(local_tris), len(local_val_tris)/(len(local_tris))

train_idx = sorted([id for tri in local_train_tris for id in truncated_inchi_to_id[tri]])
val_idx = sorted([id for tri in local_val_tris for id in truncated_inchi_to_id[tri]])
test_idx = sorted([id for tri in local_test_tris for id in truncated_inchi_to_id[tri]])
len(train_idx), len(val_idx), len(test_idx)

split_file = tgt_split_path('alchemy')
print('Writing ', split_file)
np.savez(split_file, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)


# For the next set, take the split of this dataset into account:
train_tris.update(local_train_tris)
val_tris.update(local_val_tris)
test_tris.update(local_test_tris)


# QM9

# Collect inchis where the truncated InChIs differ (QM9)
truncated_inchi_to_id = {}
for k, v in inchi_dicts['qm9'].items():
    tri = inchi_up_to_layer(v, L)
    if tri not in truncated_inchi_to_id:
        truncated_inchi_to_id[tri] = [int(k)]
    else:
        truncated_inchi_to_id[tri] += [int(k)]

local_tris = set(truncated_inchi_to_id.keys())
# Identify those tris that are already in the previous splits
local_train_tris = local_tris & set(train_tris)
local_val_tris = local_tris & set(val_tris)
local_test_tris = local_tris & set(test_tris)
len(local_train_tris), len(local_val_tris), len(local_test_tris)

remaining_tris = set(truncated_inchi_to_id.keys()) - local_train_tris - local_val_tris - local_test_tris
# Split the remaining truncated InChIs randomly into train, validation, test:
idx = random_state.permutation(len(remaining_tris))
trinchis = sorted(remaining_tris)
trinchis = [trinchis[i] for i in idx]

num_test = max(int(test_fraction * len(local_tris)) - len(local_test_tris), 0)
num_val = max(int(val_fraction * len(local_tris)) - len(local_val_tris), 0)
num_train = len(trinchis) - num_test - num_val
assert num_train > 0

local_train_tris.update(trinchis[:num_train])
local_val_tris.update(trinchis[num_train : num_train + num_val])
local_test_tris.update(trinchis[num_train + num_val :])
len(local_train_tris), len(local_val_tris), len(local_test_tris)

len(local_test_tris)/len(local_tris), len(local_val_tris)/(len(local_tris))

train_idx = sorted([id for tri in local_train_tris for id in truncated_inchi_to_id[tri]])
val_idx = sorted([id for tri in local_val_tris for id in truncated_inchi_to_id[tri]])
test_idx = sorted([id for tri in local_test_tris for id in truncated_inchi_to_id[tri]])
len(train_idx), len(val_idx), len(test_idx)

split_file = tgt_split_path('qm9')
print('Writing ', split_file)
np.savez(split_file, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)


# For the next set, take the split of this dataset into account:
train_tris.update(local_train_tris)
val_tris.update(local_val_tris)
test_tris.update(local_test_tris)


# OE62

# Collect InChIs where the truncated InChIs differ (OE62)
truncated_inchi_to_id = {}
for k, v in inchi_dicts['oe62'].items():
    tri = inchi_up_to_layer(v, L)
    if tri not in truncated_inchi_to_id:
        truncated_inchi_to_id[tri] = [int(k)]
    else:
        truncated_inchi_to_id[tri] += [int(k)]

local_tris = set(truncated_inchi_to_id.keys())
# Identify those tris that are already in the previous splits
local_train_tris = local_tris & set(train_tris)
local_val_tris = local_tris & set(val_tris)
local_test_tris = local_tris & set(test_tris)
len(local_train_tris), len(local_val_tris), len(local_test_tris)

# Split the remaining truncated InChIs randomly into train, validation, test:
remaining_tris = set(truncated_inchi_to_id.keys()) - local_train_tris - local_val_tris - local_test_tris
idx = random_state.permutation(len(remaining_tris))
trinchis = sorted(remaining_tris)
trinchis = [trinchis[i] for i in idx]

num_test = max(int(test_fraction * len(local_tris)) - len(local_test_tris), 0)
num_val = max(int(val_fraction * len(local_tris)) - len(local_val_tris), 0)
num_train = len(trinchis) - num_test - num_val
assert num_train > 0

local_train_tris.update(trinchis[:num_train])
local_val_tris.update(trinchis[num_train : num_train + num_val])
local_test_tris.update(trinchis[num_train + num_val :])
len(local_train_tris), len(local_val_tris), len(local_test_tris)

len(local_test_tris)/len(local_tris), len(local_val_tris)/(len(local_tris))

train_idx = sorted([id for tri in local_train_tris for id in truncated_inchi_to_id[tri]])
val_idx = sorted([id for tri in local_val_tris for id in truncated_inchi_to_id[tri]])
test_idx = sorted([id for tri in local_test_tris for id in truncated_inchi_to_id[tri]])
len(train_idx), len(val_idx), len(test_idx)

split_file = tgt_split_path('oe62')
print('Writing ', split_file)
np.savez(split_file, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)


# For the next set, take the split of this dataset into account:
train_tris.update(local_train_tris)
val_tris.update(local_val_tris)
test_tris.update(local_test_tris)


# HOPV

# Collect InChIs where the truncated InChIs differ (HOPV)
truncated_inchi_to_id = {}
for k, v in inchi_dicts['hopv'].items():
    tri = inchi_up_to_layer(v, L)
    if tri not in truncated_inchi_to_id:
        truncated_inchi_to_id[tri] = [int(k)]
    else:
        truncated_inchi_to_id[tri] += [int(k)]

local_tris = set(truncated_inchi_to_id.keys())
# Identify those tris that are already in the previous splits
local_train_tris = local_tris & set(train_tris)
local_val_tris = local_tris & set(val_tris)
local_test_tris = local_tris & set(test_tris)
len(local_train_tris), len(local_val_tris), len(local_test_tris)

# Split the remaining truncated InChIs randomly into train, validation, test:
remaining_tris = set(truncated_inchi_to_id.keys()) - local_train_tris - local_val_tris - local_test_tris
idx = random_state.permutation(len(remaining_tris))
trinchis = sorted(remaining_tris)
trinchis = [trinchis[i] for i in idx]

num_test = max(int(test_fraction * len(local_tris)) - len(local_test_tris), 0)
num_val = max(int(val_fraction * len(local_tris)) - len(local_val_tris), 0)
num_train = len(trinchis) - num_test - num_val
assert num_train > 0

local_train_tris.update(trinchis[:num_train])
local_val_tris.update(trinchis[num_train : num_train + num_val])
local_test_tris.update(trinchis[num_train + num_val :])
len(local_train_tris), len(local_val_tris), len(local_test_tris)

len(local_test_tris)/len(local_tris), len(local_val_tris)/(len(local_tris))

train_idx = sorted([id for tri in local_train_tris for id in truncated_inchi_to_id[tri]])
val_idx = sorted([id for tri in local_val_tris for id in truncated_inchi_to_id[tri]])
test_idx = sorted([id for tri in local_test_tris for id in truncated_inchi_to_id[tri]])
len(train_idx), len(val_idx), len(test_idx)

split_file = tgt_split_path('hopv')
print('Writing ', split_file)
np.savez(split_file, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)


# For the next set, take the split of this dataset into account:
train_tris.update(local_train_tris)
val_tris.update(local_val_tris)
test_tris.update(local_test_tris)


# Some plausibility checks
assert train_tris & test_tris == set()

assert train_tris & val_tris == set()

assert val_tris & test_tris == set()

print('Truncated InChIs in training:', len(train_tris))
print('Truncated InChIs in validation:', len(val_tris))
print('Truncated InChIs in test: ', len(test_tris))

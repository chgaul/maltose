from typing import Union, List, Tuple, Optional
import numpy as np
from schnetpack.data import AtomsData, AtomsDataSubset, ConcatAtomsData
from schnetpack.data.atoms import torchify_dict

class MultitaskAtomsData(AtomsData):
    """
    PyTorch dataset for atomistic data. The raw data is stored in the specified
    schnetpack AtomsData, AtomsDataSubset, or ConcatAtomsData.
    """
    def __init__(
            self,
            atomsdata: Union[AtomsData, AtomsDataSubset, ConcatAtomsData],
            tasksmapping: map,
            validity_column=True):
        """
        :param tasksmapping: map from task names (str) to either
            - a source name (str), or
            - a tuple of source name and factor.
        :param validity_column, bool: Whether to provide a validity column. If
            all entries are valid, the validity column may be omitted.
        """
        self.atomsdata = atomsdata
        self._available_properties = set(tasksmapping.keys())
        for v in tasksmapping.values():
            assert v is None or isinstance(v, str) or isinstance(v, tuple)
        self.tasksmap = {
            k: (v, 1.0) if isinstance(v, str) else v for k, v in tasksmapping.items()
        }
        self._load_only = None
        self.validity_column = validity_column
        for inner_k in self.tasksmap.values():
            if inner_k is None:
                assert self.validity_column
            else:
                assert inner_k[0] in self.atomsdata.available_properties
        
    def get_properties(self, idx, load_only=None):
        
        # Determine the requested properties
        requested_props = self.available_properties
        if load_only is not None:
            for lo in load_only:
                assert lo in self.available_properties
            requested_props = load_only

        # Get the properties if the original item
        at, inner_props = self.atomsdata.get_properties(idx)
        
        # Copy over everything except the original properties, i.e.,
        # _atom_numbers, _positions, ...
        outer_props = {k: v for k, v in inner_props.items() \
                       if k not in self.atomsdata.available_properties}
        
        # Add the "outer" properties
        for k in requested_props:
            mapto = self.tasksmap[k]
            if mapto is None: # This implies that ther is a validity column
                invalid, dummy = 0.0, -1.0
                outer_props[k] = np.array([[invalid], [dummy]])
            else:
                inner_k, factor = mapto
                value = inner_props[inner_k] * factor
                if self.validity_column:
                    valid = np.array([1.0])
                    outer_props[k] = np.stack([valid, value])
                else:
                    outer_props[k] = value

        return at, outer_props

    def __len__(self):
        return len(self.atomsdata)

    def __getitem__(self, idx):
        _, properties = self.get_properties(idx, self.load_only)
        properties["_idx"] = np.array([idx], dtype=np.int)

        return torchify_dict(properties)

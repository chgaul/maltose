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
            tasksmapping: List[Tuple[str, Optional[str]]]):
        self.atomsdata = atomsdata
        self._available_properties = [outer_k for outer_k, _ in tasksmapping]
        self.tasksmap = {outer_k: inner_k for outer_k, inner_k in tasksmapping}
        for inner_k in self.tasksmap.values():
            if inner_k is not None:
                assert inner_k in self.atomsdata.available_properties
        self._load_only = None
        
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
            inner_k = self.tasksmap[k]
            if inner_k is not None:
                valid = np.array([1.0])
                outer_props[k] = np.stack([valid, inner_props[inner_k]])
            else:
                invalid, dummy = 0.0, -1.0
                outer_props[k] = np.array([[invalid], [dummy]])

        return at, outer_props

    def __len__(self):
        return len(self.atomsdata)

    def __getitem__(self, idx):
        _, properties = self.get_properties(idx, self.load_only)
        properties["_idx"] = np.array([idx], dtype=np.int)

        return torchify_dict(properties)

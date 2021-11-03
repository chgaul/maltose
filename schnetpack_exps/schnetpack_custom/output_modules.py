#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 11:40:04 2021

@author: cgaul
"""
import numpy as np
import torch
from torch import nn
from torch.autograd import grad

import schnetpack
from schnetpack.atomistic.output_modules import AtomwiseError
from schnetpack import Properties

class Atomwise(nn.Module):
    """
    Predicts atom-wise contributions and accumulates global prediction, e.g. for the
    energy.

    Args:
        n_in (int): input dimension of representation
        n_out (int): output dimension of target property (default: 1)
        aggregation_mode (str): one of {sum, avg} (default: sum)
        n_layers (int): number of nn in output network (default: 2)
        n_neurons (list of int or None): number of neurons in each layer of the output
            network. If `None`, divide neurons by 2 in each layer. (default: None)
        activation (function): activation function for hidden nn
            (default: spk.nn.activations.shifted_softplus)
        property (str): name of the output property (default: "y")
        contributions (str or None): Name of property contributions in return dict.
            No contributions returned if None. (default: None)
        derivative (str or None): Name of property derivative. No derivative
            returned if None. (default: None)
        negative_dr (bool): Multiply the derivative with -1 if True. (default: False)
        stress (str or None): Name of stress property. Compute the derivative with
            respect to the cell parameters if not None. (default: None)
        create_graph (bool): If False, the graph used to compute the grad will be
            freed. Note that in nearly all cases setting this option to True is not
            needed and often can be worked around in a much more efficient way.
            Defaults to the value of create_graph. (default: False)
        mean (torch.Tensor or None): mean of property
        stddev (torch.Tensor or None): standard deviation of property (default: None)
        atomref (torch.Tensor or None): reference single-atom properties. Expects
            an (max_z + 1) x 1 array where atomref[Z] corresponds to the reference
            property of element Z. The value of atomref[0] must be zero, as this
            corresponds to the reference property for for "mask" atoms. (default: None)
        outnet (callable): Network used for atomistic outputs. Takes schnetpack input
            dictionary as input. Output is not normalized. If set to None,
            a pyramidal network is generated automatically. (default: None)

    Returns:
        tuple: prediction for property

        If contributions is not None additionally returns atom-wise contributions.

        If derivative is not None additionally returns derivative w.r.t. atom positions.

    """

    def __init__(
        self,
        n_in,
        n_out=1,
        aggregation_mode="sum",
        n_layers=2,
        n_neurons=None,
        activation=schnetpack.nn.activations.shifted_softplus,
        property="y",
        contributions=None,
        derivative=None,
        negative_dr=False,
        stress=None,
        create_graph=False,
        mean=None,
        stddev=None,
        atomref=None,
        outnet=None,
    ):
        super(Atomwise, self).__init__()

        self.n_layers = n_layers
        self.create_graph = create_graph
        self.property = property
        self.contributions = contributions
        self.derivative = derivative
        self.negative_dr = negative_dr
        self.stress = stress

        mean = torch.FloatTensor([0.0]) if mean is None else mean
        stddev = torch.FloatTensor([1.0]) if stddev is None else stddev

        # initialize single atom energies
        if atomref is not None:
            self.atomref = nn.Embedding.from_pretrained(
                torch.from_numpy(atomref.astype(np.float32))
            )
        else:
            self.atomref = None

        # build output network
        if outnet is None:
            self.out_net = nn.Sequential(
                schnetpack.nn.base.GetItem("representation"),
                schnetpack.nn.blocks.MLP(n_in, n_out, n_neurons, n_layers, activation),
            )
        else:
            self.out_net = outnet

        # build standardization layer
        self.standardize = schnetpack.nn.base.ScaleShift(mean, stddev)

        # build aggregation layer
        if aggregation_mode == "sum":
            self.atom_pool = schnetpack.nn.base.Aggregate(axis=1, mean=False)
        elif aggregation_mode == "avg":
            self.atom_pool = schnetpack.nn.base.Aggregate(axis=1, mean=True)
        elif aggregation_mode == "max":
            self.atom_pool = schnetpack.nn.base.MaxAggregate(axis=1)
        elif aggregation_mode == "softmax":
            self.atom_pool = schnetpack.nn.base.SoftmaxAggregate(axis=1)
        else:
            raise AtomwiseError(
                "{} is not a valid aggregation " "mode!".format(aggregation_mode)
            )

    def forward(self, inputs):
        r"""
        predicts atomwise property
        """
        atomic_numbers = inputs[Properties.Z]
        atom_mask = inputs[Properties.atom_mask]

        # run prediction
        yi = self.out_net(inputs)
        yi = self.standardize(yi)

        if self.atomref is not None:
            y0 = self.atomref(atomic_numbers)
            yi = yi + y0

        y = self.atom_pool(yi, atom_mask)

        # collect results
        result = {self.property: y}

        if self.contributions is not None:
            result[self.contributions] = yi

        create_graph = True if self.training else self.create_graph

        if self.derivative is not None:
            sign = -1.0 if self.negative_dr else 1.0
            dy = grad(
                result[self.property],
                inputs[Properties.R],
                grad_outputs=torch.ones_like(result[self.property]),
                create_graph=create_graph,
                retain_graph=True,
            )[0]
            result[self.derivative] = sign * dy

        if self.stress is not None:
            cell = inputs[Properties.cell]
            # Compute derivative with respect to cell displacements
            stress = grad(
                result[self.property],
                inputs["displacement"],
                grad_outputs=torch.ones_like(result[self.property]),
                create_graph=create_graph,
                retain_graph=True,
            )[0]
            # Compute cell volume
            volume = torch.sum(
                cell[:, 0, :] * torch.cross(cell[:, 1, :], cell[:, 2, :], dim=1),
                dim=1,
                keepdim=True,
            )[..., None]
            # Finalize stress tensor
            result[self.stress] = stress / volume

        return result


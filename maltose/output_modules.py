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

from torch_scatter import scatter_add
from torch_geometric.utils import softmax

class CustomAtomwise(nn.Module):
    """
    Predicts atom-wise contributions and accumulates global prediction, e.g. for the
    energy.

    Args:
        n_in (int): input dimension of representation
        n_out (int): output dimension of target property (default: len(properties))
        aggregation_mode (str): one of {sum, avg} (default: sum)
        n_layers (int): number of nn in output network (default: 2)
        n_neurons (list of int or None): number of neurons in each layer of the output
            network. If `None`, divide neurons by 2 in each layer. (default: None)
        activation (function): activation function for hidden nn
            (default: spk.nn.activations.shifted_softplus)
        properties (list(str)): names of the output properties (default: ["y"])
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
        means (torch.Tensor or None): means of properties
        stddevs (torch.Tensor or None): standard deviations of properties (default: None)
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
        n_out=None,
        aggregation_mode="sum",
        n_layers=2,
        n_neurons=None,
        activation=schnetpack.nn.activations.shifted_softplus,
        properties=["y"],
        contributions=None,
        derivative=None,
        negative_dr=False,
        stress=None,
        create_graph=False,
        means=None,
        stddevs=None,
        atomref=None,
        outnet=None,
    ):
        super(CustomAtomwise, self).__init__()

        self.n_layers = n_layers
        self.create_graph = create_graph
        self.properties = properties
        self.contributions = contributions
        self.derivative = derivative
        self.negative_dr = negative_dr
        self.stress = stress

        if n_out is None:
            n_out = len(properties)

        if means is None:
            means = torch.FloatTensor([0.0] * n_out)
        if stddevs is None:
            stddevs = torch.FloatTensor([1.0] * n_out)

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
        self.standardize = schnetpack.nn.base.ScaleShift(means, stddevs)

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
        result = {prop: y[:, i:i+1] for i, prop in enumerate(self.properties)}

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


class TwoNetsAtomwise(nn.Module):
    """
    Predicts atom-wise contributions and accumulates global prediction, e.g. for the
    energy.
    The atomic fatures are first processed by the "a_net", then aggregated to a
    molecular feature vector, and finally processed by the "mnet".
    The default corresponds to the orignal SchNet Atomwise output module, i.e.,
    anet with 2 layers and mnet with 0 layers.

    Note that the standardize operation takes place after the second MLP,
    i.e., after aggregation, not before aggregation as in the schnetpack
    Atomwise output module. This means, the means and stds provided in the
    constructor should be computed with divide_by_atoms=False.

    Args:
        n_in (int): input dimension of representation
        n_intermediate (int): intermediate dimension, at aggregation (default: n_out)
        aggregation_mode (str): one of {sum, avg} (default: sum)
        a_net_layers (int): number of nn in the 1st output network (default: 2)
        a_net_neurons (list of int or None): number of neurons in each layer of
            the 1st output network. If `None`, divide neurons by 2 in each
            layer. (default: None)
        m_net_layers (int): number of nn in the 2nd output network (default: 1)
        m_net_neurons (list of int or None): number of neurons in each layer of
            the 2nd output network. If `None`, divide neurons by 2 in each
            layer. (default: None)
        activation (function): activation function for hidden nn
            (default: spk.nn.activations.shifted_softplus)
        properties (list(str)): names of the output properties (default: ["y"])
        contributions (str or None): Name of property contributions in return dict.
            No contributions returned if None. (default: None)
        means (torch.Tensor or None): means of properties
        stddevs (torch.Tensor or None): standard deviations of properties (default: None)
        atomref (torch.Tensor or None): reference single-atom properties. Expects
            an (max_z + 1) x 1 array where atomref[Z] corresponds to the reference
            property of element Z. The value of atomref[0] must be zero, as this
            corresponds to the reference property for for "mask" atoms. (default: None)
        a_net (callable): Network used for atomistic outputs. Takes schnetpack
            input dictionary as input. Output is not normalized. If set to None,
            a pyramidal network is generated automatically. (default: None)
        m_net (callable): Network used for molecular outputs. Takes schnetpack
            input dictionary as input. Output is not normalized. If set to None,
            a pyramidal network is generated automatically. (default: None)

    Returns:
        tuple: prediction for property

        If contributions is not None additionally returns atom-wise contributions.

        If derivative is not None additionally returns derivative w.r.t. atom positions.

    """

    def __init__(
        self,
        n_in,
        n_intermediate=None,
        aggregation_mode="avg",
        a_net_layers=2,
        a_net_neurons=None,
        m_net_layers=1,
        m_net_neurons=None,
        activation=schnetpack.nn.activations.shifted_softplus,
        properties=["y"],
        contributions=None,
        means=None,
        stddevs=None,
        atomref=None,
        a_net=None,
        m_net=None,
    ):
        super(TwoNetsAtomwise, self).__init__()

        if a_net_neurons is not None:
            if not isinstance(a_net_neurons, int):
                assert len(a_net_neurons) == a_net_layers - 1,\
                    "parameter a_net_neurons: bad length"
        if m_net_neurons is not None:
            if not isinstance(m_net_neurons, int):
                assert len(m_net_neurons) == m_net_layers - 1,\
                    "parameter m_net_neurons: bad length"

        self.derivative = None
        self.stress = None
        self.properties = properties
        self.contributions = contributions

        n_out = len(properties)

        if n_intermediate is None:
            n_intermediate = n_out

        if means is None:
            means = torch.FloatTensor([0.0] * n_out)
        if stddevs is None:
            stddevs = torch.FloatTensor([1.0] * n_out)

        # build standardization layer
        self.standardize = schnetpack.nn.base.ScaleShift(means, stddevs)

        # initialize single atom energies
        if atomref is not None:
            self.atomref = nn.Embedding.from_pretrained(
                torch.from_numpy(atomref.astype(np.float32))
            )
        else:
            self.atomref = None

        # build the molecular-features network
        self.empty_m_net = False
        if m_net is None:
            if m_net_layers==0:
                self.empty_m_net = True
                assert n_intermediate == n_out
            else:
                self.m_net = schnetpack.nn.blocks.MLP(
                    n_intermediate, n_out, m_net_neurons, m_net_layers,
                    activation)
        else:
            self.m_net = m_net

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

        # build the atomic-features network
        if a_net is None:
            self.a_net = nn.Sequential(
                schnetpack.nn.base.GetItem("representation"),
                schnetpack.nn.blocks.MLP(
                        n_in, n_intermediate, a_net_neurons, a_net_layers,
                        activation))
            if not self.empty_m_net:
                self.a_net[1].out_net[-1].activation = activation
        else:
            self.a_net = a_net


    def forward(self, inputs):
        r"""
        predicts atomwise property
        """
        atomic_numbers = inputs[Properties.Z]
        atom_mask = inputs[Properties.atom_mask]

        # run prediction
        yi = self.a_net(inputs)

        y = self.atom_pool(yi, atom_mask)

        if not self.empty_m_net:
            y = self.m_net(y)

        y = self.standardize(y)

        if self.atomref is not None:
            y0 = self.atomref(atomic_numbers)
            y = y + self.atom_pool(y0, atom_mask)


        # collect results
        result = {prop: y[:, i:i+1] for i, prop in enumerate(self.properties)}

        if self.contributions is not None:
            result[self.contributions] = yi


        return result

class Set2Set(nn.Module):
    """
    Adapted from torch_geometric.nn.glob.set2set for schnetpack

    Args:
        n_in (int): input dimension of representation
        processing_steps (int): Number of iterations :math:`T`.
        num_layers (int, optional): Number of recurrent layers, *.e.g*, setting
            :obj:`num_layers=2` would mean stacking two LSTMs together to form
            a stacked LSTM, with the second LSTM taking in outputs of the first
            LSTM and computing the final results. (default: :obj:`1`)
        means (torch.Tensor or None): means of properties
        stddevs (torch.Tensor or None): standard deviations of properties (default: None)
    """
    def __init__(
        self,
        n_in,
        processing_steps,
        num_layers=1,
        m_net_layers=1,
        m_net_neurons=None,
        activation=schnetpack.nn.activations.shifted_softplus,
        properties=["y"],
        m_net=None,
        means=None,
        stddevs=None):

        super(Set2Set, self).__init__()
        self.derivative = None
        self.stress = None
        self.properties = properties

        self.n_in = n_in
        n_set2set_out = 2 * n_in
        self.processing_steps = processing_steps
        self.num_layers = num_layers

        self.input = nn.Sequential(
            schnetpack.nn.base.GetItem("representation"))

        self.lstm = torch.nn.LSTM(
                input_size=n_set2set_out,
                hidden_size=self.n_in,
                num_layers=num_layers)

        n_out = len(properties)
        if m_net is None:
            assert m_net_layers > 0, "Cannot get to the required output dimension!"
            self.m_net = schnetpack.nn.blocks.MLP(
                n_set2set_out, n_out, m_net_neurons, m_net_layers,
                activation)
        else:
            self.m_net = m_net

        self.standardize = schnetpack.nn.base.ScaleShift(means, stddevs)

    def forward(self, inputs):
        r"""
        predicts atomwise property
        """
        atom_mask = inputs[Properties.atom_mask]
        batch_size = atom_mask.shape[0] # batch_size = N_mols
        y = self.input(inputs)
        # y.shape = (batch_size, max(n_atoms), n_in),

        # The following code is adapted from torch_geometric.nn.glob.set2set.
        # First, adapt the input format: instead of the schnetpack _atoms_mask,
        # provide a 'batch' tensor, holding for each atom the index of the
        # molecule it belongs to.
        m = inputs['_atom_mask'].to(bool)
        # create the batch tensor, which holds the molecule index for each atom:
        batch = torch.Tensor(
            [i for i, mask in enumerate(m) for valid in mask if valid]).to(
                    dtype=int,
                    device=y.device)
        x = y[m]
        # x.shape = (sum(n_atoms), n_in) =: (N_atoms, n_in)

        # Input sequence to the lstm:
        q_star = y.new_zeros(batch_size, 2 * self.n_in)
        # Initial hidden state and cell state for each element of the batch:
        h = (y.new_zeros((self.num_layers, batch_size, self.n_in)),
             y.new_zeros((self.num_layers, batch_size, self.n_in)))
        # The following loop processes q_star and h, as a function of x:
        for i in range(self.processing_steps):
            q, h = self.lstm(q_star.unsqueeze(0), h)
            # q, h are output sequence and hidden states. Note that LSTM(n_)
            # q.shape = (1, N_mols, n_in)
            q = q.view(batch_size, self.n_in) # q.shape = (N_mols, n_in)
            # Use q for a per-molecule weighted sum of atom features from x
            # 1. dot product of the x_i with their respective q_b[i]:
            e = (x * q[batch]).sum(dim=-1, keepdim=True)
            #    e.shape = (N_atoms, 1)
            # 2. molecule-wise softmax (for normalization):
            a = softmax(e, batch, num_nodes=batch_size)
            #    a.shape = (N_atoms, 1)
            # 3. molecule-wise weighted sum:
            r = scatter_add(a * x, batch, dim=0, dim_size=batch_size)
            #    r.shape = (N_mols, n_in)
            # Finally, concatenate r (weigted sum of x's) to q, and continue:
            q_star = torch.cat([q, r], dim=-1)
            # q_star.shape = (N_mols, 2*n_in)

        # Apply the molecule-level network to reduce the dimension from
        # from 2*n_in to len(properties):
        y = self.m_net(q_star)
        y = self.standardize(y)
        # collect results
        result = {prop: y[:, i:i+1] for i, prop in enumerate(self.properties)}

        return result

import torch
from schnetpack.train.loss import LossFnError

def build_gated_mse_loss(properties, loss_tradeoff=None):
    """
    Build the mean squared error loss function, including a gating factor.
    Each dictionaly value in batch is expected to be of shape(b, 2, 1),
    where b is the batch size, and batch[q][:,0] are the validity factor,
    and batch[q][:,1] are the regression target values.

    Args:
        properties (list): mapping between the model properties and the
            dataset properties
        loss_tradeoff (list or None): multiply loss value of property with tradeoff
            factor

    Returns:
        mean squared error loss function

    """
    if loss_tradeoff is None:
        loss_tradeoff = [1] * len(properties)
    if len(properties) != len(loss_tradeoff):
        raise LossFnError("loss_tradeoff must have same length as properties!")

    def loss_fn(batch, result):
        loss = 0.0
        for prop, factor in zip(properties, loss_tradeoff):
            validity = batch[prop][:,0]
            target = batch[prop][:,1]
            diff = target - result[prop]
            diff = diff ** 2
            err_sq = factor * torch.mean(validity * diff)
            loss += err_sq
        return loss

    return loss_fn

import torch
import torch.nn as nn

# logging
from logging import getLogger  # noqa: E402
logger = getLogger(__name__)


# ------------------------------
# Helper Functions
# ------------------------------
def create_mlp(in_dim, out_dim, hidden_dims=[], dropout=0.0):
    """
    Create simple mlp nn.Modules (activated by ReLU())

    Parameters
    ----------
    in_dim : int
        input dimension
    out_dim : int
        output dimension
    hidden_dims : list of int
    dropout : float, default is 0.0

    Returns
    -------
    <return> : nn.Sequential Modules
    """
    hidden_dims.append(out_dim)
    hidden_dims.insert(0, in_dim)

    mlp = []
    for i in range(len(hidden_dims) - 1):
        mlp += [
            nn.Linear(hidden_dims[i], hidden_dims[i+1]),
            nn.Dropout(dropout),
            nn.ReLU(),
        ]
    return nn.Sequential(*mlp)


def masked_softmax(vector, mask, dim):
    """
    Masked softmax

    Parameters
    ----------
    vector : torch.tensor
    mask : torch.tensor
    dim : int
        softmax dimension

    Returns
    -------
    <return> : torch.tensor
    """
    exps = torch.exp(vector)
    masked_exps = exps * mask.float()
    # avoid divide 0
    masked_sums = masked_exps.sum(dim, keepdim=True) + 1e-7
    return masked_exps / masked_sums


def gen_mask(max_n_obj, n_obj_list):
    """
    Generate mask

    Parameters
    ----------
    max_n_obj : int
        the number of max objects
    n_obj_list : torch.tensor or None
        using for masked softmax (shape: batch_size)
    """

    mask = nn.functional.one_hot(n_obj_list - 1, num_classes=max_n_obj)
    # NOTE: maybe slow for small batch_size
    for i in range(max_n_obj - 1):
        mask[:, max_n_obj - i - 2] = \
            mask[:, max_n_obj - i - 1] - mask[:, max_n_obj - i - 2]
    return mask

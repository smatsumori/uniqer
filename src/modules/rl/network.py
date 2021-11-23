import os

import numpy as np
import torch
import torch.nn as nn

from utils.consts import RICH_DIM, BASE_DIM, NUM_MAX_OBJ, SPECIAL_TOKENS

# logging
from logging import getLogger
logger = getLogger(__name__)


def extract_topk_features(spatial_features, img_features, o_pred_prob, top_k):
    """
    Extracting top_k objects spatial features for Splitter Network

    * TODO: DEBUG: carefully debugging

    Parameters
    ----------
    spatial_features : torch.tensor
        (batch_size, num_max_obj, spatial_feature_dim)
    img_features : torch.tensor
        (batch_size, num_max_obj, img_feature_dim)
    o_pred_prob : torch.tensor or None
        (shape: batch_size, num_max_obj)
    top_k : int

    Returns
    -------
    obj_features : torch.tensor
        (shape: batch_size, top_k, obj_feature_dim)
    top_prob : torch.tensor
        (shape: batch_size, top_k)
    """
    # get indices from large to small
    # cands: (batch_size, top_k)
    # NOTE: torch.argsort is too slow...
    cands = torch.tensor(
        np.argsort(o_pred_prob)[:, ::-1][:, :top_k].copy()
    ).unsqueeze(-1)

    assert spatial_features.shape[2] == 12 * NUM_MAX_OBJ

    # spatial_features[:, 0] == spatial_features[:, i]  (i < NUM_MAX_OBJ)
    # 1-a. obj themselves spatial features

    b_size = o_pred_prob.shape[0]

    sp1 = spatial_features[:, 0, :5 * NUM_MAX_OBJ].clone().reshape(
        b_size, NUM_MAX_OBJ, -1
    )  # (b_size, NUM_MAX_OBJ, 5)
    sp1 = torch.gather(
        sp1, 1, cands.repeat(1, 1, 5)
    ).float()  # (b_size, top_k, 5)

    # 1-b. obj <-> obj features
    sp2 = torch.gather(
        spatial_features[:, :, 5 * NUM_MAX_OBJ:10 * NUM_MAX_OBJ],
        1,
        cands.repeat(1, 1, 5 * NUM_MAX_OBJ),
    ).reshape(b_size, top_k, NUM_MAX_OBJ, 5)
    sp2 = torch.gather(
        sp2, 2, cands.unsqueeze(1).repeat(1, top_k, 1, 5)
    ).float().reshape(b_size, top_k, -1)

    # 1-c. obj distance features
    sp3 = torch.gather(
        spatial_features[:, :, 10 * NUM_MAX_OBJ:12 * NUM_MAX_OBJ],
        1,
        cands.repeat(1, 1, 2 * NUM_MAX_OBJ),
    ).reshape(b_size, top_k, NUM_MAX_OBJ, 2)
    sp3 = torch.gather(
        sp3, 2, cands.unsqueeze(1).repeat(1, top_k, 1, 2)
    ).float().reshape(b_size, top_k, -1)

    img_features = torch.gather(
        img_features, 1, cands.repeat(1, 1, img_features.shape[2])
    )

    # concat
    obj_features = torch.cat([img_features, sp1, sp2, sp3], dim=2)

    top_prob = torch.gather(
        torch.tensor(o_pred_prob), 1, cands.squeeze(-1)
    ).unsqueeze(-1)  # for splitter
    return obj_features, top_prob

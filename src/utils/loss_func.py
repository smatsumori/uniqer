import torch
import torch.nn as nn

# logging
from logging import getLogger
logger = getLogger(__name__)


def masked_bce(pred, tgt, n_obj, eps=1e-10, mask=None):
    """
    Masked Binary Cross Entropy loss for pre-training Guesser & QA Encoder

    Parameters
    ----------
    pred : torch.tensor
        (shape: batch_size, max_obj)
    tgt : torch.tensor
        (shape: batch_size, max_obj)
    n_obj : array-like of int
        (shape: batch_size)
    eps : float
    """

    # [OPTION 1] batch loop ver. too slow (for debug purpose)
    # bce_loss = 0.0
    # for i in range(pred.shape[0]):
    #     max_idx = n_obj[i]
    #     bce_loss -= (
    #         (tgt[i, :max_idx] * torch.log(pred[i, :max_idx] + eps)).sum() +
    #         (
    #             (1 - tgt[i, :max_idx]) *
    #             torch.log(1 - pred[i, :max_idx] + eps)
    #         ).sum()
    #     ) / max_idx
    # bce_loss /= pred.shape[0]

    # # [OPTION 2]
    # # step 1: n_obj --> mask tensor
    if mask is None:
        mask = torch.zeros_like(tgt).scatter_(
            1,  # dim
            n_obj.clone().detach().unsqueeze(-1).to(tgt.device) - 1,
            1,  # torch.tensor(1).to(tgt.device),  # value
        )  # not elegant?

        # mask = 1 - torch.cumsum(mask, dim=1)
        max_n_obj = tgt.shape[1]
        for i in range(max_n_obj - 1):
            mask[:, max_n_obj - i - 2] += mask[:, max_n_obj - i - 1]

    # step 2: BCE without summing up
    bce_loss = -1 * (
        tgt * torch.log(pred + eps) + (1 - tgt) * torch.log(1 - pred + eps)
    )

    # step 3: element-wise mul --> averaging
    bce_loss *= mask
    bce_loss = torch.mean(bce_loss.sum(dim=1) / mask.sum(dim=1))
    return bce_loss


def dialogue_masked_bce(pred, tgt, mask, n_obj, eps=1e-10):
    """
    Dialogue Based Masked Binary Cross Entropy loss

    Parameters
    ----------
    pred : list of torch.tensor
        (length : dialogue_len, tensor shape: batch_size, max_obj)
        NOTE: this is not same shape with `tgt` !!!
    tgt : torch.tensor
        (shape: batch_size, dialogue_len, max_obj)
    mask : torch.LongTensor
        (shape: batch_size, dialogue_len, 1)
    n_obj : array-like of int
        (shape: batch_size)
    eps : float
    """

    # NOTE: By rewriting `masked_bce` function for the difference in dimension,
    #       the calculation may be faster, but since the loop times is small
    #       we leave it as it is for now.
    assert len(pred) == tgt.shape[1], 'shape error'
    bce_loss = 0
    for i in range(tgt.shape[1]):
        bce_loss += masked_bce(
            pred[i] * mask[:, i],
            tgt[:, i].float() * mask[:, i],
            n_obj,
            eps=eps
        )

    # We don't take averages because the weight of each question in a dialog
    # should be the same
    # TODO: further consideration is required
    return bce_loss


# DEBUG PURPOSE ONLY
if __name__ == '__main__':
    x = torch.sigmoid(torch.randn(3, 10))
    y = torch.zeros_like(x)
    n_obj = 7
    n_obj_ls = torch.tensor([7, 7, 7])

    # BCELoss
    print(f'original BCE: {nn.BCELoss()(x, y)}')
    print(f'masked original BCE: {nn.BCELoss()(x[:, :n_obj], y[:, :n_obj])}')

    # my impl.
    print(f'my masked BCE: {masked_bce(x, y, n_obj_ls)}')
    print('---   ' * 10)

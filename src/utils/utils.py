import os
import sys
import random
import json
import collections

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

# logging
from logging import getLogger
logger = getLogger(__name__)


def seed_everything(seed=76):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def invert_dict(d):
    """
    Copy from utils.py
    ref: https://github.com/facebookresearch/clevr-iep
    """
    return {v: k for k, v in d.items()}


def load_json(jsondata):
    """
    Parameters
    ----------
    jsondata : str or dict
        CLEVR dataset json data or json file path
    """
    if type(jsondata) is str:
        if os.path.isfile(jsondata):
            with open(jsondata, 'r') as f:
                return json.load(f)
        else:
            logger.error(f'This is not json path : {jsondata}')
            sys.exit()
    elif type(jsondata) is dict:
        return jsondata
    else:
        logger.error('`jsondata` should be given as a dict or string type')
        sys.exit()


def load_vocab(path):
    """
    Copy from utils.py
    ref: https://github.com/facebookresearch/clevr-iep
    """
    with open(path, 'r') as f:
        vocab = json.load(f)
        vocab['question_idx_to_token'] = invert_dict(
            vocab['question_token_to_idx']
        )
        vocab['program_idx_to_token'] = invert_dict(
            vocab['program_token_to_idx']
        )
        try:
            vocab['answer_idx_to_token'] = invert_dict(
                vocab['answer_token_to_idx']
            )
        except KeyError:
            logger.warning('vocab.json has not answer_token_to_idx')
            pass
    # Sanity check: make sure <NULL>, <START>, and <END> are consistent
    assert vocab['question_token_to_idx']['<NULL>'] == 0
    assert vocab['question_token_to_idx']['<START>'] == 1
    assert vocab['question_token_to_idx']['<END>'] == 2
    assert vocab['program_token_to_idx']['<NULL>'] == 0
    assert vocab['program_token_to_idx']['<START>'] == 1
    assert vocab['program_token_to_idx']['<END>'] == 2
    assert vocab['program_token_to_idx']['<UNK>'] == 3
    assert vocab['program_token_to_idx']['<STOP>'] == 4
    return vocab


def count_parameters(model):
    """
    Counting nn.Module trainable parameters

    ref: https://discuss.pytorch.org/t/
                how-do-i-check-the-number-of-parameters-of-a-model/4325/8

    Parameters
    ----------
    model : nn.Module
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_model(model, optimizer, model_path, device, is_dict=False):
    """
    Load PyTorch model

    Parameters
    ----------
    model : torch.nn.Module
    optimizer : torch.optim or None
    model_path : str
    device : str
    is_dict : bool
        True when the save model is collections.OrderedDict

    Returns
    -------
    model : torch.nn.Module
    optimizer : torch.optim or None
    """
    model.to('cpu')

    if is_dict:
        # if we saved model like following (officially recommended way)
        # torch.save(model.state_dict(), savepath)
        if isinstance(torch.load(model_path), collections.OrderedDict):
            model.load_state_dict(torch.load(model_path))
            logger.info('optimizer info was not saved!')
        else:
            logger.error(
                f'is_dict was True, but {model_path} is not torch.state_dict'
            )
    else:
        # torch.save(model, savepath)
        model = torch.load(model_path, map_location='cpu')
        if optimizer is not None:
            optimizer.load_state_dict(model.info_dict['optimizer'])

    model = model.to(device)
    logger.info(f'Loading {model_path}')

    n_params = count_parameters(model)
    logger.info(f'Trainable Params: {n_params}')

    try:
        # If model has `device` attribute, update device_id
        model.set_device(device)
    except AttributeError:
        logger.warning(f'set_device() function cannot found ({model_path})')
        pass
    return model, optimizer


def is_jupyter_notebook():
    """
    Checking if the running environment is jupyter notebook or not.

    Returns
    -------
    <return> : bool
        True for jupyter detected, False for normal python detected.
    """

    # 1. normal python shell
    if 'get_ipython' not in globals():
        return False

    # 2. ipython shell
    try:
        # get_ipython() is available in the global namespace by default when
        # iPython is started.
        if get_ipython().__class__.__name__ == 'TerminalInteractiveShell':
            return False

        # 3. jupyter notebook
        else:
            return True
    except NameError:
        return False


def cos_sim(a, b):
    """
    Return cosine similarity of two vectors.

    Parameters
    ----------
    a : array-like
    b : array-like

    Returns
    -------
    cos_sim : array of float or float
    """

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def input_yn():
    """
    Get user's input (yes / no)

    Returns
    -------
    <return> : bool
        True for 'yes', False for 'no'
    """
    while True:
        c = input("Please type 'yes' or 'no' [Y/n]: ")
        if c in ['Y', 'yes']:
            return True
        elif c in ['n', 'no']:
            return False


def base_10_to_n(x, n):
    """
    * TODO high-speed
    * NOTE: binary --> np.unpackbits
    * TODO: create hash-map

    Parameters
    ----------
    x : int
    n : int
        n-ary

    Returns
    -------
    out : str
        n-ary value
    """
    if (int(x/n)):
        return base_10_to_n(int(x/n), n) + str(x % n)
    return str(x % n)


def smart_2d_sort(src, idx):
    """
    Sort `src` tensor by given order `idx`

    Ex.
    >>> src = torch.tensor([[1, 0, 2, 2, 1], [0, 2, 1, 2, 0]])
    >>> idx = torch.tensor([[3, 2, 4, 1, 0], [2, 1, 4, 3, 3]])
    >>> smart(src, idx)
    output: torch.tensor([[2, 2, 1, 0, 1], [1, 2, 0, 2, 2]])

    Parameters
    ----------
    src : torch.tensor
        shape: (d1, d2)
    idx : torch.LongTensor
        shape: (d1, d2)

    Returns
    -------
    ret : torch.tensor
    """
    assert src.shape == idx.shape, 'shape unmatched error'
    assert len(src.shape) == 2, 'smart_2d_sort can be applied to only 2D!'

    d1, d2 = src.shape
    ret = src[
        torch.arange(d1).unsqueeze(1).repeat((1, d2)).flatten(),
        idx.flatten(),
    ].view(d1, d2)
    return ret


def confusion_matrix(pred, truth, n_class):
    """
    Computes confusion matrix.

    Parameters
    ----------
    pred :
        pred
    truth :
        truth
    n_class :
        n_class

    Returns
    ----------
    matrix : np.array
        .shape (truth, pred)
    """
    matrix = np.zeros((n_class, n_class))
    for p, t in zip(pred, truth):
        matrix[t, p] += 1
    return matrix


def compute_metrics(pred, truth, n_class):
    """compute precision, recall, and f1 scores.

    Parameters
    ----------
    pred : list
        Estimated values.
    truth : list
        Ground-truth values.
    n_class :
        The number of classes.

    Returns
    ----------
    pr : float
        precision score (micro-averaged)
    re : float
        recall score (micro-averaged)
    f1 : float
        f1 score (micro-averaged)
    """
    labels = [i for i in range(n_class)]
    pr = precision_score(truth, pred, labels=labels, average='micro')
    re = recall_score(truth, pred, labels=labels, average='micro')
    f1 = f1_score(truth, pred, labels=labels, average='micro')
    return pr, re, f1

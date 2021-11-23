# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Modification is applied by @sngyo

import numpy as np

from utils.consts import TOKEN_LABEL_CLEVR, TOKEN_LABEL_ASK3
from utils.consts import TOKEN_LABEL_ASK4, SPECIAL_TOKENS


def tokenize(
        s,
        delim=' ',
        add_start_token=True,
        add_end_token=True,
        punct_to_keep=None,
        punct_to_remove=None
):
    """
    Tokenize a sequence, converting a string s into a list of (string) tokens
    by splitting on the specified delimiter. Optionally keep or remove certain
    punctuation marks and add start and end tokens.
    """
    if punct_to_keep is not None:
        for p in punct_to_keep:
            s = s.replace(p, '%s%s' % (delim, p))

    if punct_to_remove is not None:
        for p in punct_to_remove:
            s = s.replace(p, '')

    tokens = s.split(delim)
    if add_start_token:
        tokens.insert(0, '<START>')
    if add_end_token:
        tokens.append('<END>')
    return tokens


def encode(seq_tokens, token_to_idx, allow_unk=False):
    seq_idx = []
    for token in seq_tokens:
        if token not in token_to_idx:
            if allow_unk:
                token = '<UNK>'
            else:
                raise KeyError('Token "%s" not in vocab' % token)
        seq_idx.append(token_to_idx[token])
    return seq_idx


def decode(seq_idx, idx_to_token, check_mode=False):
    """
    Decode indices list to original token

    Parameters
    ----------
    seq_idx : arraylike
    idx_to_token : dict
    check_mode : bool, default is False
       when True, idx < 5, i.e. [<NULL>, <START>, <END>, <UNK>, <STOP>] tokens
       will be removed here

    Returns
    -------
    <return> : str
    """
    lim_idx = 5 if not check_mode else -1
    return ' '.join([idx_to_token[idx] for idx in seq_idx if idx > lim_idx])


def list_to_tree(program_list):
    def build_subtree(cur):
        return {
            'function': cur['function'],
            'value_inputs': [x for x in cur['value_inputs']],
            'inputs': [build_subtree(program_list[i]) for i in cur['inputs']],
        }
    return build_subtree(program_list[-1])


def tree_to_list(program_tree):
    """
    Function that do the inverse of `list_to_tree()`
    """
    raise NotImplementedError


def tree_to_prefix(program_tree):
    output = []

    def helper(cur):
        output.append({
            'function': cur['function'],
            'value_inputs': [x for x in cur['value_inputs']],
        })
        for node in cur['inputs']:
            helper(node)
    helper(program_tree)
    return output


def prefix_to_program(prefix):
    """
    Function that do the inverse of `tree_to_prefix()` + `list_to_tree()`

    INFO: For now, this function works ONLY for trees that will not be divided.
          Therefore, if there is a possibility of using comparison functions,
          such as 'equal_color', we will need to re-implement this.

    FIXME: for more than one-hop tree
    FIXME: type ? function ? GROSS

    ex1. prefix like below  --> ok!
        [{'type': 'filter_material', 'value_inputs': ['metal']},
         {'type': 'filter_color', 'value_inputs': ['gray']},
         {'type': 'scene', 'value_inputs': []}]

    ex2. prefix like below  --> WRONG answer!
        [{'type': 'equal_color', 'value_inputs': []},
         {'type': 'query_color', 'value_inputs': []},
         {'type': 'unique', 'value_inputs': []},
         {'type': 'filter_shape', 'value_inputs': ['cylinder']},
         {'type': 'filter_size', 'value_inputs': ['small']},
         {'type': 'scene', 'value_inputs': []},
         {'type': 'query_color', 'value_inputs': []},
         {'type': 'unique', 'value_inputs': []},
         {'type': 'filter_shape', 'value_inputs': ['sphere']},
         {'type': 'filter_material', 'value_inputs': ['rubber']},
         {'type': 'filter_size', 'value_inputs': ['small']},
         {'type': 'scene', 'value_inputs': []}]

    Parameters
    ----------
    prefix : list
         examples are above.
    """

    # prefix[-1]['inputs'] = []
    # for i in range(len(prefix) - 1):
    #     prefix[len(prefix) - i - 2]['inputs'] = [prefix[len(prefix) - i - 1]]
    # return prefix[0]

    output = []
    for i, f in enumerate(prefix[::-1]):
        f['inputs'] = [] if i == 0 else [i - 1]
        output.append(f)
    return output


def list_to_prefix(program_list):
    return tree_to_prefix(list_to_tree(program_list))


def function_to_str(f):
    value_str = ''
    if f['value_inputs']:
        value_str = '[%s]' % ','.join(f['value_inputs'])
    return '%s%s' % (f['function'], value_str)


def str_to_function(text):
    """
    Parameters
    ----------
    text : str
        ex1. 'filter_shape[cylinder]'
        ex2. 'equal_color'

    Returns
    -------
    funct_dict : dict
        ex1. {'type': 'filter_shape', 'value_inputs': 'cylinder'}
        ex2. {'type': 'equal_color', 'value_inputs': '[]'}
    """
    funct_dict = {}
    text = text.split('[')

    if len(text) == 2:
        funct_dict['type'] = text[0]
        funct_dict['value_inputs'] = [text[1][:-1]]  # remove ']'
    elif len(text) == 1:
        funct_dict['type'] = text[0]
        funct_dict['value_inputs'] = []
    else:
        raise NotImplementedError

    return funct_dict


def list_to_str(program_list):
    return ' '.join(function_to_str(f) for f in program_list)


def str_to_list(text):
    """
    Function that do the opposite of `list_to_str()`

    Parameters
    ----------
    text : str
    """
    split_list = text.split()
    result = []

    for f in split_list:
        result.append(str_to_function(f))
    return result


def program_to_str(program):
    """
    CLEVR functional programming (list of dict) to string for sequence model
    """
    program_prefix = list_to_prefix(program)
    return list_to_str(program_prefix)


def str_to_program(text):
    """
    FIXME prefix_to_proram only works for one-hop ? tree program
    """
    return prefix_to_program(str_to_list(text))


def build_vocab(
        sequences,
        min_token_count=1,
        delim=' ',
        punct_to_keep=None,
        punct_to_remove=None,
):
    token_to_count = {}
    tokenize_kwargs = {
        'delim': delim,
        'punct_to_keep': punct_to_keep,
        'punct_to_remove': punct_to_remove,
    }
    for seq in sequences:
        # TODO: remove the following
        seq_tokens = tokenize(seq, **tokenize_kwargs,
                              add_start_token=False, add_end_token=False)
        for token in seq_tokens:
            if token not in token_to_count:
                token_to_count[token] = 0
            token_to_count[token] += 1

    token_to_idx = {}
    for token, idx in SPECIAL_TOKENS.items():
        token_to_idx[token] = idx
    for token, count in sorted(token_to_count.items()):
        if count >= min_token_count:
            token_to_idx[token] = len(token_to_idx)

    return token_to_idx


def get_q_type(seq_idx, idx_to_token, image_data_version):
    """
    Parameters
    ----------
    seq_idx : arraylike
    idx_to_token : dict
    image_data_version : str

    Returns
    -------
    out : np.array
        shape 5  # color, shape, size, material, relate
    """
    if image_data_version == 'CLEVR_Ask3':
        tk = TOKEN_LABEL_ASK3
    elif image_data_version == 'CLEVR_Ask4':
        tk = TOKEN_LABEL_ASK4
    else:
        tk = TOKEN_LABEL_CLEVR

    out = np.zeros(5)

    program = [idx_to_token[idx] for idx in seq_idx]
    for token in program:
        if token in tk.TOKEN_LABEL_COLOR:
            out[0] = 1
        elif token in tk.TOKEN_LABEL_SHAPE:
            out[1] = 1
        elif token in tk.TOKEN_LABEL_SIZE:
            out[2] = 1
        elif token in tk.TOKEN_LABEL_MATERIAL:
            out[3] = 1
        elif token in tk.TOKEN_LABEL_RELATE:
            out[4] = 1
    return out


if __name__ == '__main__':
    # DEBUG PURPOSE
    program = [
        {'function': 'scene', 'inputs': []},
        {'function': 'filter_size', 'inputs': [0], 'value_inputs': ['small']},
        {
            'function': 'filter_shape',
            'inputs': [1],
            'value_inputs': ['cylinder']
        },
        {'function': 'unique', 'inputs': [2]},
        {'function': 'scene', 'inputs': []},
        {'function': 'filter_size', 'inputs': [4], 'value_inputs': ['small']},
        {
            'function': 'filter_material',
            'inputs': [5],
            'value_inputs': ['rubber']
        },
        {
            'function': 'filter_shape',
            'inputs': [6],
            'value_inputs': ['sphere']
        },
        {'function': 'unique', 'inputs': [7]},
        {'function': 'query_color', 'inputs': [3]},
        {'function': 'query_color', 'inputs': [8]},
        {'function': 'equal_color', 'inputs': [9, 10]}
    ]

    for f in program:
        if 'value_inputs' not in f:
            f['value_inputs'] = []

    program_str = program_to_str(program)
    print('--- program_to_str version ---')
    print(program_str)

    program_token_to_idx = {
        '<NULL>': 0,
        '<START>': 1,
        '<END>': 2,
        '<UNK>': 3,
        'count': 4,
        'equal_color': 5,
        'equal_integer': 6,
        'equal_material': 7,
        'equal_shape': 8,
        'equal_size': 9,
        'exist': 10,
        'filter_color[blue]': 11,
        'filter_color[brown]': 12,
        'filter_color[cyan]': 13,
        'filter_color[gray]': 14,
        'filter_color[green]': 15,
        'filter_color[purple]': 16,
        'filter_color[red]': 17,
        'filter_color[yellow]': 18,
        'filter_material[metal]': 19,
        'filter_material[rubber]': 20,
        'filter_shape[cube]': 21,
        'filter_shape[cylinder]': 22,
        'filter_shape[sphere]': 23,
        'filter_size[large]': 24,
        'filter_size[small]': 25,
        'greater_than': 26,
        'intersect': 27,
        'less_than': 28,
        'query_color': 29,
        'query_material': 30,
        'query_shape': 31,
        'query_size': 32,
        'relate[behind]': 33,
        'relate[front]': 34,
        'relate[left]': 35,
        'relate[right]': 36,
        'same_color': 37,
        'same_material': 38,
        'same_shape': 39,
        'same_size': 40,
        'scene': 41,
        'union': 42,
        'unique': 43
    }

    print(encode(program_str.split(), program_token_to_idx))

    """
    equal_color query_color unique filter_shape[cylinder] filter_size[small]
    scene query_color unique filter_shape[sphere] filter_material[rubber]
    filter_size[small] scene
    """

    print('\n--- str_to_program version ---')
    input_str = 'filter_material[metal] filter_color[gray] scene'
    print(f'String input : {input_str}')

    try:
        from pprint import pprint
        pprint(str_to_program(input_str))
    except ImportError:
        print(str_to_program(input_str))

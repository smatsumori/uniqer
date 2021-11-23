# Available datasets
DATASETS = ['CLEVR_v1.0', 'CLEVR_Ask3', 'CLEVR_Ask4']

# Dataset Configuration
# Maximum / minimum number of objects in a scene
NUM_MAX_OBJ = 10
NUM_MIN_OBJ = 3


# Program tokens
# [WARNING] All token names that may be used in all dataset versions are
# listed, so if you want to use something like length, handle it appropriately
# at the imported file

# The following tokens are common
# special tokens
# TODO: deprecates
TOKEN_LABEL_SPECIAL = [
    '<NULL>', '<START>', '<END>', '<UNK>', '<STOP>', 'scene', 'exist'
]

# relate tokens
# TODO: deprecates
TOKEN_LABEL_RELATE = [
    'relate[behind]', 'relate[front]', 'relate[left]', 'relate[right]'
]


class TOKEN_LABEL_CLEVR:
    # colors
    TOKEN_LABEL_COLOR = [
        'filter_color[blue]',  'filter_color[brown]',
        'filter_color[cyan]', 'filter_color[gray]', 'filter_color[green]',
        'filter_color[purple]', 'filter_color[red]', 'filter_color[yellow]'
    ]

    # shapes
    TOKEN_LABEL_SHAPE = [
        'filter_shape[sphere]', 'filter_shape[cylinder]', 'filter_shape[cube]'
    ]

    # sizes
    TOKEN_LABEL_SIZE = [
        'filter_size[small]', 'filter_size[large]', 'filter_size[medium]'
    ]

    # materials
    TOKEN_LABEL_MATERIAL = [
        'filter_material[rubber]',
        'filter_material[metal]',
        'filter_material[glass]'
    ]

    # special tokens
    TOKEN_LABEL_SPECIAL = [
        '<NULL>', '<START>', '<END>', '<UNK>', '<STOP>', 'scene', 'exist'
    ]

    # relate tokens
    TOKEN_LABEL_RELATE = [
        'relate[behind]', 'relate[front]', 'relate[left]', 'relate[right]'
    ]

    ATTER_SIZE = len(
        TOKEN_LABEL_COLOR + TOKEN_LABEL_SHAPE
        + TOKEN_LABEL_SIZE + TOKEN_LABEL_MATERIAL
    )


class TOKEN_LABEL_ASK3:
    # colors
    TOKEN_LABEL_COLOR = [
        'filter_color[blue]', 'filter_color[green]', 'filter_color[red]',
    ]

    # shapes
    TOKEN_LABEL_SHAPE = [
        'filter_shape[sphere]', 'filter_shape[cylinder]', 'filter_shape[cube]'
    ]

    # sizes
    TOKEN_LABEL_SIZE = [
        'filter_size[small]', 'filter_size[large]', 'filter_size[medium]'
    ]

    # materials
    TOKEN_LABEL_MATERIAL = [
        'filter_material[rubber]',
        'filter_material[metal]',
        'filter_material[glass]'
    ]

    # special tokens
    TOKEN_LABEL_SPECIAL = [
        '<NULL>', '<START>', '<END>', '<UNK>', '<STOP>', 'scene', 'exist'
    ]

    # relate tokens
    TOKEN_LABEL_RELATE = [
        'relate[behind]', 'relate[front]', 'relate[left]', 'relate[right]'
    ]

    ATTER_SIZE = len(
        TOKEN_LABEL_COLOR + TOKEN_LABEL_SHAPE
        + TOKEN_LABEL_SIZE + TOKEN_LABEL_MATERIAL
    )


class TOKEN_LABEL_ASK4:
    # Add new attributes
    # filter_color[yellow]
    # filter_shape[cone]
    # filter_size[xsmall]
    # filter_material[marble]
    # colors
    TOKEN_LABEL_COLOR = [
        'filter_color[blue]', 'filter_color[green]',
        'filter_color[red]', 'filter_color[yellow]',
    ]

    # shapes
    TOKEN_LABEL_SHAPE = [
        'filter_shape[sphere]', 'filter_shape[cylinder]',
        'filter_shape[cube]', 'filter_shape[cone]'
    ]

    # sizes
    TOKEN_LABEL_SIZE = [
        'filter_size[xsmall]', 'filter_size[small]',
        'filter_size[medium]', 'filter_size[large]'
    ]

    # materials
    TOKEN_LABEL_MATERIAL = [
        'filter_material[rubber]',
        'filter_material[metal]',
        'filter_material[glass]',
        'filter_material[marble]',
    ]

    # special tokens
    TOKEN_LABEL_SPECIAL = [
        '<NULL>', '<START>', '<END>', '<UNK>', '<STOP>', 'scene', 'exist'
    ]

    # relate tokens
    TOKEN_LABEL_RELATE = [
        'relate[behind]', 'relate[front]', 'relate[left]', 'relate[right]'
    ]

    ATTER_SIZE = len(
        TOKEN_LABEL_COLOR + TOKEN_LABEL_SHAPE
        + TOKEN_LABEL_SIZE + TOKEN_LABEL_MATERIAL
    )


"""
Utilities for preprocessing sequence data.

Special tokens that are in all dictionaries:

<NULL>: Extra parts of the sequence that we should ignore
<START>: Goes at the start of a sequence
<END>: End of sequence. Goes at the end of a sequence, before <NULL> tokens
<UNK>: Out-of-vocabulary words
<STOP>: End of dialogue (Question Phase --> Inpherence Phase)
"""
SPECIAL_TOKENS = {
    '<NULL>': 0,
    '<START>': 1,
    '<END>': 2,
    '<UNK>': 3,
    '<STOP>': 4,
}


# Source Image Size
SRC_IMG_H = 320  # y-axis
SRC_IMG_W = 480  # x-axis


# Bounding Box Manual Configuration
DEFAULT_H = 120
DEFAULT_W = 120

BBX_BASE_COEF = 10 * 1.2
XSMALL_COEF = 0.25
SMALL_COEF = 0.5
MEDIUM_COEF = 0.75


# Spatial Feature Dim
RICH_DIM = 113
BASE_DIM = 4


# FOR Transformer encoder pos embedding
MAX_POS_EMBEDDING = 100

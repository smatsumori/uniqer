# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

"""
Utilities for working with function program representations of questions.

Some of the metadata about what question node types are available etc are
stored in a JSON metadata file.
"""
from utils.consts import DATASETS


class InvalidQuestionError(Exception):
    # TODO: subdivision
    # Currently, we raise always this error, but it might be informative
    # and useful to create a more detailed error class. Sorry.
    pass


# Handlers for answering questions. Each handler receives the scene structure
# that was output from Blender, the node, and a list of values that were output
# from each of the node's inputs; the handler should return the computed output
# value from this node.


def scene_handler(scene_struct, inputs, side_inputs):
    # Just return all objects in the scene
    return list(range(len(scene_struct['objects'])))


def make_filter_handler(attribute):
    def filter_handler(scene_struct, inputs, side_inputs):
        # assert len(inputs) == 1
        # assert len(side_inputs) == 1
        if not isinstance(inputs, list):
            raise InvalidQuestionError
        elif len(inputs) != 1 or len(side_inputs) != 1:
            raise InvalidQuestionError
        elif not isinstance(inputs[0], list):
            raise InvalidQuestionError

        value = side_inputs[0]
        output = []
        for idx in inputs[0]:
            atr = scene_struct['objects'][idx][attribute]
            if value == atr or value in atr:
                output.append(idx)
        return output
    return filter_handler


def unique_handler(scene_struct, inputs, side_inputs):
    # assert len(inputs) == 1
    if not isinstance(inputs, list):
        raise InvalidQuestionError
    elif len(inputs) != 1:
        raise InvalidQuestionError
    elif not isinstance(inputs[0], list):
        raise InvalidQuestionError

    if len(inputs[0]) != 1:
        return '__INVALID__'
    return inputs[0][0]


def vg_relate_handler(scene_struct, inputs, side_inputs):
    # assert len(inputs) == 1
    # assert len(side_inputs) == 1
    if not isinstance(inputs, list):
        raise InvalidQuestionError
    elif not isinstance(side_inputs, list):
        raise InvalidQuestionError
    elif len(inputs) != 1 or len(side_inputs) != 1:
        raise InvalidQuestionError

    output = set()
    for rel in scene_struct['relationships']:
        if rel['predicate'] == side_inputs[0] and \
           rel['subject_idx'] == inputs[0]:
            output.add(rel['object_idx'])
    return sorted(list(output))


def relate_handler(scene_struct, inputs, side_inputs):
    # assert len(inputs) == 1
    # assert len(side_inputs) == 1
    if not isinstance(inputs, list):
        raise InvalidQuestionError
    elif not isinstance(side_inputs, list):
        raise InvalidQuestionError
    elif len(inputs) != 1 or len(side_inputs) != 1:
        raise InvalidQuestionError
    elif not isinstance(inputs[0], list):
        raise InvalidQuestionError

    relation = side_inputs[0]

    iset = inputs[0]
    if 1 < len(iset):
        # if multiple candidates are given
        _objs = [scene_struct['relationships'][relation][i] for i in iset]
        objs = list(set(sum(_objs, [])))
    elif len(iset) == 1:
        objs = scene_struct['relationships'][relation][iset[0]]
    else:
        objs = []
    return objs


def union_handler(scene_struct, inputs, side_inputs):
    # assert len(inputs) == 2
    # assert len(side_inputs) == 0
    if not isinstance(inputs, list):
        raise InvalidQuestionError
    elif len(inputs) != 2 or len(side_inputs) != 0:
        raise InvalidQuestionError

    return sorted(list(set(inputs[0]) | set(inputs[1])))


def intersect_handler(scene_struct, inputs, side_inputs):
    # assert len(inputs) == 2
    # assert len(side_inputs) == 0
    if not isinstance(inputs, list):
        raise InvalidQuestionError
    elif len(inputs) != 2 or len(side_inputs) != 0:
        raise InvalidQuestionError

    return sorted(list(set(inputs[0]) & set(inputs[1])))


def count_handler(scene_struct, inputs, side_inputs):
    # assert len(inputs) == 1

    if not isinstance(inputs, list):
        raise InvalidQuestionError
    elif len(inputs) != 1:
        raise InvalidQuestionError
    elif not isinstance(inputs[0], list):
        raise InvalidQuestionError

    return len(inputs[0])


def make_same_attr_handler(attribute):
    def same_attr_handler(scene_struct, inputs, side_inputs):
        cache_key = '_same_%s' % attribute
        if cache_key not in scene_struct:
            cache = {}
            for i, obj1 in enumerate(scene_struct['objects']):
                same = []
                for j, obj2 in enumerate(scene_struct['objects']):
                    if i != j and obj1[attribute] == obj2[attribute]:
                        same.append(j)
                cache[i] = same
            scene_struct[cache_key] = cache

        cache = scene_struct[cache_key]
        # assert len(inputs) == 1
        # assert len(side_inputs) == 0
        if not isinstance(inputs, list):
            raise InvalidQuestionError
        elif not isinstance(side_inputs, list):
            raise InvalidQuestionError
        elif len(inputs) != 1 or len(side_inputs) != 0:
            raise InvalidQuestionError

        return cache[inputs[0]]
    return same_attr_handler


def make_query_handler(attribute):
    def query_handler(scene_struct, inputs, side_inputs):
        # assert len(inputs) == 1
        # assert len(side_inputs) == 0
        if not isinstance(inputs, list):
            raise InvalidQuestionError
        elif len(inputs) != 1 or len(side_inputs) != 0:
            raise InvalidQuestionError

        idx = inputs[0]
        obj = scene_struct['objects'][idx]
        # assert attribute in obj
        try:
            val = obj[attribute]
        except ValueError:
            raise InvalidQuestionError

        if type(val) == list and len(val) != 1:
            return '__INVALID__'
        elif type(val) == list and len(val) == 1:
            return val[0]
        else:
            return val
    return query_handler


def exist_handler(scene_struct, inputs, side_inputs):
    # assert len(inputs) == 1
    # assert len(side_inputs) == 0
    if not isinstance(inputs, list):
        raise InvalidQuestionError
    elif len(inputs) != 1 or len(side_inputs) != 0:
        raise InvalidQuestionError
    elif not isinstance(inputs[0], list):
        raise InvalidQuestionError

    return len(inputs[0]) > 0


def equal_handler(scene_struct, inputs, side_inputs):
    # assert len(inputs) == 2
    # assert len(side_inputs) == 0
    if not isinstance(inputs, list):
        raise InvalidQuestionError
    elif len(inputs) != 2 or len(side_inputs) != 0:
        raise InvalidQuestionError

    return inputs[0] == inputs[1]


def less_than_handler(scene_struct, inputs, side_inputs):
    # assert len(inputs) == 2
    # assert len(side_inputs) == 0
    if not isinstance(inputs, list):
        raise InvalidQuestionError
    elif len(inputs) != 2 or len(side_inputs) != 0:
        raise InvalidQuestionError

    return inputs[0] < inputs[1]


def greater_than_handler(scene_struct, inputs, side_inputs):
    # assert len(inputs) == 2
    # assert len(side_inputs) == 0
    if not isinstance(inputs, list):
        raise InvalidQuestionError
    elif len(inputs) != 2 or len(side_inputs) != 0:
        raise InvalidQuestionError

    return inputs[0] > inputs[1]


# Register all of the answering handlers here.
# TODO maybe this would be cleaner with a function decorator that takes
# care of registration? Not sure. Also what if we want to reuse the same engine
# for different sets of node types?
execute_handlers = {
    'scene': scene_handler,
    'filter_color': make_filter_handler('color'),
    'filter_shape': make_filter_handler('shape'),
    'filter_material': make_filter_handler('material'),
    'filter_size': make_filter_handler('size'),
    'filter_objectcategory': make_filter_handler('objectcategory'),
    'unique': unique_handler,
    'relate': relate_handler,
    'union': union_handler,
    'intersect': intersect_handler,
    'count': count_handler,
    'query_color': make_query_handler('color'),
    'query_shape': make_query_handler('shape'),
    'query_material': make_query_handler('material'),
    'query_size': make_query_handler('size'),
    'exist': exist_handler,
    'equal_color': equal_handler,
    'equal_shape': equal_handler,
    'equal_integer': equal_handler,
    'equal_material': equal_handler,
    'equal_size': equal_handler,
    'equal_object': equal_handler,
    'less_than': less_than_handler,
    'greater_than': greater_than_handler,
    'same_color': make_same_attr_handler('color'),
    'same_shape': make_same_attr_handler('shape'),
    'same_size': make_same_attr_handler('size'),
    'same_material': make_same_attr_handler('material'),
}


def answer_question(
    question, scene_struct, all_outputs=False, cache_outputs=True
):
    """
    Use structured scene information to answer a structured question. Most of
    the heavy lifting is done by the execute handlers defined above.

    We cache node outputs in the node itself; this gives a nontrivial speedup
    when we want to answer many questions that share nodes on the same scene
    (such as during question-generation DFS). This will NOT work if the same
    nodes are executed on different scenes.
    """
    # all_input_types, all_output_types = [], []
    node_outputs = []
    for node in question['nodes']:
        if cache_outputs and '_output' in node:
            node_output = node['_output']
        else:
            node_type = node['type']
            # TODO:
            # msg = 'Could not find handler for "%s"' % node_type
            # assert node_type in execute_handlers, msg
            handler = execute_handlers[node_type]
            node_inputs = [node_outputs[idx] for idx in node['inputs']]
            side_inputs = node.get('side_inputs', [])
            node_output = handler(scene_struct, node_inputs, side_inputs)
            if cache_outputs:
                node['_output'] = node_output
        node_outputs.append(node_output)
        if node_output == '__INVALID__':
            break

    if all_outputs:
        return node_outputs
    else:
        return node_outputs[-1]


def insert_scene_node(nodes, idx):
    # First make a shallow-ish copy of the input
    new_nodes = []
    for node in nodes:
        new_node = {
            'type': node['type'],
            'inputs': node['inputs'],
        }
        if 'side_inputs' in node:
            new_node['side_inputs'] = node['side_inputs']
        new_nodes.append(new_node)

    # Replace the specified index with a scene node
    new_nodes[idx] = {'type': 'scene', 'inputs': []}

    # Search backwards from the last node to see which nodes are actually used
    output_used = [False] * len(new_nodes)
    idxs_to_check = [len(new_nodes) - 1]
    while idxs_to_check:
        cur_idx = idxs_to_check.pop()
        output_used[cur_idx] = True
        idxs_to_check.extend(new_nodes[cur_idx]['inputs'])

    # Iterate through nodes, keeping only those whose output is used;
    # at the same time build up a mapping from old idxs to new idxs
    old_idx_to_new_idx = {}
    new_nodes_trimmed = []
    for old_idx, node in enumerate(new_nodes):
        if output_used[old_idx]:
            new_idx = len(new_nodes_trimmed)
            new_nodes_trimmed.append(node)
            old_idx_to_new_idx[old_idx] = new_idx

    # Finally go through the list of trimmed nodes and change the inputs
    for node in new_nodes_trimmed:
        new_inputs = []
        for old_idx in node['inputs']:
            new_inputs.append(old_idx_to_new_idx[old_idx])
        node['inputs'] = new_inputs

    return new_nodes_trimmed


def is_degenerate(
        question, scene_struct, answer=None, verbose=False
):
    """
    A question is degenerate if replacing any of its relate nodes with a scene
    node results in a question with the same answer.
    """
    if answer is None:
        answer = answer_question(question, scene_struct)

    for idx, node in enumerate(question['nodes']):
        if node['type'] == 'relate':
            new_question = {
                'nodes': insert_scene_node(question['nodes'], idx)
            }
            new_answer = answer_question(new_question, scene_struct)
            if verbose:
                print('here is truncated question:')
                for i, n in enumerate(new_question['nodes']):
                    name = n['type']
                    if 'side_inputs' in n:
                        name = '%s[%s]' % (name, n['side_inputs'][0])
                    print(i, name, n['_output'])
                print('new answer is: ', new_answer)

            if new_answer == answer:
                return True

    return False


def node_shallow_copy(node):
    new_node = {
        'type': node['type'],
        'inputs': node['inputs'],
    }
    if 'side_inputs' in node:
        new_node['side_inputs'] = node['side_inputs']
    return new_node


def find_filter_options(object_idxs, scene_struct, metadata):
    # Keys are tuples (size, color, shape, material) (where some may be None)
    # and values are lists of object idxs that match the filter criterion

    if '_filter_options' not in scene_struct:
        precompute_filter_options(scene_struct, metadata)

    attribute_map = {}
    object_idxs = set(object_idxs)
    for k, vs in scene_struct['_filter_options'].items():
        attribute_map[k] = sorted(list(object_idxs & vs))
    return attribute_map


def precompute_filter_options(scene_struct, metadata):
    # Keys are tuples (size, color, shape, material) (where some may be None)
    # and values are lists of object idxs that match the filter criterion
    attribute_map = {}

    if metadata['dataset'] in DATASETS:
        attr_keys = ['size', 'color', 'material', 'shape']
    else:
        assert False, 'Unrecognized dataset'

    # Precompute masks
    masks = []
    for i in range(2 ** len(attr_keys)):
        mask = []
        for j in range(len(attr_keys)):
            mask.append((i // (2 ** j)) % 2)
        masks.append(mask)

    for object_idx, obj in enumerate(scene_struct['objects']):
        if metadata['dataset'] in DATASETS:
            keys = [tuple(obj[k] for k in attr_keys)]

        for mask in masks:
            for key in keys:
                masked_key = []
                for a, b in zip(key, mask):
                    if b == 1:
                        masked_key.append(a)
                    else:
                        masked_key.append(None)
                masked_key = tuple(masked_key)
                if masked_key not in attribute_map:
                    attribute_map[masked_key] = set()
                attribute_map[masked_key].add(object_idx)

    scene_struct['_filter_options'] = attribute_map

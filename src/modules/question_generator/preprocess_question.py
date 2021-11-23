import json

import numpy as np
import h5py

from . import program_manipulates_func as pmf
from .program_manipulates_func import build_vocab, program_to_str

# logging
from logging import getLogger
logger = getLogger(__name__)

# Default Parameters
# it's not elegant to write them down here
UNK_THRESHOLD = 1


def preprocess_question(
        json_path, save_h5_path, vocab_json, train=False, max_num_obj=10,
):
    """
    This function is based on `preprocess_questions.py`
    ref: https://github.com/facebookresearch/clevr-iep

    Parameters
    ----------
    json_path : str
        json_question file path
    save_h5_path : str
        savepath
    vocab_json : str
        vocab_json path that contains the vocabulary if train==False
        when train == True, the path will be used to save the information
    train : bool, default is False
    max_num_obj : int, default is 10
    """
    logger.info('Loading questions')
    with open(json_path, 'r') as f:
        questions = json.load(f)['questions']

    # create vocab.json for training dataset
    if train:
        if 'answer' in questions[0]:
            # NOTE: this will only works for 'exsits'
            # answer_token_to_idx =
            # build_vocab((q['answer'] for q in questions))
            ex_answer_token_to_idx = {
                'exist_no': 0, 'exist_yes': 1
            }
        # INFO In the original code, `answer` were also tokenized, but since
        # the `answer` in our dataset are ObjectSet, we remove the tokenization
        # part from the code
        question_token_to_idx = build_vocab(
            (q['question'] for q in questions),
            min_token_count=UNK_THRESHOLD,
            punct_to_keep=[';', ','], punct_to_remove=['?', '.']
        )

        all_program_strs = []
        for q in questions:
            if 'program' not in q:
                continue
            program_str = program_to_str(q['program'])
            if program_str is not None:
                all_program_strs.append(program_str)
        program_token_to_idx = build_vocab(all_program_strs)

        logger.info('program: {}'.format(program_token_to_idx))
        vocab = {
            'question_token_to_idx': question_token_to_idx,
            'program_token_to_idx': program_token_to_idx,
            'ex_answer_token_to_idx': ex_answer_token_to_idx
        }

        with open(vocab_json, 'w') as f:
            logger.info(f'Vocab.json dumped to {vocab_json}')
            json.dump(vocab, f)

    # for validation dataset, test dataset
    else:
        with open(vocab_json, 'r') as f:
            vocab = json.load(f)

    # encode questions and programs
    logger.info('Encoding Questions and Programs')
    questions_encoded = []
    programs_encoded = []
    obj_answers = []
    question_families = []
    orig_idxs = []
    image_idxs = []
    ex_answers = []
    for orig_idx, q in enumerate(questions):
        question = q['question']

        orig_idxs.append(orig_idx)
        image_idxs.append(q['image_index'])
        if 'question_family_index' in q:
            question_families.append(q['question_family_index'])
        question_tokens = pmf.tokenize(
            question,
            punct_to_keep=[';', ','],
            punct_to_remove=['?', '.']
        )
        question_encoded = pmf.encode(
            question_tokens,
            vocab['question_token_to_idx'],
            allow_unk=False
        )
        questions_encoded.append(question_encoded)

        if 'program' in q:
            # Drop `exist` node
            program = q['program'][:-1]
            ans_objs = program[-1]['_output']
            ans_objs_enc = [
                1 if i in ans_objs else 0 for i in range(max_num_obj)
            ]
            obj_answers.append(ans_objs_enc)

            program_str = program_to_str(program)
            program_tokens = pmf.tokenize(program_str)
            program_encoded = pmf.encode(
                program_tokens,
                vocab['program_token_to_idx']
            )
            programs_encoded.append(program_encoded)

        if 'answer' in q:
            # TODO(->@smatsumori): refactor
            ans = {False: 'exist_no', True: 'exist_yes'}[q['answer']]
            ex_answers.append(vocab['ex_answer_token_to_idx'][ans])

    # Pad encoded questions and programs
    max_question_length = max(len(x) for x in questions_encoded)
    for qe in questions_encoded:
        while len(qe) < max_question_length:
            qe.append(vocab['question_token_to_idx']['<NULL>'])

    if len(programs_encoded) > 0:
        max_program_length = max(len(x) for x in programs_encoded)
        for pe in programs_encoded:
            while len(pe) < max_program_length:
                pe.append(vocab['program_token_to_idx']['<NULL>'])

    # Create h5 file
    logger.info('Writing output')
    questions_encoded = np.asarray(questions_encoded, dtype=np.int32)
    programs_encoded = np.asarray(programs_encoded, dtype=np.int32)
    obj_answers = np.array(obj_answers, dtype=np.int32)
    logger.info(f'Encoded Question shape: {questions_encoded.shape}')
    logger.info(f'Encoded Program shape: {programs_encoded.shape}')
    logger.info(f'Encoded Object Answers shape: {obj_answers.shape}')
    with h5py.File(save_h5_path, 'w') as f:
        f.create_dataset('questions', data=questions_encoded)
        f.create_dataset('image_idxs', data=np.asarray(image_idxs))
        f.create_dataset('orig_idxs', data=np.asarray(orig_idxs))

        if len(programs_encoded) > 0:
            f.create_dataset('programs', data=programs_encoded)
        if len(question_families) > 0:
            f.create_dataset(
                'question_families',
                data=np.asarray(question_families)
            )
        if len(obj_answers) > 0:
            f.create_dataset('answers', data=obj_answers)

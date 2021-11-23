import torch
import numpy as np

# Import consts
from utils.consts import TOKEN_LABEL_ASK3
from utils.consts import TOKEN_LABEL_ASK4
from utils.consts import TOKEN_LABEL_CLEVR

# logging
from logging import getLogger
logger = getLogger(__name__)

# Available Rewards
# 1. Correct reward
# 2. Informativeness reward
# 3. Progressive reward
# 4. Optimize reward
# 5. Descriptive reward
# 6. Turn discount penalty
# 7. Turn penalty


def calculate_batch_reward_sp(
    o_pred_probs, prev_o_pred_probs, n_turns, batch_size,
    is_currq_eoq, is_currq_eod, state, tgt_id, imfn, params, oracle,
):
    """
    Calculate batch reward

    # TODO: update docstrings

    Parameters
    ----------
    o_pred_probs :
        o_pred_probs
    prev_o_pred_probs :
        prev_o_pred_probs
    n_turns :
        n_turns
    batch_size :
        batch_size
    is_currq_eoq :
        is_currq_eoq
    is_currq_eod :
        is_currq_eod
    state :
        state
    tgt_id :
        Target object id
    imfn :
        imfn
    params :
        params
    oracle :
        oracle

    Returns
    -------
    reward : np.array
        (shape: n_steps, n_batches)
    """
    # The pseudo variables are set to use token wise
    # calculate_batch_reward function.
    pseudo_n_step = 1
    # If true set a zero index
    pseudo_c_eoq_idxs = [0 if c else -1 for c in is_currq_eoq]
    pseudo_c_eod_idxs = [0 if c else -1 for c in is_currq_eod]

    _zip_r = calculate_batch_reward(
        o_pred_probs, prev_o_pred_probs, n_turns, batch_size,
        pseudo_c_eoq_idxs, pseudo_c_eod_idxs, state, tgt_id,
        imfn, params, oracle, pseudo_n_step
    )

    def helper(r):
        # (n_timesteps, n_batches) -> (n_batches)
        return np.sum(r, axis=0)
    zip_r = [helper(rw) for rw in _zip_r]
    return zip_r


def calculate_batch_reward(
    o_pred_probs, prev_o_pred_probs, n_turns, batch_size,
    c_eoq_idxs, c_eod_idxs, state, tgt_id, imfn, params, oracle,
    n_token_step
):
    """
    Calculates batch reward.

    # TODO: update docstrings

    Parameters
    ----------
    o_pred_probs : torch.tensor
        shape: (n_batches, n_max_objects)
    prev_o_pred_probs :
        The previous o_pred_probs.
    n_turns : list of int
        The net number of questions in current batches.
    batch_size : int
        The current batch size.
    c_eoq_idxs : list of int
        The current end of question indices.
        If no eoq in the current question generation,
        then -1 will be inserted.
    c_eod_idxs : list of int
        The current end of dialogue indices.
        If no eod in the current question gneration,
        then -1 will be inserted.
    state : dict
    tgt_id : list of int
    imfn : list of str
        A list of filenames.
    params : Parameters for reward calculations.
    oracle : Oracle instance
    n_step : int
        The max number of current question generation step.
    n_token_step : int
        The max number of tokens in the current question.

    Returns
    -------
    """
    batch_len = len(o_pred_probs)
    assert batch_len == batch_size

    # compute rewards.shape: (n_batches, n_timesteps)
    corr_r = []
    info_r = []
    prog_r = []
    opti_r = []
    turn_p = []
    disc_p = []
    desc_r = []

    def helper(hot_idxs, r):
        return [r if i in hot_idxs else 0 for i in range(n_token_step)]

    # calculate rewards on each batch
    for b in range(batch_len):
        # get eoq and eod indices in current generation steps
        eoqi, eodi = c_eoq_idxs[b], c_eod_idxs[b]

        # List of available rewards:
        # corr_r, info_r, prog_r, opti_r, turn_p, disc_p, desc_r
        # TODO: implement disc_p. rename
        c, i, p, o, t, di, de = calculate_reward(
            state['curr_question'][b],
            state['n_obj'][b],
            state['dialogue'][b],
            o_pred_probs[b], prev_o_pred_probs[b],
            0 <= eoqi, 0 <= eodi, n_turns[b],
            oracle, imfn[b],
            tgt_id[b], params
        )
        cb = helper([eodi], c)  # rewards are given when EOD
        ib = helper([eoqi], i)  # rewards are given when eoq
        pb = helper([eoqi], p)  # rewards are given when eoq
        ob = helper([eodi], o)  # rewards are given when EOD
        tb = helper([eoqi], t)  # rewards are given when eoq
        dib = helper([eoqi], di)  # rewards are given when eoq
        deb = helper([eoqi], de)  # rewards are given when eoq
        # rb = [sum(z) for z in zip(cb, ib, pb, ob, tb)]

        corr_r.append(cb)
        info_r.append(ib)
        prog_r.append(pb)
        opti_r.append(ob)
        turn_p.append(tb)
        disc_p.append(dib)
        desc_r.append(deb)
        # rewards.append(rb)

    # transpose to (n_timesteps, n_batches)
    _corr_r = np.array(corr_r).T
    _info_r = np.array(info_r).T
    _prog_r = np.array(prog_r).T
    _opti_r = np.array(opti_r).T
    _turn_p = np.array(turn_p).T
    _disc_p = np.array(disc_p).T
    _desc_r = np.array(desc_r).T
    # _rewards = np.array(rewards).T
    return _corr_r, _info_r, _prog_r, _opti_r, _turn_p, _disc_p, _desc_r


def calculate_reward(
    curr_question, n_obj, dialogue,
    o_pred_prob, prev_o_pred_prob, eoq, eod, n_turns,
    oracle, imfn, tgt_id, params
):
    """
    Calculates rewards. This function works for single batch.
    Calculation for batches is implemented in `calculate_batch_reward`.

    Parameters
    ----------
    curr_question : list of int
        The current question for a batch.
    n_obj : int
        The number of objects in the scene.
    dialogue :
        The current dialogue history.
    o_pred_prob : torch.tensor
        shape (n_max_obj)
    prev_o_pred_prob :
        The previous predicted object probabilities.
        shape (n_max_obj)
    eoq : bool
        end of question flag
    eod : bool
        end of dialogue flag
    n_turns : list of int
        the number of turns for the current episode
    oracle : Oracle instance
    imfn : str
        An image filename.
    tgt_id : int
        The target object id.
    params : dict
        Parameters for rewarding.


    Returns
    -------
    corr_r : float
        reward depends on correctly discorvered referenced object
    info_r : float
        reward depends on a question was meaningful
    prog_r : float
        reward depends on guesser's internal state
    opti_r : float
        be rewarded when the number of questions is optim
    turn_p : float
        be negatively-rewarded when the agent consumes the question turn
    disc_p : float
        be negatively-rewarded gradually when the agent consumes
        the question trun
    desc_r : float
        be rewarded when the generated question describes object
        in detail.
    """
    # TODO: dict management ?
    corr_r = 0.0
    info_r = 0.0
    prog_r = 0.0
    opti_r = 0.0
    turn_p = 0.0
    disc_p = 0.0  # turn discount
    desc_r = 0.0  # descriptive reward

    # TODO: undefined
    if eoq:
        # image_filename required
        info_r = get_informativeness_reward(
            curr_question, n_obj, imfn, oracle, eta=0.2  # TODO: params
        )
        prog_r = get_progressive_reward(
            o_pred_prob, prev_o_pred_prob, tgt_id
        )
        turn_p = get_turn_penalty()
        desc_r = get_descriptive_reward(
            curr_question, oracle,
            params['image_data_version'],
            params['descriptive_coeff']
        )
    if eod:
        corr_r = get_corr_reward(
            o_pred_prob, tgt_id
        )
        opti_r = get_opti_reward(
            n_obj, len(dialogue)
        )
        disc_p = get_turn_discount(
            n_turns, params['n_max_questions'], params['turn_discount_coeff']
        )

    return corr_r, info_r, prog_r, opti_r, turn_p, disc_p, desc_r


# R1. Correct Reward
def get_corr_reward(o_pred_prob, tgt_id):
    """
    The basic reward. If the prediction is correct, give 1.0 else 0.0.

    Supposed to be called in `calculate_reward()` when `eod` is True.

    Parameters
    ----------
    o_pred_prob : torch.tensor
        shape : max_n_object
    tgt_id : int
        the target id for the current batch
    """
    # NOTE: Do not count the last `eod` generated turn.
    #       Assuming there is no question - answer in the turn
    o_pred = torch.argmax(o_pred_prob)
    # TODO: batch
    if o_pred == tgt_id:
        r = 1.0
        # if turn discount is enabled
    else:
        r = 0.0
    return r


# R2. Informativeness reward
def get_informativeness_reward(
    curr_question, n_obj, imfn, oracle, eta
):
    """
    If the answer to a question is same for all objects, then the question
    is meaningless. Therefore, give a const. reward for the question with
    generating at least one different answer.

    Supposed to be called in `calculate_reward()` when `eoq` is True.

    Parameters
    ----------
    eta : float, default is 0.1
        TODO: parameter search
        const informativeness reward

    Returns
    -------
    infor : list of floats
        Informativeness rewards. (batch)
    """
    if oracle.meaningful_question(
        imfn, curr_question, n_obj, require_pp=True
    ):
        return eta
    else:
        return 0.0


# R3. Progressive reward
# TODO: check if this works
def get_progressive_reward(
    o_pred_prob, prev_o_pred_prob, tgt_id
):
    """
    Progressive Reward

    This reward should be given each time `o_pred_prob` is updated.
    For now, this is limited when eoq is generated.
    WARNING: This reward could be negative.

    Parameters
    ----------
        o_pred_prob : torch.tensor
        prev_o_pred_prob : torch.tensor
        info : dict

    """
    # handle if no previous o_preb_prob
    if prev_o_pred_prob is None:
        return 0.0
    elif type(prev_o_pred_prob) == list and not prev_o_pred_prob:
        return 0.0
    else:
        return (o_pred_prob[tgt_id] - prev_o_pred_prob[tgt_id]).item()


# R4. Optimize reward
def get_opti_reward(n_obj, curr_dial_len):
    """
    Question number optimal Reward

    Supposed to be called in `calculate_reward()` when `eod` is True.
    1. Optimal Question Times Reward
    Assuming that the minimum number of questions required for the
    prediction is log_2_(n_objct), the reward is determined by the
    difference between the actual number of questions

    2. (temporal / experimental)
    Giving additional negative rewards if no questions are asked to avoid
    random prediction.

    Parameters
    ----------
    n_obj : LongTensor
        Cast to numpy?
    curr_dial_len : int
        The current length of dialogue

    Returns
    -------
    opti_r : float
    """
    # 1.
    opti_r = 0.0

    # optimal number of questions
    opt_n_q = torch.ceil(torch.log2(n_obj.type(torch.FloatTensor)))
    opti_r += -torch.abs(opt_n_q - curr_dial_len) / opt_n_q

    # 2.
    # TODO: should use question_count ?
    opti_r = opti_r - 1 if curr_dial_len == 0 else opti_r
    return opti_r.item()


# R5. Descriptive reward
def get_descriptive_reward(
    curr_question, oracle, image_data_version, coeff=0.5
):
    """ Gets descriptive rewards.
    Give rewards when the attributes used in the question is rare.
    That is the higher reward is given for the rarer attributes
    such as colors and otherwise lower for the more common attributes
    such as materials and shapes.
    N_color: 8
    N_shape: 3
    N_material: 2
    N_size: 2
    Total: 15 attributes

    Parameters
    ----------
    curr_question : list of int
    oracle : Oracle
    image_data_version : str
    coeff : float
    """
    # set tokens
    if image_data_version == 'CLEVR_Ask3':
        tk = TOKEN_LABEL_ASK3
    elif image_data_version == 'CLEVR_Ask4':
        tk = TOKEN_LABEL_ASK4
    else:
        tk = TOKEN_LABEL_CLEVR
    # curr_q: list of int (unique)
    curr_qi = set(curr_question)
    q_len = len(curr_question)

    r = 0.0
    for qi in curr_qi:
        # get question index from question string
        qs = oracle.vocab['program_idx_to_token'][qi]
        if qs in tk.TOKEN_LABEL_COLOR:
            r += len(tk.TOKEN_LABEL_COLOR) / tk.ATTER_SIZE
        elif qs in tk.TOKEN_LABEL_SIZE:
            r += len(tk.TOKEN_LABEL_SIZE) / tk.ATTER_SIZE
        elif qs in tk.TOKEN_LABEL_MATERIAL:
            r += len(tk.TOKEN_LABEL_MATERIAL) / tk.ATTER_SIZE
        elif qs in tk.TOKEN_LABEL_SHAPE:
            r += len(tk.TOKEN_LABEL_SHAPE) / tk.ATTER_SIZE
        elif qs in tk.TOKEN_LABEL_SPECIAL:
            pass
        elif qs in tk.TOKEN_LABEL_RELATE:
            pass
        else:
            # TODO: define exception
            logger.error('Unknown token <{}>. Fix this.'.format(qs))
    # NOTE: ideally we shoud also normalize this with
    # maximum length of steps in a question
    return (r * coeff) / q_len


# R6. Turn discount penalty
def get_turn_discount(n_turns, n_max_questions, coeff=0.5):
    """
    Turn discount penalty given at the end of the dialogue.

    Parameters
    ----------
    coeff : float
        A penalty coefficient.
    """
    # Discount the value by the number of turns.
    # This encourages to reduce the redundant questions.
    # TODO: batch mode. Also you can get the n_turns from self.__dialogue
    return -1.0 * coeff * n_turns / n_max_questions


# R7. Turn penalty
def get_turn_penalty(penalty=0.2):
    """
    Not at the end of the dialogue (proposed in get_corr_reward()),
    but as an intermediate reward for each eoq token.

    Supposed to be called in `calculate_reward()` when `eoq` is True.

    Parameters
    ----------
    penalty : float
        TODO: parameter search
              const ? increasing ?
    """
    return -1 * penalty

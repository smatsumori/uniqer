import os
import json

import numpy as np

import modules.question_generator.program_manipulates_func as pmf
from utils.utils import cos_sim
from modules.oracle.oracle import Oracle
from modules.rl.environment import Environment
from functools import reduce

# TODO: update SP mode
from modules.rl.envrewards import calculate_batch_reward_sp

# logging
from logging import getLogger
logger = getLogger(__name__)


class NoTargetCandsWarning(Exception):
    # Used to move forward to next scene when there are no reference object
    # candidates (New Object Evaluation)
    pass


class EnvironmentSp(Environment):
    """
    An environment class for SplitNetwork. Inherits Enviornment class.

    * TODO: refactoring (inherits functions)
    """
    def __init__(
            self, args, is_valid=False, is_eval_obj=False, is_eval_img=False
    ):
        """
        Parameters
        ----------
        args : ArgumentParser
        is_valid : bool, default is False
            Validation mode flag
        is_eval_obj : bool, default is False
            New object evaluation mode flag
        is_eval_img : bool, default is False
            New image evaluation mode flag
        """
        # Settings here
        self.args = args
        # new object evalution
        self.eval_obj_mode = is_eval_obj

        # TODO: remove flags?
        self.proposed_model_mode = args.proposed_model_mode
        self.beta_generator_mode = args.beta_generator_mode

        self.path_metadata = args.metadata_path
        imdp = os.path.join('./data', args.image_data_version)
        scene_train_path = os.path.join(imdp, args.scene_train_path)
        scene_val_path = os.path.join(imdp, args.scene_val_path)
        scene_test_path = os.path.join(imdp, args.scene_test_path)

        if is_valid:
            # validation dataset
            self.path_scenes = scene_val_path
        elif is_eval_img:
            # evaluation dataset
            self.path_scenes = scene_test_path
        else:
            # training dataset (for train_mode, is_eval_obj mode)
            self.path_scenes = scene_train_path

        self.vocab_path = args.vocab_path

        # Get restricted objects
        if is_valid or is_eval_img:
            # Case1: No restricted objects
            # The "restricted object" setting is not applicable to
            # validation mode or evaluation mode (new image).
            rest_objs = None
        else:
            # Case2: Restricted objects (train or is_eval_obj)
            # Restricted objects are only for train dataset
            # with new object evaluation mode
            root_savepath = os.path.join(
                './data', args.question_data_version, 'savedata'
            )
            rest_path = os.path.join(root_savepath, 'restricted.json')
            rest_objs = self.__get_restricted_objects(rest_path)

        # Initialize oracle
        self.oracle = Oracle(
            self.path_metadata,
            self.path_scenes,
            self.vocab_path,
            args.image_data_version,
            rest_objs,
            args.idk_mode,
        )
        self.tokens = self.oracle.vocab['program_token_to_idx']

        # image_target_dict: {'image_filename': reference_object_id, ...}
        self.image_target_dict = self.oracle.ref_objects

        # reward type flag
        self.activ_info_r = args.informativeness_reward
        self.activ_prog_r = args.progressive_reward
        self.activ_opti_r = args.optimal_reward
        self.activ_turn_p = args.turn_penalty
        self.activ_disc_p = args.discounted_corr
        self.activ_desc_r = args.descriptive_reward

        # coefficients
        self.descriptive_coeff = args.descriptive_coeff
        self.turn_discount_coeff = args.turn_discount_coeff

        # placeholders
        self.o_pred_probs = []  # (n_batches, n_question, n_obj)
        self.prev_o_pred_probs = []  # (n_batches, n_objs)
        self.results = []  # (n_batches)  True: success, False, fail
        self.split_actions = []  # (n_natches, n_question, n_bits)

        # Make sure to initialize batch_size in init_batches
        self.batch_size = None
        self.n_max_questions = args.n_max_questions

    def init_batches(self, n_batches):
        """init_batches.

        Parameters
        ----------
        n_batches :
            n_batches
        """
        self.batch_size = n_batches
        self.o_pred_probs = [[] for _ in range(n_batches)]
        self.prev_o_pred_probs = [[] for _ in range(n_batches)]
        self.results = [False for _ in range(n_batches)]
        self.split_actions = [[] for _ in range(n_batches)]

    def __get_restricted_objects(self, path):
        with open(path, 'r') as f:
            robj = json.load(f)
        if not self.eval_obj_mode:
            # training mode
            return robj['train']
        else:
            # evaluation new object mode
            return robj['test']

    def qstep(self):
        """
        Initialize eoq_list, eod_list
        """
        raise NotImplementedError

    def set_state(
        self, data: dict, dialogue: list,
        curr_question: list, curr_answer: list, idxs: list
    ):
        self.__context = data['image_feature'][idxs]
        self.__cropped = data['obj_image'][idxs]
        self.__n_obj = data['n_obj'][idxs]
        self.__dialogue = dialogue  # dialogue contains both ques and ans
        self.__curr_question = curr_question
        self.__curr_answer = curr_answer  # 0: 'no', 1: 'yes', 2: 'none'

        if self.proposed_model_mode:
            self.__obj_f = data['obj_features'][idxs]
            self.__org_spatial_f = data['org_spatial_features'][idxs]

            if self.beta_generator_mode:
                self.__g_obj_features = data['g_obj_features'][idxs]
            else:
                self.__g_obj_features = None
        else:
            self.__spatial = data['spatial'][idxs]

    @property
    def state(self):
        if self.proposed_model_mode:
            state = {
                'context': self.__context,
                'cropped': self.__cropped,
                'obj_features': self.__obj_f,
                'n_obj': self.__n_obj,
                'dialogue': self.__dialogue,
                'curr_question': self.__curr_question,
                'curr_ans': self.__curr_answer,
                'org_spatial_features': self.__org_spatial_f,
                'g_obj_features': self.__g_obj_features,
            }
        else:
            state = {
                'context': self.__context,
                'cropped': self.__cropped,
                'spatial': self.__spatial,
                'n_obj': self.__n_obj,
                'dialogue': self.__dialogue,
                'curr_question': self.__curr_question,
                'curr_ans': self.__curr_answer,
            }
        return state

    def next_scene(self, data: dict) -> dict:
        """
        Update info for next image (next episode)

        Parameters
        ----------
        data : dict
            - image : image itself
            - image_feature : pre-calculcated image features
            - image_filename : image_filename
            - n_obj : the number of object (will be used to making objects)
            - obj_image : NotImplemented
            - obj_feature : pre-calculcated image features
            - spatial : spatial info of each object (x, y, w, h)

        Returns
        -------
        state : dict
        """
        self.imfn = []
        _imfn = [os.path.basename(x) for x in data['image_filename']]
        idxs = []
        # Check candidate errors
        for i, fn in enumerate(_imfn):
            if self.image_target_dict[fn] == -1:
                # Skip if no target candidate
                continue
            else:
                self.imfn.append(fn)
                idxs.append(i)
        if len(self.imfn) < len(_imfn):
            logger.debug(
                'Skipped {} scenes.'.format(len(_imfn) - len(self.imfn))
            )
        self.batch_size = len(self.imfn)

        self.scene_id = [
            self.get_sceneidx_from_image_name(x) for x in self.imfn
        ]
        self.tgt_id = [self.image_target_dict[x] for x in self.imfn]

        # TODO: dialogue, curr_question, curr_answer batch impl.
        # TODO: torch.tensor ? list?
        self.set_state(
            data,
            dialogue=[[] for _ in range(self.batch_size)],
            curr_question=[
                [self.tokens['<START>']] for _ in range(self.batch_size)
            ],
            curr_answer=[2 for _ in range(self.batch_size)],  # 2: None
            idxs=idxs
        )
        return self.state, self.batch_size

    def update(
        self, actions, bit_actions, questions,
        current_q_eod, is_last_question=False,
    ):
        """
        Based on the question update the current answer.
        This function is called on every question generation.

        Parameters
        ----------
        actions : list of int (batch of actions)
            length of list : self.batch_size
        bit_actions : list of list (batch of split actions)
            length of list : self.batch_size
        questions : list of list (batch of questions)
            length of list : self.batch_size
        current_q_eod : list of bool
            Indicates if the question generation steps were terminated
            in the current question generation.
            This is equivalent to producing <STOP> token.
        is_last_question : bool
            Indicates whether the action belongs to the last question.

        Returns
        -------
        state : dict
        """

        for i, (a, ba, q) in enumerate(zip(actions, bit_actions, questions)):
            # 0. Append the current question. i is a batch_id.
            self.__curr_question[i] = q
            self.split_actions[i].append(ba)

            # Check if the question generation has terminated.
            if a is None:
                self.__curr_answer[i] = 2  # set answer invalid
                continue

            # 1. Handle violated cases
            # Note that all violated cases will get invald answers.
            # [PREM1] The token for the last questions should be <STOP>.
            if is_last_question and not current_q_eod[i]:
                self.__curr_answer[i] = 2  # set answer invalid
                continue
            elif not self.is_question_valid(q):
                self.__curr_answer[i] = 2  # set answer invalid

            # 2. Set the answer to the questions if it's sound
            if current_q_eod[i]:
                # Regards <STOP> was produced in the current qgen
                # submits the answer
                self.__curr_answer[i] = 2  # set answer invalid (N/A)
            else:
                # TODO: check the generated syntax of the question
                self.__curr_answer[i] = self.get_answer(self.imfn[i], q)

        state = self.state
        return state

    def is_question_valid(self, question):
        """Returns if the question is valid.
        Herein the following premises are considered:
        1. <START> token should be placed at first.
        2. <STOP> token should be produced alone.
            (i.e. <START><STOP> is the only acceptable case)
        3. <END> token should be produced except in the prem 2.

        Parameters
        ----------
        question : list

        Returns
        -------
        check : bool
            True: is valid, False: is not valid
        """
        # Check if the question starts properly
        assert question[0] == self.tokens['<START>']
        check = True
        if self.tokens['<STOP>'] in question:
            if question[1] != self.tokens['<STOP>']:
                check = False
        elif self.tokens['<END>'] not in question:
            # Invalid if no <END> token in the question
            check = False
        return check

    def get_answer(self, imfn, question):
        """
        Parameters
        ----------
        imfn : str
            image file name
        question : list of int

        Returns
        -------
        answer : int
            {0: 'no', 1: 'yes', 2: 'none'}
            invalid?
        """
        return self.oracle.answering_yn(imfn, question, require_pp=True)

    def reset_batch_questions(self):
        for b in range(self.batch_size):
            self.reset_question(b)

    def reset_question(self, idx):
        """
        Parameters
        ----------
        idx : int
            batch index
        """
        self.__dialogue[idx].append(
            # Append current question and answer pair
            (self.__curr_question[idx].tolist(), self.__curr_answer[idx])
        )
        # N.B. <START> token is produced in looper.get_env_questions
        self.__curr_question[idx] = [self.tokens['<START>']]
        self.__curr_answer[idx] = 2  # TODO: remove the magic number

    def reinit_reference_object(self):
        """
        changing the reference objects for each epoch
        """
        self.oracle.reset_ref_objects()
        self.image_target_dict = self.oracle.ref_objects

    # TODO: deprecated?
    def get_len_cur_question(self, idx):
        """
        Parameters
        ----------
        idx : int
            batch index
        """
        # <START> token is not our agent's action
        # thus subtracting one is required
        return len(self.__curr_question[idx]) - 1

    def get_batch_reward(
            self, o_pred_probs, question_counts, q_count,
            is_currq_eoq, is_currq_eod
    ):
        """
        get_batch_reward.

        Parameters
        ---------
        o_pred_probs : list
            The target prediction probabilities.
        question_counts : list
            The number of generated questions for each batch.
            N.B. This does only include questions BEFORE eod.
        q_count : int
            The count of current question in train_rl loop.
            This is used to eliminate the random prediction.
        is_currq_eoq : list
            A list indicates whether the current question
            produces eoq.
        is_currq_eod : list
            A list indicates whether the current question
            produces eod.
        """
        # Parameters for reward calculations
        params = {
            'n_max_questions': self.n_max_questions,
            'descriptive_coeff': self.descriptive_coeff,
            'turn_discount_coeff': self.turn_discount_coeff,
            'image_data_version': self.args.image_data_version,
        }
        n_qcounts = [i for i in question_counts]
        zip_rewards = calculate_batch_reward_sp(
            o_pred_probs, self.prev_o_pred_probs, n_qcounts, self.batch_size,
            is_currq_eoq, is_currq_eod, self.state,
            self.tgt_id, self.imfn, params, self.oracle
        )
        # TODO: Refactor. List iteration might cause degrade in performance.
        # Register predicted probabilities (batch)
        for ib, prob in enumerate(o_pred_probs):
            # cuda tensor -> list
            self.o_pred_probs[ib].append(prob.detach().cpu().tolist())
        self.prev_o_pred_probs = o_pred_probs  # (n_batches, n_max_objs)

        # rewards.shape: (n_batches)
        (
            _corr_r, _info_r, _prog_r, _opti_r, _turn_p, _disc_p, _desc_r
        ) = zip_rewards

        # TODO: move this outside of get_rewards?
        # Update results (success or fail) based on correct reward
        success_idxs = np.where(0 < _corr_r)[0]  # get positive batch idxs
        for i in success_idxs:
            self.results[i] = True

        # Do not give rewards for random prediction (q_count = 1)
        _corr_r *= np.array(q_count != 1)
        # Pass the rewards based on flags
        _info_r *= np.array(self.activ_info_r)
        _prog_r *= np.array(self.activ_prog_r)
        _opti_r *= np.array(self.activ_opti_r)
        _turn_p *= np.array(self.activ_turn_p)
        _disc_p *= np.array(self.activ_disc_p)  # Turn discounted
        _desc_r *= np.array(self.activ_desc_r)  # Descriptive

        zip_rewards = (
            _corr_r + _info_r + _prog_r + _opti_r
            + _turn_p + _disc_p + _desc_r,
            _corr_r, _info_r, _prog_r, _opti_r, _turn_p, _disc_p, _desc_r
        )
        return zip_rewards

    # utils
    def get_sceneidx_from_image_name(self, image_fn: str) -> int:
        return self.oracle.image_name_to_scene[image_fn]

    def get_token_variety(self):
        """
        Return the token variety of the current batch episodes.
        """
        # TODO: Too deep. Need to be sophisticated.
        # 1. Compute bag of words
        bow = {}
        for dials in self.__dialogue:
            for dial in dials:
                token_idxs, _ = dial[0], dial[1]  # get question token
                for tk in token_idxs:
                    try:
                        bow[tk] += 1
                    except KeyError:
                        bow[tk] = 1

        # 2. Calculate entropy
        n_tokens = reduce(lambda a, b: a + b, bow.values())
        ent = 0.0
        for k, v in bow.items():
            p = v / n_tokens  # probability p(x)
            ent += - p * np.log(p)  # compute p * log(1/p)
        return ent

    def get_question_variety(self, idx):
        """
        Return question variety score (that is caluculated by ...) for the
        `__dialogue`
        * Metric : Cosine-Similarity

        TODO: better metrics ?

        Parameters
        ----------
        idx : int
            batch index

        Returns
        -------
        score : float
        """
        len_dialogue = len(self.__dialogue[idx])
        if len_dialogue < 2:
            # To avoid "Zero Division", but...
            return 0

        # [[token_idx, token_idx, ...], [...], [...], ...]
        # 1. Bag-of-words vectorization
        bow = np.zeros((len_dialogue, len(self.tokens)))
        for i in range(len_dialogue):
            # __dialogue.shape: (n_batches, n_questions, 2)
            # dialogue contains both question and answers.
            for token_idx in self.__dialogue[idx][i][0]:  # get question token
                # NOTE: Be careful not to add tokens after "end-of-question"
                #       when implementing batch mode
                bow[i, token_idx] += 1

        # 2. Score Calculation
        sim_score = 0.0
        for i in range(len_dialogue):
            for j in range(i + 1, len_dialogue):
                # 0 <= cs <= 1
                sim_score += cos_sim(bow[i], bow[j])
        avg_sim_score = 2 * sim_score / (len_dialogue * (len_dialogue - 1))
        return 1 - avg_sim_score

    def is_curr_question_invalid(self, idx: int) -> bool:
        """
        Parameters
        ----------
        idx : int. an index of batch.
        """

        return self.__curr_answer[idx] == 2

    def is_curr_question_related(self, idx: int) -> bool:
        """
        Parameters
        ----------
        idx : int. an index of batch.
        """
        meaningful = self.oracle.meaningful_question(
            self.imfn[idx],
            self.__curr_question[idx],
            self.__n_obj[idx],
            require_pp=True
        )
        return meaningful

    def summarize_current_episodes(self) -> list:
        """
        Note that this function should be called when
        the all episodes are terminated.
        Returns
        -------

        """
        # -[x] Get image filename
        # -[x] Get questions
        # -[x] Get o_pred_probs
        # -[x] Get answers from oracle
        # -[x] Get questions in program token
        # -[x] Get final prediction results (rewards)

        def pmf_decoder(questions):
            raw_str_programs = []
            str_programs = []
            programs = []
            for q in questions:
                raw_str_programs.append(
                    pmf.decode(
                        q,
                        self.oracle.vocab['program_idx_to_token'],
                        check_mode=True
                    )
                )
                str_programs.append(
                    pmf.decode(
                        q,
                        self.oracle.vocab['program_idx_to_token']
                    )
                )
                programs.append(
                    pmf.str_to_program(str_programs[-1]))
            return raw_str_programs, str_programs, programs

        summaries = []
        for bid in range(self.batch_size):
            questions = [d[0] for d in self.__dialogue[bid]]
            # Computer sorted split_actions
            split_actions = []
            _split_actions = [
                a.tolist() for a in self.split_actions[bid]
            ]
            for sa, op in zip(_split_actions, self.o_pred_probs[bid]):
                # [(0, p0), (1, p1), ....]
                _pred_tup_list = [(i, v) for i, v in enumerate(op)]
                _sorted_tup = sorted(
                    _pred_tup_list, key=lambda t: t[1], reverse=True
                )
                _sorted_keys = [t[0] for t in _sorted_tup]  # get keys
                # sort indecies by o_pred_prob values
                _sa_tup_list = [(i, v) for i, v in zip(_sorted_keys, sa)]
                _sa_sorted = sorted(_sa_tup_list, key=lambda x: x[0])
                sa_sorted = [s[1] for s in _sa_sorted]
                split_actions.append(sa_sorted)

            rsp, sp, p = pmf_decoder(questions)
            summary_dict = {
                'image_filename': self.imfn[bid],
                'questions': questions,
                'raw_str_programs': rsp,
                'str_programs': sp,
                'programs': p,
                'answers': [d[1] for d in self.__dialogue[bid]],
                'target_id': self.tgt_id[bid],
                'predictions': self.o_pred_probs[bid],
                'results': self.results[bid],
                'submitted_id':
                int(
                    np.argmax(self.o_pred_probs[bid][-1])
                ),
                'split_actions': split_actions
            }
            summaries.append(summary_dict)
        return summaries

import os
import json

import numpy as np

import modules.question_generator.program_manipulates_func as pmf
from functools import reduce
from utils.utils import cos_sim
from modules.oracle.oracle import Oracle
from modules.rl.envrewards import calculate_batch_reward

# logging
from logging import getLogger
logger = getLogger(__name__)


class NoTargetCandsWarning(Exception):
    # Used to move forward to next scene when there are no reference object
    # candidates (New Object Evaluation)
    pass


class Environment():
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
        self.c_eoq_idxs = []
        self.c_eod_idxs = []
        self.o_pred_probs = []  # (n_batches, n_question, n_obj)
        self.prev_o_pred_probs = []  # (n_batches, n_objs)
        self.results = []  # (n_batches)  True: success, False, fail

        # Make sure to initialize batch_size in init_batches
        self.batch_size = None
        self.n_max_questions = args.n_max_questions

    def init_batches(self, n_batches):
        """
        Initializes placeholders w.r.t. batches.

        Parameters
        ----------
        n_batches: int
            Size of batches.
        """
        self.batch_size = n_batches
        self.c_eoq_idxs = [-1 for _ in range(n_batches)]
        self.c_eod_idxs = [-1 for _ in range(n_batches)]
        self.o_pred_probs = [[] for _ in range(n_batches)]
        self.prev_o_pred_probs = [[] for _ in range(n_batches)]
        self.results = [False for _ in range(n_batches)]

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
        self.c_eoq_idxs = [-1 for _ in range(self.batch_size)]
        self.c_eod_idxs = [-1 for _ in range(self.batch_size)]

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

    # TODO: udpate docstrings
    def update(
        self, action, c_istep, eoq_status, eod_status,
        curr_istep, is_last_token=False, is_last_question=False,
    ):
        """
        Based on `action`, update state of the environment.
        This function is called on every token generation.

        Parameters
        ----------
        action : list of int (batch of actions)
            length of list : self.batch_size
        c_istep : current question token steps.
            Bear in mind that this is not global steps!
        eoq_status : list of int
        eod_status : list of int
        curr_istep :
            TODO: deprecates
        is_last_token : bool
            indicates the action is the last token
        is_last_question : bool
            indicates the action belongs to the last question

        Returns
        -------
        state : dict
        """
        if is_last_question:
            pass
        for i, act in enumerate(action):
            # 0. append action. i is batch_id
            self.__curr_question[i].append(act)

            # 1. Handle violated cases.
            # All violated cases will get invalid answers from oracle.
            # [PREM1]
            # The last token of the question should be <END>.
            # In this violated case, the invalid will be returned
            # from the oracle.
            if is_last_token and not eoq_status[i]:
                # self.__curr_question[i].append(act)
                self.__curr_answer[i] = 2  # set answer invalid
                self.c_eoq_idxs[i] = -1  # set eoq index (no eoq)
                continue

            # [PREM2]
            # The token for the last question should be <STOP>.
            # In this violated case, the invalid will be returned
            # from oracle and the correct_reward will be disabled
            # even if the guesser prediction is correct.
            # TODO: disable correct reward.
            elif is_last_question and not eod_status[i]:
                # self.__curr_question[i].append(act)
                self.__curr_answer[i] = 2  # set answer invalid
                assert c_istep == 0  # this should be always 0
                self.c_eod_idxs[i] = -1  # set eod index (no eod)
                continue

            # [PERM3]
            # The <STOP> token should be produced alone.
            # Thus <STOP> token generated other than step 0 is
            # treated as a violated case.
            elif act == self.tokens['<STOP>'] and c_istep != 0:
                self.__curr_answer[i] = 2  # set answer invalid
                continue

            # 2. Set actions if the question is sound.
            # [CASE2] if recieves EOQ token get answer from oracle
            # <END> : end of sentence
            if act == self.tokens['<END>']:
                # self.__curr_question[i].append(act)  # append <END>
                # get answer from oracle
                self.__curr_answer[i] = self.get_answer(
                    self.imfn[i], self.__curr_question[i]
                )
                self.c_eoq_idxs[i] = c_istep  # end of question

            # [CASE3] if recieve EOD token (<STOP>),
            # terminate the current episode and update params
            elif act == self.tokens['<STOP>']:
                # TODO: !!! check this impact !!!!
                # send <END> instead of <STOP> to QA encoder
                # to prevent hidden state corruption
                # -> NOTE: MODIFIED
                # self.__curr_question[i].append(self.tokens['<END>'])
                # self.__curr_question[i].append(act)
                self.c_eod_idxs[i] = c_istep  # end of dialogue

            # [CASE0] insert <NULL> for the terminated batch
            elif act == self.tokens['<NULL>']:
                pass
                # self.__curr_question[i].append(act)

            # [CASE1] continue question generation
            else:
                pass
                # self.__curr_question[i].append(act)

        state = self.state
        return state

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
            (self.__curr_question[idx], self.__curr_answer[idx])
        )
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

        Returns
        -------
        out : int
        """
        # <START> token is not our agent's action
        # thus subtracting one is required
        return len(self.__curr_question[idx]) - 1

    def get_batch_reward(
            self, o_pred_probs, question_counts, n_token_step, q_count
    ):
        """
        get_batch_reward.

        Parameters
        ----------
        o_pred_probs : list
            The target prediction probabilities.
        question_counts : list
            The number of generated questions for each batch.
            N.B. This does only include questions BEFORE eod.
        n_token_step : int
            The token step in the current question generation.
        q_count : int
            The count of current question in train_rl loop.
        """
        # Parameters for reward calculations
        params = {
            'n_max_questions': self.n_max_questions,
            'descriptive_coeff': self.descriptive_coeff,
            'turn_discount_coeff': self.turn_discount_coeff,
            'image_data_version': self.args.image_data_version,
        }
        # TODO: check n_turns.
        # calculate question turns for each batch
        # n_qcounts = [l for l in question_counts]
        zip_rewards = calculate_batch_reward(
            o_pred_probs, self.prev_o_pred_probs, question_counts,
            self.batch_size, self.c_eoq_idxs, self.c_eod_idxs, self.state,
            self.tgt_id, self.imfn, params, self.oracle, n_token_step
        )
        # TODO: Refactor. List iteration might cause degrade in performance.
        # Register predicted probabilities (batch)
        for ib, prob in enumerate(o_pred_probs):
            # cuda tensor -> list
            self.o_pred_probs[ib].append(prob.detach().cpu().tolist())
        self.prev_o_pred_probs = o_pred_probs  # (n_batches, n_max_objs)

        # rewards: (n_timesteps, n_batches)
        (
            _corr_r, _info_r, _prog_r, _opti_r, _turn_p, _disc_p, _desc_r
        ) = zip_rewards

        # TODO: move this outside of get_rewards?
        # Update results (success or fail) based on correct reward
        summed_rw = np.sum(_corr_r, axis=0)
        success_idxs = np.where(0 < summed_rw)[0]
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
            # dialogue containes both question and answers.
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
                    np.argmax(np.array(self.o_pred_probs[bid][-1]))
                ),
            }
            summaries.append(summary_dict)
        return summaries

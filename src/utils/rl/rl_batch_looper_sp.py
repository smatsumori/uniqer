import numpy as np
from utils.rl.rl_batch_looper import BatchLooper
from utils.rl.rl_batch_looper import EpisodeTracker
# logging
from logging import getLogger
logger = getLogger(__name__)


class BatchLooperSp(BatchLooper):
    def __init__(
        self, n_max_question: int, tokens: dict, stop_ids: list,
        force_stop: bool
    ):
        """__init__.

        Parameters
        ----------
        n_max_question : int
            n_max_question
        tokens : dict
            tokens
        stop_ids : list
        force_stop : bool
            enables force stop mode.
        """
        # Note that n_max_question includes the answer submission.
        self.n_max_question = n_max_question
        self.tokens = tokens
        self.stop_ids = stop_ids
        self.activ_force_stop = force_stop

        # Placeholders
        self.n_batches = 0  # will be overwritten in init_batches
        self.batches = []  # a placeholder for batches
        self.n_bstep = 0  # steps count within a question (token gen)
        self.n_bqstep = 1  # question steps count
        if force_stop:
            logger.warn('Force stop mode is activated.')

    # Available functions from super-class
    # eod_status
    # get_env_actions
    # is_all_batch_eod
    # set_predictions
    # loss_masks (property)
    # current_loss_masks (property)
    # question_counts (property)
    # get_n_all_steps <- perhaps we should rename this

    # override
    def init_batches(self, n_batches):
        self.n_bqstep = 1
        self.n_batches = n_batches
        self.batches = [
            EpisodeTrackerSp(bid, self.tokens, self.stop_ids)
            for bid in range(n_batches)
        ]

    # override
    def set_actions(self, actions: list):
        """set_actions.

        Parameters
        ----------
        actions : list
            A batch of actions.
        """
        assert len(actions) == self.n_batches
        # actions.shape: (n_batches, n_act_space)
        for a, b in zip(actions, self.batches):
            # NOTE: this is for debugging purposes
            # eventually this settings should be removed!!
            if self.activ_force_stop and self.is_last_question:
                b.set_action(self.stop_ids[0], loss_mask=0)
            else:
                b.set_action(a)

    def set_questions(self, questions):
        """set_questions.

        Parameters
        ----------
        questions : list
            A batch of question tokens.
            Shape is (n_batches, n_tokens)
        """
        assert len(questions) == self.n_batches
        for q, b in zip(questions, self.batches):
            # NOTE: this is for debugging purposes
            # eventually this settings should be removed!!
            if self.activ_force_stop and self.is_last_question:
                b.set_question(
                    np.array([
                        self.tokens['<START>'],
                        self.tokens['<STOP>']
                    ])
                )
            else:
                b.set_question(q.copy())

    def get_env_questions(self):
        """Returns a list of questions compatible with
        the format in the environment.
        Note that <START> token is produced by the agent
        automatically.
        """
        questions = []
        for b in self.batches:
            qs = b.questions[-1].copy()
            if len(qs) == 0:
                # At least we need 2 tokens
                token_len = max(self.get_currq_max_token_len(), 2)
                qs = np.zeros(token_len).astype(np.int)
                # <START> token is required for the environment
                qs[0] = self.tokens['<START>']
                # Padding with <NULL> tokens
                qs[1:] = self.tokens['<NULL>']
            questions.append(qs)
        return questions

    def get_is_currq_eod(self):
        """Returns a list indicates if the current
        question produces EOD (End of dialogue).
        """
        return [b.is_currq_eod for b in self.batches]

    def get_is_currq_eoq(self):
        """Returns a list indicates if the current
        question produces EOQ (End of question).
        Note well we regard as so by only checking
        if the current question is terminated and
        this does not ensure the <END> token
        is produced appropriately.
        """
        return [not b.is_ep_eod for b in self.batches]

    def qstep(self):
        """Resets current question status.
        This will be called when the current question
        generation steps have finished.
        """
        self.n_bqstep += 1  # question count
        for b in self.batches:
            # terminate current question generation
            if b.is_currq_eod:
                b.is_ep_eod = True
            b.is_currq_eod = False

    def get_currq_max_token_len(self):
        """Returns the max length of tokens in the current questions
        """
        return max([len(b.questions[-1]) for b in self.batches])

    @property
    def is_last_question(self):
        return self.n_bqstep == self.n_max_question

    # ====================================
    #   Unused properties and functions
    # ===================================
    def reset_current_question_status(self):
        raise NotImplementedError

    @property
    def eoq_status(self):
        raise NotImplementedError

    @property
    def is_all_batch_eoq(self):
        raise NotImplementedError

    @property
    def n_mask_actiavated(self):
        raise NotImplementedError

    @property
    def eoq_idxs(self):
        raise NotImplementedError

    @property
    def eod_idxs(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    @property
    def is_last_token(self):
        raise NotImplementedError


class EpisodeTrackerSp(EpisodeTracker):
    """Episode tracker for each batch.
    """
    def __init__(self, bid: int, tokens: dict, stop_ids: list):
        """__init__.
        Parameters
        ----------
        bid : int
            A batch id.
        tokens : dict
        stop_ids : list
        """
        self.bid = bid  # batch id (must be unique)
        self.tokens = tokens
        self.stop_ids = stop_ids
        self.is_ep_eod = False  # is eod in the current episode
        self.is_currq_eod = False  # is eod in current question generation
        self.question_count = 0
        self.questions = []  # (n_questions)
        self.actions = []  # (n_questions)
        self.o_pred_prob_list = []  # (n_questions)
        self.loss_mask = []  # (n_curr_max_questions)

    # TODO: determine how to handle <STOP> token
    def set_action(self, act: int, loss_mask=None):
        """set_action.
        Parameters
        ----------
        act : int
            action in integer
        """
        if self.is_ep_eod:
            # If the current dialogue is already terminated,
            # i.e. the <STOP> token was produced before current generation,
            # set loss mask 0 to mask out the loss value
            # and set None to actions for debugging purposes.
            # TODO: maybe we should use the other token.
            self.actions.append(None)
            if not loss_mask:
                self.loss_mask.append(0)
        else:
            # Otherwise append actions
            self.actions.append(act)
            if not loss_mask:
                self.loss_mask.append(1)
        if loss_mask:
            self.loss_mask.append(loss_mask)

        # Update the current status for the special tokens
        # Set eod <STOP> token is properly produce at the first time
        if not self.is_ep_eod and self.is_stop(act):
            # The status `is_ep_eod` will be updated in BatchLooper.qstep()
            self.is_currq_eod = True
            self.eoq_idx = self.curr_qidx  # Get the current index

    def set_question(self, question: list):
        """set_question.

        Parameters
        ----------
        question : list
            question
        """
        if self.is_ep_eod:
            self.questions.append([])
        else:
            self.questions.append(question)

    def set_prediction(self, o_pred_prob):
        """Sets current prediction.
        Skip when question generation is already terminated,
        except no prediction has stored.
        This is the case when the <STOP> token is produced at the
        first question generation.

        Parameters
        ----------
        o_pred_prob : list
            A list of predicted probabilities.
        """
        # case1: When eod already or eod was produced in the current generation
        # do not add a prediction.
        if (not self.is_currq_eod) and (not self.is_ep_eod):
            self.o_pred_prob_list.append(o_pred_prob)
            self.question_count += 1
        # case2: When eod is produced but no prediction has been set.
        # (i.e. Random prediction)
        elif not self.o_pred_prob_list:
            self.o_pred_prob_list.append(o_pred_prob)

    def is_stop(self, act: int) -> bool:
        """Detects whether the <STOP> token is produced
        in the current action.

        Parameters
        ----------
        act : int
            An action.

        Returns
        -------
        is_stop : bool

        """
        # TODO: update this part if necessary!
        return act in self.stop_ids

    @property
    def n_steps(self) -> int:
        # TODO: check this effect
        try:
            step = len(self.questions[-1])
        except IndexError:
            step = 0
        return step

    @property
    def curr_qidx(self) -> int:
        """Returns the current index for question.
        Returns
        -------
        int
        """
        return len(self.actions) - 1

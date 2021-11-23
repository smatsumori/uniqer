import numpy as np


class BatchLooper():
    def __init__(
        self, n_max_question: int,
        n_max_steps_per_question: int, tokens: dict
    ):
        """__init__.
        Parameters
        ----------
        n_max_question : int
            n_max_question
        n_max_steps_per_question : int
            n_max_steps_per_question
        tokens : dict
            tokens
        """
        self.n_max_question = n_max_question
        self.n_max_steps_per_question = n_max_steps_per_question
        self.tokens = tokens

        # Placeholders
        self.n_batches = 0  # will be overwritten in init_batches
        self.batches = []  # a placeholder for batches
        self.n_bstep = 0  # steps count within a question (token gen)
        self.n_bqstep = 0  # question steps count

    def init_batches(self, n_batches):
        """init_batches.
        Parameters
        ----------
        n_batches : int
            n_batches
        """
        self.n_bqstep = 0
        self.n_bstep = 0
        self.n_batches = n_batches
        self.batches = [
            EpisodeTracker(bid, self.tokens) for bid in range(n_batches)
        ]

    @property
    def eod_status(self):
        """Returns eod status of batches.
        """
        return [b.is_ep_eod for b in self.batches]

    @property
    def eoq_status(self):
        """Returns eoq status of batches
        """
        return [b.is_currq_eoq for b in self.batches]

    # GOTO: question encoder
    @property
    def is_all_batch_eoq(self):
        """Returns True if all batches are eoq
        """
        return all(self.eoq_status)

    @property
    def is_all_batch_eod(self):
        """Returns True if all batches are eod
        """
        return all(self.eod_status)

    def reset_current_question_status(self):
        """ Resets eoq status of the batchs.
        This will be called when the current question
        generation steps have finished.
        """
        for b in self.batches:
            if not b.is_ep_eod:
                b.is_currq_eoq = False
                b.is_currq_eod = False
                b.currq_eoq_idx = -1
                b.currq_eod_idx = -1

    def set_actions(self, actions: list, cistep: int):
        """Set actions.
        If the current question generation is terminated
        (either eoq or eod), <Null> token will be inserted.
        (See Episode Tracker for the details)
        Parameters
        ----------
        actions : list
            list of actions whose shape is (n_batchs, 1)
        cistep : int
            The step in the current question generation.
        """
        assert len(actions) == len(self.batches)
        assert len(actions) == self.n_batches
        # actions.shape: (n_batches)
        for a, b in zip(actions, self.batches):
            b.set_action(a, cistep)

    def get_env_actions(self):
        """Get current actions that are passed to environment.
        Note that some actions are overwritten to <Null>,
        based on the criteria in `set_actions` function.
        """
        actions = [b.actions[-1] for b in self.batches]
        return actions

    def set_predictions(self, o_pred_probs: list):
        """Sets predcitons unless if eod token (<STOP> token) is produced.
        """
        assert len(o_pred_probs) == len(self.batches)
        assert len(o_pred_probs) == self.n_batches
        for o, b in zip(o_pred_probs, self.batches):
            b.set_prediction(o)

    @property
    def predictions(self) -> list:
        """Returns current predictions.
        If a batch is terminated, it will return the submitted prediction,
        otherwise the lates prediction.

        Returns:
            list:
        """
        latest_o_pred_probs = [b.o_pred_prob_list[-1] for b in self.batches]
        return latest_o_pred_probs

    def get_n_all_steps(self):
        """get_n_all_steps.
        """
        steps = [b.n_steps for b in self.batches]
        return sum(steps)

    @property
    def loss_masks(self) -> np.ndarray:
        """loss_masks.
        Parameters
        ----------
        Returns
        -------
        masks : np.ndarray
            masks.shape: (n_batches, n_timesteps)
            1 for not masked, 0 for masked
        """
        masks = np.array([b.loss_mask for b in self.batches])
        return masks

    @property
    def current_loss_masks(self) -> np.ndarray:
        """Returns current_loss_masks.
        Parameters
        ----------
        Returns
        -------
        masksT[-1] : np.ndarray
            .shape: (n_batches)
            1 for not masked, 0 for masked
        """
        masksT = self.loss_masks.T
        return masksT[-1]

    # compute the number of masks activated
    @property
    def n_mask_actiavated(self):
        """n_mask_actiavated.
        """
        lm = np.array(self.loss_masks)
        n_activated = lm.sum()
        return n_activated

    @property
    def eoq_idxs(self):
        """Returns eoq indices in the current question generation.
        """
        idxs = [b.currq_eoq_idx for b in self.batches]
        return idxs

    @property
    def eod_idxs(self):
        """Returns eod indices in the current question generation.
        """
        idxs = [b.currq_eod_idx for b in self.batches]
        return idxs

    # token generation steps (atomic)
    def step(self):
        """Increments the question generation steps (an atomic action)
        """
        self.n_bstep += 1
        assert self.n_bstep <= self.n_max_steps_per_question

    # question submission steps (generate new question)
    def qstep(self):
        """Increments the question steps.
        """
        # initializes steps within the question
        self.n_bstep = 0  # token steps within the question
        # resets end of question status for the new question generation
        self.reset_current_question_status()
        self.n_bqstep += 1  # question count
        assert self.n_bqstep <= self.n_max_question

    @property
    def is_last_token(self) -> bool:
        """is_last_token.
        Returns
        -------
        bool
        """
        ilt = (self.n_bstep == self.n_max_steps_per_question)
        return ilt

    @property
    def is_last_question(self) -> bool:
        """is_last_question.
        Returns
        -------
        bool
        """
        ilq = (self.n_bqstep == self.n_max_question)
        return ilq

    @property
    def question_counts(self) -> list:
        """Returns net question counts.
        This count only includes question before eod.
        Returns
        -------
        list
        """
        return [b.question_count for b in self.batches]


class EpisodeTracker:
    """Episode tracker for each batch.
    """
    def __init__(self, bid: int, tokens: dict):
        """__init__.
        Parameters
        ----------
        bid : int
            A batch id.
        tokens : dict
        """
        self.bid = bid  # batch id (must be unique)
        self.tokens = tokens
        self.is_ep_eod = False  # is eod in the current episode
        self.is_currq_eoq = False  # is eoq in current question generation
        self.is_currq_eod = False  # is eod in current question generation
        self.currq_eoq_idx = -1  # local eoq idx in the current question
        self.currq_eod_idx = -1  # local eod idx in the current question
        self.question_count = 0
        self.actions = []
        self.o_pred_prob_list = []
        self.loss_mask = []

    def set_action(self, act: int, cstep: int):
        """set_action.
        Parameters
        ----------
        act : int
            action in integer
        cstep : int
            an action step in the current question generation.
            (<BOS> is not counted)
        """
        if self.curr_qg_terminated:
            # If the current question generation step is terminated
            # (i.e. the eoq token was produced in the current generation steps)
            # (i.e. the eod token was produeed before)
            # set loss mask 0 to mask out the loss and
            # <NULL> to actions for debugging purposes.
            # TODO: maybe we should use the other token.
            self.actions.append(self.tokens['<NULL>'])
            self.loss_mask.append(0)
        else:
            # Otherwise append actions
            self.actions.append(act)
            self.loss_mask.append(1)

        # Set current status for the special tokens
        if act == self.tokens['<END>']:  # when end token is produced
            # Submits current question
            if not self.is_currq_eoq:
                self.is_currq_eoq = True
                self.currq_eoq_idx = self.curr_qidx  # question index
        elif act == self.tokens['<STOP>']:  # stop token is produced
            # Only single <STOP> token generation is allowed
            # Otherwise it will be substituted as <END>
            # e.g. w1, w2, w3, <STOP> is not allowed
            # Submits the answer
            if cstep == 1 and not self.is_ep_eod:
                self.is_ep_eod = True
                self.is_currq_eod = True  # will be reset in the next qgen
                self.currq_eod_idx = self.curr_qidx  # question index

    def set_prediction(self, o_pred_prob):
        """Sets current prediction.
        Skip when question generation is already terminated,
        except no prediction has stored.
        This is the case when the <STOP> token is produced at the
        first question generation.
        """
        # TODO: do we need to insert dummy o_pred_prob? after eod?
        # case1: When eod already, do not add a prediction
        if not self.is_ep_eod:
            self.o_pred_prob_list.append(o_pred_prob)
            self.question_count += 1
        # case2: When eod is produced but no prediction has been set.
        elif not self.o_pred_prob_list:
            self.o_pred_prob_list.append(o_pred_prob)

    @property
    def curr_qg_terminated(self) -> bool:
        """Returns True either if the current question generation
            is terminated (eoq) or the current dialogue is
            terminated (eod).
        Returns
        -------
        bool
        """
        return self.is_ep_eod or self.is_currq_eoq

    @property
    def n_steps(self) -> int:
        """n_steps.
        Returns
        -------
        int
        """
        return len(self.loss_mask)

    @property
    def curr_qidx(self) -> int:
        """Returns the current index for question.
        Returns
        -------
        int
        """
        return len(self.actions) - 1

import numpy as np
import torch
from torch.distributions import Categorical


class Agent:
    def __init__(self, policy, split_mode=False):
        '''
        Parameters
        ----------
        policy : Policy()
            an instance of policy
        split_mode : bool, default is False

        Returns
        -------
        '''
        self.policy = policy
        self.split_obj_model_mode = split_mode
        # placeholders
        self.o_pred_prob = None

    def act(self, state, is_train):
        """
        Parameters
        ----------
        state : dict
        is_train : bool

        Returns
        -------
        action.item() : <list of int>
        """
        # DEBUG: cuda causes unresolved error
        #        (--> therefore, we send `action_prob` to cpu)

        if self.split_obj_model_mode:
            # action_prob : shape(batch_size, action_class_size)
            action_prob = self.policy.update_splitter(
                state, self.o_pred_prob
            ).cpu()
        else:
            # action_prob : shape(batch_size, vocab_size)
            action_prob = self.policy.update_generator(state).cpu()

        # TODO: perhaps move this to policy.py
        # epsilon greedy exploration
        if is_train:
            # (batch_size, action_size)
            m = Categorical(action_prob)
            # (batch_size)
            action = m.sample()

            # store an action history
            # m.log_prob(action).shape (batch_size)
            self.policy.history_log_probs.append(m.log_prob(action))

            # TODO: remove this
            # TODO: move this to policy
            # self.policy.returns.append(0)  # the value will be updated later
            # Add batch size of the placeholder
            # self.policy.returns.append(np.zeros_like(action, dtype=float))

            return action.detach().cpu().tolist()
        else:
            action_prob = action_prob.detach().cpu().numpy()
            action = np.argmax(action_prob, axis=-1)
            self.policy.history_log_probs.append(
                torch.tensor(action_prob.max(axis=-1))
            )

            # TODO: remove this
            # TODO: move this to policy
            # self.policy.returns.append(0)  # the value will be updated later

            return action.tolist()

    def model_initialize(self, state, is_train=True):
        """
        Parameters
        ----------
        state : dict
        is_train : bool, default is True

        Returns
        -------
        o_pred_prob : torch.tensor
        """
        self.o_pred_prob = self.policy.model_initialize(state, is_train)
        return self.o_pred_prob

    def generate_question(self, state, actions):
        """
        Execute seq2seq_generator and return a question

        Parameters
        ----------
        state : dict
        actions : array-like (list ?)
            (shape: batch_size)

        Returns
        -------
        programs : np.array
        split_actions : np.array
        """
        programs, split_actions = self.policy.exec_seq2seq_qgen(
            state, actions, self.o_pred_prob
        )
        return programs, split_actions

    def eoq_update(self, state):
        """
        End-of_Question

        Parameters
        ----------
        state : dict

        Returns
        -------
        o_pred_prob : torch.tensor
        """
        self.o_pred_prob = self.policy.update_state_encoder(state)
        return self.o_pred_prob

    def eod_update(self, state):
        """
        End-of-Dialogues
        For now, just return the last o_pred_prob

        Parameters
        ----------
        state : dict

        Returns
        -------
        o_pred_prob : torch.tensor
        """
        return self.o_pred_prob

    def save_models(self, args, save_dict, epoch, overwrite):
        """
        Saving learned models

        Parameters
        ----------
        args : ArgumentParser
        save_dict : dict
        epoch : int
        over_write : bool
            If true, then overwrite the previous saved models.
        """
        self.policy.save_models(args, save_dict, epoch, overwrite)

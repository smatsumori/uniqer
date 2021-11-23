import torch
import torch.optim as optim
import numpy as np

# from .network import BasicPolicyNetwork
from .single_tf_network import SingleTFPolicyNetwork

# Logging
from logging import getLogger
logger = getLogger(__name__)


class Policy:
    def __init__(self, args, device, vocab_size):
        """
        Parameters
        ----------
        args : ArugmentParser
        device : str
        vocab_size : int
        """
        # self.rewards will be updated in `train_rl.py`
        self.rewards = []
        self.ep_rewards = []
        self.imm_returns = []

        # TODO: DEBUG: split-mode
        # --> [(batch_size, n_max_obj), (batch_size, n_max_obj), (...), ...]
        # history_log_probs : [[batch_size], [batch_size], ... ]
        self.history_log_probs = []

        if args.rl_single_tf_mode:
            self.networks = SingleTFPolicyNetwork(args, device)
        else:
            raise NotImplementedError

        self.qgen_current_hidden_output = None

        self.gamma = args.gamma
        self.split_obj_model_mode = args.split_obj_model_mode

        # TODO: separately managed optimizer ?
        # TODO: impl get_optimizer func in network.py
        # TODO: deprecates rl_freeze
        self.rl_freeze = args.rl_freeze
        if self.rl_freeze:
            if self.split_obj_model_mode:
                self.optimizer = optim.Adam(
                    self.networks.splitter.parameters(), lr=args.rl_lr
                )
            else:
                self.optimizer = optim.Adam(
                    self.networks.generator.parameters(), lr=args.rl_lr
                )

        else:
            if self.split_obj_model_mode:
                self.optimizer = optim.Adam(
                    list(self.networks.generator.parameters()) +
                    list(self.networks.guesser.parameters()) +
                    list(self.networks.state_encoder.parameters()) +
                    list(self.networks.qa_encoder.parameters()) +
                    list(self.networks.splitter.parameters()),
                    lr=args.rl_lr
                )
            else:
                self.optimizer = optim.Adam(
                    list(self.networks.generator.parameters()) +
                    list(self.networks.guesser.parameters()) +
                    list(self.networks.state_encoder.parameters()) +
                    list(self.networks.qa_encoder.parameters()),
                    lr=args.rl_lr
                )

        self.clip_value = args.clip_value
        if self.clip_value > 0:
            logger.info(f'Gradient clipping value: {self.clip_value}')

    def reset_history(self):
        self.rewards = []
        self.ep_rewards = []
        self.imm_returns = []
        self.history_log_probs = []

    def model_initialize(self, state, is_train=True):
        """
        Parameters
        ----------
        state : dict

        Returns
        -------
        o_pred_prob : torch.tensor
            Initial probability, shape: (batch_size, n_obj)
        """
        return self.networks.model_initialize(state, is_train=is_train)

    def update_generator(self, state):
        """
        step generator (not split-mode)

        Parameters
        ----------
        state : dict

        Returns
        -------
        action_prob : torch.tensor
            shape: (batch_size, vocab_size)
        """
        action_prob, hidden_output = self.networks.gen_next_token_prob(state)

        # register the current hidden_output
        self.q_gen_current_hidden_output = hidden_output

        return action_prob

    def update_splitter(self, state, o_pred_prob):
        """
        step splitter (split-mode)

        Parameters
        ----------
        state : dict
        o_pred_prob : torch.tensor
            shape: (batch_size, n_obj)

        Returns
        -------
        action_prob : torch.tensor
            shape: (batch_size, split_class_size)
        """
        # TODO: how value_estimator works in this mode?
        action_prob = self.networks.gen_split_prob(state, o_pred_prob)
        return action_prob

    def exec_seq2seq_qgen(self, state, actions, o_pred_prob):
        """
        Execute seq2seq_generator and return a question

        Parameters
        ----------
        state : dict
        actions : array-like (list ?)
            split info
            (shape: batch_size)
        o_pred_prob : torch.tensor
            shape: (batch_size, n_obj)

        Returns
        -------
        programs : np.array
        split_actions : np.array
        """
        programs, split_actions = self.networks.run_seq2seq_qgen(
            state, actions, o_pred_prob
        )
        return programs, split_actions

    def update_state_encoder(self, state):
        """
        Function called when End of Question token <END> is generated.
        TODO: rename func ? this func manages not only updating state encoder,
              but also encoding Q&A, update generator's hidden state

        Parameters
        ----------
        state : dict

        Returns
        -------
        o_pred_prob : torch.tensor
            shape: (batch_size, n_obj)
        """
        return self.networks.encode_qa(state)

    def register_rewards(self, eps_rewards, imm_rewards):
        """
        Registers two types of rewards.
        - called in train_rl.py each time a single question is generated
        - self.split_obj_model_mode --> shape: (n_batches)

        Parameters
        ----------
        eps_rewards : np.array
            shape : (n_steps, n_batches) or (n_batches)
            Episodic rewards. The values are stored in self.eps_rewards
            and will be used to calculate ep_returns in update_params().

        imm_rewards : np.array
            shape : (n_steps, n_batches) or (n_batches)
            Immediate rewards. The return value will only be calculated
            back to the current question generation steps.
        """
        # 1. Episode Rewards
        # Store episode rewards whose shape is (n_gross_steps, n_batches)
        # The returns will be calculated in update_params().
        if self.split_obj_model_mode:
            assert eps_rewards.ndim == 1, 'Invalid dim'
            # eps_rewards.shape: (n_batches)
            self.ep_rewards.append(eps_rewards)
        else:
            # eps_rewards.shape: (n_steps, n_batches)
            # Extends alongside n_steps
            self.ep_rewards.extend(eps_rewards)

        # 2. Immediate Rewards
        if self.split_obj_model_mode:
            assert imm_rewards.ndim == 1, 'Invalid dim'
            # imm_rewards.shape: (n_batches)
            self.imm_returns.append(imm_rewards)
        else:
            # Compute intermediate rewards
            # ret.shape: (n_batches)
            ret = np.zeros(imm_rewards.shape[1])
            curr_returns = []
            for r in imm_rewards[::-1]:
                ret += r + ret*self.gamma
                curr_returns.insert(0, ret)

            # curr_returns.shape: (n_steps, n_batches)
            self.imm_returns.extend(curr_returns)

    def update_params(self, loss_mask, no_grad=False, eps=1e-7):
        """
        Updates parameters in the policy network by policy gradient.

        Parameters
        ----------
        loss_mask : list of list
            shape : (n_steps, n_batches)
            Mask out the losses for the actions produced
            after <END> or <STOP> token.
        no_grad : bool, default is False
            Mode to only calculate loss value (no back propagation)
        eps : float, default is 1e-7
        """
        # Calculate returns for episodic rewards
        # self.eps_rewards.shape: (n_timesteps, n_batches)
        n_batches = np.array(self.ep_rewards).shape[1]
        ret = np.zeros(n_batches)
        curr_ep_returns = []  # shape: (n_steps, n_batches)

        # TODO: gamma should be chosen 1.0 or self.gamma based on loss_mask
        # currently, one very long question can make the reward for other
        # tokens extra small (depends of self.gamma)
        for r in self.ep_rewards[::-1]:
            ret = r + ret * self.gamma
            curr_ep_returns.insert(0, ret)

        # Check shapes for both returns are the same (n_steps, n_batches)
        imm_shape = np.array(self.imm_returns).shape
        ep_shape = np.array(curr_ep_returns).shape
        assert imm_shape == ep_shape
        # Check steps are the same (n_steps)
        assert len(self.history_log_probs) == len(curr_ep_returns)

        # Calculate returns. shape: (n_steps, n_batches)
        returns = torch.tensor(
            np.array(self.imm_returns) + np.array(curr_ep_returns),
            dtype=torch.float
        )

        policy_loss = torch.zeros(n_batches, dtype=torch.float)

        # TODO: DEBUG
        # returns = (returns - returns.mean()) / torch.max(
        #     returns.std(), torch.tensor(eps)
        # )

        # Check if the shape is same
        # transpose (n_batches, n_steps) -> (n_steps, n_batches)
        # loss_mask.shape: (n_steps, n_batches)
        loss_mask = torch.t(torch.tensor(loss_mask, dtype=torch.float))
        assert loss_mask.size() == returns.size()

        # history_log_probs.shape: (n_steps, n_batches)
        if no_grad:
            with torch.no_grad():
                for log_prob, R, m in zip(
                    self.history_log_probs, returns, loss_mask
                ):
                    policy_loss += (-log_prob * R * m)

        else:
            for log_prob, R, m in zip(
                self.history_log_probs, returns, loss_mask
            ):
                policy_loss += (-log_prob * R * m)

            # update params
            self.optimizer.zero_grad()
            policy_loss.mean().backward()

            # gradient clipping
            if self.clip_value > 0:
                if self.split_obj_model_mode and self.rl_freeze:
                    params = self.networks.splitter.parameters()
                elif self.rl_freeze:
                    params = self.networks.generator.parameters()
                else:
                    params = list(self.networks.guesser.parameters()) + \
                        list(self.networks.generator.parameters()) + \
                        list(self.networks.state_encoder.parameters()) + \
                        list(self.networks.qa_encoder.parameters())

                    if self.split_obj_model_mode:
                        params += list(self.networks.splitter.parameters())

                torch.nn.utils.clip_grad_norm_(params, self.clip_value)
            self.optimizer.step()

        self.reset_history()

        return policy_loss.mean().item()

    def save_models(self, args, save_dict, epoch, overwrite):
        """
        Saving learned models

        Parameters
        ----------
        args : ArgumentParser
        save_dict : dict
        epoch : int
        overwrite : bool
            Whether to overwrite the previous saves.
        """
        self.networks.save_models(args, save_dict, epoch, overwrite)

import os

import numpy as np
import torch

from modules.single_tf import SingleNet
from modules.object_targeting_module.otm_gru import CandsSplitNetwork
from modules.object_targeting_module.otm_trm import CandsSplitTransformer
from modules.rl.network import extract_topk_features
from utils.utils import load_model, load_vocab, smart_2d_sort
from utils.consts import RICH_DIM, BASE_DIM, SPECIAL_TOKENS

# logging
from logging import getLogger
logger = getLogger(__name__)


class SingleTFPolicyNetwork:
    """
    Single Transformer Policy Network Model

    Basically, every function takes `state` (dict form) as input
    (parameter x) and retrieves the necessary information internally.

    * TODO: update the following explanation
    - 0. GuesserMLP running, Generator and StateEncoder Initialize
    - 1. Generator generates one token
    - (2. environment update)
    - 3-1. end of question
           --> * Question & Answer encode
               * StateEncoder update
               * Generator hidden states update
    - 3-2. end of dialogue --> no update model for now

    Attributes
    ----------
    device : str
        pytorch  device type
    """

    def __init__(self, args, device):
        """
        Parameters
        ----------
        args : ArugmentParser
        device : str
        """
        self.device = device
        self.args = args
        self.top_k = args.proposed_q_gen_top_k
        self.proposed_model_mode = args.proposed_model_mode
        logger.info(f'Proposed Model Mode: {self.proposed_model_mode}')

        # Modeling
        # [COMPONENT1]  Transformer Encoder-Decoder Model
        vocab = load_vocab(args.vocab_path)
        spatial_dim = RICH_DIM if args.proposed_model_mode else BASE_DIM
        self.net = SingleNet(
            device,
            args.cropped_img_enc_dim + spatial_dim,
            vocab,
            args.ans_vocab_size,
            args.gen_seq_len,
            args.load_guesser,
            args.load_generator,
            args.proposed_q_gen_top_k,
            args.split_class_size,
            args.image_data_version,
            max_batch_size=args.batch_size,
            obj_memory_only=args.obj_memory_only,
        )

        # [COMPONENT2] splitter
        if args.otm_type == 'GRU':
            OTM = CandsSplitNetwork
        elif args.otm_type == 'transformer':
            logger.warning('You are using transformer OTM mode!')
            OTM = CandsSplitTransformer
        splitter = OTM(
            args,
            self.device,
            args.cropped_img_enc_dim + (5 + (5 + 2) * self.top_k),  # sorry
        ).to(self.device)
        self.splitter = self._load_model(splitter, args.load_splitter)

        # Placeholders
        self.memory = None
        self.dialogue = None

        self.sv_prev_qgen_path = None
        self.sv_prev_guesser_path = None
        self.sv_se_path = None
        self.sv_qa_path = None

    def _load_model(self, model, savepath, is_dict=False):
        """
        loading pre-trained model if exists

        Parameters
        ----------
        model : nn.Module
        savepath : str
        is_dict : bool, default is False
            If True, load model as state_dict()
        """
        if savepath is not None:
            model, _ = load_model(model, None, savepath, self.device, is_dict)
        else:
            logger.warning(
                'Loading pre-trained model is highly recommended!'
            )
            logger.warning(model)
        return model

    def model_initialize(self, x, is_train=True):
        """

        Parameters
        ----------
        x : dict
        is_train : bool, default is True
            if True, models will be set to train() mode

        Returns
        -------
        o_pred_prob : torch.tensor
            initial predicted object probability
        """
        self.is_train = is_train

        # nn.Modules mode setting
        if is_train:
            self.set_train_mode()
        else:
            self.set_eval_mode()

        b_size = x['context'].shape[0]

        # Run Encoder and extract `o_pred_prob` & `memory`
        # TODO: DEBUG: obj_order is correctly kept ?
        if self.proposed_model_mode:
            spatial = x['obj_features'][:, :, :RICH_DIM].float()
            obj_features = torch.cat(
                [x['cropped'].float(), spatial], dim=2
            ).to(self.device)
        else:
            obj_features = torch.cat(
                [x['cropped'], x['spatial']], dim=2
            ).float().to(self.device)

        with torch.no_grad():
            memory, o_pred_prob = self.net.encoder(
                obj_features,
                self.net.cls_token[:b_size],
                n_obj=x['n_obj'].long().to(self.device),
                is_pretrain=False,
            )
        self.memory = memory.transpose(1, 0).clone()

        # update dialogue
        self.dialogue = self.net.cls_token[:b_size].clone()
        return o_pred_prob

    def gen_split_prob(self, x, o_pred_prob):
        """

        Parameters
        ----------
        x : dict
        o_pred_prob : torch.tensor
            (shape: batch_size, n_max_obj)

        Returns
        -------
        action_prob : torch.tensor
            * out_size : class_size ** top_k + 1  (<STOP>)
            (shape: batch_size, out_size)
        """

        # TODO: deprecates this `if` (should always be in Rich-spatial-mode)
        if self.proposed_model_mode:
            obj_features, top_o_preds = extract_topk_features(
                x['org_spatial_features'].float(),
                x['cropped'].float(),
                o_pred_prob.detach().cpu().numpy(),
                self.top_k,
            )
            obj_features = obj_features.to(self.device)
            top_o_preds = top_o_preds.to(self.device)
        else:
            logger.error('Deprecated')
            raise NotImplementedError

        # question index (times)
        # the number of questions should be the same for the entire batch
        iq = len(x['dialogue'][0])

        if self.is_train:
            action_prob = self.splitter(top_o_preds, obj_features, iq)
        else:
            with torch.no_grad():
                action_prob = self.splitter(top_o_preds, obj_features, iq)
        return action_prob

    def run_seq2seq_qgen(self, x, actions, o_pred_prob):
        """

        Parameters
        ----------
        x : dict
        actions : array-like (list ?)
            (shape: batch_size)

        Returns
        -------
        program : np.array
            (shape: batch_size, seq_len)]

        split_ids : np.array
        """

        split_ids, eod_mask = self.splitter.actions2splits(actions)

        # re-arrange obj_features based on split_ids order
        # (Splitter Network's in-out order is based on o_pred_prob top_k)
        _cands = np.argsort(o_pred_prob.detach().cpu().numpy())[:, ::-1]
        cands = np.argsort(_cands)

        # re-arrange split_ids for obj_order
        reorder_split_ids = smart_2d_sort(split_ids, cands)

        with torch.no_grad():
            # NOTE: is_tf_inference will change nothing if LSTM is used
            #       as decoder model
            program_prob = self.net.decoder(
                torch.tensor(x['curr_question'])[:, 0].to(self.device),
                self.memory,
                split_ids=reorder_split_ids,
                is_train=False,
            ).permute(0, 2, 1)

        # post-process
        # torch.argmax is much slower than np.argmax (torch version 1.3)
        program_prob = program_prob.detach().cpu().numpy()
        program = np.argmax(program_prob, axis=1)

        # eod_mask apply
        # HARD-coded
        eod = np.zeros(program.shape[1])
        # eod[1:] = SPECIAL_TOKENS['<NULL>']
        eod[0] = SPECIAL_TOKENS['<STOP>']

        program[eod_mask] = eod

        # TODO: DEBUG: should I set here ?
        # <START> token append
        _program = np.zeros(
            (program.shape[0], program.shape[1] + 1)
        ).astype(np.int)
        _program[:, 1:] = program
        _program[:, 0] = SPECIAL_TOKENS['<START>']

        # update dialogue
        tmp = torch.tensor(_program).long().to(self.device)
        tmp = self._clean_program(tmp)

        self.dialogue = torch.cat([self.dialogue, tmp], dim=1)
        return _program, split_ids.cpu().numpy().copy()

    def _clean_program(self, program):
        """
        Cleaning Generated Program for Encoder

        Parameters
        ----------
        program : torch.LongTensor
            shape: (batch_size, token_len)

        Returns
        -------
        clean : torch.LongTensor
            shape: (batch_size, token_len)
        """
        mask = (program == SPECIAL_TOKENS['<END>']).cumsum(axis=1)
        clean = program.clone()
        clean[mask.cumsum(axis=1) > 1] = SPECIAL_TOKENS['<NULL>']
        return clean

    def encode_qa(self, x):
        """

        Parameters
        ----------
        x : dict

        Returns
        -------
        o_pred_prob : torch.tensor
            (shape: batch_size, n_max_obj)
        """
        # Run Encoder and extract `o_pred_prob` & `memory`
        if self.proposed_model_mode:
            spatial = x['obj_features'][:, :, :RICH_DIM].float()
            obj_features = torch.cat(
                [x['cropped'].float(), spatial], dim=2
            ).to(self.device)
        else:
            obj_features = torch.cat(
                [x['cropped'], x['spatial']], dim=2
            ).float().to(self.device)

        # create qa_info
        a = torch.tensor(x['curr_ans']).long().to(self.device).unsqueeze(-1)
        a += self.net.vocab_size
        self.dialogue = torch.cat([self.dialogue, a], dim=1)

        with torch.no_grad():
            memory, o_pred_prob = self.net.encoder(
                obj_features,
                self.dialogue,
                n_obj=x['n_obj'].long().to(self.device),
                is_pretrain=False
            )
        self.memory = memory.transpose(1, 0).clone()
        return o_pred_prob

    def set_train_mode(self):
        """
        nn.Module mode set to train()
        """
        self.splitter.train()

    def set_eval_mode(self):
        """
        nn.Module mode set to eval()
        """
        self.net.set_model_mode(False)  # eval mode
        self.splitter.eval()

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
        # TODO: we do not training except generator (or splitter), which means
        # there is no need to save other models

        def _update_save_path(path, epoch):
            return path.replace('.pt', f'_{epoch}.pt')

        self.splitter.info_dict = save_dict
        splitter_path = _update_save_path(args.save_splitter, epoch)
        os.makedirs(os.path.dirname(splitter_path), exist_ok=True)
        torch.save(self.splitter, splitter_path)

        enc_path = _update_save_path(args.save_guesser, epoch)
        dec_path = _update_save_path(args.save_generator, epoch)
        self.net.save_models(save_dict, enc_path, dec_path)

        # else:
        #     # TODO: update loading models when test
        #     # save only learned-model, that is generator here
        #     self.generator.info_dict = save_dict
        #     qgen_path = _update_save_path(args.save_generator, epoch)
        #     os.makedirs(os.path.dirname(qgen_path), exist_ok=True)
        #     torch.save(self.generator, qgen_path)

        # Track saved_path
        self.sv_prev_qgen_path = dec_path
        self.sv_prev_guesser_path = enc_path
        self.sv_se_path = None
        self.sv_qa_path = None

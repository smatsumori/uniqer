import os
import random

import numpy as np
import torch
import torch.nn as nn

from modules.guesser.tf_guesser import TF_Encoder
from modules.question_generator.tf_generator import TF_Decoder
import modules.question_generator.program_manipulates_func as pmf
from utils.utils import load_model
from utils.loss_func import dialogue_masked_bce
from utils.consts import RICH_DIM, NUM_MAX_OBJ, BASE_DIM, SPECIAL_TOKENS

# logging
from logging import getLogger
logger = getLogger(__name__)


class SingleNet:
    """
    Preparing Single Transformer Encoder-Decoder Model
    """
    def __init__(
            self,
            device,
            obj_feature_dim,
            vocab,
            ans_size,
            dec_out_seq_len,
            load_enc_path,
            load_dec_path,
            top_k,
            cluster_size,
            image_data_version,
            dropout=0.0,
            max_batch_size=1024,
            obj_memory_only=False,
    ):
        """
        Parameters
        ----------
        device : str
        obj_feature_dim : int
            object feature dimension + split_cluster_dim
        vocab : dict
        ans_size : int
            answer vocabulary size
        load_enc_path : str
            args.load_guesser
        load_deC_path : str
            args.load_generator
        cluster_size : int
            the number of split classes
        image_data_version : str
        dropout : float, default is 0.0
        max_batch_size : int
            args.batch_size
        obj_memory_only : bool, default is False
            all of the encoder's output will be used as decoder's memory
            when False, use only memory[:NUM_MAX_OBJ]
        """
        self.device = device
        self.vocab = vocab
        self.vocab_size = len(self.vocab['program_token_to_idx'])
        self.top_k = top_k
        self.image_data_version = image_data_version

        # ---------------------------------------------------------------------
        # Encoder

        # ------------------------------------
        # Hard-code
        d_model = 512
        enc_n_head = 8
        enc_n_hid = 512
        enc_n_layers = 3
        # ------------------------------------

        self.encoder = TF_Encoder(
            self.device,
            obj_feature_dim,
            self.vocab_size,
            ans_size,
            d_model,
            enc_n_head,
            enc_n_hid,
            enc_n_layers,
            max_batch_size=max_batch_size,
            pass_all_memory_to_dec=not obj_memory_only,
            dropout=dropout,
        ).to(self.device)

        if load_enc_path is not None:
            self.encoder, _ = load_model(
                self.encoder, None, load_enc_path, self.device
            )

        self.enc_criterion = dialogue_masked_bce

        # ---------------------------------------------------------------------
        # Decoder

        # ------------------------------------
        # Hard-code
        dec_n_head = 8
        dec_n_hid = 512
        dec_n_layers = 3
        # ------------------------------------

        self.decoder = TF_Decoder(
            self.device,
            self.vocab_size,
            d_model,
            dec_n_head,
            dec_n_hid,
            dec_n_layers,
            d_model,  # memory dim
            dec_out_seq_len,
            cluster_size,
            dropout=dropout,
        ).to(self.device)

        if load_dec_path is not None:
            self.decoder, _ = load_model(
                self.decoder, None, load_dec_path, self.device
            )

        self.dec_criterion = nn.CrossEntropyLoss(ignore_index=0)

        # ---------------------------------------------------------------------
        # placeholder
        self.cls_token = (
            torch.ones(max_batch_size, 1) * self.encoder.cls_id
        ).long().to(self.device)

        # for check_mode
        self.type_name = ['color', 'shape', 'size', 'material', 'relative']
        self.dec_res, self.f1_counts = self.init_results()

    def init_results(self):
        """
        Initialize (or Reset) results placeholder

        Returns
        -------
        dec_res : dict
            Evaluate generated questions' quality
        f1_counts : np.array
            [tp, tn, fp, fn]
        """
        dec_res = {
            'total_q': 0,  # total question count
            'perfect_q': 0,  # ans set is equal to split set
            'invalid_q': 0,  # invalid question count
            'correct_q': 0,  # n_question that can be regarded as correct one
            'total_q_type': np.zeros(len(self.type_name)),
            'perfect_q_type': np.zeros(len(self.type_name)),
            'correct_q_type': np.zeros(len(self.type_name)),
        }
        f1_counts = np.zeros(4)
        return dec_res, f1_counts

    def _prepare_enc_input_qa(self, tgt, ans):
        """
        * add <CLS> token
        * concat and reshape for transformer encoder input

        Parameters
        ----------
        tgt : torch.LongTensor
            shape : (batch_size, n_question, n_max_tokens)
        ans : torch.LongTensor
            shape : (batch_size, n_question)

        Returns
        -------
        input_qa : torch.LongTensor
            shape : (batch_size, (n_max_tokens + 1) * n_question)
        """
        batch_size = tgt.shape[0]
        # to embed Q and A into same space
        _ans = (ans + self.vocab_size).unsqueeze(-1)
        input_qa = torch.cat([tgt, _ans], dim=2).reshape(batch_size, -1)
        input_qa = torch.cat([self.cls_token[:batch_size], input_qa], dim=1)
        return input_qa

    def _add_nullqa(self, obj_answer, mask, n_obj):
        """
        Update `obj_answer` & `mask` by adding null answer since
        no questions are entered to the encoder at the first iteration.

        Parameters
        ----------
        obj_answer : torch.tensor
            shape: (batch_size, n_question, n_max_objects)
        mask : torch.tensor
            shape: (batch_size, n_question, 1)
        n_obj : torch.tensor
            shape: (batch_size)

        Returns
        -------
        obj_answer : torch.tensor
            shape: (batch_size, n_question + 1, n_max_objects)
        mask : torch.tensor
            shape: (batch_size, n_question + 1, 1)
        """
        batch_size = n_obj.shape[0]

        # update obj_answer (before input first Q&A, all objects should be
        # equally treated!)
        init_o_ans = torch.eye(NUM_MAX_OBJ + 1)[n_obj][:, :NUM_MAX_OBJ]
        init_o_ans = (init_o_ans.cumsum(dim=1) != 1).long().to(self.device)
        obj_answer = torch.cat([init_o_ans.unsqueeze(1), obj_answer], dim=1)

        # update mask for the same reason
        init_mask = torch.ones(batch_size, 1, 1).long().to(self.device)
        mask = torch.cat([init_mask, mask], dim=1)
        return obj_answer, mask

    def _gen_pseudo_splits(self, org_obj_answer, obj_answer, n_obj):
        """
        Generate split_idx
        randomly assign cluster_id = 1 or cluster_id = 2
        <randomly> means cluster_id does not represent 'Yes' obj nor 'No' obj

        Parameters
        ----------
        org_obj_answer : torch.tensor
            original (that is not updated by dummy dialogue) object answer
            shape: (batch_size, n_question, n_max_objects)
        obj_answer : torch.tensor
            object answer which is updated by dummy dialogue
            shape: (batch_size, n_question, n_max_objects)
        n_obj : torch.tensor
            shape: (batch_size)

        Returns
        -------
        splits : torch.tensor
            shape: (batch_size, n_question, n_max_objects)
        """
        batch_size, n_question, _ = org_obj_answer.shape
        splits = torch.zeros_like(org_obj_answer).long().to(self.device)

        for i in range(n_question):
            splits[:, i] = get_dummy_splits_clusters(
                org_obj_answer[:, i],
                n_obj,
                self.top_k,  # which is not implemented in that function
                self.device,
            )

        # splits.shape : (batch_size, n_question, n_max_objects)

        # TODO: move following code to `get_dummy_splits_clusters`

        # ---------------------------------------------------------------------
        # pseudo top-k update
        # 1. creating null candidates mask
        n_obj = n_obj.cpu()
        null_mask = torch.zeros_like(splits)

        # each turn, mask one object (with replacement to simplify code)
        for i in range(NUM_MAX_OBJ - self.top_k):
            tmp_shape = null_mask[
                (n_obj - self.top_k - i) > 0, :, :self.top_k + i + 1
            ].shape

            # choose which objects should be masked
            tmp_index = torch.randint(0, tmp_shape[2], tmp_shape[:2])

            # ignore some replacement
            null_mask[
                (n_obj - self.top_k - i) > 0, :, :self.top_k + i + 1
            ] += torch.eye(tmp_shape[2])[tmp_index].long().to(self.device)

        null_mask = (null_mask > 0).long()

        # 2. update null mask itself
        # NOTE: If the objects are answered up to the previous turn's question,
        #       they are considered to have high priority, and they are masked,
        #       the mask will be removed here.
        # DEBUG: We need to examine whether my method reproduces the ideal
        #        split (mask) for RL
        null_mask[:, 1:] -= obj_answer[:, :-1]
        null_mask = (null_mask > 0).long()

        # 3. update splits by null mask
        splits[null_mask == 1] = 0
        return splits

    def set_model_mode(self, is_train):
        """
        Parameters
        ----------
        is_train : bool
        """
        if is_train:
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()

    def save_models(self, save_dict, enc_path, dec_path):
        """
        Parameters
        ----------
        save_dict : dict
        enc_path : str
            encoder save path
        enc_path : str
            decoder save path
        """
        self.encoder.info_dict = save_dict
        self.decoder.info_dict = save_dict
        os.makedirs(os.path.dirname(enc_path), exist_ok=True)
        os.makedirs(os.path.dirname(dec_path), exist_ok=True)
        torch.save(self.encoder, enc_path)
        torch.save(self.decoder, dec_path)

    def pretrain_net(
            self,
            data,
            is_train=True,
            enc_only=False,
            proposed_model_mode=True,
            idk_mode=False,
    ):
        """
        Parameters
        ----------
        data : batch data
        is_train : bool, default is True
        enc_only : bool, default is False
            when True, training only encoder
        proposed_model_mode : bool, default is True
        idk_mode : bool, default is False

        Returns
        -------
        loss : torch.tensor
        """
        self.set_model_mode(is_train)

        n_obj = data['n_obj'].to(self.device)

        if proposed_model_mode:
            # extract only RICH_DIM spatial features
            spatial = data['obj_features'][:, :, :RICH_DIM].float()
            obj_features = torch.cat(
                [data['obj_image'].float(), spatial], dim=2,
            ).to(self.device)

        else:
            obj_features = torch.cat(
                [data['obj_image'], data['spatial']], dim=2
            ).float().to(self.device)

        # shape has `n_question` as dim=1
        tgt = data['program'].to(self.device)
        obj_answer = data['answer'].to(self.device)

        # tgt.shape : (batch_size, n_question, n_max_tokens)
        # obj_answer: (batch_size, n_question, n_max_objects)
        # dummy_yn : (batch_size, n_question)
        # mask : (batch_size : n_question, 1)  <- For loss calculation
        tgt, obj_answer, dummy_yn, mask, org_obj_answer = choose_questions(
            tgt, obj_answer, self.device, idk_mode=idk_mode, add_eod=False,
        )
        batch_size, n_question, n_max_tokens = tgt.shape
        splits = self._gen_pseudo_splits(org_obj_answer, obj_answer, n_obj)
        obj_answer, mask = self._add_nullqa(obj_answer, mask, n_obj)

        # Encoder's QA input data
        # input_tgt.shape : (batch_size, (n_max_tokens + 1) * n_question)
        input_tgt = self._prepare_enc_input_qa(tgt, dummy_yn)

        # placeholder
        sig_obj_ls = []
        program_ls = []

        # dialogue loop
        for i_q in range(n_question + 1):
            # Encoder
            # initial run (without question)
            if i_q == 0:
                # only a <CLS> token will be given
                if is_train:
                    memory, sigmoid_obj = self.encoder(
                        obj_features,
                        self.cls_token[:batch_size],
                        is_pretrain=True,
                    )
                else:
                    with torch.no_grad():
                        memory, sigmoid_obj = self.encoder(
                            obj_features,
                            self.cls_token[:batch_size],
                            is_pretrain=True,
                        )

            else:
                # 1 + () : <CLS token>
                # n_max_tokens + 1 : question length + ans
                if is_train:
                    memory, sigmoid_obj = self.encoder(
                        obj_features,
                        input_tgt[:, :1 + i_q * (n_max_tokens + 1)],
                        is_pretrain=True,
                    )
                else:
                    with torch.no_grad():
                        memory, sigmoid_obj = self.encoder(
                            obj_features,
                            input_tgt[:, :1 + i_q * (n_max_tokens + 1)],
                            is_pretrain=True,
                        )
            memory = memory.transpose(1, 0)
            sig_obj_ls.append(sigmoid_obj)

            # Decoder
            if not enc_only:
                if i_q < n_question:
                    if is_train:
                        program = self.decoder(
                            tgt[:, i_q],
                            memory,
                            split_ids=splits[:, i_q],
                            is_train=is_train,
                        ).permute(0, 2, 1)
                    else:
                        with torch.no_grad():
                            program = self.decoder(
                                tgt[:, i_q],
                                memory,
                                split_ids=splits[:, i_q],
                                is_train=is_train,
                            ).permute(0, 2, 1)

                    program_ls.append(program)

        # calc_loss
        # 1. sigmoid obj prediction loss
        sig_obj_loss = self.enc_criterion(
            sig_obj_ls, obj_answer.float(), mask.float(), n_obj
        )

        # 2. Question Generation loss
        program_loss = torch.tensor([0.0]).to(self.device)
        if not enc_only:
            for i in range(n_question):
                program_loss += self.dec_criterion(
                    program_ls[i], tgt[:, i, 1:]
                )

        # 3. coeff
        sig_obj_loss *= 10  # TODO: argument ?
        return sig_obj_loss, program_loss

    def check_results(
            self,
            data,
            oracle,
            proposed_model_mode=True,
            idk_mode=False,
            show_example=False,
    ):
        """
        Parameters
        ----------
        data : batch data
        oracle : Oracle()
        proposed_model_mode : bool, default is True
        idk_mode : bool, default is False
        show_example : bool, default is False

        Returns
        -------
        loss : torch.tensor
        """
        # evaluation mode
        self.set_model_mode(False)

        n_obj = data['n_obj'].to(self.device)

        # TODO: create pre-process fucntion and share with pretrain_net()
        if proposed_model_mode:
            # extract only RICH_DIM spatial features
            spatial = data['obj_features'][:, :, :RICH_DIM].float()
            obj_features = torch.cat(
                [data['obj_image'].float(), spatial], dim=2,
            ).to(self.device)

        else:
            obj_features = torch.cat(
                [data['obj_image'], data['spatial']], dim=2
            ).float().to(self.device)

        # shape has `n_question` as dim=1
        tgt = data['program'].to(self.device)
        obj_answer = data['answer'].to(self.device)

        # tgt.shape : (batch_size, n_question, n_max_tokens)
        # obj_answer.(batch_size, n_question, n_max_objects)
        # dummy_yn : (batch_size, n_question)
        # mask : (batch_size, n_question, 1)  <- For loss calculation
        tgt, obj_answer, dummy_yn, mask, org_obj_answer = choose_questions(
            tgt, obj_answer, self.device, idk_mode=idk_mode, add_eod=False,
        )
        batch_size, n_question, n_max_tokens = tgt.shape
        splits = self._gen_pseudo_splits(org_obj_answer, obj_answer, n_obj)
        obj_answer, mask = self._add_nullqa(obj_answer, mask, n_obj)

        # Encoder's QA input data
        # input_tgt.shape : (batch_size, (n_max_tokens + 1) * n_question)
        input_tgt = self._prepare_enc_input_qa(tgt, dummy_yn)

        # placeholder
        sig_obj_ls = []
        program_ls = []

        # =====================================================================
        # inference part
        # =====================================================================
        with torch.no_grad():
            # dialogue loop
            for i_q in range(n_question + 1):
                # Encoder
                if i_q == 0:
                    memory, sigmoid_obj = self.encoder(
                        obj_features,
                        self.cls_token[:batch_size],
                        is_pretrain=True,
                    )
                else:
                    memory, sigmoid_obj = self.encoder(
                        obj_features,
                        input_tgt[:, :1 + i_q * (n_max_tokens + 1)],
                        is_pretrain=True,
                    )
                memory = memory.transpose(1, 0)
                sig_obj_ls.append(sigmoid_obj)

                # Decoder
                if i_q < n_question:
                    program = self.decoder(
                        tgt[:, i_q],
                        memory,
                        split_ids=splits[:, i_q],
                        is_train=False,
                    ).permute(0, 2, 1)
                    program_ls.append(program)

        # =====================================================================
        # results check part
        # =====================================================================

        # 1. encoder
        for i in range(batch_size):
            image_filename = os.path.basename(data['image_filename'][i])

            for t in range(n_question):  # dialogue loop
                if mask[i][t] == 0:
                    break
                sigmoid_obj = sig_obj_ls[t]
                ans_obj_set = []
                for obj_idx in range(n_obj[i]):
                    if obj_answer[i][t][obj_idx] == 1:
                        ans_obj_set.append(obj_idx)

                for obj_idx in range(n_obj[i]):
                    if sigmoid_obj[i, obj_idx] > 0.5 and \
                       obj_idx in ans_obj_set:
                        self.f1_counts[0] += 1  # tp
                    elif sigmoid_obj[i, obj_idx] > 0.5:
                        self.f1_counts[2] += 1  # fp
                    elif obj_idx not in ans_obj_set:
                        self.f1_counts[1] += 1  # tn
                    else:
                        self.f1_counts[3] += 1  # fn

        # 2. decoder
        splits = splits.cpu().numpy()
        for t in range(n_question):  # dialogue loop
            programs = program_ls[t].cpu().numpy()
            programs = np.argmax(programs, axis=1)
            for i in range(len(programs)):
                self.dec_res['total_q'] += 1
                image_filename = os.path.basename(data['image_filename'][i])
                ans = set(oracle._answering_question(
                    image_filename, programs[i], require_pp=True
                ))

                # question_type
                q_type = pmf.get_q_type(
                    programs[i],
                    self.vocab['program_idx_to_token'],
                    self.image_data_version
                )
                self.dec_res['total_q_type'] += q_type

                if ans == {-1}:
                    # question is invalid
                    self.dec_res['invalid_q'] += 1
                else:
                    cls1 = set(np.where(splits[i, t] == 1)[0])
                    cls2 = set(np.where(splits[i, t] == 2)[0])
                    # perfectly divided (strict)
                    if cls1 == ans or cls2 == ans:
                        self.dec_res['perfect_q'] += 1
                        self.dec_res['perfect_q_type'] += q_type

                    # regard as correct question
                    if (cls1.issubset(ans) and len(cls2 & ans) == 0) or \
                       (cls2.issubset(ans) and len(cls1 & ans) == 0):
                        self.dec_res['correct_q'] += 1
                        self.dec_res['correct_q_type'] += q_type

        # 3. encoder example
        if show_example:
            tgt = tgt.detach().cpu().numpy()
            for idx in range(min(batch_size, 10)):
                image_filename = os.path.basename(data['image_filename'][idx])

                logger.info(f'========== Question {idx} ==========')
                logger.info(f'image_filename: {image_filename}')
                for t in range(tgt.shape[1]):
                    if mask[idx][t] == 0:
                        break
                    str_program = pmf.decode(
                        tgt[idx, t], self.vocab['program_idx_to_token']
                    )
                    logger.info(f'str_program [ {t} ]: {str_program}')
                    logger.info(f'[0-No, 1-Yes, 2-Nan]: {dummy_yn[idx, t]}')
                    ans_obj_set = []
                    pred_obj_set = []
                    for obj_idx in range(n_obj[idx]):
                        if obj_answer[idx][t][obj_idx] == 1:
                            ans_obj_set.append(obj_idx)
                        if sig_obj_ls[t][idx, obj_idx] >= 0.5:
                            pred_obj_set.append(obj_idx)
                    logger.info(f'\tAnswer Object Set   : {ans_obj_set}')
                    logger.info(f'\tPredicted Object Set: {pred_obj_set}')
                    logger.info(
                        'Predicted sigmoid Obj: '
                        f'{sig_obj_ls[t][idx, :n_obj[idx]]}'
                    )

        # 4. decoder example
        if show_example:
            _program_ls = []
            for t in range(n_question):
                programs = program_ls[t].cpu().numpy()
                programs = np.argmax(programs, axis=1)
                _program_ls.append(programs)

            for i in range(min(len(programs), 10)):
                image_filename = os.path.basename(data['image_filename'][i])
                logger.info(f'===== Image: {image_filename} =====')
                for t in range(n_question):
                    programs = _program_ls[t]
                    raw_str_program = pmf.decode(
                        programs[i],
                        self.vocab['program_idx_to_token'],
                        check_mode=True,
                    )
                    logger.info(
                        f'str_program - {i} - turn {t} : {raw_str_program}'
                    )

    def show_total_results(self):
        # Encoder
        logger.info('=' * 80)
        tp = self.f1_counts[0]
        fp = self.f1_counts[1]
        tn = self.f1_counts[2]
        fn = self.f1_counts[3]

        logger.info(f'tp_count : {tp}')
        logger.info(f'tn_count : {fp}')
        logger.info(f'fp_count : {tn}')
        logger.info(f'fn_count : {fn}')
        logger.info(f'F1_score : {tp / (tp + (tn + fn) / 2):.3f}')

        # Decoder
        total_q = self.dec_res['total_q']
        perfect_q = self.dec_res['perfect_q']
        invalid_q = self.dec_res['invalid_q']
        correct_q = self.dec_res['correct_q']
        total_q_type = self.dec_res['total_q_type']
        perfect_q_type = self.dec_res['perfect_q_type']
        correct_q_type = self.dec_res['correct_q_type']

        # TODO: color ? material ? shape ?
        logger.info('=' * 80)
        logger.info(f'Total question count: {total_q}')
        for i in range(len(self.type_name)):
            logger.info(f'Total {self.type_name[i]} Q count: {total_q_type[i]}'
                        f'--> {total_q_type[i] * 100 / total_q:.2f} %')

        # perfect (strictly correct)
        logger.info('-' * 80)
        logger.info(f'Perfect Q count: {perfect_q} '
                    f'--> {perfect_q * 100 / total_q:.2f} %')

        for i in range(len(self.type_name)):
            logger.info(f'Perfect {self.type_name[i]} Q /Perfect Q = '
                        f'{perfect_q_type[i] * 100 / perfect_q:.2f} %')

        # correct
        logger.info('-' * 80)
        logger.info(f'Correct Q count: {correct_q} '
                    f'--> {correct_q * 100 / total_q:.2f} %')
        for i in range(len(self.type_name)):
            logger.info(f'Correct {self.type_name[i]} Q /Correct Q = '
                        f'{correct_q_type[i] * 100 / correct_q:.2f} %')

        # invalid
        logger.info('-' * 80)
        logger.info(f'Invalid Q count: {invalid_q} '
                    f'--> {invalid_q * 100 / total_q:.2f} %')

def choose_questions(
        tgt, obj_answer, device, turn_limit=5, idk_mode=False, add_eod=True,
):
    """
    Creating pseudo-dialogue-datasets
    * NOTE: When calculating loss value, n_obj will be considered and masked,
            therefore, we don't care n_obj here
    * TODO: eod should be generated just after len(sum(obj_answer)) == 1
    ~~~ STEPs ~~~
    1. Randomly rearrange the questions which are tied to the image
    2. After the length of ans_obj_set is one, that is, no object in the image
       can be a candidate, mask subsequent questions and dummy answers.
    3. When the length of ans_obj_set has been narrowed down to 1, update
       the next question as <STOP>(end-of-dialogue) token, then mask subsequent
       questions and dummy answers.
       NOTE: This <STOP> token will be useful when training Question
             Generator. We do not care while training Guesser, QAEncoder, and
             StateEncoder.
    Parameters
    ----------
    tgt : torch.LongTensor
        (shape: batch_size, n_src_questions, n_max_tokens)
    obj_answer : torch.LongTensor
        (shape: batch_size, n_src_questions, n_max_objects)
    device : string
    turn_limit : int, default is 5
        maximum turn in a dialogue (padded if shorter dialogue are generated)
    idk_mode : bool, default is False
        When "I don't know" mode is activated, special treatement for
        obj_answer will be applied, details are in the following code.
    add_eod : bool, default is True
        True --> adding <STOP> <STOP> at the last
        False --> [<NULL> ... <NULL>]
    Returns
    -------
    tgt : torch.LongTensor
        Generated dialogue.
        (shape: batch_size, turn_limit, n_max_tokens)
    obj_answer : torch.LongTensor
        Represented by 1 (candidate) or 0 (not candidate).
        NOTE: To remove the bias in making dataset, we generate `dummy_yn`
              randomly, then naturally, there are many cases in which no object
              fits to the dialogue at the end, but if the Guesser model is
              good enough, this should not be a problem.
        (shape: batch_size, turn_limit, n_max_objects)
    dummy_yn : torch.LongTensor
        Randomly generated [yes/no] (1/0) sequence for the pseudo-dialogue.
        (shape: batch_size, turn_limit)
    mask : torch.LongTensor
        1: valid_question, 0: should be ignored
        This mask should be used when calculating loss
        (shape: batch_size, turn_limit, 1)
    org_obj_answer : torch.LongTensor
        original obj_answer (extract corresponds to the questions chosen)
        (shape: batch_size, turn_limit, n_max_objects)
    """
    batch_size, src_q_len, n_max_tokens = tgt.shape

    # safety
    if src_q_len < turn_limit:
        logger.error(
            f'You should choose `turn_limit` variable be less than {src_q_len}'
        )
        sys.exit()

    if idk_mode:
        # dummy answer ({1: 'yes', 0: 'no', 3: 'IDK'})
        _dummy_yn = np.random.choice([0, 1, 3], (batch_size, turn_limit))
        dummy_yn = torch.tensor(_dummy_yn).to(device)
    else:
        # dummy answer ({1: 'yes', 0: 'no'})
        dummy_yn = torch.randint(0, 2, (batch_size, turn_limit)).to(device)

    # ----------------
    #     Step 1
    # ----------------
    # randomly choose `turn_limit`(= max turn) questions (in random ordered)
    lens = torch.LongTensor(
        [random.sample(range(tgt.shape[1]), turn_limit)]
    ).unsqueeze(-1).to(device)

    # extraction
    lens_tgt = lens.repeat(batch_size, 1, tgt.shape[2])
    tgt = torch.gather(tgt, 1, lens_tgt)
    lens_ans = lens.repeat(batch_size, 1, obj_answer.shape[2])
    obj_answer = torch.gather(obj_answer, 1, lens_ans)
    org_obj_answer = obj_answer.clone()

    # Update-1: obj_answer based on dummy yn (yes: 1, no: 0)
    # The original `obj_answer` is assumed to be an object set for
    # `yes`, so it needs to be inverted for the input of `no`.
    obj_answer[dummy_yn == 0] = 1 - obj_answer[dummy_yn == 0]

    # Update-2: (only for idk_mode)
    # If it was the first question, [1, 1,...]
    # Otherwise, do not update from previous results (which means, here,
    # copy the previous obj_answer)
    if idk_mode:
        # NOTE: In idk_mode, we are no longer able to use BCELoss
        obj_answer = obj_answer.float()
        for i in range(turn_limit):
            # helper matrix
            helper = torch.ones_like(obj_answer)
            helper[:, 1:] = obj_answer[:, :-1].clone()
            obj_answer[:, i][dummy_yn[:, i] == 3] = \
                helper[:, i][dummy_yn[:, i] == 3].clone()

    # Update-3: obj_answer for a series of questions
    for i in range(obj_answer.shape[1] - 1):
        obj_answer[:, i + 1] = obj_answer[:, i + 1] * obj_answer[:, i]

    # ----------------
    #     Step 2
    # ----------------
    # After the length of ans_obj_set is one, that is, no object in the image
    # can be a candidate, mask subsequent questions and dummy answers.
    # NOTE: There is still a possibility that the question will proceed
    #       with the number of ans_objects being 1 by following scripts.
    #       Although it is totally possible to exclude these possibilities,
    #       we decided to ignore the influence here considering the
    #       computational cost. (trade-off)
    #       You may be able to create a "perfect" dataset at first and save it,
    #       then reuse.
    # TODO: Once a candidate has been narrowed down to one, it is probably
    #       better to learn so that the next question produces an EOD.
    tmp = obj_answer.sum(dim=2)  # shape: batch_size, turn_limit
    tmp = tmp < 1  # "n_ans is less than 1" flag

    # mask represents valid questions in a "dialogue"
    mask = (tmp == 0).unsqueeze(-1).long()  # shape: batch_size, turn_limit, 1
    tgt = tgt * mask

    # ----------------
    #     Step 3 [optional for add_eod]
    # ----------------
    # When the length of ans_obj_set has been narrowed down to 1, update
    # the next question as <STOP>(end-of-dialogue) token, then mask subsequent
    # questions and dummy answers.
    # eod_mask.shape: (batch_size, turn_limit, 1)
    #     A mask matrix of only the parts that need to be put eod (<STOP>)
    #     right after <START> token.
    #     The position where the mask matrix switches from 1 to 0 is where
    #     eod token should be placed.
    if add_eod:
        _mask = mask.clone().repeat(
            1, 1, tgt.shape[2]
        ).reshape([batch_size, -1])
        eod_mask = torch.zeros_like(_mask).to(device)
        eod_mask[:, 2:] += (_mask[:, 1:-1] - _mask[:, :-2])
        eod_mask = eod_mask.reshape([batch_size, turn_limit, n_max_tokens])
        tgt += (SPECIAL_TOKENS['<STOP>'] * (eod_mask != 0).long())
        tgt[:, :, 0] = SPECIAL_TOKENS['<START>']
    return tgt, obj_answer, dummy_yn, mask, org_obj_answer

def get_dummy_splits_clusters(obj_answer, n_obj, top_k, device, discard=False):
    """
    Create dummy cluster indices for pre-training phase
    (In RL phase, these indices will be predicted by RL Network)
    Parameters
    ----------
    obj_answer : torch.LongTensor
        (shape: batch_size, n_max_objects)
    n_obj : torch.LongTensor
        (shape: batch_size)
    top_k : int
        only k objects will be divided into clusters
        currently, `top` does not mean anything in the pre-training (random)
    device : string
    discard : bool, default is False
        when True, cluster_id can be 3 to represents discarded candidate obj
    Returns
    -------
    splits: torch.LongTensor
        (shape: batch_size, n_max_objects)
        cluster_id = {
            0: null,
            1 & 2: cluster_id,
            3: [OPTIONAL] discarded candidate objects,  # TODO
        }
    """
    if discard:
        # TODO: impl.
        raise NotImplementedError

    batch_size = obj_answer.shape[0]

    # splits: shape
    splits = torch.zeros_like(obj_answer).long().to(device)

    # randomly assign cluster_id = 1 or cluster_id = 2
    # 'randomly' means cluster_id does not represent 'Yes' obj nor 'No' obj
    random_assign = torch.randint(0, 2, (batch_size,)).to(device)

    # [random_assign : 0] 'Yes' obj will be cluster_id = 1
    # obj_answer2: Yes --> 1, NO --> 2
    obj_answer2 = 2 - obj_answer
    splits[random_assign == 0] = obj_answer2[random_assign == 0]

    # [random_assign : 1] 'Yes' obj will be cluster_id = 2
    # obj_answer3: Yes --> 2, No --> 1
    obj_answer3 = 1 + obj_answer
    splits[random_assign == 1] = obj_answer3[random_assign == 1]

    # null_mask (shape: batch_size, max_n_obj)
    null_mask = torch.eye(NUM_MAX_OBJ + 1)[n_obj][:, :NUM_MAX_OBJ]
    null_mask = null_mask.cumsum(dim=1)
    splits[null_mask > 0] = 0  # null

    # TODO: discarded cands

    # TODO: top_k : how ?
    # k_over_mask : (shape: batch_size)
    # k_over_mask = (splits > 0).sum(axis=1) > top_k
    # n_obj[k_over_mask]
    return splits
import torch
import torch.nn as nn

import modules.transformer.transformer as trm  # pytorch1.2 base
from modules.guesser.guesser import gen_mask, masked_softmax
# from modules.question_generator.tf_generator import PositionalEncoder1D
# import modules.transformer.transformer_1_7 as trm  # pytorch1.7 base
from utils.consts import MAX_POS_EMBEDDING, NUM_MAX_OBJ, SPECIAL_TOKENS

# logging
from logging import getLogger
logger = getLogger(__name__)


class PositionalEncoder2D(nn.Module):
    """
    2D Positional Encoder (for TF)

        * TODO Future Works
    """
    pass


class TF_Encoder(nn.Module):
    """
    Transformer Guesser
        * TODO: rename ? or deprecate tf_generator.py/TF_Encoder(nn.Module) ?
        * TODO: context image ?
        * TODO: docstring
    """
    def __init__(
            self,
            device,
            obj_feature_dim,
            vocab_size,
            ans_size,
            d_model,
            n_head,
            n_hid,
            n_layers,
            max_batch_size=1024,
            pass_all_memory_to_dec=True,
            dropout=0.0,
    ):
        """
        Parameters
        ----------
        device : str
        obj_feature_dim : int
            object feature dimension + split_cluster_dim
        vocab_size : int
            question vocabulary size
        ans_size : int
            answer vocabulary size
        d_model : int
            the number of expected features in the input
        n_head : int
            the number of heads in the multi-head-attention models
            if n_head > 1, attention weight will be averaged for now
        n_hid : int
            the dimension of the feed-forward network
        n_layers : int
        max_batch_size : int
            maybe not good idea
        pass_all_memory_to_dec : bool, default is True
            all of the encoder's output will be used as decoder's memory
            when False, use only memory[:NUM_MAX_OBJ]
        dropout : float, default is 0.0
        """
        super(TF_Encoder, self).__init__()
        self.device = device
        self.pass_all_memory_to_dec = pass_all_memory_to_dec

        # CLS token will be used as sentence embedding
        logger.warning('We add <CLS> token as a special token here')
        self.vocab_size = vocab_size
        self.add_vocab_size = 1  # for <CLS> token
        self.cls_id = self.vocab_size + ans_size

        # ---------------------------------------------------------------------
        # 1. Core Encoder (Transformer)
        encoder_layer = trm.TransformerEncoderLayer(
            d_model, n_head, n_hid, dropout=dropout,
            # activation='relu'  # torch-1.2 does not support
        )
        # Transformer Encoder Model
        self.tf_encoder = trm.TransformerEncoder(encoder_layer, n_layers)

        # ---------------------------------------------------------------------
        # 2. Obj Feature Embedding
        self.objf_2_tf = nn.Sequential(
            nn.Linear(obj_feature_dim, d_model),
            nn.ReLU(),
        )

        # ---------------------------------------------------------------------
        # 3. QA Embedding
        # NOTE: ans value should be update (+self.vocab_size)
        #       --> see single_tf._prepare_enc_input_qa()
        self.qa_2_tf = nn.Embedding(
            self.vocab_size + ans_size + self.add_vocab_size, d_model
        )

        # ---------------------------------------------------------------------
        # 4. Belief Extracting
        # NOTE: no longer used self.out_2_opred
        self.out_2_opred = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

        # TODO: rename ?
        self.obj_convert = nn.Linear(d_model, d_model // 4)
        self.cls_convert = nn.Linear(d_model, d_model // 4)

        # ---------------------------------------------------------------------
        # 5. Positional Encoding for QA
        # NOTE: 0 for obj_features, [1, 2...] for dialogue
        # NOTE: It can be buggy if a length that has never been used
        #       during training comes up during inference.
        # self.pos_encoder = PositionalEncoder1D(d_model)
        self.pos_encoder = nn.Embedding(MAX_POS_EMBEDDING, d_model)
        self.pos_enc_base = self._init_pos_enc_base(max_batch_size)

        # ---------------------------------------------------------------------
        # 6. Segment Encoding ?
        # 0: null, 1: Image, 2: Q&A
        self.seg_encoder = nn.Embedding(3, d_model)

    def _init_pos_enc_base(self, max_batch_size):
        """
        Get base tensor to calculate pos_enc

        Parameters
        ----------
        max_batch_size : int

        Returns
        -------
        base : torch.tensor
            shape: (max_batch_size, MAX_POS_EMBEDDING)
        """
        base = torch.zeros(max_batch_size, MAX_POS_EMBEDDING)
        _base = torch.ones(max_batch_size, MAX_POS_EMBEDDING - NUM_MAX_OBJ)
        _base = torch.cumsum(_base, axis=1)
        base[:, NUM_MAX_OBJ:] = _base
        return base.long().to(self.device)

    def set_device(self, device):
        """
        Set device attribute

        Parameters
        ----------
        device : str
        """
        self.device = device
        self.pos_enc_base = self.pos_enc_base.to(self.device)
        logger.info(f'TF_Encoder device updated to {self.device}')

    def get_belief(self, memory, n_obj=None, is_pretrain=True):
        """
        Somehow, extracting o_pred_prob

        Parameters
        ----------
        memory : torch.tensor
            [option1 & 2]
                shape: (NUM_MAX_OBJ, batch_size, d_model)
            [option3]
                objects + <CLS> token
                shape: (NUM_MAX_OBJ + 1, batch_size, d_model)
        n_obj : torch.LongTensor, default is None
        is_pretrain : bool, default is True
            True --> Sigmoid
            False (RL-mode) --> Softmax

        Returns
        -------
        o_pred_prob : torch.tensor
            is_pretrain True  --> Sigmoid
                        False --> Sigmoid -> Softmax
        """

        # # FAIL: [option 1] softmax(QK)
        # # >>> o_pred_prob : torch.Size([batch_size, query_len, key_len])
        # TODO: how to get (batch_size, NUM_MAX_OBJ) ?
        # NOTE: what we need is not attention while pre-training,
        #       we should do multi-label classification(?)
        # NEXT: softmax(<CLS> <OBJECTS>) would be an idea
        #       (though we still have a same problem while pre-training)
        # o_pred_prob = self.tf_encoder.get_attention()

        # # FAIL: [option 2] output --> Linear
        # m = memory.transpose(1, 0)
        # o_pred_prob = self.out_2_opred(m).squeeze(-1)

        # # FAIL: [option 3] <CLS> --> score (direct)
        # m = memory.transpose(1, 0)  # batch_size, n_obj + 1, d_model
        # score = (m[:, :-1] * m[:, -1].unsqueeze(1)).sum(dim=2)
        # o_pred_prob = torch.sigmoid(score)

        # [option 3-1] <CLS> --> score (with Linear)
        m = memory.transpose(1, 0)  # batch_size, n_obj + 1, d_model
        _obj = self.obj_convert(m[:, :-1])
        _cls = self.cls_convert(m[:, -1].unsqueeze(1))
        o_pred_prob = torch.sigmoid((_obj * _cls).sum(dim=2))

        # RL mode --> probability
        if not is_pretrain:
            mask = gen_mask(NUM_MAX_OBJ, n_obj)
            o_pred_prob = masked_softmax(o_pred_prob, mask, dim=1)

        return o_pred_prob

    def forward(self, obj_feature, qa, n_obj=None, is_pretrain=True):
        """
        Parameters
        ----------
        obj_feature : torch.tensor
            object feature (spatial + cropped image feature)
            shape: (batch_size, NUM_MAX_OBJ, feature_dim)
        qa : torch.LongTensor
            question & answer (token id)
                answer should be updated by vocab_size to embed correctly
            shape: (batch_size, qa_len)
        n_obj : torch.LongTensor, default is None
        is_pretrain : bool, default is True

        Returns
        -------
        memory : torch.tensor
            (NUM_MAX_OBJ + qa_len, batch_size, d_dim)
        o_pred_prob : torch.tensor
            (batch_size, NUM_MAX_OBJ)
        """
        b_size, qa_len = qa.shape

        # obj_features
        obj_feature = self.objf_2_tf(obj_feature)

        # question and answer
        qa_embed = self.qa_2_tf(qa)

        # position encoding
        pos_embed = self.pos_encoder(
            self.pos_enc_base[:b_size, :qa_len + NUM_MAX_OBJ]
        )

        # segment encoding
        seg_base = torch.zeros(b_size, NUM_MAX_OBJ + qa_len)
        seg_base[:, :NUM_MAX_OBJ] = 1
        seg_base[:, NUM_MAX_OBJ:][qa != SPECIAL_TOKENS['<NULL>']] = 2
        seg_embed = self.seg_encoder(seg_base.long().to(self.device))

        tf_input = torch.cat([obj_feature, qa_embed], dim=1)
        tf_input = tf_input + pos_embed + seg_embed
        tf_input = tf_input.transpose(1, 0)

        # transformer
        memory = self.tf_encoder(tf_input)

        # objects + <CLS>
        o_pred_prob = self.get_belief(
            memory[:NUM_MAX_OBJ + 1], n_obj, is_pretrain
        )

        # TODO: deprecates [Temporary]
        if not hasattr(self, 'pass_all_memory_to_dec'):
            self.pass_all_memory_to_dec = True

        if not self.pass_all_memory_to_dec:
            # TODO: <CLS> will not be important ?
            # NOTE: maybe we can generate <STOP> by decoder non ?
            # Send only the output corresponding to the part of the input
            # object feature to the decoder as memory
            memory = memory[:NUM_MAX_OBJ + 1]

        return memory, o_pred_prob


if __name__ == '__main__':
    # DEBUG PURPOSE
    obj_feature_dim = 300
    batch_size = 24
    seq_len = 15
    vocab_size = 30
    ans_size = 3
    d_model = 256
    n_head = 4
    n_hid = 128
    n_layers = 3

    obj_feature = torch.rand(batch_size, NUM_MAX_OBJ, obj_feature_dim)
    qa = torch.randint(0, vocab_size, (batch_size, seq_len))

    model = TF_Encoder(
        'cpu',
        obj_feature_dim,
        vocab_size,
        ans_size,
        d_model,
        n_head,
        n_hid,
        n_layers,
    )

    memory, o_pred_prob = model(obj_feature, qa)
    print(f'>>> memory : {memory.shape}')
    print(f'>>> o_pred_prob : {o_pred_prob.shape}')
    # memory : num_max_obj + seq_len, batch_size, d_model
    # >>> memory : torch.Size([25, 24, 256])
    # >>> o_pred_prob : torch.Size([24, 10])
    print(o_pred_prob[0].sum())

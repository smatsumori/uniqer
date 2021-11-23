import math

import torch
import torch.nn as nn

import modules.transformer.transformer as trm  # pytorch1.2 base
# import modules.transformer.transformer_1_7 as trm  # pytorch1.7 base
from utils.consts import SPECIAL_TOKENS, NUM_MAX_OBJ

# logging
from logging import getLogger
logger = getLogger(__name__)


class PositionalEncoder1D(nn.Module):
    """
    1D Positional Encoder (for TF)

    * TODO: move to transformer.py
    """
    def __init__(self, d_model, dropout=0.1, max_len=100):
        """
        * TODO: max_len should be specified by args.seqlen?

        Parameters
        ----------
        d_model : int
            the number of expected features in the input
        dropout : float, default is 0.0
        max_len : int, default is 100
        """
        super(PositionalEncoder1D, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.tensor
            shape: (batch_size, tgt_len, d_model)

        Returns
        -------
        x : torch.tensor
            shape: (batch_size, tgt_len, d_model)
        """
        x = x + self.pe[:, :x.size(1)]
        x = self.dropout(x)
        return x


class TF_Encoder(nn.Module):
    """
    TODO: similar to `guesser.py TF_Guesser() model` --> refactoring!
    Transformer Encoder(for seq2seq) Question Generator Model

        * NOTE: positional encoding is not used currently
        * NOTE: maybe pos-enc for object position ?
    """
    def __init__(
            self,
            device,
            d_model,
            n_head,
            n_hid,
            n_layers,
            obj_feature_dim,
            dropout=0.0,
    ):
        """
        Parameters
        ----------
        device : str
        d_model : int
            the number of expected features in the input
        n_head : int
            the number of heads in the multi-head-attention models
        n_hid : int
            the dimension of the feed-forward network
        n_layers : int
        obj_feature_dim : int
            object feature dimension + split_cluster_dim
        dropout : float, default is 0.0
        """
        super(TF_Encoder, self).__init__()
        self.device = device

        # TODO: dropout default is 0.1 (PyTorch documentation)
        encoder_layer = trm.TransformerEncoderLayer(
            d_model, n_head, n_hid, dropout=dropout,
            # activation='relu'  # torch-1.2 does not support
        )

        # Transformer Encoder Model
        self.tf_encoder = trm.TransformerEncoder(encoder_layer, n_layers)

        # Embedding input to transformer input dim
        self.objf_2_tf = nn.Sequential(
            nn.Linear(obj_feature_dim, d_model),
            nn.ReLU(),
        )

    def set_device(self, device):
        """
        Set device attribute

        Parameters
        ----------
        device : str
        """
        self.device = device
        logger.info(f'TF_Encoder device updated to {self.device}')

    def forward(self, obj_features):
        """
        Parameters
        ----------
        obj_features : torch.tensor
            shape: (batch_size, num_max_obj, obj_feature_dim)

        Returns
        -------
        out : torch.tensor
            shape: (batch_size, num_max_obj, d_model)
        """
        embed_f = self.objf_2_tf(obj_features)
        embed_f = embed_f.transpose(1, 0)
        out = self.tf_encoder(embed_f).transpose(1, 0)
        return out


class TF_Decoder(nn.Module):
    """
    Transformer Decoder Question Generator Model
    """
    def __init__(
            self,
            device,
            vocab_size,
            d_model,
            n_head,
            n_hid,
            n_layers,
            obj_feature_dim,
            out_seq_len,
            cluster_size,
            dropout=0.0,
    ):
        """
        Parameters
        ----------
        device : str
        vocab_size : int
            vocabulary size
        d_model : int
            the number of expected features in the input
        n_head : int
            the number of heads in the multi-head-attention models
        n_hid : int
            the dimension of the feed-forward network
        n_layers : int
        obj_feature_dim : int
            object feature dimension
        out_seq_len : int
            Output Sequence length
        cluster_size : int
            the number of split classes
        dropout : float, default is 0.0
        """
        super(TF_Decoder, self).__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.cluster_size = cluster_size
        self.out_seq_len = out_seq_len

        # TODO: dropout default is 0.1 (PyTorch documentation)
        decoder_layer = trm.TransformerDecoderLayer(
            d_model, n_head, n_hid, dropout=dropout,
            # activation='relu'  # torch-1.2 does not support
        )

        # Positional Encoding
        self.pos_encoder = PositionalEncoder1D(d_model)

        # Transformer Decoder Model
        self.tf_decoder = trm.TransformerDecoder(decoder_layer, n_layers)

        # Embedding token to input transformer
        self.enc_4_tf = nn.Embedding(self.vocab_size, d_model)

        # Encoding split_id (for single-transformer mode)
        # 0, 1, ..., cluster_size - 1 for objects, cluster_size for QA Part
        self.split_encoder = nn.Embedding(self.cluster_size + 1, d_model)

        # From d_model to vocab_size probability
        self.tf_2_out = nn.Sequential(
            nn.Linear(d_model, self.vocab_size),
            nn.Softmax(dim=2)
        )

        # Simple Version (We may replace this submodule with TF_encoder later)
        # TODO: deprecates?
        #       seq2seq model uses trm_encoder or GRU_encoder and train
        #       together, therefore, self.obj_linear may not be necessary.
        self.obj_linear = nn.Sequential(
            nn.Linear(obj_feature_dim, d_model),
            nn.ReLU(),
        )

    def set_device(self, device):
        """
        Set device attribute

        Parameters
        ----------
        device : str
        """
        self.device = device
        logger.info(f'TF_Decoder device updated to {self.device}')

    @staticmethod
    def gen_square_mask(sz):
        """
        Generate a square mask for the sequence.
            Original is nn.Transformer().generate_square_subsequent_mask()

        Parameters
        ----------
        size : int

        Returns
        -------
        mask : torch.tensor
            The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(
            mask == 0, float('-inf')
        ).masked_fill(mask == 1, float(0.0))
        return mask

    def gen_split_ids_embedding(self, split_ids, src_len):
        """

        Parameters
        ----------
        split_ids : torch.LongTensor, default is None
            shape: (batch_size, num_max_obj)

        Returns
        -------
        split_embedding : torch.tensor
            shape: (src_len, batch_size, d_model)
        """
        base = torch.ones(split_ids.shape[0], src_len) * (self.cluster_size)
        base = base.long().to(self.device)
        base[:, :NUM_MAX_OBJ] = split_ids
        split_embedding = self.split_encoder(base)
        return split_embedding.transpose(1, 0)

    def forward(self, tgt, obj_features, split_ids=None, is_train=True):
        """
        * NOTE: tgt is now NOT one-hot vector!

        Parameters
        ----------
        tgt : torch.tensor
            shape: (batch_size, tgt_len)
        obj_features : torch.tensor
            or memory (encoder output)
            shape: (batch_size, memory_dim, obj_feature_dim)
        split_ids : torch.LongTensor, default is None
            for single transformer mode
            shape: (batch_size, num_max_obj)
        is_train: bool, default is True

        Returns
        -------
        out : torch.tensor
            shape: (batch_size, tgt_len, vocab_size)
        """
        batch_size, memory_dim, _ = obj_features.shape
        obj_features = obj_features.transpose(1, 0)

        # memory part
        # NOTE: Does it make sense to add positional embedding here?
        embed_obj_f = self.obj_linear(obj_features)
        # embed_obj_f = self.pos_encoder(embed_obj_f)

        if split_ids is not None:
            split_embedding = self.gen_split_ids_embedding(
                split_ids, memory_dim
            )
            embed_obj_f += split_embedding

        if is_train:
            # Training Mode
            # -------------
            embed_tgt = self.enc_4_tf(tgt[:, :-1])  # TODO: DEBUG
            embed_tgt = self.pos_encoder(embed_tgt)
            embed_tgt = embed_tgt.transpose(1, 0)

            # mask
            tgt_mask = self.gen_square_mask(embed_tgt.shape[0]).to(self.device)

            # TODO: DEBUG: why ?
            # logger.warning('check tgt_mask behavior')

            # Transformer decoder
            out = self.tf_decoder(embed_tgt, embed_obj_f, tgt_mask=tgt_mask)

            # probability
            out_prob = self.tf_2_out(out)
            output = out_prob.transpose(1, 0)
        else:
            # Inference Mode
            # --------------
            # token output
            output_token = torch.zeros(
                batch_size, self.out_seq_len + 1
            ).long().to(self.device) * SPECIAL_TOKENS['<NULL>']
            output_token[:, 0] = SPECIAL_TOKENS['<START>']

            # probability
            output = torch.zeros(
                batch_size, self.out_seq_len, self.vocab_size
            ).to(self.device)

            for t in range(1, self.out_seq_len + 1):
                embed_tgt = self.enc_4_tf(output_token[:, :t])
                embed_tgt = self.pos_encoder(embed_tgt)
                embed_tgt = embed_tgt.transpose(1, 0)

                # NOTE: inference mode does not require tgt_mask non?
                tgt_mask = self.gen_square_mask(t).to(self.device)
                out = self.tf_decoder(
                    embed_tgt,
                    embed_obj_f,
                    tgt_mask=tgt_mask
                )

                # out_prob : shape (batch_size, vocab_size)
                out_prob = self.tf_2_out(out)[-1]
                output_t = out_prob.data.topk(1)[1].squeeze()
                output_token[:, t] = output_t
                output[:, t - 1] = out_prob
        return output


if __name__ == '__main__':
    # DEBUG PURPOSE
    # ADD the following codes
    # -------------------------------------------------------------------------
    # import sys
    # sys.path.append('src/')
    # -------------------------------------------------------------------------

    # dummy data
    batch_size = 4
    vocab_size = 15
    d_model = 256
    n_head = 4
    n_hid = 128
    n_layers = 2
    obj_feature_dim = 128
    out_seq_len = 30

    tgt_len = 50

    obj_f = torch.rand(batch_size, 10, obj_feature_dim)  # obj_max_size = 10

    print('>>>>> TF Encoder <<<<<')

    encoder = TF_Encoder(
        'cpu',
        d_model,
        n_head,
        n_hid,
        n_layers,
        obj_feature_dim,
        dropout=0.0
    )
    output = encoder(obj_f)
    print(f'>>> output.shape: {output.shape}')
    # >>> output.shape: torch.Size([4, 10, 256])

    print('>>>>> TF Decoder <<<<<')

    decoder = TF_Decoder(
        'cpu',
        vocab_size,
        d_model,
        n_head,
        n_hid,
        n_layers,
        obj_feature_dim,
        out_seq_len,
        dropout=0.0,
    )

    tgt = torch.randint(0, vocab_size, (batch_size, tgt_len))

    print('>>> Training Mode')
    output = decoder(tgt, obj_f)
    print(f'>>> output.shape: {output.shape}')
    # >>> output.shape: torch.Size([4, 49, 15])

    print('>>> Inference Mode')
    output = decoder(None, obj_f, is_train=False)
    print(f'>>> output.shape: {output.shape}')
    # >>> output.shape: torch.Size([4, 30, 15])

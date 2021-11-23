import warnings
import copy
from typing import Optional, Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn.functional as F


# TODO: where to place norm ?
# TODO: warmup learning rate


class Transformer(nn.Module):
    r"""
    A transformer model. User is able to modify the attributes as needed.
    The architecture is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
    Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is
    all you need. In Advances in Neural Information Processing Systems,
    pages 6000-6010. Users can build the BERT(https://arxiv.org/abs/1810.04805)
    model with corresponding parameters.

    Args:
        d_model: the number of expected features in the encoder/decoder inputs
                 (default=512)
        nhead: the number of heads in the multiheadattention models (default=8)
        num_encoder_layers: the number of sub-encoder-layers in the encoder
                            (default=6)
        num_decoder_layers: the number of sub-decoder-layers in the decoder
                            (default=6)
        dim_feedforward: the dimension of the feedforward network model
                         (default=2048)
        dropout: the dropout value (default=0.1).
        activation: the activation function of encoder/decoder intermediate
                    layer, relu or gelu (default=relu)
        custom_encoder: custom encoder (default=None)
        custom_decoder: custom decoder (default=None)

    Examples::
        >>> transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
        >>> src = torch.rand((10, 32, 512))
        >>> tgt = torch.rand((20, 32, 512))
        >>> out = transformer_model(src, tgt)

    Note: A full example to apply nn.Transformer module for the word language
          model is available in
    https://github.com/pytorch/examples/tree/master/word_language_model
    """

    def __init__(
            self, d_model: int = 512,
            nhead: int = 8,
            num_encoder_layers: int = 6,
            num_decoder_layers: int = 6,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            activation: str = "relu",
            custom_encoder: Optional[Any] = None,
            custom_decoder: Optional[Any] = None,
    ) -> None:
        super(Transformer, self).__init__()

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout, activation
            )
            encoder_norm = nn.LayerNorm(d_model)
            self.encoder = TransformerEncoder(
                encoder_layer, num_encoder_layers, encoder_norm
            )

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayer(
                d_model, nhead, dim_feedforward, dropout, activation
            )
            decoder_norm = nn.LayerNorm(d_model)
            self.decoder = TransformerDecoder(
                decoder_layer, num_decoder_layers, decoder_norm
            )

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def forward(
            self,
            src: Tensor,
            tgt: Tensor,
            src_mask: Optional[Tensor] = None,
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        r"""Take in and process masked source/target sequences.

        Args:
            src: the sequence to the encoder (required).
            tgt: the sequence to the decoder (required).
            src_mask: the additive mask for the src sequence (optional).
            tgt_mask: the additive mask for the tgt sequence (optional).
            memory_mask: the additive mask for the encoder output (optional).
            src_key_padding_mask:
                the ByteTensor mask for src keys per batch (optional).
            tgt_key_padding_mask:
                the ByteTensor mask for tgt keys per batch (optional).
            memory_key_padding_mask:
                the ByteTensor mask for memory keys per batch (optional).

        Shape:
            - src: :math:`(S, N, E)`.
            - tgt: :math:`(T, N, E)`.
            - src_mask: :math:`(S, S)`.
            - tgt_mask: :math:`(T, T)`.
            - memory_mask: :math:`(T, S)`.
            - src_key_padding_mask: :math:`(N, S)`.
            - tgt_key_padding_mask: :math:`(N, T)`.
            - memory_key_padding_mask: :math:`(N, S)`.

            Note: [src/tgt/memory]_mask ensures that position i is allowed to
            attend the unmasked positions. If a ByteTensor is provided,
            the non-zero positions are not allowed to attend while the zero
            positions will be unchanged. If a BoolTensor is provided, positions
            with ``True`` are not allowed to attend while ``False`` values will
            be unchanged. If a FloatTensor is provided, it will be added to
            the attention weight.
            [src/tgt/memory]_key_padding_mask provides specified elements
            in the key to be ignored by the attention. If a ByteTensor is
            provided, the non-zero positions will be ignored while the zero
            positions will be unchanged. If a BoolTensor is provided, the
            positions with the value of ``True`` will be ignored while the
            position with the value of ``False`` will be unchanged.

            - output: :math:`(T, N, E)`.

            Note: Due to the multi-head attention architecture in the
            transformer model, the output sequence length of a transformer is
            same as the input sequence (i.e. target) length of the decode.

            where S is the source sequence length, T is the target sequence
            length, N is the batch size, E is the feature number

        Examples:
            >>> output = transformer_model(
                    src, tgt, src_mask=src_mask, tgt_mask=tgt_mask
                )
        """

        if src.size(1) != tgt.size(1):
            raise RuntimeError("the batch number of src and tgt must be equal")

        if src.size(2) != self.d_model or tgt.size(2) != self.d_model:
            raise RuntimeError(
                "the feature number of src and tgt must be equal to d_model"
            )

        memory = self.encoder(
            src, mask=src_mask, src_key_padding_mask=src_key_padding_mask
        )
        output = self.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        return output

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        r"""Generate a square mask for the sequence. The masked positions are
            filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
            mask == 1, float(0.0)
        )
        return mask

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class
                       (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=6
            )
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
            self,
            src: Tensor,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask:
                the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for mod in self.layers:
            output = mod(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class
                       (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(
                decoder_layer, num_layers=6
            )
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
            self,
            tgt: Tensor,
            memory: Tensor,
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch
                                  (optional).
            memory_key_padding_mask: the mask for the memory keys per batch
                                     (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper
    "Attention Is All You Need". Ashish Vaswani, Noam Shazeer, Niki Parmar,
    Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and
    Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or
    implement in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model
                         (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu
                    (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(
            self,
            d_model,
            nhead,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu"
    ):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(
            self,
            src: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch
                                  (optional)

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerDecoderLayer(nn.Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and
    feedforward network. This standard decoder layer is based on the paper
    "Attention Is All You Need". Ashish Vaswani, Noam Shazeer, Niki Parmar,
    Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and
    Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or
    implement in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model
                         (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer,
                    relu or gelu (default=relu).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    def __init__(
            self,
            d_model,
            nhead,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu"
    ):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(
            d_model, nhead, dropout=dropout
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(
            self,
            tgt: Tensor,
            memory: Tensor,
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch
                                  (optional).
            memory_key_padding_mask: the mask for the memory keys per batch
                                     (optional).

        Shape:
            see the docs in Transformer class.
        """
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(
        "activation should be relu/gelu, not {}".format(activation)
    )


class MultiheadAttention(nn.Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)

    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in value. Default: None.

        Note: if kdim and vdim are None, they will be set to embed_dim such
        that query, key, and value have the same number of features.

    Examples::

        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(
                query, key, value
            )
    """
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(
            self,
            embed_dim,
            num_heads,
            dropout=0.,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            kdim=None, vdim=None
    ):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and \
            self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, \
            "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(
                torch.empty(3 * embed_dim, embed_dim)
            )
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # self.out_proj = _LinearWithBias(embed_dim, embed_dim)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.q_proj_weight)
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by
        # v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None):
        # # type: (Tensor, Tensor, Tensor, Optional[Tensor], bool,
        #          Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]
        r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an
                           output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key
            will be ignored by the attention. When given a binary mask and a
            value is True, the corresponding value on the attention layer will
            be ignored. When given a byte mask and a value is non-zero,
            the corresponding value on the attention layer will be ignored
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions.
            A 2D mask will be broadcasted for all the batches while a 3D mask
            allows to specify a different mask for the entries of each batch.

    Shape:
        - Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length,
          N is the batch size, E is the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length,
          N is the batch size, E is the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size,
          S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored
          while the position with the zero positions will be unchanged.
          If a BoolTensor is provided, the positions with the value of ``True``
          will be ignored while the position with the value of ``False``
          will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence
          length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size,
          L is the target sequence length,
          S is the source sequence length. attn_mask ensure that position
          i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions
          are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is
          provided, positions with ``True``
          is not allowed to attend while ``False`` values will be unchanged.
          If a FloatTensor is provided, it will be added to the attention
          weight.

        - Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length,
          N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """
        if not self._qkv_same_embed_dim:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight
            )
        else:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask
            )


def multi_head_attention_forward(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        embed_dim_to_check: int,
        num_heads: int,
        in_proj_weight: Tensor,
        in_proj_bias: Tensor,
        bias_k: Optional[Tensor],
        bias_v: Optional[Tensor],
        add_zero_attn: bool,
        dropout_p: float,
        out_proj_weight: Tensor,
        out_proj_bias: Tensor,
        training: bool = True,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        use_separate_proj_weight: bool = False,
        q_proj_weight: Optional[Tensor] = None,
        k_proj_weight: Optional[Tensor] = None,
        v_proj_weight: Optional[Tensor] = None,
        static_k: Optional[Tensor] = None,
        static_v: Optional[Tensor] = None
):  # #  -> Tuple[Tensor, Optional[Tensor]]:
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an
            output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at
                        dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key
            will be ignored by the attention. This is an binary mask.
            When the value is True, the corresponding value on the attention
            layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions.
            A 2D mask will be broadcasted for all the batches while a 3D mask
            allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for
            query, key, and value in different forms. If false, in_proj_weight
            will be used, which is a combination of q_proj_weight,
            k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input
            projection weight and bias.
        static_k, static_v: static key and value used for attention operators.


    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length,
            N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length,
            N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length,
            N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size,
            S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored
            while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with
          the value of ``True`` will be ignored while the position with the
          value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence
          length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
          attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are
          not allowed to attend while the zero positions will be unchanged.
          If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged.
          If a FloatTensor is provided, it will be added to the attention
          weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`,
            where S is the source sequence length,
          N is the batch size, E is the embedding dimension.
            E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`,
          where S is the source sequence length,
          N is the batch size, E is the embedding dimension.
          E/num_heads is the head dimension.

        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length,
          N is the batch size, E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """
    # if not torch.jit.is_scripting():
    #     tens_ops = (
    #         query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v,
    #         out_proj_weight, out_proj_bias
    #     )
    #     if any([type(t) is not Tensor for t in tens_ops]) and \
    #        has_torch_function(tens_ops):
    #         return handle_torch_function(
    #             multi_head_attention_forward, tens_ops, query, key, value,
    #             embed_dim_to_check, num_heads, in_proj_weight, in_proj_bias,
    #             bias_k, bias_v, add_zero_attn, dropout_p, out_proj_weight,
    #             out_proj_bias, training=training,
    #             key_padding_mask=key_padding_mask,
    #             need_weights=need_weights, attn_mask=attn_mask,
    #             use_separate_proj_weight=use_separate_proj_weight,
    #             q_proj_weight=q_proj_weight, k_proj_weight=k_proj_weight,
    #             v_proj_weight=v_proj_weight,
    #             static_k=static_k, static_v=static_v,
    #         )
    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, \
        "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if not use_separate_proj_weight:
        if torch.equal(query, key) and torch.equal(key, value):
            # self-attention
            q, k, v = F.linear(
                query, in_proj_weight, in_proj_bias
            ).chunk(3, dim=-1)

        elif torch.equal(key, value):
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight
            # and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:

                # This is inline in_proj function with in_proj_weight and
                # in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = F.linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and
            # in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and
            # in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = F.linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and
            # in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = F.linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        if in_proj_bias is not None:
            q = F.linear(
                query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim]
            )
            k = F.linear(
                key,
                k_proj_weight_non_opt,
                in_proj_bias[embed_dim:(embed_dim * 2)]
            )
            v = F.linear(
                value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2):]
            )
        else:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = F.linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = F.inear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling

    if attn_mask is not None:
        assert attn_mask.dtype == torch.float32 or \
            attn_mask.dtype == torch.float64 or \
            attn_mask.dtype == torch.float16 or \
            attn_mask.dtype == torch.uint8 or \
            attn_mask.dtype == torch.bool, \
            ('Only float, byte, and bool types are supported for attn_mask, '
             'not {}'.format(attn_mask.dtype))
        if attn_mask.dtype == torch.uint8:
            warnings.warn(
                "Byte tensor for attn_mask in nn.MultiheadAttention is "
                "deprecated. Use bool tensor instead."
            )
            attn_mask = attn_mask.to(torch.bool)

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError(
                    'The size of the 2D attn_mask is not correct.'
                )
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [
                    bsz * num_heads, query.size(0), key.size(0)
            ]:
                raise RuntimeError(
                    'The size of the 3D attn_mask is not correct.'
                )
        else:
            raise RuntimeError(
                "attn_mask's dimension {} is not supported".format(
                    attn_mask.dim()
                )
            )
        # attn_mask's dim is 3 now.

    # convert ByteTensor key_padding_mask to bool
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn(
            "Byte tensor for key_padding_mask in nn.MultiheadAttention is "
            "deprecated. Use bool tensor instead."
        )
        key_padding_mask = key_padding_mask.to(torch.bool)

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros(
            (k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device
        )], dim=1)
        v = torch.cat([v, torch.zeros(
            (v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device
        )], dim=1)
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [
        bsz * num_heads, tgt_len, src_len
    ]

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_output_weights.masked_fill_(attn_mask, float('-inf'))
        else:
            attn_output_weights += attn_mask

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(
            bsz, num_heads, tgt_len, src_len
        )
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )
        attn_output_weights = attn_output_weights.view(
            bsz * num_heads, tgt_len, src_len
        )

    attn_output_weights = F.softmax(
        attn_output_weights, dim=-1)
    attn_output_weights = F.dropout(
        attn_output_weights, p=dropout_p, training=training
    )

    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(
        tgt_len, bsz, embed_dim
    )
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(
            bsz, num_heads, tgt_len, src_len
        )
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None

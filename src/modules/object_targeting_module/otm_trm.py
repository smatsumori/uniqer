import numpy as np
import torch
import torch.nn as nn
import modules.transformer.transformer as trm  # pytorch1.2 base
from utils.utils import base_10_to_n
from utils.consts import NUM_MAX_OBJ

# logging
from logging import getLogger
logger = getLogger(__name__)


class CandsSplitTransformer(nn.Module):
    """
    Candidate Splits Network Model (Object Targeting Module)
    For ablation study, I implemented transformer encoder version

    Only the differences from CandsSplitNetwork are
      - __init__(): some parameters, self.net, etc.
      - forward(): model in/out shape are different
    The other helper functions are copied from the original functions
    """
    def __init__(self, args, device, obj_feature_dim):
        """
        Parameters
        ----------
        args : ArgumentParser
        device : str
        obj_feature_dim : int
        """
        super(CandsSplitTransformer, self).__init__()
        self.device = device
        self.top_k = args.proposed_q_gen_top_k

        # TODO: sorry for hard-coded parameters
        # -------------------------------------
        self.hidden_dim = 32  # transformer's n_hid
        n_head = 4
        o_pred_input_dim = 4  # o_pred_prob embedding dim
        obj_feature_input_dim = 64  # obj_features embedding dim
        iq_input_dim = 4  # question index (times) embedding dim
        d_model = o_pred_input_dim + obj_feature_input_dim + iq_input_dim
        self.n_layers = 2
        dropout = 0.0
        # -------------------------------------

        # output size
        # TODO: class_size 2 pre-training should be considered
        self.class_size = args.split_class_size  # cluster1, cluster2, null
        # <STOP> token is considered to have been generated when out class
        # index is 0 (split=000...0) or the maximum index (split=222...2)
        # where no division occurs
        self.out_size = self.class_size ** self.top_k

        # ---------------------------------------------------------------------
        # [Module 1] o_pred_prob encoder
        self.pred_embed = nn.Linear(1, o_pred_input_dim)

        # ---------------------------------------------------------------------
        # [Module 2] obj features encoder
        # TODO: an MLP for obj_features or update self.embed for concatenating
        # object feature and object predicted probability
        self.obj_embed = nn.Sequential(
            nn.Linear(obj_feature_dim, obj_feature_input_dim),
            nn.ReLU(),
        )

        # ---------------------------------------------------------------------
        # [Module 3] question index encoder
        self.iq_embed = nn.Embedding(args.n_max_questions + 1, iq_input_dim)

        # ---------------------------------------------------------------------
        # [Module 4] Core network (Transformer Encoder)
        encoder_layer = trm.TransformerEncoderLayer(
            d_model, n_head, self.hidden_dim, dropout=dropout,
        )
        self.net = trm.TransformerEncoder(encoder_layer, self.n_layers)

        # ---------------------------------------------------------------------
        # [Module 5] output decoder ?
        # TODO: better network architecture ?
        self.hidden2output = nn.Sequential(
            nn.Linear(d_model * self.top_k, self.out_size),
            nn.Softmax(dim=-1),
        )

        self.hash_splits = self.init_hash_10_n()

    def set_device(self, device):
        """
        Set device attribute

        Parameters
        ----------
        device : str
        """
        self.device = device
        logger.info(f'Splitter device updated to {self.device}')

    def init_hash_10_n(self):
        """
        Preparing hash map for 10-ary to n-ary (class_size)

        * TODO: refactoring base_10_n (as it uses recursive method)

        Parameters
        ----------

        Returns
        -------
        hash_10_n : np.array
            (shape: out_size, num_max_obj)
        """
        hash_10_n = np.zeros((self.out_size, NUM_MAX_OBJ)).astype(np.int)

        for i in range(self.out_size):
            # TODO: refactoring
            if self.class_size == 3:
                corr = 0
            elif self.class_size == 2:
                corr = 1
            else:
                raise NotImplementedError
            # 18 --> '200' --> '00200'
            hash_10_n[i, :self.top_k] = np.array(
                list(base_10_to_n(i, self.class_size).zfill(self.top_k))
            ).astype(np.int) + corr
        return hash_10_n

    def forward(self, o_pred_prob, obj_features, iq, init_state=None):
        """
        Parameters
        ----------
        o_pred_prob : torch.tensor
            The output of Guesser, i.e. the estimated reference probability
            for each object
            (shape: batch_size, n_max_obj, 1)
        obj_features: torch.tensor
            Object Features (spatial + image features)
            (shape: batch_size, n_max_obj, feature_dim)
        iq : int
            question index (times)
        init_state : torch.tensor, default is None
            GRU version use this parameter, but trm mode just ignore this
            (shape: 2 * n_layers, batch_size, hidden_dim)

        Returns
        -------
        out : torch.tensor
            (shape: batch_size, out_size)
            Each bit represents a division method or <STOP> token
        """
        batch_size, n_obj, _ = o_pred_prob.shape

        # shape: batch_size, n_max_obj, pred_embed_dim
        o_pred_emb = self.pred_embed(o_pred_prob)
        # shape: batch_size, n_max_obj, obj_feature_embed_dim
        obj_features = self.obj_embed(obj_features)

        # question index embedding
        # shape (batch_size, n_max_obj, iq_embed_dim)
        iq_emb = (torch.ones(batch_size, n_obj) * iq).long().to(self.device)
        iq_emb = self.iq_embed(iq_emb)
        x = torch.cat([iq_emb, o_pred_emb, obj_features], dim=-1)
        x = x.transpose(1, 0)  # (n_max_obj, batch_size, iq_embed_dim)

        # out.shape: (batch_size, n_max_obj, self.hidden_dim)
        out = self.net(x).transpose(1, 0)

        # concat obj directions
        out = out.reshape(batch_size, -1)
        out = self.hidden2output(out)
        return out

    def actions2splits(self, actions):
        """
        Parameters
        ----------
        actions : array-like (list ?)
            (shape: batch_size)

        Returns
        -------
        split_ids : torch.tensor
            (shape: batch_size, n_max_obj)
        eod_mask : np.array
            (shape: batch_size)
        """
        actions = np.array(actions)

        # TODO: actions == `11111`(3) should also treaded as <STOP> ?
        eod_mask = (actions == 0) | (actions == (self.out_size - 1))

        # create split_ids
        # (shape: batch_size, n_max_obj)
        split_ids = torch.tensor(self.hash_splits[actions])
        return split_ids, eod_mask


if __name__ == '__main__':
    # DEBUG purpose only
    # ADD this top : import sys; sys.path.append('../../../src')
    # [WARNING] You should comment out importing self-built modules
    b_size = 4
    obj_feature_dim = 32
    index_question = 4

    class DummyArgs:
        proposed_q_gen_top_k = 5
        split_class_size = 3
        n_max_questions = 5

    a = DummyArgs()

    net = CandsSplitTransformer(a, 'cpu', obj_feature_dim)
    o_pred_prob = torch.rand(b_size, a.proposed_q_gen_top_k, 1)
    obj_features = torch.rand(b_size, a.proposed_q_gen_top_k, obj_feature_dim)

    out = net(o_pred_prob, obj_features, index_question)
    print(f'>>> out.shape: {out.shape}')
    # >>> out.shape: torch.Size([4, 243])

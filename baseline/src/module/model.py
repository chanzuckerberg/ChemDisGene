import numpy as np
import json
import math
import sys
import torch
from torch import nn, Tensor

from transformers import AutoModel

from module.neural import TransformerInterEncoder
from module.neural import RNNEncoder
from module.utils import log1mexp

__all__ = [
    "Model"
]


def orthonormal_initializer(input_size, output_size):
    """from https://github.com/patverga/bran/blob/32378da8ac339393d9faa2ff2d50ccb3b379e9a2/src/tf_utils.py#L154"""
    I = np.eye(output_size)
    lr = .1
    eps = .05/(output_size + input_size)
    success = False
    tries = 0
    while not success and tries < 10:
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
        for i in range(100):
            QTQmI = Q.T.dot(Q) - I
            loss = np.sum(QTQmI**2 / 2)
            Q2 = Q**2
            Q -= lr*Q.dot(QTQmI) / (np.abs(Q2 + Q2.sum(axis=0,
                                                       keepdims=True) + Q2.sum(axis=1, keepdims=True) - 1) + eps)
            if np.max(Q) > 1e6 or loss > 1e6 or not np.isfinite(loss):
                tries += 1
                lr /= 2
                break
        success = True
    if success:
        print('Orthogonal pretrainer loss: %.2e' % loss)
    else:
        print('Orthogonal pretrainer failed, using non-orthogonal random matrix')
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
    return Q.astype(np.float32)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerConv(nn.Module):

    def __init__(self, dim: int, vocab_size: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dim = dim
        self.token_embeddings = nn.Embedding(vocab_size, dim)
        self.token_embeddings.weight.data.uniform_(-0.1, 0.1)

        self.pos_encoder = PositionalEncoding(dim)
        # self.encoder = AutoModel.from_pretrained(
        #     "bert-base-cased", output_hidden_states=True)
        # self.encoder.init_weights()

        self.transformer1 = torch.nn.TransformerEncoderLayer(
            d_model=self.dim, nhead=4, batch_first=True, dropout=dropout)
        self.conv11 = torch.nn.Conv1d(self.dim, self.dim, 1, padding='same')
        self.conv12 = torch.nn.Conv1d(self.dim, self.dim, 5, padding='same')
        self.conv13 = torch.nn.Conv1d(self.dim, self.dim, 1, padding='same')
        #torch.nn.MultiheadAttention(self.dim, 4, batch_first=True),
        self.transformer2 = torch.nn.TransformerEncoderLayer(
            d_model=self.dim, nhead=4, batch_first=True, dropout=dropout)
        self.conv21 = torch.nn.Conv1d(self.dim, self.dim, 1, padding='same')
        self.conv22 = torch.nn.Conv1d(self.dim, self.dim, 5, padding='same')
        self.conv23 = torch.nn.Conv1d(self.dim, self.dim, 1, padding='same')

        self.relu = torch.nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_ids: Tensor, attention_mask: Tensor = None) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        src = self.token_embeddings(input_ids) * math.sqrt(self.dim)
        #print("token shape", src.shape)
        src = self.pos_encoder(src)
        src = self.dropout(src)
        h = self.transformer1(src, src_key_padding_mask=attention_mask)
        h = h.transpose(1, 2)
        h = self.conv11(h)
        h = self.conv12(h)
        h = self.conv13(h)
        h = h.transpose(1, 2)
        h = self.transformer2(h, src_key_padding_mask=attention_mask)
        h = h.transpose(1, 2)
        h = self.conv21(h)
        h = self.conv22(h)
        h = self.conv23(h)
        h = h.transpose(1, 2)
        return h


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.dim = config["dim"]
        self.num_rel = len(json.loads(
            open(config["data_path"] + "/relation_map.json").read()))
        if self.config["encoder_type"] == "transformer_conv":
            self.encoder = TransformerConv(self.dim, self.config["vocabsize"])
            self.D = self.dim
        elif self.config["encoder_type"] == "transformer":
            self.encoder = AutoModel.from_pretrained(
                "bert-base-cased", output_hidden_states=True)
            self.encoder.init_weights()
            self.D = self.encoder.config.hidden_size
        else:
            self.encoder = AutoModel.from_pretrained(config["encoder_type"])
            self.D = self.encoder.config.hidden_size

        if self.config["model"] == "biaffine":
            self.head_layer0 = torch.nn.Linear(self.D, self.D)
            self.head_layer1 = torch.nn.Linear(self.D, self.dim)
            self.tail_layer0 = torch.nn.Linear(self.D, self.D)
            self.tail_layer1 = torch.nn.Linear(self.D, self.dim)
            self.relu = torch.nn.ReLU()
            mat = orthonormal_initializer(self.dim, self.dim)[:, None, :]
            biaffine_mat = np.concatenate([mat] * (self.num_rel + 1), axis=1)
            self.biaffine_mat = torch.nn.Parameter(torch.tensor(
                biaffine_mat), requires_grad=True)  # (dim, R, dim)
            self.multi_label = config["multi_label"]
            self.softmax = torch.nn.Softmax(dim=1)
            self.sigmoid = torch.nn.Sigmoid()

        elif self.config["model"] == "dot":
            self.layer1 = torch.nn.Linear(self.D * 2, self.dim)
            self.layer2 = torch.nn.Linear(self.dim, self.num_rel + 1)
            self.relu = torch.nn.ReLU()
            self.multi_label = config["multi_label"]
            self.softmax = torch.nn.Softmax(dim=1)
            self.sigmoid = torch.nn.Sigmoid()
            self.dropout = torch.nn.Dropout(p=config["dropout_rate"])

    def bi_affine(self, e1_vec, e2_vec):
        # e1_vec: batchsize, text_length, dim
        # e2_vec: batchsize, text_length, dim
        # output: batchsize, text_length, text_length, R
        batchsize, text_length, dim = e1_vec.shape

        # (batchsize * text_length, dim) (dim, R*dim) -> (batchsize * text_length, R*dim)
        lin = torch.matmul(
            torch.reshape(e1_vec, [-1, dim]),
            torch.reshape(self.biaffine_mat, [dim, (self.num_rel + 1) * dim])
        )
        # (batchsize, text_length * R, D) (batchsize, D, text_length) -> (batchsize, text_length * R, text_length)
        bilin = torch.matmul(
            torch.reshape(lin, [batchsize, text_length *
                          (self.num_rel + 1), self.dim]),
            torch.transpose(e2_vec, 1, 2)
        )

        output = torch.reshape(
            bilin, [batchsize, text_length, self.num_rel + 1, text_length])
        output = torch.transpose(output, 2, 3)
        return output

    def forward(self, input_ids, attention_mask, ep_mask, e1_indicator, e2_indicator):
        # input_ids: (batchsize, text_length)
        # attention_mask: (batchsize, text_length)
        # ep_mask: (batchsize, num_ep, text_length, text_length)
        # e1_indicator: not used
        # e2_indicator: not used
        if self.config["encoder_type"] == "transformer_conv":
            h = self.encoder(input_ids=input_ids.long(),
                             attention_mask=attention_mask.long())
        elif self.config["encoder_type"] == "transformer":
            h = self.encoder(input_ids=input_ids.long(),
                             attention_mask=attention_mask.long())[2][2]  # Two hidden layer
        else:
            h = self.encoder(input_ids=input_ids.long(), attention_mask=attention_mask.long())[
                0]  # (batchsize, text_length, D)

        if self.config["model"] == "biaffine":
            e1_vec = self.head_layer1(self.relu(self.head_layer0(h)))
            e2_vec = self.tail_layer1(self.relu(self.tail_layer0(h)))

            # (batchsize, 1, text_length, text_length, R + 1)
            pairwise_scores = self.bi_affine(e1_vec, e2_vec).unsqueeze(1)
            # pairwise_scores = torch.nn.functional.softmax(pairwise_scores, dim=3)
            # # above line Commented, was used in original Bran code: https://github.com/patverga/bran/blob/32378da8ac339393d9faa2ff2d50ccb3b379e9a2/src/models/transformer.py#L468
            # (batchsize, num_ep, text_length, text_length, 1)
            ep_mask = ep_mask.unsqueeze(4)
            # batchsize, num_ep, text_length, text_length, R + 1
            pairwise_scores = pairwise_scores + ep_mask
            pairwise_scores = torch.logsumexp(
                pairwise_scores, dim=[2, 3])  # batchsize, num_ep, R + 1

        elif self.config["model"] == "dot":
            # (batchsize, num_ep, text_length, 1)
            e1_indicator = e1_indicator.unsqueeze(3)
            e2_indicator = e2_indicator.unsqueeze(3)

            e1_indicator_mask = 1000 * \
                torch.ones_like(e1_indicator) * (1 - e1_indicator)
            e2_indicator_mask = 1000 * \
                torch.ones_like(e2_indicator) * (1 - e2_indicator)

            # (batchsize, num_ep, text_length, D)
            e1_vec = (h * e1_indicator - e1_indicator_mask)
            # (batchsize, num_ep, text_length, D)
            e2_vec = (h * e2_indicator - e2_indicator_mask)
            e1_vec = e1_vec.max(2)[0]  # (batchsize, num_ep, D)
            e2_vec = e2_vec.max(2)[0]  # (batchsize, num_ep, D)

            e1e2_vec = self.dropout(
                self.relu(self.layer1(torch.cat([e1_vec, e2_vec], 2))))  # (batchsize, num_ep, 2D)
            pairwise_scores = self.layer2(e1e2_vec)  # (batchsize, num_ep, R+1)

        if self.multi_label == True:
            # (batchsize, num_ep, R)
            pairwise_scores = pairwise_scores[:, :, :-1]

        return pairwise_scores

import torch
import torch.nn as nn
from torch.nn import Linear, LeakyReLU, Dropout
from models.LSTM import LSTMSequence


class Decoder(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, pred_len):
        super(Decoder, self).__init__()
        hidden_size = in_dim // 2
        self.rnn = LSTMSequence(in_dim, hidden_size // 2, dropout=dropout)
        self.linear1 = Linear(hidden_size, hidden_size)
        self.dropout = Dropout(p=dropout)
        self.act_fn = LeakyReLU()
        self.linear2 = Linear(hidden_size, out_dim)

    def forward(self, x):
        # x [BS, L, D]
        x = self.rnn(x)
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.act_fn(x)
        x = self.linear2(x)
        return x


class Decoder_Varb(nn.Module):
    def __init__(self, in_dim, pred_len, dropout, n_var=2):
        super(Decoder_Varb, self).__init__()
        hidden_size = in_dim * 2
        self.rnn1 = LSTMSequence(in_dim, hidden_size // 2, dropout=dropout)
        self.linear1 = Linear(hidden_size, hidden_size)
        self.dropout = Dropout(p=dropout)
        self.act_fn = LeakyReLU()
        self.linear2 = Linear(hidden_size, pred_len)
        self.n_var = n_var

    def forward(self, x, dim=0):
        # x [BS, covar, hidden_size]
        x = self.rnn1(x)[:, dim, :]  # [BS, hidden_size]
        x = self.linear1(x)
        x = self.dropout(x)  # [BS, hidden_size]
        x = self.act_fn(x)  # [BS, hidden_size]
        x = self.linear2(x)  # [BS, pred_len]
        return x  # [BS, pred_len]


class Decoder_Time(nn.Module):
    def __init__(self, in_dim, dropout, n_var=1, pred_len=24):
        super(Decoder_Time, self).__init__()
        hidden_size = in_dim // 2
        self.rnn = LSTMSequence(in_dim, hidden_size // 2, dropout=dropout)
        self.linear1 = Linear(hidden_size, hidden_size)
        self.dropout = Dropout(p=dropout)
        self.act_fn = LeakyReLU()
        self.linear2 = Linear(hidden_size, 1)
        self.pred_len = pred_len

    def forward(self, x):
        # x  [BS, seq_len + pred_len, d_model]
        x = self.rnn(x)[:, -self.pred_len :, :]  # [BS, pred_len, hidden_size]
        x = self.linear1(x)  # [BS, pred_len, hidden_size]
        x = self.dropout(x)  # [BS, pred_len, hidden_size]
        x = self.act_fn(x)  # [BS, pred_len, hidden_size]
        x = self.linear2(x)  # [BS, pred_len, 1]
        return x.squeeze(-1)  # [BS, pred_len]


class Decoder_SAMFormer(nn.Module):
    def __init__(self, enc_in, seq_len, dropout, pred_len=24):
        super(Decoder_SAMFormer, self).__init__()
        self.rnn = LSTMSequence(enc_in, enc_in, dropout=dropout)
        self.linear1 = Linear(seq_len, pred_len)
        self.dropout = Dropout(p=dropout)
        self.act_fn = LeakyReLU()
        self.linear2 = Linear(pred_len, pred_len)
        self.pred_len = pred_len
        self.enc_in = enc_in

    def forward(self, x):
        # x [bs, enc_in, seq_len]
        x = self.rnn(x.permute(0, 2, 1))  # [BS, seq_len, enc_in * 2]
        x = (
            torch.stack([x[..., : self.enc_in], x[..., self.enc_in :]], dim=0)
            .mean(0)
            .permute(0, 2, 1)
        )  # [BS, enc_in, seq_len]
        x = self.linear1(x)  # [BS, enc_in, pred_len]
        x = self.dropout(x)  # [BS, enc_in, pred_len]
        x = self.act_fn(x)  # [BS, enc_in, pred_len]
        x = self.linear2(x)  # [BS, enc_in, pred_len]
        return x  # [BS, enc_in, pred_len]

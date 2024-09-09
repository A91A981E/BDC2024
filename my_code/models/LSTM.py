import torch
import torch.nn as nn


class LSTMSequence(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1, dropout=0.2):
        super(LSTMSequence, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )
        self.reset_params()
        self.dropout = nn.Dropout(p=dropout)

    def reset_params(self):
        for i in range(self.rnn.num_layers):
            nn.init.orthogonal_(getattr(self.rnn, "weight_hh_l%s" % i))
            nn.init.kaiming_normal_(getattr(self.rnn, "weight_ih_l%s" % i))
            nn.init.constant_(getattr(self.rnn, "bias_hh_l%s" % i), val=0)
            nn.init.constant_(getattr(self.rnn, "bias_ih_l%s" % i), val=0)
            # getattr(self.rnn, 'bias_hh_l%s' % i).chunk(4)[1].fill_(1)     # unable to change in-place

            if self.rnn.bidirectional:
                nn.init.orthogonal_(getattr(self.rnn, "weight_hh_l%s_reverse" % i))
                nn.init.kaiming_normal_(getattr(self.rnn, "weight_ih_l%s_reverse" % i))
                nn.init.constant_(getattr(self.rnn, "bias_hh_l%s_reverse" % i), val=0)
                nn.init.constant_(getattr(self.rnn, "bias_ih_l%s_reverse" % i), val=0)
                # getattr(self.rnn, 'bias_hh_l%s_reverse' % i).chunk(4)[1].fill_(1)

    def forward(self, x):
        x = self.dropout(x)
        h0 = torch.zeros(2 * self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(2 * self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, (h0, c0))
        return out

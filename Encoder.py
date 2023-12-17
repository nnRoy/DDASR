import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init

class RNNEncoder(nn.Module):
    def __init__(self, input_emb, input_size, hidden_size, bidir, n_layers, dropout=0.5, noise_radius=0.2):
        super(RNNEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.noise_radius = noise_radius
        self.n_layers = n_layers
        self.bidir = bidir
        self.dropout = dropout
        self.input_emb = input_emb
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=n_layers, batch_first=True, bidirectional=bidir)
        self.init_h = nn.Parameter(torch.randn(self.n_layers * (1 + self.bidir), 1, self.hidden_size),
                                   requires_grad=True)
        self.init_weights()

    def init_weights(self):
        for name, param in self.rnn.named_parameters():
            if 'weight' in name or 'bias' in name:
                param.data.uniform_(-0.1, 0.1)

    def forward(self, inputs):
        if self.input_emb is not None:
            inputs = self.input_emb(inputs)
        batch_size, seq_len, emb_size = inputs.size()
        inputs = F.dropout(inputs, self.dropout, self.training)
        hids, (h_n, c_n) = self.rnn(inputs)
        h_n = h_n.view(self.n_layers, (1 + self.bidir), batch_size, self.hidden_size)
        h_n = h_n[-1]
        enc = h_n.view(batch_size, -1)
        return enc, hids


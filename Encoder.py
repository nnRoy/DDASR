import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForMaskedLM, AutoModelForSeq2SeqLM
import math
from collections import namedtuple
import copy
import numpy as np
import config as config
from transformers import RobertaTokenizer, T5ForConditionalGeneration


def get_attn_mask(tokens, PAD_ID):
    device = tokens.device
    attn_mask = torch.zeros(tokens.size())
    attn_mask[tokens == PAD_ID] = 0
    attn_mask[tokens != PAD_ID] = 1
    return attn_mask.to(device)


def get_key_padding_mask(tokens, PAD_ID):
    device = tokens.device
    key_padding_mask = torch.zeros(tokens.size())
    key_padding_mask[tokens == PAD_ID] = 1
    key_padding_mask[tokens != PAD_ID] = 0
    return key_padding_mask.to(device)

def get_mask(tokens):
    device = tokens.device
    mask = (torch.triu(torch.ones(tokens.size(-1), tokens.size(-1))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.to(device)
    mask = mask.data.eq(-torch.inf)
    return mask

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()

        # 初始化Shape为(max_len, d_model)的PE (positional encoding)
        pe = torch.zeros(max_len, d_model)
        # 初始化一个tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len).unsqueeze(1)
        # 这里就是sin和cos括号中的内容，通过e和ln进行了变换
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # 计算PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 为了方便计算，在最外面在unsqueeze出一个batch
        pe = pe.unsqueeze(0)
        # 如果一个参数不参与梯度下降，但又希望保存model的时候将其保存下来
        # 这个时候就可以用register_buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x 为embedding后的inputs，例如(1,7, 128)，batch size为1,7个单词，单词维度为128
        """
        # 将x和positional encoding相加。
        # x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        # return self.dropout(x)
        return self.pe[:,:x.size(1)].requires_grad_(False)


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

def __init__(self, vocab_size, seq_len, d_model, n_heads, d_ff, n_layers, dropout) -> None:
        super().__init__()
        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = PositionalEncoding(d_model, seq_len*10)
        self.dropout = nn.Dropout(dropout)
        self.encoder_layers = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layers, n_layers)
    def forward(self, inputs):
        batch_size, input_len, seq_len = inputs.size()
        for i in range(input_len):
            input = inputs[:,i,:].view(batch_size, seq_len)
            input_emb = self.word_emb(input)
            input_pos = self.pos_emb(input_emb)
            input_enc = input_emb+input_pos
            input_key_padding_mask = get_key_padding_mask(input, 0)
            enc_outputs = self.encoder(input_enc, src_key_padding_mask=input_key_padding_mask)
            enc_temp = enc_outputs.mean(dim=1).unsqueeze(1) if i==0 else torch.cat([enc_temp,enc_outputs.mean(dim=1).unsqueeze(1)],dim=1)
        return enc_temp


class LLMsEncoder(nn.Module):
    def __init__(self, encoder, tokenizer) -> None:
        super().__init__()
        self.desc_encoder = encoder
        self.tokenizer = tokenizer
    def forward(self, inputs):
        PAD_ID = self.tokenizer.pad_token_id
        attn_mask = get_attn_mask(inputs, PAD_ID)
        if config.pretrain_type=="codet5":
            enc_outputs = self.desc_encoder.encoder(inputs).last_hidden_state
            enc_key_padding_mask = get_key_padding_mask(inputs, PAD_ID)
            return enc_outputs, enc_key_padding_mask
        elif config.pretrain_type=="graphcodebert":
            enc_outputs = self.desc_encoder.roberta(inputs).last_hidden_state
            enc_key_padding_mask = get_key_padding_mask(inputs, PAD_ID)
            return enc_outputs, enc_key_padding_mask
        elif config.pretrain_type=="plbart":
            enc_outputs = self.desc_encoder.model.encoder(inputs).last_hidden_state
            enc_key_padding_mask = get_key_padding_mask(inputs, PAD_ID)
            return enc_outputs, enc_key_padding_mask
        elif config.pretrain_type=="unixcoder":
            enc_outputs = self.desc_encoder(inputs).last_hidden_state
            enc_key_padding_mask = get_key_padding_mask(inputs, PAD_ID)
            return enc_outputs, enc_key_padding_mask
        else:
            enc_outputs = self.desc_encoder(inputs, attn_mask)
            enc_key_padding_mask = get_key_padding_mask(inputs, PAD_ID)
            return enc_outputs[0], enc_key_padding_mask



import torch.nn as nn
import torch
import config
import json
from array import array
import numpy as np
from helper import PAD_ID
from Encoder import RNNEncoder
from Decoder import RNNDecoder
from LossLongtail import Loss_Longtail,Loss_Longtail_Division
from utils import getWeight, getWeight1

class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()
        self.desc_vocab_size = config.desc_vocab_size
        self.api_vocab_size = config.api_vocab_size

        self.max_desc_len = config.max_desc_length
        self.max_api_len = config.max_api_length
        self.max_longtail_len = config.max_longtail_length

        self.Loss_Longtail = Loss_Longtail_Division()
        self.loss_weight = getWeight1()

        self.desc_emb = nn.Embedding(self.desc_vocab_size, config.emb_dim, padding_idx=PAD_ID)
        self.api_emb = nn.Embedding(self.api_vocab_size, config.emb_dim, padding_idx=PAD_ID)

        self.encoder = RNNEncoder(self.desc_emb, config.emb_dim, config.n_hidden,
                                  True, config.n_layers, config.noise_radius)

        self.ctx2dec = nn.Sequential(
            nn.Linear(2 * config.n_hidden, config.n_hidden),
            nn.Tanh(),
        )

        self.ctx2dec.apply(self.init_weights)

        self.decoder = RNNDecoder(self.api_emb, config.emb_dim, config.n_hidden,
                                  self.api_vocab_size, 1, config.dropout)


    def init_weights(self, m):  # Initialize Linear Weight for GAN
        if isinstance(m, nn.Linear):
            m.weight.data.uniform_(-0.08, 0.08)  # nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0.)


    def forward(self, inputs, outputs, longtail, batchsize = config.batch_size):
        c,hids = self.encoder(inputs)
        init_h, hids = self.ctx2dec(c), self.ctx2dec(hids)
        src_pad_mask = inputs.eq(PAD_ID)
        output, _ = self.decoder(init_h, hids, src_pad_mask, outputs[:, :-1])
        dec_target = outputs[:, 1:].clone()
        longtail = longtail[:, 1:].clone()
        #loss = nn.CrossEntropyLoss()(output.view(-1, self.api_vocab_size), dec_target.view(-1).long())
        #loss,_ =self.Loss_Longtail(output.view(-1, self.api_vocab_size), dec_target.view(batchsize * (self.max_api_len - 1)), longtail.view(batchsize * (self.max_api_len - 1), config.max_longtail_length))
        loss, _ = self.Loss_Longtail(output.view(-1, self.api_vocab_size),
                                     dec_target.view(-1, 1),
                                     longtail.view(-1, config.max_longtail_length), self.loss_weight)

        return loss

    def valid(self, inputs, outputs, longtail):
        self.eval()
        loss = self.forward(inputs, outputs, longtail, config.valid_batch_size)
        loss = torch.mean(loss)
        return {'valid_loss': loss.item()}


    def adjust_lr(self):
        # self.lr_scheduler_AE.step()
        return None

    def sample(self, src_seqs, n_samples):
        self.eval()
        src_pad_mask = src_seqs.eq(PAD_ID)
        c, hids = self.encoder(src_seqs)
        init_h, hids = self.ctx2dec(c), self.ctx2dec(hids)
        sample_words, sample_lens, _ = self.decoder.beam_decode(init_h, hids, src_pad_mask, 12, self.max_api_len,
                                                                n_samples)
        sample_words, sample_lens = sample_words[0], sample_lens[0]
        return sample_words, sample_lens



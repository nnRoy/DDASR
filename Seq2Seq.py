import torch.nn as nn
import torch
import config
import json
from array import array
import numpy as np
from helper import PAD_ID
from Encoder import Encoder
from Decoder import Decoder
from LossLongtail import Loss_Longtail,Loss_Longtail_Division

class Seq2SeqModel(nn.Module):
    def __init__(self, src_max_len, tokenizer, tgt_vocab_size, tgt_max_len, d_model, n_heads, d_ff, n_layers, dropout) -> None:
        super().__init__()
        if config.pretrain_type=="codet5":
            self.automodel = T5ForConditionalGeneration.from_pretrained(config.pretrain_file)
        elif config.pretrain_type=="graphcodebert":
            self.automodel = AutoModelForMaskedLM.from_pretrained(config.pretrain_file)
        elif config.pretrain_type=="plbart":
            self.automodel = AutoModelForSeq2SeqLM.from_pretrained(config.pretrain_file)
        else:
            self.automodel = AutoModel.from_pretrained(config.pretrain_file)
        #self.autoconfig = AutoConfig.from_pretrained(config.pretrain_file)
        self.encoder = Multi_Encoder(self.automodel, tokenizer, src_max_len, tgt_vocab_size, tgt_max_len, d_model, n_heads, d_ff, n_layers, dropout)
        self.decoder = Decoder(tgt_vocab_size, tgt_max_len, d_model, n_heads, d_ff, n_layers, dropout)
        self.project = nn.Linear(d_model, tgt_vocab_size)
        self.tokenizer = tokenizer
        self.tgt_vocab_size = tgt_vocab_size
        self.tgt_max_len = tgt_max_len

    #def forward(self, src_inputs, retrieve, tgt_inputs):
    def forward(self, src_inputs, api_sim, tgt_inputs):
        batch_size = tgt_inputs.size(0)
        enc_outputs, enc_key_mask = self.encoder(src_inputs, api_sim)

        for di in range(self.tgt_max_len-1):
            dec_inputs = tgt_inputs[:,0:di+1].clone()
            dec_inputs = dec_inputs.view(batch_size, -1)
            out_scores = self.decoder(dec_inputs, enc_outputs, enc_key_mask)
            out_scores = self.project(out_scores[:,-1,:]).unsqueeze(1)
            out = out_scores if di==0 else torch.cat([out, out_scores],1)

        dec_target = tgt_inputs[:,1:].clone()
        loss, _ = self.Loss_Longtail(output.view(-1, self.tgt_vocab_size),
                                     dec_target.view(-1, 1),
                                     longtail.view(-1, config.max_longtail_length), self.loss_weight)
        return loss
    def valid(self, inputs, api_sim, outputs):
    #def valid(self, inputs, retrieve, outputs):
        self.eval()
        #loss = self.forward(inputs, retrieve, outputs)
        loss = self.forward(inputs, api_sim, outputs)
        loss = torch.mean(loss)
        return {'valid_loss': loss.item()}

    def beam_search(self, inputs, api_sim, beam_size, max_len, top_k):
    #def beam_search(self, inputs, retrieve, beam_size, max_len, top_k):
        batch_size = inputs.size(0)
        device = inputs.device
        Hypothesis = namedtuple('Hypothesis', ['value', 'score'])
        
        enc_inputs, enc_key_padding_mask = self.encoder(inputs, api_sim)
        # SOS_ID = self.tokenizer.cls_token_id
        # EOS_ID = self.tokenizer.sep_token_id
        # PAD_ID = self.tokenizer.pad_token_id

        SOS_ID = 1
        EOS_ID = 2
        PAD_ID = 0


        hypotheses = [copy.deepcopy(torch.full((1,1), SOS_ID, dtype=torch.long, device=device)) for _ in range(batch_size)]
        completed_hypotheses = [copy.deepcopy([]) for _ in range(batch_size)]
        hyp_scores = [copy.deepcopy(torch.full((1,), 0, dtype=torch.long, device=device)) for _ in range(batch_size)]

        for iter in range(max_len-1):
            if all([len(completed_hypotheses[i]) == beam_size for i in range(batch_size)]): break
            cur_beam_sizes, last_tokens, model_encodings_l, src_mask_l = [], [], [], []
            for i in range(batch_size):
                if hypotheses[i] is None:
                    cur_beam_sizes += [0]
                    continue
                cur_beam_size, decoded_len = hypotheses[i].shape
                cur_beam_sizes += [cur_beam_size]
                last_tokens += [hypotheses[i]]
                model_encodings_l += [enc_inputs[i:i+1]] * cur_beam_size
                src_mask_l += [enc_key_padding_mask[i:i+1]] * cur_beam_size
            model_encodings_cur = torch.cat(model_encodings_l, dim=0)
            src_mask_cur = torch.cat(src_mask_l, dim=0)
            tgt_ids = torch.cat(last_tokens, dim=0).to(device)
            # tgt_emb = self.decoder_emb(tgt_ids)
            # tgt_pos = self.decoder_pos(tgt_emb)
            # tgt_inputs = tgt_emb+tgt_pos
            # #tgt_inputs = self.dropout(tgt_inputs)
            # tgt_key_padding_mask = get_key_padding_mask(tgt_ids)
            # tgt_mask = get_mask(tgt_ids)

            # out = self.transformer(model_encodings_cur, tgt_inputs, tgt_mask = tgt_mask, src_key_padding_mask = src_mask_cur, tgt_key_padding_mask = tgt_key_padding_mask)
            out = self.decoder(tgt_ids, model_encodings_cur, src_mask_cur)
            log_prob = self.project(out[:,-1,:]).unsqueeze(1)
            _, decoded_len, vocab_sz = log_prob.shape
            log_prob = torch.split(log_prob, cur_beam_sizes, dim=0)
            new_hypotheses, new_hyp_scores = [], []
            for i in range(batch_size):
                if hypotheses[i] is None or len(completed_hypotheses[i]) >= beam_size:
                    new_hypotheses += [None]
                    new_hyp_scores += [None]
                    continue
                cur_beam_sz_i, dec_sent_len, vocab_sz = log_prob[i].shape
                #"shape (vocab_sz,)"
                cumulative_hyp_scores_i = (hyp_scores[i].unsqueeze(-1).unsqueeze(-1) \
                                          .expand((cur_beam_sz_i, 1, vocab_sz)) + log_prob[i])\
                                          .view(-1)

                
                live_hyp_num_i = beam_size - len(completed_hypotheses[i])
                #"shape (cur_beam_sz,). Vals are between 0 and 50002 vocab_sz"
                top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(cumulative_hyp_scores_i, k=live_hyp_num_i)
                #"shape (cur_beam_sz,). prev_hyp_ids vals are 0 <= val < cur_beam_sz. hyp_word_ids vals are 0 <= val < vocab_len"
                prev_hyp_ids, hyp_word_ids = top_cand_hyp_pos // self.tgt_vocab_size, top_cand_hyp_pos % self.tgt_vocab_size

                new_hypotheses_i, new_hyp_scores_i = [],[] # Removed live_hyp_ids_i, which is used in the LSTM decoder to track live hypothesis ids
                for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                    prev_hyp_id, hyp_word_id, cand_new_hyp_score = \
                        prev_hyp_id.item(), hyp_word_id.item(), cand_new_hyp_score.item()

                    new_hyp_sent = torch.cat((hypotheses[i][prev_hyp_id], torch.tensor([hyp_word_id], device=device)))
                    if hyp_word_id == EOS_ID:
                        completed_hypotheses[i].append(Hypothesis(
                            value=new_hyp_sent[:],
                            score=cand_new_hyp_score))
                    else:
                        new_hypotheses_i.append(new_hyp_sent.unsqueeze(-1))
                        new_hyp_scores_i.append(cand_new_hyp_score)
                
                if len(new_hypotheses_i) > 0:
                    hypotheses_i = torch.cat(new_hypotheses_i, dim=-1).transpose(0,-1).to(device)
                    hyp_scores_i = torch.tensor(new_hyp_scores_i, dtype=torch.float, device=device)
                else:
                    hypotheses_i, hyp_scores_i = None, None
                new_hypotheses += [hypotheses_i]
                new_hyp_scores += [hyp_scores_i]
            hypotheses, hyp_scores = new_hypotheses, new_hyp_scores
        
        for i in range(batch_size):
            hyps_to_add = beam_size - len(completed_hypotheses[i])
            if hyps_to_add > 0:
                scores, ix = torch.topk(hyp_scores[i], k=hyps_to_add)
                for score, id in zip(scores, ix):
                    completed_hypotheses[i].append(Hypothesis(
                    value=hypotheses[i][id][:],
                    score=score))
            completed_hypotheses[i].sort(key=lambda hyp: hyp.score, reverse=True)

        decoded=[]
        for i in range(batch_size):
            decoded_cur=[]
            for j in range(top_k):
                cur = completed_hypotheses[i][j].value
                cur = cur.cpu().detach().numpy()
                for m in range(cur.shape[0], max_len):
                    cur = np.append(cur,[PAD_ID])
                decoded_cur.append(cur)
            decoded.append(decoded_cur)
        decoded = torch.LongTensor(decoded).view(batch_size, top_k, -1).to(device)
        return decoded

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

        self.encoder = Encoder(self.desc_emb, config.emb_dim, config.n_hidden,
                                  True, config.n_layers, config.noise_radius)

        self.ctx2dec = nn.Sequential(
            nn.Linear(2 * config.n_hidden, config.n_hidden),
            nn.Tanh(),
        )

        self.ctx2dec.apply(self.init_weights)

        self.decoder = Decoder(self.api_emb, config.emb_dim, config.n_hidden,
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



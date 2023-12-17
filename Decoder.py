import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init

import operator
from helper import SOS_ID, EOS_ID, PAD_ID
from queue import PriorityQueue
import numpy as np
from Attention import Attention
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
        
class BeamHupotheses(object):
    def __init__(self, beam_width, max_len):
        self.max_len = max_len
        self.beam_width = beam_width
        self.beams = []
        self.worst_score = 1e9
    
    def __len__(self):
        return len(self.beams)

    def add(self, hyp, sum_logprobs):
        score = sum_logprobs/len(hyp)
        if len(self) < self.beam_width or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self)>self.beam_width:
                sorted_scores = sorted([(s, idx) for idx, (s,_) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len=None):
        if len(self) < self.beam_width:
            return False
        else:
            if cur_len is None:
                cur_len = self.max_len
            cur_score = best_sum_logprobs/cur_len
            ret = self.worst_score>=cur_score
            return ret
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, tgt_seq_len, d_model, n_heads, d_ff, n_layers, dropout) -> None:
        super().__init__()
        #self.api_emb = encoder.embeddings()
        #self.encoder = encoder
        #self.tokenizer = tokenizer
        self.word_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        self.pos_emb = PositionalEncoding(d_model, tgt_seq_len*10)
        self.dropout = nn.Dropout(dropout)
        self.decoder_layers = nn.TransformerDecoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layers, n_layers)
    
    def forward(self, dec_inputs, enc_outputs, enc_key_padding_mask):
        #PAD_ID = self.tokenizer.pad_token_id
        PAD_ID = 0
        #dec_outputs = self.encoder.embeddings(dec_inputs)
        dec_emb = self.word_emb(dec_inputs)
        dec_pos = self.pos_emb(dec_emb)
        dec_outputs = dec_emb+dec_pos
        dec_outputs = self.dropout(dec_outputs)
        dec_mask = get_mask(dec_inputs)
        dec_key_padding_mask = get_key_padding_mask(dec_inputs, PAD_ID=PAD_ID)
        dec_outputs = self.decoder(dec_outputs, enc_outputs, tgt_mask = dec_mask, tgt_key_padding_mask=dec_key_padding_mask, memory_key_padding_mask=enc_key_padding_mask)
        return dec_outputs


class RNNDecoder(nn.Module):
    def __init__(self, output_emb, output_size, hidden_size, vocab_size, n_layers=1, dropout=0.5):
        super(RNNDecoder, self).__init__()
        self.output_emb = output_emb
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.rnn = nn.GRU(output_size, hidden_size, batch_first=True)
        self.project = nn.Linear(hidden_size, vocab_size)

        self.attn = Attention(self.hidden_size, self.hidden_size, self.hidden_size)
        self.x_context = nn.Linear(hidden_size + output_size, output_size)
        self.rnn = nn.GRU(output_size + hidden_size, hidden_size, batch_first=True)
        self.project = nn.Linear(2*self.hidden_size, vocab_size)

        self.init_weights()

    def init_weights(self):
        for w in self.rnn.parameters(): # initialize the gate weights with orthogonal
            if w.dim()>1:
                weight_init.orthogonal_(w)
        self.project.weight.data.uniform_(-0.1, 0.1)#nn.init.xavier_normal_(self.out.weight)
        nn.init.constant_(self.project.bias, 0.)
        self.x_context.weight.data.uniform_(-0.1, 0.1)  # nn.init.xavier_normal_(self.out.weight)
        nn.init.constant_(self.x_context.bias, 0.)

    def forward(self, init_h, enc_hids, src_pad_mask, outputs):
        if self.output_emb is not None:
            outputs = self.output_emb(outputs)
        batch_size, maxlen, _ = outputs.size()
        outputs = F.dropout(outputs, self.dropout, self.training)
        h = init_h.unsqueeze(0)

        attn_ctx, _, _ = self.attn(init_h.unsqueeze(1), enc_hids, enc_hids, src_pad_mask)
        for di in range(maxlen):
            x = outputs[:,di,:].unsqueeze(1)
            x = torch.cat((x, attn_ctx), 2)
            h_n, h = self.rnn(x, h)
            attn_ctx, _, comb_attn_ctx = self.attn(h_n, enc_hids, enc_hids, src_pad_mask)
            logits = self.project(torch.cat((h_n, attn_ctx), 2))
            decoded = logits if di==0 else torch.cat([decoded, logits], 1)

        '''

        hids, h = self.rnn(outputs, h)
        decoded = self.project(hids.contiguous().view(-1, self.hidden_size))  # reshape before linear over vocab
        decoded = decoded.view(batch_size, maxlen, self.vocab_size)
        '''
        return decoded, h

    def beam_decode(self, init_h, enc_hids, src_pad_mask, beam_width, max_unroll, topk = 10):
        device = init_h.device
        batch_size = init_h.size(0)
        decoded_words = np.zeros((batch_size, topk, max_unroll), dtype=np.int)
        sample_lens = np.zeros((batch_size, topk), dtype=np.int)
        scores = np.zeros((batch_size,topk))

        for idx in range(batch_size):
            if isinstance(init_h, tuple):
                h = (init_h[0][idx, :].view(1, 1, -1), init_h[1][idx, :].view(1, 1, -1))
            else:
                h = init_h[idx, :].view(1, 1, -1)

            if enc_hids is not None:
                enc_outs = enc_hids
                enc_outs = enc_outs[idx, :, :].unsqueeze(0)
                src_pad_mask = src_pad_mask[idx, :].unsqueeze(0)

            x = torch.zeros((1, 1), dtype=torch.long, device=device)
            attn_ctx, _, _ = self.attn(h, enc_hids, enc_hids, src_pad_mask)
            endnodes = []
            number_required = min((topk + 1), topk - len(endnodes))

            node = BeamNode(h, None, x, 0, 1)
            nodes = PriorityQueue()

            nodes.put((-node.eval(), node))
            qsize = 1

            while True:
                if qsize > 2000: break
                score, n =nodes.get()
                x = n.wordid
                h = n.h
                qsize -= 1

                if n.wordid.item() == EOS_ID and n.prevNode != None:
                    endnodes.append((score, n))
                    # if we reached maximum # of sentences required
                    if len(endnodes) >= number_required:
                        break
                    else:
                        continue

                if self.output_emb is not None:
                    x = self.output_emb(x)
                x = torch.cat((x, attn_ctx), 2)

                h_n, h = self.rnn(x, h)
                attn_ctx, _, comb_attn_ctx = self.attn(h_n, enc_hids, enc_hids, src_pad_mask)
                logits = self.project(torch.cat((h_n, attn_ctx),2))
                logits = logits.squeeze(1)
                logits = F.log_softmax(logits, 1)
                log_prob, indexes = torch.topk(logits, beam_width)
                for new_k in range(beam_width):
                    decoded_t = indexes[0][new_k].view(1, -1)
                    log_p = log_prob[0][new_k].item()
                    node = BeamNode(h, n, decoded_t, n.logp + log_p, n.len + 1)
                    score = -node.eval()
                    nodes.put((score, node))  # put them into queue
                    qsize += 1  # increase qsize

            if len(endnodes) == 0:
                endnodes = [nodes.get() for _ in range(topk)]

            uid = 0
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                utterance, length = [], n.len
                utterance.append(n.wordid)
                # back trace
                while n.prevNode != None:
                    n = n.prevNode
                    utterance.append(n.wordid)
                utterance = utterance[::-1]  # reverse
                utterance, length = utterance[1:], length - 1  # remove <sos>
                utterance = [tensor.cpu() for tensor in utterance]
                decoded_words[idx, uid, :min(length, max_unroll)] = utterance[:min(length, max_unroll)]

                sample_lens[idx, uid] = min(length, max_unroll)
                scores[idx, uid] = score
                uid = uid + 1

        return decoded_words, sample_lens, scores



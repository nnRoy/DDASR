import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init

import operator
from helper import SOS_ID, EOS_ID, PAD_ID
from queue import PriorityQueue
import numpy as np
from Attention import Attention

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



import torch
import torch.nn as nn
from torch.autograd import Variable
import config
from utils import get_PL_dist

class Loss_Longtail(nn.Module):
    def __init__(self):
        super(Loss_Longtail, self).__init__()
    def forward(self, output, target, longtail, loss_weight):

        target = target.long()
        current_loss_weight = torch.squeeze(torch.gather(torch.tensor(loss_weight).cuda(), 0, target.view(-1, 1)))
        target_scores = torch.squeeze(torch.gather(output, 1, target.view(-1, 1)))
        below = -torch.log(torch.sum(torch.exp(output), 1))
        log_pl = target_scores + below
        temp = []
        for i in range(len(longtail)):
            temp_longtail = [x for x in longtail[i] if x > 0]
            if len(temp_longtail) > 0:
                tail_idx = Variable(torch.LongTensor(temp_longtail))
                if self.cuda: tail_idx = tail_idx.cuda()
                scores = torch.index_select(output[i,:], 0 , tail_idx)
                other_scores_tail = torch.sum(torch.exp(output[i,:])) - (torch.exp(target_scores[i]) + torch.sum(torch.exp(scores)))
                log_pl_tail = get_PL_dist(scores, other_scores_tail)
                log_pl[i] = log_pl[i].clone() + log_pl_tail
                temp.append((i, log_pl_tail))

        neg_like = -log_pl + current_loss_weight

        return neg_like, temp


class Loss_Longtail_Division(nn.Module):
    def __init__(self):
        super(Loss_Longtail_Division, self).__init__()
    def forward(self, output, target, longtail, loss_weight):
        target = target.long()
        loss_weight = torch.tensor(loss_weight).cuda()
        loss_weight = loss_weight.view(-1, config.api_vocab_size)
        output = torch.div(output, loss_weight)
        target_scores = torch.squeeze(torch.gather(output, 1, target.view(-1, 1)))
        below = -torch.log(torch.sum(torch.exp(output), 1))
        log_pl = target_scores + below
        temp = []
        for i in range(len(longtail)):
            temp_longtail = [x for x in longtail[i] if x > 0]
            if len(temp_longtail) > 0:
                tail_idx = Variable(torch.LongTensor(temp_longtail))
                if self.cuda: tail_idx = tail_idx.cuda()
                scores = torch.index_select(output[i,:], 0 , tail_idx)
                other_scores_tail = torch.sum(torch.exp(output[i,:])) - (torch.exp(target_scores[i]) + torch.sum(torch.exp(scores)))
                log_pl_tail = get_PL_dist(scores, other_scores_tail)
                log_pl[i] = log_pl[i].clone() + log_pl_tail
                temp.append((i, log_pl_tail))

        neg_like = -log_pl

        return neg_like, temp


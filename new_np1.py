import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import math
def cross_entropy(logits, target, size_average=True):
    if size_average:
        return torch.mean(torch.sum(- target * F.log_softmax(logits, -1), -1))
    else:
        return torch.sum(torch.sum(- target * F.log_softmax(logits, -1), -1))


class NpairLoss(nn.Module):
    """the multi-class n-pair loss"""
    def __init__(self, l2_reg=0.02):
        super(NpairLoss, self).__init__()
        self.l2_reg = l2_reg

    def forward(self, embeddings, target, n2, n4):
        indices11 = torch.LongTensor([0, 4, 8, 12, 16, 20, 24, 28])
        indices12 = torch.LongTensor([1, 5, 9, 13, 17, 21, 25, 29])
        indices21 = torch.LongTensor([2, 6, 10, 14, 18, 22, 26, 30])
        indices22 = torch.LongTensor([3, 7, 11, 15, 19, 23, 27, 31])
        anchor1 = torch.index_select(embeddings, 0, indices11.cuda())
        positive1 = torch.index_select(embeddings, 0, indices12.cuda())
        anchor2 = torch.index_select(embeddings, 0, indices21.cuda())                      ####CHANGED######
        positive2 = torch.index_select(embeddings, 0, indices22.cuda())
        batch_size = anchor1.size(0)
        target = target.view(target.size(0)*target.size(1), 1)
        target = torch.index_select(target, 0, indices11)
        polarity_score = n2
        emotion_score = n4
        polarity = target < 4
        anchor1_score_p = torch.index_select(polarity_score, 0, indices11.cuda())
        #print(anchor1_score_p)
        positive1_score_p = torch.index_select(polarity_score, 0, indices12.cuda())                ###CHANGED###
        anchor2_score_p = torch.index_select(polarity_score, 0, indices21.cuda())
        positive2_score_p = torch.index_select(polarity_score, 0, indices22.cuda())

        anchor1_score_e = torch.index_select(emotion_score, 0, indices11.cuda()).long()
        positive1_score_e = torch.index_select(emotion_score, 0, indices12.cuda()).long()              ###CHANGED####
        anchor2_score_e = torch.index_select(emotion_score, 0, indices21.cuda()).long()
        positive2_score_e = torch.index_select(emotion_score, 0, indices22.cuda()).long()

        weight_polarity_anchor1 = torch.zeros(8, 2)
        weight_polarity_positive1 = torch.zeros(8, 2)
        weight_polarity_anchor2 = torch.zeros(8, 2)
        weight_polarity_positive2 = torch.zeros(8, 2)

        #print(polarity)

        prev=True
        main=polarity.squeeze().data.cpu().numpy()
        x=0
        for x in range(main.shape[0]-1):
          if main[x]==False and main[x+1]==False:
            pass
          else:
            break

        main=np.array([main[x],main[x+1]])
        for i in range(8):
           # print(anchor1_score_p[i,polarity.squeeze().data.cpu().numpy()].size())
            weight_polarity_anchor1[i] = anchor1_score_p[i, main]
            weight_polarity_positive1[i] = positive1_score_p[i, main]
            weight_polarity_anchor2[i] = anchor2_score_p[i, main]
            weight_polarity_positive2[i] = positive2_score_p[i, main]

        weight_emotion_anchor1 = torch.zeros(8, 8).long()
        weight_emotion_positive1 = torch.zeros(8, 8).long()
        weight_emotion_anchor2 = torch.zeros(8, 8).long()
        weight_emotion_positive2 = torch.zeros(8, 8).long()

        m=target.long()
        for i in range(8):
            weight_emotion_anchor1[i] = anchor1_score_e[i][m].squeeze()
            weight_emotion_positive1[i] = positive1_score_e[i][m].squeeze()
            weight_emotion_anchor2[i] = anchor2_score_e[i][m].squeeze()
            weight_emotion_positive2[i] = positive2_score_e[i][m].squeeze()
        tgidx = target.long()
        label = torch.LongTensor([0, 0, 0, 0, 0, 0, 0, 0])
        label = label.view(label.size(0), 1)
        for i in range(0, 8):
            label[tgidx[i]] = i
        target = (target == torch.transpose(target, 0, 1)).float()
        target = target / torch.sum(target, dim=1, keepdim=True).float()


        logit1 = torch.matmul(anchor1, torch.transpose(positive1, 0, 1))
        logit2 = torch.matmul(anchor2, torch.transpose(positive2, 0, 1))

       # print(weight_polarity_anchor1.size()," ",weight_polarity_positive1.size())
        weight_polarity1 = torch.matmul(torch.exp(weight_polarity_anchor1), torch.exp(torch.transpose(weight_polarity_positive1, 0, 1)))
        weight_polarity2 = torch.matmul(torch.exp(weight_polarity_anchor2), torch.exp(torch.transpose(weight_polarity_positive2, 0, 1)))

        for i in range(0, 8):
            for j in range(0, 8):
                if polarity[i] == polarity[j]:
                    weight_polarity1[i][j] = 1
                    weight_polarity2[i][j] = 1


        weight_emotion1 = torch.matmul(torch.exp(weight_emotion_anchor1.float()), torch.exp(torch.transpose(weight_emotion_positive1.float(), 0, 1)))
        weight_emotion2 = torch.matmul(torch.exp(weight_emotion_anchor2.float()), torch.exp(torch.transpose(weight_emotion_positive2.float(), 0, 1)))


        for i in range(0, 8):
            weight_emotion1[i][i] = 1
            weight_emotion2[i][i] = 1

        logit1 = logit1 * weight_polarity1.cuda()
        logit2 = logit2 * weight_polarity2.cuda()     ########## CHANGED###

        logit1 = logit1 * weight_emotion1.cuda()
        logit2 = logit2 * weight_emotion2.cuda()


        logit1_2 = torch.zeros(8, 2).cuda()            ###CHANGED###
        logit2_2 = torch.zeros(8, 2).cuda()

        index_1 = torch.LongTensor([[label[0], label[1], label[2], label[3]]])
        index_2 = torch.LongTensor([[label[4], label[5], label[6], label[7]]])

        lossp1 = torch.cat([logit1[:, index_1][index_1, :].squeeze(0).squeeze(1), logit1[:, index_2][index_2, :].squeeze(0).squeeze(1)], 0)
        lossp2 = torch.cat([logit2[:, index_1][index_1, :].squeeze(0).squeeze(1), logit2[:, index_2][index_2, :].squeeze(0).squeeze(1)], 0)



        for i in range(0, 8):
            if tgidx[i] - 4 < 0:
                logit1_2[i, 0] = (logit1[i, label[0:4]].sum() - logit1[i, label[tgidx[i]]])/3
                logit2_2[i, 0] = (logit2[i, label[0:4]].sum() - logit2[i, label[tgidx[i]]])/3
                logit1_2[i, 1] = (logit1[i, label[4:8]].sum())/4
                logit2_2[i, 1] = (logit2[i, label[4:8]].sum())/4


            else:
                logit1_2[i, 0] = (logit1[i, label[4:8]].sum() - logit1[i, label[tgidx[i]]])/3
                logit2_2[i, 0] = (logit2[i, label[4:8]].sum() - logit2[i, label[tgidx[i]]])/3
                logit1_2[i, 1] = (logit1[i, label[0:4]].sum())/4
                logit2_2[i, 1] = (logit2[i, label[0:4]].sum())/4


        tg2 = torch.zeros(8, 2).cuda()     #CHANGED###
        tg2[:, 0] = 1
        tg3 = torch.cat([torch.eye(4), torch.eye(4)], 0).cuda()      ###CHANGED##

        loss_ce1 = 0.5 * cross_entropy(logit1_2, tg2) + 0.5 * cross_entropy(lossp1, tg3)
        loss_ce2 = 0.5 * cross_entropy(logit2_2, tg2) + 0.5 * cross_entropy(lossp2, tg3)

        l2_loss1 = torch.sum(anchor1 ** 2) / batch_size + torch.sum(positive1 ** 2) / batch_size
        l2_loss2 = torch.sum(anchor2 ** 2) / batch_size + torch.sum(positive2 ** 2) / batch_size

        loss = (loss_ce1 + loss_ce2 + self.l2_reg * l2_loss1 * 0.25 + self.l2_reg * l2_loss2 * 0.25)/2
        return loss

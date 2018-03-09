'''

args:
input: (N, num_classes)
target: (N, 1)
alpha(data balance parameter): (num_classes, 1)
gamma(weight decay parameter): scalar

output:
focalloss

'''

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):

    def __init__(self, alpha=None, gamma=2, size_average=False):
        super(FocalLoss, self).__init__()
        if alpha is not None:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        else:
            self.alpha = None
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target):
        dim = input.dim()
        if dim != 2:
            raise ValueError('Expected 2 dimensions (got {})'.format(dim))
        target = target.view(-1, 1)

        log_p = F.log_softmax(input, dim=1)
        log_p = log_p.gather(1, target)
        log_p = log_p.view(-1, 1)
        p = Variable(log_p.data.exp())

        if self.alpha is not None:
            if input.is_cuda and not self.alpha.is_cuda:
                self.alpha = self.alpha.cuda()
            al = self.alpha.gather(0, target)
            al = al.view(-1, 1)
            log_p = log_p * al
        
        loss = -1 * pow(1 - p, self.gamma) * log_p

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

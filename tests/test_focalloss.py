from __future__ import print_function

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from context import *

max_error = 0.0

for i in xrange(100):
    input = Variable(torch.rand(2, 5))
    target = Variable(torch.LongTensor(2).random_(5))
    alpha1 = 0.2 * torch.ones(5, 1)
    alpha2 = 1 * torch.ones(5, 1)

    if torch.cuda.is_available():
        input, target, alpha1, alpha2 = input.cuda(), target.cuda(), alpha1.cuda(), alpha2.cuda()
    
    celoss1 = nn.CrossEntropyLoss(weight=alpha1)
    celoss2 = nn.CrossEntropyLoss(weight=alpha1)
    nllloss1 = nn.NLLLoss(weight=alpha1)
    nllloss2 = nn.NLLLoss(weight=alpha2)
    focalloss1 = FocalLoss(alpha=alpha1, gamma=0, size_average=True)
    focalloss2 = FocalLoss(alpha=alpha2, gamma=0, size_average=True)

    
    output_nll1 = nllloss1(F.log_softmax(input), target)
    output_nll2 = nllloss2(F.log_softmax(input), target)
    output_fc1 = focalloss1(input, target)
    output_fc2 = focalloss2(input, target)
    output_ce1 = celoss1(input, target)
    output_ce2 = celoss2(input, target)

    error = abs(output_ce1.data[0] - output_fc1.data[0])
    if error > max_error:
        max_error = error

print('input: ', input)
print('target: ', target)
print('alpha1: ', alpha1)
print('alpha2: ', alpha2)
print('log_softmax: ', F.log_softmax(input))
print('max error: {error: .4f}'.format(error=max_error))
print(output_fc1.data[0])
print(output_fc2.data[0])
print(output_ce1.data[0])
print(output_ce2.data[0])
print(output_nll1.data[0])
print(output_nll2.data[0])
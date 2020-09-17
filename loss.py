import torch
import torch.nn as nn
import torch.nn.functional as F

class betweenLoss(nn.Module):
    def __init__(self, gamma=[1,1,1,1,1,1], loss=nn.L1Loss()):
        super(betweenLoss, self).__init__()
        self.gamma = gamma
        self.loss = loss

    def forward(self, outputs, targets):
        assert len(outputs)
        assert len(outputs) == len(targets)

        length = len(outputs)
        
        res = sum([self.gamma[i]*self.loss(outputs[i], targets[i]) for i in range(length)])

        return res

def CrossEntropy(outputs, targets):
    log_softmax_outputs = F.log_softmax(outputs, dim=1)
    softmax_targets = F.softmax(targets, dim=1)

    return -(log_softmax_outputs*softmax_targets).sum(dim=1).mean()


class discriminatorLoss(nn.Module):
    def __init__(self, models, loss=nn.BCEWithLogitsLoss()):
        super(discriminatorLoss, self).__init__()
        self.models = models
        self.loss = loss

    def forward(self, outputs, targets):
        inputs = [torch.cat((i,j),0) for i, j in zip(outputs, targets)]
        inputs = torch.cat(inputs, 1)
        batch_size = inputs.size(0)
        target = torch.FloatTensor([[1, 0] for _ in range(batch_size//2)] + [[0, 1] for _ in range(batch_size//2)])
        target = target.to(inputs[0].device)
        output = self.models(inputs)
        res = self.loss(output, target)
        return res


class discriminatorFakeLoss(nn.Module):
    def forward(self, outputs, targets):
        res = (0*outputs[0]).sum()
        return res






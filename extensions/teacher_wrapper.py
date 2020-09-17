import torch
from torch import nn
from torch.nn import functional as F

import random
import numpy as np


class ModelDistillationWrapper(nn.Module):
    """Convenient wrapper class to train a model with soft label ."""

    def __init__(self, model, teacher):
        super().__init__()
        self.model = model
        self.teachers_0 = teacher
        self.combine = True

        # Since we don't want to back-prop through the teacher network,
        # make the parameters of the teacher network not require gradients. This
        # saves some GPU memory.

        for model in self.teachers_0:
            for param in model.parameters():
                param.requires_grad = False

        self.false = False

    @property
    def LR_REGIME(self):
        # Training with soft label does not change learing rate regime.
        # Return's wrapped model lr regime.
        return self.model.LR_REGIME

    def state_dict(self):
        return self.model.state_dict()

    def forward(self, input, before=False):
        if self.training:
            if len(self.teachers_0) == 3 and self.combine == False:
                index = [0,1,1,2,2]
                idx = random.randint(0, 4)
                soft_labels_ = self.teachers_0[index[idx]](input)
                soft_labels = F.softmax(soft_labels_, dim=1)

            elif self.combine:
                soft_labels_ = [ torch.unsqueeze(self.teachers_0[idx](input), dim=2) for idx in range(len(self.teachers_0))]
                soft_labels_softmax = [F.softmax(i, dim=1) for i in soft_labels_]                
                soft_labels_ = torch.cat(soft_labels_, dim=2).mean(dim=2)                
                soft_labels = torch.cat(soft_labels_softmax, dim=2).mean(dim=2)

            else:
                idx = random.randint(0, len(self.teachers_0)-1)
                soft_labels_ = self.teachers_0[idx](input)
                soft_labels = F.softmax(soft_labels_, dim=1)

            # soft_labels = F.softmax(soft_labels_, dim=1)
            model_output = self.model(input)

            if before:
                return (model_output, soft_labels, soft_labels_)

            return (model_output, soft_labels)

        else:
            return self.model(input)

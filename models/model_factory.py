"""Utility functions to construct a model."""

import torch
from torch import nn

import random

from extensions import data_parallel
from extensions import teacher_wrapper
from extensions import kd_loss
from models import resnet50
import torchvision.models as models
# import pretrainedmodels
import timm

MODEL_NAME_MAP = {
    'resnet50': resnet50.ResNet50,
}

def _create_single_cpu_model(model_name, state_file=None):
    model = _create_model(model_name, teacher=False, pretrain=True)
    if state_file is not None:
        model.load_state_dict(torch.load(state_file))
    # model = torch.nn.DataParallel(model).cuda()
    return model

def _create_checkpoint_model(model_name, state_file=None):
    model = _create_model(model_name, teacher=False, pretrain=True)
    # model = timm.create_model(model_name.lower(), pretrained=False)
    if state_file is not None:
        model.load_state_dict(torch.load(state_file))
        # model = torch.nn.DataParallel(model).cuda()
    return model

def _create_model(model_name, teacher=False, pretrain=True):
    if pretrain:
        print("=> teacher" if teacher else "=> student", end=":")
        print(" using pre-trained model '{}'".format(model_name))

        # model = models.__dict__[model_name.lower()](pretrained=True)
        model = timm.create_model(model_name.lower(), pretrained=True)
    else:
        print("=> creating model '{}'".format(model_name))
        # model = models.__dict__[model_name.lower()]()
        model = timm.create_model(model_name.lower(), pretrained=False)

    if model_name.startswith('alexnet') or model_name.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    if teacher:
        for p in model.parameters():
            p.requires_grad = False
        model.eval()

    return model


def teachers(teachers=['resnet50'], state_file=None):
    if state_file is not None:
        return [_create_single_cpu_model(t, state_file).cuda() for t in teachers]
    else:
        return [_create_model(t, teacher=True).cuda() for t in teachers]


def create_model(model_name, student_state_file=None, gpus=[], teacher=None,
                 teacher_state_file=None):
    # model = _create_model(model_name)
    model = _create_checkpoint_model(model_name, student_state_file)
    model.LR_REGIME = [1, 100, 0.01, 101, 300, 0.001] # LR_REGIME 
    if teacher is not None:
        # assert teacher_state_file is not None, "Teacher state is None."

        teacher = teachers(teacher.split(","), teacher_state_file)
        model = teacher_wrapper.ModelDistillationWrapper(model, teacher)
        loss = kd_loss.KLLoss()
    else:
        loss = nn.CrossEntropyLoss()

    return model, loss

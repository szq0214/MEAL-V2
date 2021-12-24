#!/usr/bin/env python3
"""Script to train a model through soft labels on ImageNet's train set."""

import argparse
import logging
import pprint
import os
import sys
import time
import math
import numpy as np

import torch
from torch import nn

from loss import discriminatorLoss

import imagenet
from models import model_factory
from models import discriminator
import opts
import test
import utils
from utils_FKD import Recover_soft_label

def parse_args(argv):
    """Parse arguments @argv and return the flags needed for training."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)

    group = parser.add_argument_group('General Options')
    opts.add_general_flags(group)

    group = parser.add_argument_group('Dataset Options')
    opts.add_dataset_flags(group)

    group = parser.add_argument_group('Model Options')
    opts.add_model_flags(group)

    group = parser.add_argument_group('Soft Label Options')
    opts.add_teacher_flags(group)

    group = parser.add_argument_group('Training Options')
    opts.add_training_flags(group)

    group = parser.add_argument_group('CutMix Training Options')
    opts.add_cutmix_training_flags(group)

    args = parser.parse_args(argv)

    return args


class LearningRateRegime:
    """Encapsulates the learning rate regime for training a model.

    Args:
        @intervals (list): A list of triples (start, end, lr). The intervals
            are inclusive (for start <= epoch <= end, lr will be used). The
            start of each interval must be right after the end of its previous
            interval.
    """

    def __init__(self, regime):
        if len(regime) % 3 != 0:
            raise ValueError("Regime length should be devisible by 3.")
        intervals = list(zip(regime[0::3], regime[1::3], regime[2::3]))
        self._validate_intervals(intervals)
        self.intervals = intervals
        self.num_epochs = intervals[-1][1]

    @classmethod
    def _validate_intervals(cls, intervals):
        if type(intervals) is not list:
            raise TypeError("Intervals must be a list of triples.")
        elif len(intervals) == 0:
            raise ValueError("Intervals must be a non empty list.")
        elif intervals[0][0] != 1:
            raise ValueError("Intervals must start from 1: {}".format(intervals))
        elif any(end < start for (start, end, lr) in intervals):
            raise ValueError("End of intervals must be greater or equal than their"
                             " start: {}".format(intervals))
        elif any(intervals[i][1] + 1 != intervals[i + 1][0]
                 for i in range(len(intervals) - 1)):
            raise ValueError("Start of each each interval must be the end of its "
                             "previous interval plus one: {}".format(intervals))

    def get_lr(self, epoch):
        for (start, end, lr) in self.intervals:
            if start <= epoch <= end:
                return lr
        raise ValueError("Invalid epoch {} for regime {!r}".format(
            epoch, self.intervals))


def _set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def _get_learning_rate(optimizer):
    return max(param_group['lr'] for param_group in optimizer.param_groups)


def train_for_one_epoch(model, g_loss, discriminator_loss, train_loader, optimizer, epoch_number, args):
    model.train()
    g_loss.train()

    data_time_meter = utils.AverageMeter()
    batch_time_meter = utils.AverageMeter()
    g_loss_meter = utils.AverageMeter(recent=100)
    d_loss_meter = utils.AverageMeter(recent=100)
    top1_meter = utils.AverageMeter(recent=100)
    top5_meter = utils.AverageMeter(recent=100)

    timestamp = time.time()
    for i, (images, labels, soft_labels) in enumerate(train_loader):
        batch_size = args.batch_size

        # Record data time
        data_time_meter.update(time.time() - timestamp)

        images = torch.cat(images, dim=0)
        soft_labels = torch.cat(soft_labels, dim=0)
        labels = torch.cat(labels, dim=0)

        if args.soft_label_type == 'ori':
            soft_labels = soft_labels.cuda()
        else:
            soft_labels = Recover_soft_label(soft_labels, args.soft_label_type, args.num_classes)
            soft_labels = soft_labels.cuda()

        if utils.is_model_cuda(model):
            images = images.cuda()
            labels = labels.cuda()

        if args.w_cutmix == True:
            r = np.random.rand(1)
            if args.beta > 0 and r < args.cutmix_prob:
                # generate mixed sample
                lam = np.random.beta(args.beta, args.beta)
                rand_index = torch.randperm(images.size()[0]).cuda()
                target_a = soft_labels
                target_b = soft_labels[rand_index]
                bbx1, bby1, bbx2, bby2 = utils.rand_bbox(images.size(), lam)
                images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))

        # Forward pass, backward pass, and update parameters.
        output = model(images)
        # output, soft_label, soft_no_softmax = outputs
        if args.w_cutmix == True:
            g_loss_output1 = g_loss((output, target_a), labels)
            g_loss_output2 = g_loss((output, target_b), labels)
        else:
            g_loss_output = g_loss((output, soft_labels), labels)
        if args.use_discriminator_loss:
            # Our stored label is "after softmax", this is slightly different from original MEAL V2 
            # that used probibilaties "before softmax" for the discriminator.
            output_softmax = nn.functional.softmax(output)
            if args.w_cutmix == True:
                d_loss_value = discriminator_loss([output_softmax], [target_a]) * lam + discriminator_loss([output_softmax], [target_b]) * (1 - lam)
            else:
                d_loss_value = discriminator_loss([output_softmax], [soft_labels])

        # Sometimes loss function returns a modified version of the output,
        # which must be used to compute the model accuracy.
        if args.w_cutmix == True:
            if isinstance(g_loss_output1, tuple):
                g_loss_value1, output1 = g_loss_output1
                g_loss_value2, output2 = g_loss_output2
                g_loss_value = g_loss_value1 * lam + g_loss_value2 * (1 - lam)
            else:
                g_loss_value = g_loss_output1 * lam + g_loss_output2 * (1 - lam)
        else:
            if isinstance(g_loss_output, tuple):
                g_loss_value, output = g_loss_output
            else:
                g_loss_value = g_loss_output

        if args.use_discriminator_loss:
            loss_value = g_loss_value + d_loss_value
        else:
            loss_value = g_loss_value 

        loss_value.backward()

        # Update parameters and reset gradients.
        optimizer.step()
        optimizer.zero_grad()

        # Record loss and model accuracy.
        g_loss_meter.update(g_loss_value.item(), batch_size)
        d_loss_meter.update(d_loss_value.item(), batch_size)

        top1, top5 = utils.topk_accuracy(output, labels, recalls=(1, 5))
        top1_meter.update(top1, batch_size)
        top5_meter.update(top5, batch_size)

        # Record batch time
        batch_time_meter.update(time.time() - timestamp)
        timestamp = time.time()

        if i%20 == 0:
            logging.info(
                'Epoch: [{epoch}][{batch}/{epoch_size}]\t'
                'Time {batch_time.value:.2f} ({batch_time.average:.2f})   '
                'Data {data_time.value:.2f} ({data_time.average:.2f})   '
                'G_Loss {g_loss.value:.3f} {{{g_loss.average:.3f}, {g_loss.average_recent:.3f}}}    '
                'D_Loss {d_loss.value:.3f} {{{d_loss.average:.3f}, {d_loss.average_recent:.3f}}}    '
                'Top-1 {top1.value:.2f} {{{top1.average:.2f}, {top1.average_recent:.2f}}}    '
                'Top-5 {top5.value:.2f} {{{top5.average:.2f}, {top5.average_recent:.2f}}}    '
                'LR {lr:.5f}'.format(
                    epoch=epoch_number, batch=i + 1, epoch_size=len(train_loader),
                    batch_time=batch_time_meter, data_time=data_time_meter,
                    g_loss=g_loss_meter, d_loss=d_loss_meter, top1=top1_meter, top5=top5_meter,
                    lr=_get_learning_rate(optimizer)))
    # Log the overall train stats
    logging.info(
        'Epoch: [{epoch}] -- TRAINING SUMMARY\t'
        'Time {batch_time.sum:.2f}   '
        'Data {data_time.sum:.2f}   '
        'G_Loss {g_loss.average:.3f}     '
        'D_Loss {d_loss.average:.3f}     '
        'Top-1 {top1.average:.2f}    '
        'Top-5 {top5.average:.2f}    '.format(
            epoch=epoch_number, batch_time=batch_time_meter, data_time=data_time_meter,
            g_loss=g_loss_meter, d_loss=d_loss_meter, top1=top1_meter, top5=top5_meter))


def save_checkpoint(checkpoints_dir, model, optimizer, epoch):
    model_state_file = os.path.join(checkpoints_dir, 'model_state_{:02}.pytar'.format(epoch))
    optim_state_file = os.path.join(checkpoints_dir, 'optim_state_{:02}.pytar'.format(epoch))
    torch.save(model.state_dict(), model_state_file)
    torch.save(optimizer.state_dict(), optim_state_file)


def create_optimizer(model,  discriminator_parameters, momentum=0.9, weight_decay=0):
    # Get model parameters that require a gradient.
    parameters = [{'params': model.parameters()}, discriminator_parameters]
    optimizer = torch.optim.SGD(parameters, lr=0,
                                momentum=momentum, weight_decay=weight_decay)
    return optimizer

def create_discriminator_criterion(args):
    d = discriminator.Discriminator(outputs_size=1000, K=8).cuda()
    d = torch.nn.DataParallel(d)
    update_parameters = {'params': d.parameters(), "lr": args.d_lr}
    discriminators_criterion = discriminatorLoss(d).cuda()
    if len(args.gpus) > 1:
        discriminators_criterion = torch.nn.DataParallel(discriminators_criterion, device_ids=args.gpus)
    return discriminators_criterion, update_parameters

def main(argv):
    """Run the training script with command line arguments @argv."""
    args = parse_args(argv)
    utils.general_setup(args.save, args.gpus)

    logging.info("Arguments parsed.\n{}".format(pprint.pformat(vars(args))))

    # convert to TRUE number of loading-images since we use multiple crops from the same image within a minbatch
    args.batch_size = math.ceil(args.batch_size / args.num_crops)

    # Create the train and the validation data loaders.
    train_loader = imagenet.get_train_loader_FKD(args.imagenet, args.batch_size,
                                             args.num_workers, args.image_size, args.num_crops, args.softlabel_path)
    val_loader = imagenet.get_val_loader(args.imagenet, args.batch_size,
                                         args.num_workers, args.image_size)
    # Create model with optional teachers.
    model, loss = model_factory.create_model(
        args.model, args.student_state_file, args.gpus, args.teacher_model,
        args.teacher_state_file, True)
    logging.info("Model:\n{}".format(model))

    discriminator_loss, update_parameters = create_discriminator_criterion(args)

    optimizer = create_optimizer(model, update_parameters, args.momentum, args.weight_decay)

    for epoch in range(args.start_epoch, args.epochs, args.num_crops):
        adjust_learning_rate(optimizer, epoch, args)
        train_for_one_epoch(model, loss, discriminator_loss, train_loader, optimizer, epoch, args)
        test.test_for_one_epoch(model, loss, val_loader, epoch)
        save_checkpoint(args.save, model, optimizer, epoch)


if __name__ == '__main__':
    main(sys.argv[1:])

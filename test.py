#!/usr/bin/env python3
"""Script to test a pytorch model on ImageNet's validation set."""

import argparse
import logging
import pprint
import sys
import time

import torch
from torch import nn

import imagenet
from models import model_factory
import opts
import utils


def parse_args(argv):
    """Parse arguments @argv and return the flags needed for training."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)

    group = parser.add_argument_group('General Options')
    opts.add_general_flags(group)

    group = parser.add_argument_group('Dataset Options')
    opts.add_dataset_flags(group)

    group = parser.add_argument_group('Model Options')
    opts.add_model_flags(group)

    args = parser.parse_args(argv)

    if args.student_state_file is None:
        parser.error("You should set --model-state-file (student) to reload a model "
                     "state.")

    return args


def test_for_one_epoch(model, loss, test_loader, epoch_number):
    model.eval()
    loss.eval()

    data_time_meter = utils.AverageMeter()
    batch_time_meter = utils.AverageMeter()
    loss_meter = utils.AverageMeter(recent=100)
    top1_meter = utils.AverageMeter(recent=100)
    top5_meter = utils.AverageMeter(recent=100)

    timestamp = time.time()
    for i, (images, labels) in enumerate(test_loader):
        batch_size = images.size(0)

        if utils.is_model_cuda(model):
            images = images.cuda()
            labels = labels.cuda()

        # Record data time
        data_time_meter.update(time.time() - timestamp)

        # Forward pass without computing gradients.
        with torch.no_grad():
            outputs = model(images)
            loss_output = loss(outputs, labels)

        # Sometimes loss function returns a modified version of the output,
        # which must be used to compute the model accuracy.
        if isinstance(loss_output, tuple):
            loss_value, outputs = loss_output
        else:
            loss_value = loss_output

        # Record loss and model accuracy.
        loss_meter.update(loss_value.item(), batch_size)
        top1, top5 = utils.topk_accuracy(outputs, labels, recalls=(1, 5))
        top1_meter.update(top1, batch_size)
        top5_meter.update(top5, batch_size)

        # Record batch time
        batch_time_meter.update(time.time() - timestamp)
        timestamp = time.time()

        logging.info(
            'Epoch: [{epoch}][{batch}/{epoch_size}]\t'
            'Time {batch_time.value:.2f} ({batch_time.average:.2f})   '
            'Data {data_time.value:.2f} ({data_time.average:.2f})   '
            'Loss {loss.value:.3f} {{{loss.average:.3f}, {loss.average_recent:.3f}}}    '
            'Top-1 {top1.value:.2f} {{{top1.average:.2f}, {top1.average_recent:.2f}}}    '
            'Top-5 {top5.value:.2f} {{{top5.average:.2f}, {top5.average_recent:.2f}}}    '.format(
                epoch=epoch_number, batch=i + 1, epoch_size=len(test_loader),
                batch_time=batch_time_meter, data_time=data_time_meter,
                loss=loss_meter, top1=top1_meter, top5=top5_meter))
    # Log the overall test stats
    logging.info(
        'Epoch: [{epoch}] -- TESTING SUMMARY\t'
        'Time {batch_time.sum:.2f}   '
        'Data {data_time.sum:.2f}   '
        'Loss {loss.average:.3f}     '
        'Top-1 {top1.average:.2f}    '
        'Top-5 {top5.average:.2f}    '.format(
            epoch=epoch_number, batch_time=batch_time_meter, data_time=data_time_meter,
            loss=loss_meter, top1=top1_meter, top5=top5_meter))


def main(argv):
    """Run the test script with command line arguments @argv."""
    args = parse_args(argv)
    utils.general_setup(args.save, args.gpus)

    logging.info("Arguments parsed.\n{}".format(pprint.pformat(vars(args))))

    # Create the validation data loaders.
    val_loader = imagenet.get_val_loader(args.imagenet, args.batch_size,
                                         args.num_workers)
    # Create model and the loss.
    model, loss = model_factory.create_model(
        args.model, args.student_state_file, args.gpus)
    logging.info("Model:\n{}".format(model))

    # Test for one epoch.
    test_for_one_epoch(model, loss, val_loader, epoch_number=1)


if __name__ == '__main__':
    main(sys.argv[1:])

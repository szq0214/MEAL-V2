import collections
import logging
import os
import sys

import torch


def general_setup(checkpoints_dir=None, gpus=[]):
    if checkpoints_dir is not None:
        os.makedirs(checkpoints_dir, exist_ok=True)
    if len(gpus) > 0:
        torch.cuda.set_device(gpus[0])
    # Setup python's logging module.
    log_formatter = logging.Formatter(
        '%(levelname)s %(asctime)-20s:\t %(message)s')
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    # Add a console handler to write to stdout.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    # Add a file handler to write to log.txt.
    log_filepath = os.path.join(checkpoints_dir, 'log.txt')
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)


def is_model_cuda(model):
    # Check if the first parameter is on cuda.
    return next(model.parameters()).is_cuda


def topk_accuracy(outputs, labels, recalls=(1, 5)):
    """Return @recall accuracies for the given recalls."""

    _, num_classes = outputs.size()
    maxk = min(max(recalls), num_classes)

    _, pred = outputs.topk(maxk, dim=1, largest=True, sorted=True)
    correct = (pred == labels[:,None].expand_as(pred)).float()

    topk_accuracy = []
    for recall in recalls:
        topk_accuracy.append(100 * correct[:, :recall].sum(1).mean())
    return topk_accuracy


class AverageMeter:
    """Helper class to track the running average (and optionally the recent k
    items average of a sequence)."""

    def __init__(self, recent=None):
        self._recent = recent
        if recent is not None:
            self._q = collections.deque()
        self.reset()

    def reset(self):
        self.value = 0
        self.sum = 0
        self.count = 0
        if self._recent is not None:
            self.sum_recent = 0
            self.count_recent = 0
            self._q.clear()

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n

        if self._recent is not None:
            self.sum_recent += value * n
            self.count_recent += n
            self._q.append((n, value))
            while len(self._q) > self._recent:
                (n, value) = self._q.popleft()
                self.sum_recent -= value * n
                self.count_recent -= n

    @property
    def average(self):
        if self.count > 0:
            return self.sum / self.count
        else:
            return 0

    @property
    def average_recent(self):
        if self.count_recent > 0:
            return self.sum_recent / self.count_recent
        else:
            return 0

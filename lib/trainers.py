from __future__ import print_function, absolute_import
import time
from time import gmtime, strftime
from datetime import datetime
import gc
import os.path as osp
import sys
from PIL import Image
import numpy as np

import torch
from torchvision import transforms

from . import evaluation_metrics
from .evaluation_metrics import Accuracy, EditDistance
from .utils import to_numpy
from .utils.meters import AverageMeter
from .utils.serialization import load_checkpoint, save_checkpoint

metrics_factory = evaluation_metrics.factory()

from config import get_args
global_args = get_args(sys.argv[1:])

class BaseTrainer(object):
  def __init__(self, model, metric, logs_dir, iters=0, best_res=-1, grad_clip=-1, use_cuda=True, loss_weights={}):
    super(BaseTrainer, self).__init__()
    self.model = model
    self.metric = metric
    self.logs_dir = logs_dir
    self.iters = iters
    self.best_res = best_res
    self.grad_clip = grad_clip
    self.use_cuda = use_cuda
    self.loss_weights = loss_weights

    self.device = torch.device("cuda" if use_cuda else "cpu")

  def train(self, epoch, data_loader, optimizer, current_lr=0.0, 
            print_freq=100, train_tfLogger=None, is_debug=False,
            evaluator=None, test_loader=None, eval_tfLogger=None,
            test_dataset=None, test_freq=1000):

    self.model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()

    for i, inputs in enumerate(data_loader):
      self.model.train()
      self.iters += 1

      data_time.update(time.time() - end)

      input_dict = self._parse_data(inputs)
      output_dict = self._forward(input_dict)

      batch_size = input_dict['images'].size(0)

      total_loss = 0
      loss_dict = {}
      for k, loss in output_dict['losses'].items():
        loss = loss.mean(dim=0, keepdim=True)
        total_loss += self.loss_weights[k] * loss
        loss_dict[k] = loss.item()
        # print('{0}: {1}'.format(k, loss.item()))

      losses.update(total_loss.item(), batch_size)

      optimizer.zero_grad()
      total_loss.backward()
      if self.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
      optimizer.step()

      # # debug: check the parameters fixed or not.
      # print(self.model.parameters())
      # for tag, value in self.model.named_parameters():
      #   if tag == 'module.base.resnet.layer4.0.conv1.weight':
      #     print(value[:10,0,0,0])
      #   if tag == 'module.rec_head.decoder.attention_unit.sEmbed.weight':
      #     print(value[0, :10])

      batch_time.update(time.time() - end)
      end = time.time()

      if self.iters % print_freq == 0:
        print('[{}]\t'
              'Epoch: [{}][{}/{}]\t'
              'Time {:.3f} ({:.3f})\t'
              'Data {:.3f} ({:.3f})\t'
              'Loss {:.3f} ({:.3f})\t'
              # .format(strftime("%Y-%m-%d %H:%M:%S", gmtime()),
              .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                      epoch, i + 1, len(data_loader),
                      batch_time.val, batch_time.avg,
                      data_time.val, data_time.avg,
                      losses.val, losses.avg))

      #====== TensorBoard logging ======#
      if self.iters % print_freq*10 == 0:
        if train_tfLogger is not None:
          step = epoch * len(data_loader) + (i + 1)
          info = {
            'lr': current_lr,
            'loss': total_loss.item(), # this is total loss
          }
          ## add each loss
          for k, loss in loss_dict.items():
            info[k] = loss
          for tag, value in info.items():
            train_tfLogger.scalar_summary(tag, value, step)

          # if is_debug and (i + 1) % (print_freq*100) == 0: # this time-consuming and space-consuming
          #   # (2) Log values and gradients of the parameters (histogram)
          #   for tag, value in self.model.named_parameters():
          #     tag = tag.replace('.', '/')
          #     train_tfLogger.histo_summary(tag, to_numpy(value.data), step)
          #     train_tfLogger.histo_summary(tag+'/grad', to_numpy(value.grad.data), step)

          # # (3) Log the images
          # images, _, pids, _ = inputs
          # offsets = to_numpy(offsets)
          # info = {
          #   'images': to_numpy(images[:10])
          # }
          # for tag, images in info.items():
          #   train_tfLogger.image_summary(tag, images, step)

      #====== evaluation ======#
      if self.iters % test_freq == 0:
        # only symmetry branch
        if 'loss_rec' not in output_dict['losses']:
          is_best = True
          # self.best_res is alwarys equal to 1.0 
          self.best_res = evaluator.evaluate(test_loader, step=self.iters, tfLogger=eval_tfLogger, dataset=test_dataset)
        else:
          res = evaluator.evaluate(test_loader, step=self.iters, tfLogger=eval_tfLogger, dataset=test_dataset)

          if self.metric == 'accuracy':
            is_best = res > self.best_res
            self.best_res = max(res, self.best_res)
          elif self.metric == 'editdistance':
            is_best = res < self.best_res
            self.best_res = min(res, self.best_res)
          else:
            raise ValueError("Unsupported evaluation metric:", self.metric)

          print('\n * Finished iters {:3d}  accuracy: {:5.1%}  best: {:5.1%}{}\n'.
            format(self.iters, res, self.best_res, ' *' if is_best else ''))

        # if epoch < 1:
        #   continue
        save_checkpoint({
          'state_dict': self.model.module.state_dict(),
          'iters': self.iters,
          'best_res': self.best_res,
        }, is_best, fpath=osp.join(self.logs_dir, 'checkpoint.pth.tar'))


    # collect garbage (not work)
    # gc.collect()

  def _parse_data(self, inputs):
    raise NotImplementedError

  def _forward(self, inputs, targets):
    raise NotImplementedError


class Trainer(BaseTrainer):
  def _parse_data(self, inputs):
    input_dict = {}
    imgs, label_encs, lengths = inputs
    images = imgs.to(self.device)
    if label_encs is not None:
      labels = label_encs.to(self.device)

    input_dict['images'] = images
    input_dict['rec_targets'] = labels
    input_dict['rec_lengths'] = lengths
    return input_dict

  def _forward(self, input_dict):
    self.model.train()
    output_dict = self.model(input_dict)
    return output_dict
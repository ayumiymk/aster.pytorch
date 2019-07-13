from __future__ import print_function, absolute_import
import time
from time import gmtime, strftime
from datetime import datetime
from collections import OrderedDict

import torch

import numpy as np
from random import randint
from PIL import Image
import sys

from . import evaluation_metrics
from .evaluation_metrics import Accuracy, EditDistance, RecPostProcess
from .utils.meters import AverageMeter
from .utils.visualization_utils import recognition_vis, stn_vis

metrics_factory = evaluation_metrics.factory()

from config import get_args
global_args = get_args(sys.argv[1:])

class BaseEvaluator(object):
  def __init__(self, model, metric, use_cuda=True):
    super(BaseEvaluator, self).__init__()
    self.model = model
    self.metric = metric
    self.use_cuda = use_cuda
    self.device = torch.device("cuda" if use_cuda else "cpu")

  def evaluate(self, data_loader, step=1, print_freq=1, tfLogger=None, dataset=None, vis_dir=None):
    self.model.eval()

    batch_time = AverageMeter()
    data_time  = AverageMeter()

    # forward the network
    images, outputs, targets, losses = [], {}, [], []
    file_names = []

    end = time.time()
    for i, inputs in enumerate(data_loader):
      data_time.update(time.time() - end)

      input_dict = self._parse_data(inputs)
      output_dict = self._forward(input_dict)

      batch_size = input_dict['images'].size(0)

      total_loss_batch = 0.
      for k, loss in output_dict['losses'].items():
        loss = loss.mean(dim=0, keepdim=True)
        total_loss_batch += loss.item() * batch_size

      images.append(input_dict['images'])
      targets.append(input_dict['rec_targets'])
      losses.append(total_loss_batch)
      if global_args.evaluate_with_lexicon:
        file_names += input_dict['file_name']
      for k, v in output_dict['output'].items():
        if k not in outputs:
          outputs[k] = []
        outputs[k].append(v.cpu())

      batch_time.update(time.time() - end)
      end = time.time()

      if (i + 1) % print_freq == 0:
        print('[{}]\t'
              'Evaluation: [{}/{}]\t'
              'Time {:.3f} ({:.3f})\t'
              'Data {:.3f} ({:.3f})\t'
              # .format(strftime("%Y-%m-%d %H:%M:%S", gmtime()),
              .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                      i + 1, len(data_loader),
                      batch_time.val, batch_time.avg,
                      data_time.val, data_time.avg))

    if not global_args.keep_ratio:
      images = torch.cat(images)
      num_samples = images.size(0)
    else:
      num_samples = sum([subimages.size(0) for subimages in images])
    targets = torch.cat(targets)
    losses = np.sum(losses) / (1.0 * num_samples)
    for k, v in outputs.items():
      outputs[k] = torch.cat(outputs[k])

    # save info for recognition
    if 'pred_rec' in outputs:
      # evaluation with metric
      if global_args.evaluate_with_lexicon:
        eval_res = metrics_factory[self.metric+'_with_lexicon'](outputs['pred_rec'], targets, dataset, file_names)
        print('lexicon0: {0}, {1:.3f}'.format(self.metric, eval_res[0]))
        print('lexicon50: {0}, {1:.3f}'.format(self.metric, eval_res[1]))
        print('lexicon1k: {0}, {1:.3f}'.format(self.metric, eval_res[2]))
        print('lexiconfull: {0}, {1:.3f}'.format(self.metric, eval_res[3]))
        eval_res = eval_res[0]
      else:
        eval_res = metrics_factory[self.metric](outputs['pred_rec'], targets, dataset)
        print('lexicon0: {0}: {1:.3f}'.format(self.metric, eval_res))
      pred_list, targ_list, score_list = RecPostProcess(outputs['pred_rec'], targets, outputs['pred_rec_score'], dataset)

      if tfLogger is not None:
        # (1) Log the scalar values
        info = {
          'loss': losses,
          self.metric: eval_res,
        }
        for tag, value in info.items():
          tfLogger.scalar_summary(tag, value, step)

    #====== Visualization ======#
    if vis_dir is not None:
      # recognition_vis(images, outputs['pred_rec'], targets, score_list, dataset, vis_dir)
      stn_vis(images, outputs['rectified_images'], outputs['ctrl_points'], outputs['pred_rec'],
              targets, score_list, outputs['pred_score'] if 'pred_score' in outputs else None, dataset, vis_dir)
    return eval_res


  def _parse_data(self, inputs):
    raise NotImplementedError

  def _forward(self, inputs):
    raise NotImplementedError
    

class Evaluator(BaseEvaluator):
  def _parse_data(self, inputs):
    input_dict = {}
    if global_args.evaluate_with_lexicon:
      imgs, label_encs, lengths, file_name = inputs
    else:
      imgs, label_encs, lengths = inputs

    with torch.no_grad():
      images = imgs.to(self.device)
      if label_encs is not None:
        labels = label_encs.to(self.device)

    input_dict['images'] = images
    input_dict['rec_targets'] = labels
    input_dict['rec_lengths'] = lengths
    if global_args.evaluate_with_lexicon:
      input_dict['file_name'] = file_name
    return input_dict

  def _forward(self, input_dict):
    self.model.eval()
    with torch.no_grad():
      output_dict = self.model(input_dict)
    return output_dict
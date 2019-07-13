from __future__ import print_function, absolute_import
import json
import os
import sys
# import moxing as mox
import os.path as osp
import shutil

import torch
from torch.nn import Parameter

from .osutils import mkdir_if_missing

from config import get_args
global_args = get_args(sys.argv[1:])

if global_args.run_on_remote:
  import moxing as mox


def read_json(fpath):
  with open(fpath, 'r') as f:
    obj = json.load(f)
  return obj


def write_json(obj, fpath):
  mkdir_if_missing(osp.dirname(fpath))
  with open(fpath, 'w') as f:
    json.dump(obj, f, indent=4, separators=(',', ': '))


def save_checkpoint(state, is_best, fpath='checkpoint.pth.tar'):
  print('=> saving checkpoint ', fpath)
  if global_args.run_on_remote:
    dir_name = osp.dirname(fpath)
    if not mox.file.exists(dir_name):
      mox.file.make_dirs(dir_name)
      print('=> makding dir ', dir_name)
    local_path = "local_checkpoint.pth.tar"
    torch.save(state, local_path)
    mox.file.copy(local_path, fpath)
    if is_best:
      mox.file.copy(local_path, osp.join(dir_name, 'model_best.pth.tar'))
  else:
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
      shutil.copy(fpath, osp.join(osp.dirname(fpath), 'model_best.pth.tar'))


def load_checkpoint(fpath):
  if global_args.run_on_remote:
    mox.file.shift('os', 'mox')
    checkpoint = torch.load(fpath)
    print("=> Loaded checkpoint '{}'".format(fpath))
    return checkpoint
  else:
    load_path = fpath

    if osp.isfile(load_path):
      checkpoint = torch.load(load_path)
      print("=> Loaded checkpoint '{}'".format(load_path))
      return checkpoint
    else:
      raise ValueError("=> No checkpoint found at '{}'".format(load_path))


def copy_state_dict(state_dict, model, strip=None):
  tgt_state = model.state_dict()
  copied_names = set()
  for name, param in state_dict.items():
    if strip is not None and name.startswith(strip):
      name = name[len(strip):]
    if name not in tgt_state:
      continue
    if isinstance(param, Parameter):
      param = param.data
    if param.size() != tgt_state[name].size():
      print('mismatch:', name, param.size(), tgt_state[name].size())
      continue
    tgt_state[name].copy_(param)
    copied_names.add(name)

  missing = set(tgt_state.keys()) - copied_names
  if len(missing) > 0:
      print("missing keys in state_dict:", missing)

  return model
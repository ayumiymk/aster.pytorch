from __future__ import absolute_import
import os
import errno


def mkdir_if_missing(dir_path):
  try:
    os.makedirs(dir_path)
  except OSError as e:
    if e.errno != errno.EEXIST:
      raise


def make_symlink_if_not_exists(real_path, link_path):
  '''
  param real_path: str the path linked
  param link_path: str the path with only the symbol
  '''
  try:
    os.makedirs(real_path)
  except OSError as e:
    if e.errno != errno.EEXIST:
      raise

  cmd = 'ln -s {0} {1}'.format(real_path, link_path)
  os.system(cmd)
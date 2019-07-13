from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append('./')

import six
import os
import os.path as osp
import math
import argparse


parser = argparse.ArgumentParser(description="Softmax loss classification")
# data
parser.add_argument('--synthetic_train_data_dir', nargs='+', type=str, metavar='PATH',
                    default=['/share/zhui/reg_dataset/NIPS2014'])
parser.add_argument('--real_train_data_dir', type=str, metavar='PATH',
                    default='/data/zhui/benchmark/cocotext_trainval')
parser.add_argument('--extra_train_data_dir', nargs='+', type=str, metavar='PATH',
                    default=['/share/zhui/reg_dataset/CVPR2016'])
parser.add_argument('--test_data_dir', type=str, metavar='PATH',
                    default='/share/zhui/reg_dataset/IIIT5K_3000')
parser.add_argument('--MULTI_TRAINDATA', action='store_true', default=False,
                    help='whether use the extra_train_data for training.')
parser.add_argument('-b', '--batch_size', type=int, default=128)
parser.add_argument('-j', '--workers', type=int, default=8)
parser.add_argument('--height', type=int, default=64,
                    help="input height, default: 256 for resnet*, ""64 for inception")
parser.add_argument('--width', type=int, default=256,
                    help="input width, default: 128 for resnet*, ""256 for inception")
parser.add_argument('--keep_ratio', action='store_true', default=False,
                    help='length fixed or lenghth variable.')
parser.add_argument('--voc_type', type=str, default='ALLCASES_SYMBOLS',
                    choices=['LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS'])
parser.add_argument('--mix_data', action='store_true',
                    help="whether combine multi datasets in the training stage.")
parser.add_argument('--num_train', type=int, default=math.inf)
parser.add_argument('--num_test', type=int, default=math.inf)
parser.add_argument('--aug', action='store_true', default=False,
                    help='whether use data augmentation.')
parser.add_argument('--lexicon_type', type=str, default='0', choices=['0', '50', '1k', 'full'],
                    help='which lexicon associated to image is used.')
parser.add_argument('--image_path', type=str, default='',
                    help='the path of single image, used in demo.py.')
parser.add_argument('--tps_inputsize', nargs='+', type=int, default=[32, 64])
parser.add_argument('--tps_outputsize', nargs='+', type=int, default=[32, 100])
# model
parser.add_argument('-a', '--arch', type=str, default='ResNet_ASTER')
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--max_len', type=int, default=100)
parser.add_argument('--n_group', type=int, default=1)
parser.add_argument('--STN_ON', action='store_true',
                    help='add the stn head.')
parser.add_argument('--tps_margins', nargs='+', type=float, default=[0.05,0.05])
parser.add_argument('--stn_activation', type=str, default='none')
parser.add_argument('--num_control_points', type=int, default=20)
parser.add_argument('--stn_with_dropout', action='store_true', default=False)
## lstm
parser.add_argument('--with_lstm', action='store_true', default=False,
                    help='whether append lstm after cnn in the encoder part.')
parser.add_argument('--decoder_sdim', type=int, default=512,
                    help="the dim of hidden layer in decoder.")
parser.add_argument('--attDim', type=int, default=512,
                    help="the dim for attention.")
# optimizer
parser.add_argument('--lr', type=float, default=1,
                    help="learning rate of new parameters, for pretrained "
                         "parameters it is 10 times smaller than this")
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=0.0) # the model maybe under-fitting, 0.0 gives much better results.
parser.add_argument('--grad_clip', type=float, default=1.0)
parser.add_argument('--loss_weights', nargs='+', type=float, default=[1,1,1])
# training configs
parser.add_argument('--resume', type=str, default='', metavar='PATH')
parser.add_argument('--evaluate', action='store_true',
                    help="evaluation only")
parser.add_argument('--epochs', type=int, default=6)
parser.add_argument('--start_save', type=int, default=0,
                    help="start saving checkpoints after specific epoch")
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=100)
parser.add_argument('--cuda', default=True, type=bool,
                    help='whether use cuda support.')
# testing configs
parser.add_argument('--evaluation_metric', type=str, default='accuracy')
parser.add_argument('--evaluate_with_lexicon', action='store_true', default=False)
parser.add_argument('--beam_width', type=int, default=5)
# misc
working_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
parser.add_argument('--logs_dir', type=str, metavar='PATH',
                    default=osp.join(working_dir, 'logs'))
parser.add_argument('--real_logs_dir', type=str, metavar='PATH',
                    default='/media/mkyang/research/recognition/selfattention_rec')
parser.add_argument('--debug', action='store_true',
                    help="if debugging, some steps will be passed.")
parser.add_argument('--vis_dir', type=str, metavar='PATH', default='',
                    help="whether visualize the results while evaluation.")
parser.add_argument('--run_on_remote', action='store_true', default=False,
                    help="run the code on remote or local.")

def get_args(sys_args):
  global_args = parser.parse_args(sys_args)
  return global_args
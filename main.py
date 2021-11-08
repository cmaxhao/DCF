import argparse
import numpy as np
from time import time
from data_loader import load_data
from train import train
import logging
import os
import logging.config
from itemKNN import ItemBasedCF
np.random.seed(555)

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, default='fgim2', help='which model to evaluate')
parser.add_argument('--routing_times', type=int, default=4, help='routing times')
parser.add_argument('--cap_k', type=int, default=4, help='capsule num')
'''
# movie
parser.add_argument('--dataset', type=str, default='movie', help='which dataset to use')
parser.add_argument('--dim', type=int, default=32, help='dimension of user and entity embeddings')
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use')
parser.add_argument('--n_epochs', type=int, default=20, help='the number of epochs')
parser.add_argument('--neighbor_sample_size', type=int, default=16, help='the number of neighbors to be sampled')
parser.add_argument('--neighbor_item', type=int, default=16, help='the number of neighbors to be sampled')
parser.add_argument('--n_iter', type=int, default=1, help='number of iterations when computing entity representation')
parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
parser.add_argument('--ui_iter', type=int, default=1, help='number of iterations when computing entity representation')
parser.add_argument('--l2_weight', type=float, default=1e-6, help='weight of l2 regularization')
parser.add_argument('--ls_weight', type=float, default=0, help='weight of LS regularization')
parser.add_argument('--ratio', type=float, default=1, help='size of training dataset')
'''

# book
parser.add_argument('--dataset', type=str, default='book', help='which dataset to use')
parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use')
parser.add_argument('--n_epochs', type=int, default=10, help='the number of epochs')
parser.add_argument('--neighbor_sample_size', type=int, default=10, help='the number of neighbors to be sampled')
parser.add_argument('--neighbor_item', type=int, default=20, help='the number of neighbors to be sampled')
parser.add_argument('--dim', type=int, default=64, help='dimension of user and entity embeddings')
parser.add_argument('--n_iter', type=int, default=1, help='number of iterations when computing entity representation')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--ui_iter', type=int, default=1, help='number of iterations when computing entity representation')
parser.add_argument('--l2_weight', type=float, default=2e-5, help='weight of l2 regularization')
parser.add_argument('--ls_weight', type=float, default=0, help='weight of LS regularization')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--ratio', type=float, default=1, help='size of training dataset')

'''
# music
parser.add_argument('--dataset', type=str, default='music', help='which dataset to use')
parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use')
parser.add_argument('--n_epochs', type=int, default=30, help='the number of epochs')
parser.add_argument('--neighbor_sample_size', type=int, default=10, help='the number of neighbors to be sampled')
parser.add_argument('--neighbor_item', type=int, default=30, help='the number of neighbors to be sampled')
parser.add_argument('--dim', type=int, default=64, help='dimension of user and entity embeddings')
parser.add_argument('--n_iter', type=int, default=1, help='number of iterations when computing entity representation')
parser.add_argument('--ui_iter', type=int, default=1, help='number of iterations when computing user or item representation')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--l2_weight', type=float, default=5e-4, help='weight of l2 regularization')
parser.add_argument('--ls_weight', type=float, default=5e-4, help='weight of LS regularization')
parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')
parser.add_argument('--ratio', type=float, default=1.0, help='size of training dataset')
'''
show_loss = True
show_time = True
show_topk = True

t = time()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
args = parser.parse_args()
data = load_data(args)
logging.config.fileConfig('logging.conf')
logger = logging.getLogger()
logger.info('model_type=%s, dataset=%s, cap_k=%d, routing_times=%d' % (
args.model_type, args.dataset, args.cap_k, args.routing_times))
logger.info(
    'dim=%d, lr=%.4f, aggregator=%s,  n_epochs=%d, neighbor_sample_size=%s, neighbor_item=%d, n_iter=%d, ui_iter=%d, batch_size=%d, l2_weight=%.4f, ls_weight=%.4f, ratio=%.2f'
    % (args.dim, args.lr, args.aggregator, args.n_epochs, args.neighbor_sample_size, args.neighbor_item, args.n_iter,
       args.ui_iter, args.batch_size, args.l2_weight, args.ls_weight, args.ratio))

train(args, logger,data, show_loss, show_topk)

if show_time:
    print('time used: %d s' % (time() - t))

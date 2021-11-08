import tensorflow as tf
import numpy as np
import math
from DICF import DICF
from ODCF import ODCF
from FGIM import FGIM
from FGIM1 import FGIM1
from FGIM2 import FGIM2
from CKE import CKE
import scipy.sparse as sp
import pandas as pd

def train(args, logger, data, show_loss, show_topk):
    n_user, n_item, n_entity, n_relation = data[0], data[1], data[2], data[3]
    train_data, eval_data, test_data = data[4], data[5], data[6]
    adj_entity, adj_relation = data[7], data[8]

    # top-K evaluation settings
    user_list, train_record, test_record, item_set, k_list = topk_settings(show_topk, train_data, test_data, n_item)
    user_history, item_history = get_user_item_dict(train_data)
    adj_user = load_adj(args, user_history,n_user)
    adj_item = load_adj(args, item_history, n_item)
    user_adj_mat = get_adj_mat('../data/'+args.dataset,n_user, n_item, user_history)
    if args.model_type=='fgim':
        model = FGIM(args, n_user, n_item, n_entity, n_relation, adj_entity, adj_relation, adj_user, adj_item)
    elif args.model_type=='fgim1':
        model = FGIM1(args, n_user, n_item, n_entity, n_relation, adj_entity, adj_relation, adj_user, adj_item)
    elif args.model_type=='fgim2':
        model = FGIM2(args, n_user, n_entity, n_relation, adj_entity, adj_relation, adj_user, adj_item)
    elif args.model_type=='odcf':
        model = ODCF(args, n_user, n_entity, n_relation, adj_entity, adj_relation, adj_user, adj_item)
    elif args.model_type=='dicf':
        model = DICF(args, n_user, n_entity, n_relation, adj_entity, adj_relation, adj_user, adj_item)
    elif args.model_type=="cke":
        model = CKE(args, n_user, n_item,n_entity, n_relation, adj_entity, adj_relation)
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    with tf.Session(config=gpu_config) as sess:
        sess.run(tf.global_variables_initializer())
        #sne_eval(sess, model, train_record, args.batch_size, 0)
        for step in range(args.n_epochs):
            all_loss = 0
            # training
            np.random.shuffle(train_data)
            start = 0
            # skip the last incomplete minibatch if its size < batch size
            while start + args.batch_size <= train_data.shape[0]:
                _, loss= model.train(sess, get_feed_dict(model, train_data, start, start + args.batch_size))
                start += args.batch_size
                all_loss += loss
            # CTR evaluation
            train_auc, train_f1 = ctr_eval(sess, model, train_data, args.batch_size)
            eval_auc, eval_f1 = ctr_eval(sess, model, eval_data, args.batch_size)
            test_auc, test_f1 = ctr_eval(sess, model, test_data, args.batch_size)
            logger.info('epoch %d    train auc: %.4f  f1: %.4f    eval auc: %.4f  f1: %.4f    test auc: %.4f  f1: %.4f'
                  % (step, train_auc, train_f1, eval_auc, eval_f1, test_auc, test_f1))
            if show_loss:
                print(all_loss)
            # top-K evaluation
            if show_topk and (step+1)%10==0:
                sne_eval(sess, model, train_record, args.batch_size, step)
                precision, recall, ndcg = topk_eval(
                    sess, model, user_list, train_record, test_record, item_set, k_list, args.batch_size)
                logger.info('precision: %s' %(' '.join(str(i) for i in precision)))
                logger.info('recall: %s' % (' '.join(str(i) for i in recall)))
                logger.info('ndcg: %s' % (' '.join(str(i) for i in ndcg)))


def topk_settings(show_topk, train_data, test_data, n_item):
    if show_topk:
        user_num = 1000
        k_list = [1, 2, 5, 10, 20]
        train_record = get_user_record(train_data, True)
        test_record = get_user_record(test_data, False)
        user_list = list(set(train_record.keys()) & set(test_record.keys()))
        if len(user_list) > user_num:
            user_list = np.random.choice(user_list, size=user_num, replace=False)
        item_set = set(list(range(n_item)))
        return user_list, train_record, test_record, item_set, k_list
    else:
        return [None] * 5


def get_feed_dict(model, data, start, end):
    feed_dict = {model.user_indices: data[start:end, 0],
                 model.item_indices: data[start:end, 1],
                 model.labels: data[start:end, 2]}
    return feed_dict


def ctr_eval(sess, model, data, batch_size):
    start = 0
    auc_list = []
    f1_list = []
    while start + batch_size <= data.shape[0]:
        auc, f1 = model.eval(sess, get_feed_dict(model, data, start, start + batch_size))
        auc_list.append(auc)
        f1_list.append(f1)
        start += batch_size
    return float(np.mean(auc_list)), float(np.mean(f1_list))
#Book-Crossing
def sne_eval(sess, model,train_record, batch_size, step):
    id = 9668
    train_item_list = list(train_record[id])
    neigh, p, item_emb, arg = model.get_sne(sess, {model.user_indices: [id] * batch_size,
                                                model.item_indices: train_item_list + [train_item_list[-1]] * (
                               batch_size - len(train_item_list))})
    emb = []
    for i in range(len(neigh[0])):
        inde = train_item_list.index(neigh[0,i])
        emb.append(item_emb[inde])
    temp = np.concatenate((np.array(emb), np.reshape(np.array(arg[0,]),(-1,1))), axis=1)
    df = pd.DataFrame(temp)
    print(p[0, :, :])
    df.to_csv(str(step) + "middle_data_" + str(id) +".csv", sep=',', header=True, index=True)
def topk_eval(sess, model, user_list, train_record, test_record, item_set, k_list, batch_size):
    precision_list = {k: [] for k in k_list}
    recall_list = {k: [] for k in k_list}
    ndcg_list = {k: [] for k in k_list}

    for user in user_list:
        test_item_list = list(item_set - train_record[user])
        item_score_map = dict()
        start = 0
        while start + batch_size <= len(test_item_list):
            items, scores = model.get_scores(sess, {model.user_indices: [user] * batch_size,
                                                    model.item_indices: test_item_list[start:start + batch_size]})
            for item, score in zip(items, scores):
                item_score_map[item] = score
            start += batch_size

        # padding the last incomplete minibatch if exists
        if start < len(test_item_list):
            items, scores = model.get_scores(
                sess, {model.user_indices: [user] * batch_size,
                       model.item_indices: test_item_list[start:] + [test_item_list[-1]] * (
                               batch_size - len(test_item_list) + start)})
            for item, score in zip(items, scores):
                item_score_map[item] = score

        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]

        for k in k_list:
            hit_num = len(set(item_sorted[:k]) & test_record[user])
            # if k ==1 and hit_num==1:
            #     print(user)
            precision_list[k].append(hit_num / k)
            recall_list[k].append(hit_num / len(test_record[user]))
            ndcg_list[k].append(getNDCG(item_sorted[:k], test_record[user]))

    precision = [round(np.mean(precision_list[k]), 4) for k in k_list]
    recall = [round(np.mean(recall_list[k]), 4) for k in k_list]
    ndcg = [round(np.mean(ndcg_list[k]), 4) for k in k_list]

    return precision, recall, ndcg

def getNDCG(rank_list, target_items):
    dcg = 0
    idcg = IDCG(len(target_items))
    for i in range(len(rank_list)):
        item_id = rank_list[i]
        if (item_id not in target_items):
            continue
        rank = i + 1
        dcg += 1 / math.log(rank + 1, 2)
    return dcg / idcg

def IDCG(n):
    idcg = 0
    for i in range(n):
        idcg += 1 / math.log(i + 2, 2)
    return idcg

def get_user_record(data, is_train):
    user_history_dict = dict()
    for interaction in data:
        user = interaction[0]
        item = interaction[1]
        label = interaction[2]
        if is_train or label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict

def get_user_item_dict(data):
    user_history_dict = dict()
    item_history_dict = dict()
    for interaction in data:
        user = interaction[0]
        item = interaction[1]
        label = interaction[2]
        if label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
            if item not in item_history_dict:
                item_history_dict[item] = set()
            item_history_dict[item].add(user)
    return user_history_dict, item_history_dict

def load_adj(args, history_dict, num):
    print('constructing adjacency matrix ...')
    # each line of adj_user stores the sampled neighbor item for a given user
    adj = np.zeros([num, args.neighbor_item], dtype=np.int64)
    for user in range(num):
        if user in history_dict:
            neighbors = list(history_dict[user])
            n_neighbors = len(neighbors)
            if n_neighbors >= args.neighbor_item:
                sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.neighbor_item, replace=False)
            else:
                sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.neighbor_item, replace=True)
            adj[user] = np.array([neighbors[i] for i in sampled_indices])
    return adj

def get_adj_mat(path, user, item, user_adj):
    try:
        adj_mat = sp.load_npz(path + '/s_adj_mat.npz')
        norm_adj_mat = sp.load_npz(path + '/s_norm_adj_mat.npz')
        mean_adj_mat = sp.load_npz(path + '/s_mean_adj_mat.npz')
        print('already load adj matrix', adj_mat.shape)

    except Exception:
        adj_mat, norm_adj_mat, mean_adj_mat = create_adj_mat(user, item, user_adj)
        sp.save_npz(path + '/s_adj_mat.npz', adj_mat)
        sp.save_npz(path + '/s_norm_adj_mat.npz', norm_adj_mat)
        sp.save_npz(path + '/s_mean_adj_mat.npz', mean_adj_mat)
    return norm_adj_mat

def create_adj_mat(user, item, user_adj):
    adj_mat = sp.dok_matrix((user + item, user + item), dtype=np.float32)
    adj_mat = adj_mat.tolil()
    R = sp.dok_matrix((user, item), dtype=np.float32)
    for key in user_adj:
        for value in user_adj[key]:
            R[key,value]=1.
    R = R.tolil()

    adj_mat[:user, user:] = R
    adj_mat[user:, :user] = R.T
    adj_mat = adj_mat.todok()
    print('already create adjacency matrix', adj_mat.shape)

    def normalized_adj_single(adj):
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        # norm_adj = adj.dot(d_mat_inv)
        print('generate single-normalized adjacency matrix.')
        return norm_adj.tocoo()
    norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
    mean_adj_mat = normalized_adj_single(adj_mat)

    print('already normalize adjacency matrix')
    return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

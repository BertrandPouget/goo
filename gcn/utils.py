import numpy as np
import pandas as pd
import scipy.sparse as sp
import tensorflow as tf
import scipy.io as sio

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj)
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def OO_construct_feed_dict(features, support, labels, labels_mask, placeholders, alpha, beta, f_u):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    feed_dict.update({placeholders['alpha']: alpha})
    feed_dict.update({placeholders['beta']: beta})
    feed_dict.update({placeholders['f_u']: f_u})
    return feed_dict


def split_CV(mask, y, n_fold = 5, seed = 0):
    n_p = len(mask[mask[0] & (y[0] == 1)])
    n_u = len(mask[mask[0] & (y[1] == 1)])
    n = n_p + n_u

    n_val = np.array([(n // n_fold) + 1] * (n % n_fold) + [n // n_fold] * (n_fold - (n % n_fold)))
    n_p_val = np.array([(n_p // n_fold) + 1] * (n_p % n_fold) + [n_p // n_fold] * (n_fold - (n_p % n_fold)))
    n_u_val = n_val - n_p_val

    p_ids = np.where(mask[0] & (y[0] == 1))[0]
    u_ids = np.where(mask[0] & (y[1] == 1))[0]
    ids = np.concatenate((p_ids, u_ids))

    val_mask_CV = pd.DataFrame()
    train_mask_CV = pd.DataFrame()

    np.random.seed(seed)
    for i in range(n_fold):
        p_val_ids = np.random.choice(p_ids, n_p_val[i], replace=False)
        u_val_ids = np.random.choice(u_ids, n_u_val[i], replace=False)

        val_ids = np.sort(np.concatenate((p_val_ids, u_val_ids)))
        val_mask_n = pd.DataFrame([False] * len(mask))
        val_mask_n.iloc[val_ids] = True
        val_mask_CV = pd.concat([val_mask_CV, val_mask_n], axis=1)

        train_ids = np.setdiff1d(ids, val_ids)
        train_mask_n = pd.DataFrame([False] * len(mask))
        train_mask_n.iloc[train_ids] = True
        train_mask_CV = pd.concat([train_mask_CV, train_mask_n], axis=1)

        p_ids = np.setdiff1d(p_ids, p_val_ids)
        u_ids = np.setdiff1d(u_ids, u_val_ids)

    return train_mask_CV, val_mask_CV


def load_data(folder):
    a = sio.mmread(f'{folder}/a.mtx')
    a = a.tocsr().astype(np.intc)
    e = pd.read_table(f'{folder}/e.txt', sep='\t', header=None)
    y = pd.read_table(f'{folder}/y.txt', sep='\t', header=None)
    train_mask = pd.read_table(f'{folder}/train_mask.txt', sep='\t', header=None)
    test_mask = pd.read_table(f'{folder}/test_mask.txt', sep='\t', header=None)

    return a, e, y, train_mask, test_mask

def load_data_x(folder):
    a = sio.mmread(f'{folder}/a.mtx')
    a = a.tocsr().astype(np.intc)
    x = sio.mmread(f'{folder}/x.mtx')
    y = pd.read_table(f'{folder}/y.txt', sep='\t', header=None)
    train_mask = pd.read_table(f'{folder}/train_mask.txt', sep='\t', header=None)
    test_mask = pd.read_table(f'{folder}/test_mask.txt', sep='\t', header=None)

    return a, x, y, train_mask, test_mask


def compute_f_u(e, y, train_mask, kappa):
    non_train_mask = ~train_mask.values.flatten()[0:e.shape[1]]
    M = e.max().max()
    eM = e.copy()
    eM.loc[:, non_train_mask] = M
    neigh = eM.apply(lambda x: x.argsort(), axis=1).loc[:,0:(kappa-1)]
    f_u = pd.DataFrame(neigh.applymap(lambda val: np.argmax(y[val, :])).sum(axis=1) / kappa)
    return f_u
    

def compute_error(cm):
    err = 2*cm[0,1] + 2.5*cm[1,0] + cm[0, 2] + cm[1, 2]
    n_p = cm[0, 0] + cm[0, 1] + cm[0, 2]
    n_u = cm[1, 0] + cm[1, 1] + cm[1, 2]

    dum0 = 2.5*n_u
    dum1 = 2*n_p
    dum2 = n_u + n_p
    err_dum = min(dum0, dum1, dum2)

    err = err/err_dum
    return np.round(err * 100, decimals=2)


def OO_softmax(B, alpha, beta, f_u):

    B = tf.stack([
        alpha * B[:, 0] + (1 - alpha) * (tf.ones_like(f_u[:, 0]) - f_u[:, 0]),
        beta * B[:, 1] + (1 - beta) * f_u[:, 0]],
    axis=1)
    
    return tf.nn.softmax(B)


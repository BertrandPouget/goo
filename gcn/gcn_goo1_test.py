from __future__ import division
from __future__ import print_function

import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
import numpy as np

from utils import *
from models import *

#Suppress warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 8, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 5, 'Tolerance for early stopping (# of epochs).')

# Load data
a, e, y, train_mask, test_mask = load_data('../data')
# Create masks for cross-validation
n_fold = 5
train_masks_CV, val_masks_CV = split_CV(train_mask, y, n_fold = n_fold, seed = 0)

# Some preprocessing
train_mask = train_mask.values.flatten()
test_mask = test_mask.values.flatten()
y = y.values.astype(np.float32)

# Define model parameters
support = [preprocess_adj(a)]
num_supports = 1
model_func = GCN

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=(None, None)),
    'labels': tf.placeholder(tf.float32, shape=(None, y.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
}

# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss], feed_dict=feed_dict_val)
    return outs_val[0], (time.time() - t_test)

def evaluate_with_predictions(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.predict()], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)

kappa = 5

theta_1 = 0.6
theta_2 = 0.6

# Testing
f_u = compute_f_u(e, y, train_masks_CV.iloc[:, [0]], kappa)
x = OneHotEncoder().fit_transform(f_u)
features = preprocess_features(x)
model = model_func(placeholders, input_dim=features[2][1], logging=True)

cost_val = []
sess = tf.Session()
sess.run(tf.global_variables_initializer())

train_mask_0 = np.array(train_masks_CV.iloc[:, 0].values.flatten())
val_mask_0 = np.array(val_masks_CV.iloc[:, 0].values.flatten())

for epoch in range(FLAGS.epochs):

    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, y, train_mask_0, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss], feed_dict=feed_dict)

    # Validation
    cost, duration = evaluate(features, support, y, val_mask_0, placeholders)
    cost_val.append(cost)

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        break

_, test_Z, _ = evaluate_with_predictions(features, support, y, test_mask, placeholders)

y_hat = np.zeros(test_Z.shape[0])
y_hat[test_Z[:, 0] > theta_2] = 0
y_hat[test_Z[:, 0] <= theta_1] = 1
y_hat[np.logical_and(test_Z[:, 0] > theta_1, test_Z[:, 0] <= theta_2)] = 2

cm = confusion_matrix(y[test_mask].argmax(1), y_hat[test_mask], labels=[0, 1, 2])
cm = np.delete(cm, 2, axis=0)
print('Confusion Matrix:\n', cm)
print('Error:', compute_error(cm))

np.savetxt('../results.txt', y_hat, delimiter='\t')
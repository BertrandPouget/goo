from __future__ import division
from __future__ import print_function

import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from sklearn.metrics import confusion_matrix
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
a, e, y, train_mask, test_mask = load_data('../data/python')

# Create masks for cross-validation
n_fold = 5
train_masks_CV, val_masks_CV = split_CV(train_mask, y, n_fold = n_fold, seed = 0)

# Some preprocessing
train_mask = train_mask.values.flatten()
test_mask = test_mask.values.flatten()
y = y.values.astype(np.float32)

features = sp.lil_matrix(np.identity(a.shape[0], dtype=np.float32))
features = preprocess_features(features)

# Define model parameters
support = [preprocess_adj(a)]
num_supports = 1
model_func = OO_GCN

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
    'alpha': tf.placeholder(tf.float32, shape=()),
    'beta': tf.placeholder(tf.float32, shape=()),
    'f_u': tf.placeholder(tf.float32, shape=(e.shape[0], 1))
}

# Create model
model = model_func(placeholders, input_dim=features[2][1], logging=True)

# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders, alpha, beta, f_u):
    t_test = time.time()
    feed_dict_val = OO_construct_feed_dict(features, support, labels, mask, placeholders, alpha, beta, f_u)
    outs_val = sess.run([model.loss], feed_dict=feed_dict_val)
    return outs_val[0], (time.time() - t_test)

def evaluate_with_predictions(features, support, labels, mask, placeholders, alpha, beta, f_u):
    t_test = time.time()
    feed_dict_val = OO_construct_feed_dict(features, support, labels, mask, placeholders, alpha, beta, f_u)
    outs_val = sess.run([model.loss, model.predict()], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)

# Define hyperparameters
kappas = [3, 5, 7]
alphas = [0, 0.25, 0.5, 0.75, 1]
betas = [0, 0.25, 0.5, 0.75, 1]

start_theta = 0
stop_theta = 1
step_theta = 0.2
thetas_combos = [(theta_1, theta_2) for theta_1 in np.arange(start_theta, stop_theta + step_theta, step_theta)
                                  for theta_2 in np.arange(theta_1, stop_theta + step_theta, step_theta)]
thetas_combos = np.round(thetas_combos, 1)

f_u = compute_f_u(e, y, train_masks_CV.iloc[:, [0]], 3)

err_dict = {}
for i in range(n_fold):
    print('CV fold:', i+1)
    train_mask_CV = np.array(train_masks_CV.iloc[:, i].values.flatten())
    val_mask_CV = np.array(val_masks_CV.iloc[:, i].values.flatten())

    for kappa in kappas:
        f_u = compute_f_u(e, y, train_masks_CV.iloc[:, [i]], kappa)
        
        for alpha in alphas:
            for beta in betas:

                # Train model
                cost_val = []
                sess = tf.Session()
                sess.run(tf.global_variables_initializer())

                for epoch in range(FLAGS.epochs):

                    # Construct feed dictionary
                    feed_dict = OO_construct_feed_dict(features, support, y, train_mask_CV, placeholders, alpha, beta, f_u)
                    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

                    # Training
                    outs = sess.run([model.opt_op, model.loss], feed_dict=feed_dict)

                    # Validation
                    cost, duration = evaluate(features, support, y, val_mask_CV, placeholders, alpha, beta, f_u)
                    cost_val.append(cost)

                    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
                        break

                _, val_Z, _ = evaluate_with_predictions(features, support, y, val_mask_CV, placeholders, alpha, beta, f_u)
                val_Z = val_Z[val_mask_CV]

                for thetas_combo in thetas_combos:

                    theta_1 = thetas_combo[0]
                    theta_2 = thetas_combo[1]

                    y_hat = np.zeros(val_Z.shape[0])
                    y_hat[val_Z[:, 0] > theta_2] = 0
                    y_hat[val_Z[:, 0] <= theta_1] = 1
                    y_hat[np.logical_and(val_Z[:, 0] > theta_1, val_Z[:, 0] <= theta_2)] = 2

                    cm = confusion_matrix(y[val_mask_CV].argmax(1), y_hat, labels=[0, 1, 2])
                    cm = np.delete(cm, 2, axis=0)

                    key = (kappa, alpha, beta, tuple(thetas_combo))
                    if key not in err_dict:
                        err_dict[key] = []
                    err_dict[key].append(compute_error(cm))

print("Optimization Finished!")

for key, values in err_dict.items():
    average = sum(values) / len(values)
    err_dict[key] = average

best_params = min(err_dict, key=err_dict.get)
best_kappa = best_params[0]
best_f_u = compute_f_u(e, y, train_masks_CV.iloc[:, [0]], best_kappa)
best_alpha = best_params[1]
best_beta = best_params[2]
best_theta_1 = best_params[3][0]
best_theta_2 = best_params[3][1]

print('Best kappa:', best_kappa, 'Best alpha:', best_alpha, 'Best beta:', best_beta, 'Best theta_1:', best_theta_1, 'Best theta_2:', best_theta_2)
# Testing
cost_val = []
sess = tf.Session()
sess.run(tf.global_variables_initializer())

train_mask_0 = np.array(train_masks_CV.iloc[:, 0].values.flatten())
val_mask_0 = np.array(val_masks_CV.iloc[:, 0].values.flatten())

for epoch in range(FLAGS.epochs):

    # Construct feed dictionary
    feed_dict = OO_construct_feed_dict(features, support, y, train_mask_0, placeholders, best_alpha, best_beta, best_f_u)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss], feed_dict=feed_dict)

    # Validation
    cost, duration = evaluate(features, support, y, val_mask_0, placeholders, best_alpha, best_beta, best_f_u)
    cost_val.append(cost)

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        break

_, test_Z, _ = evaluate_with_predictions(features, support, y, test_mask, placeholders, best_alpha, best_beta, best_f_u)

y_hat = np.zeros(test_Z.shape[0])
y_hat[test_Z[:, 0] > best_theta_2] = 0
y_hat[test_Z[:, 0] <= best_theta_1] = 1
y_hat[np.logical_and(test_Z[:, 0] > best_theta_1, test_Z[:, 0] <= best_theta_2)] = 2

cm = confusion_matrix(y[test_mask].argmax(1), y_hat[test_mask], labels=[0, 1, 2])
cm = np.delete(cm, 2, axis=0)
print('Confusion Matrix:\n', cm)
print('Error:', compute_error(cm))

np.savetxt('../results.txt', y_hat, delimiter='\t')
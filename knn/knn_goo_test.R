rm(list=ls())
graphics.off()

library(sf)
library(progress)
library(igraph)
library(Matrix)
library(caret)
library(tictoc)

load('data/r/data.Rdata')
load('utils.Rdata')

Ns = compute_Ns(y)
N = Ns[1]; N_l = Ns[2]; N_p = Ns[3]; N_u = Ns[4]
rm(Ns)

n_fold = 5
ids_CV = split_CV(ids = train_ids, y = y)
train_ids_CV = ids_CV$train
val_ids_CV = ids_CV$val
rm(ids_CV)

#LIMITED
a = a[1:N_l,]
c = c
e = e
p = p[1:N_l,]

# HYPER_PARAMS
zeta = 0.5
kappa = 5
theta_1 = 0.4
theta_2 = 0.4

d = e^zeta * c^(1-zeta)

f_u = compute_f_u(d,y,train_ids,kappa)
f_p = 1 - f_u

lev = c(0,1,2)
preds = ifelse(f_p <= theta_1, 1, ifelse(f_p > theta_2, 0, 2))[test_ids]
true = ifelse(y[test_ids,2] == 1, 1, ifelse(y[test_ids,1] == 1, 0, 2))

cm = table(true = factor(true, levels = lev),
           preds = factor(preds, levels = lev))
cm = cm[-3,]

err_test = compute_err(cm)
cm
err_test

results = ifelse(f_p <= theta_1, 1, ifelse(f_p > theta_2, 0, 2))
write.table(results, "results.txt", sep = "\t", row.names = FALSE, col.names = FALSE)
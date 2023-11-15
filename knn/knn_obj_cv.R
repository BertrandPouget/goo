rm(list=ls())
graphics.off()

library(sf)
library(progress)
library(igraph)
library(Matrix)
library(caret)

load('data/data.Rdata')
load('knn/utils.Rdata')

n_fold = 5
ids_CV = split_CV(ids = train_ids, y = data$osm_surf)
train_ids_CV = ids_CV$train
val_ids_CV = ids_CV$val
rm(ids_CV)

#LIMITED (taking into account first N_l lines error on the test set can be computed)
a = a[1:N_l,]
c = c[1:N_l,]
e = e
p = p[1:N_l,]

# HYPER_PARAMS
kappas = c(3, 5, 7)
n_kappas = length(kappas)

thetas = get_thetas(start = 0, stop = 1, step = 0.2)
n_thetas = dim(thetas)[1]

# Cross_validation
err_array = array(0, dim = c(n_kappas, n_thetas))
pb = progress_bar$new(total = n_kappas * n_thetas * n_fold)

for (i_kappa in 1:n_kappas)
{
  kappa = kappas[i_kappa]
  for (i_theta in 1:n_thetas)
  {
    theta_1 = thetas[i_theta, 1]
    theta_2 = thetas[i_theta, 2]
    
    err_val_n = vector('numeric', n_fold)
    for (n in 1:n_fold)
    {
      f_u = compute_f_u(e,data$osm_surf,train_ids_CV[[n]],kappa)
      f_p = 1 - f_u
      
      lev = c(0,1,2)
      preds = ifelse(f_p <= theta_1, 1, ifelse(f_p > theta_2, 0, 2))[val_ids_CV[[n]]]
      true = ifelse(data$osm_surf[val_ids_CV[[n]]] == 'unpaved', 1, ifelse(data$osm_surf[val_ids_CV[[n]]] == 'paved', 0, 2))
      
      cm = table(true = factor(true, levels = lev),
                 preds = factor(preds, levels = lev))
      cm = cm[-3,]
      
      err_val_n[n] = compute_err(cm)
      pb$tick()
    }
    
    err = mean(err_val_n)
    err_array[i_kappa,i_theta] = err
  }
}

best_combo = arrayInd(which.min(err_array),c(n_kappas,n_thetas))
best_kappa = kappas[best_combo[1]]
best_theta_1 = thetas[best_combo[2],1]
best_theta_2 = thetas[best_combo[2],2]

f_u = compute_f_u(e,data$osm_surf,train_ids,best_kappa)
f_p = 1 - f_u

lev = c(0,1,2)
preds = ifelse(f_p <= best_theta_1, 1, ifelse(f_p > best_theta_2, 0, 2))[test_ids]
true = ifelse(data$osm_surf[test_ids] == 'unpaved', 1, ifelse(data$osm_surf[test_ids] == 'paved', 0, 2))

cm = table(true = factor(true, levels = lev),
           preds = factor(preds, levels = lev))
cm = cm[-3,]

err_test = compute_err(cm)

best_kappa
best_theta_1
best_theta_2
cm
err_test

results = ifelse(f_p <= best_theta_1, 1, ifelse(f_p > best_theta_2, 0, 2))
write.table(results, "results.txt", sep = "\t", row.names = FALSE, col.names = FALSE)
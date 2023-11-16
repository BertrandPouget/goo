rm(list=ls())
graphics.off()

library(sf)
library(igraph)
library(Matrix)

load('data/r/data.Rdata')
gamma = setdiff(ls(),c('test_ids','train_ids'))

load('utils.Rdata')
Ns = compute_Ns(y)
N = Ns[1]; N_l = Ns[2]; N_p = Ns[3]; N_u = Ns[4]
rm(Ns)

#Split
set.seed(110)

paved_ids = sort(which(y[,1] == 1))
unpaved_ids = sort(which(y[,2] == 1))

paved_prop = N_p / N_l
unpaved_prop = N_u / N_l

N_train = round(0.8 * N_l)
N_test = N_l - N_train

train_paved_ids = sample(paved_ids, size = round(N_train * paved_prop), replace = FALSE)
train_unpaved_ids = sample(unpaved_ids, size = round(N_train * unpaved_prop), replace = FALSE)
train_ids = sort(c(train_paved_ids, train_unpaved_ids))

test_paved_ids = setdiff(paved_ids, train_ids)
test_unpaved_ids = setdiff(unpaved_ids, train_ids)
test_ids = sort(c(test_paved_ids,test_unpaved_ids))

train_mask = rep(FALSE, N)
train_mask[train_ids] = TRUE

test_mask = rep(FALSE, N)
test_mask[test_ids] = TRUE

#Python
write.table(train_mask, "data/python/train_mask.txt", sep = "\t", row.names = FALSE, col.names = FALSE)
write.table(test_mask, "data/python/test_mask.txt", sep = "\t", row.names = FALSE, col.names = FALSE)

#R
rm(list=setdiff(ls(),c(gamma,'test_ids','train_ids')))
save.image("data/r/data.Rdata")
import os, sys
import torch
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist


@torch.no_grad()
def multiLID(args, X, X_adv, feature_extractor, model, activation=None, lid_dim=1, k=10, batch_size=100, device='cpu'):
    """
    Get the local intrinsic dimensionality of each Xi in X_adv
    estimated by k close neighbours in the random batch it lies in.
    :param X: normal images
    :param X_noisy: noisy images
    :param X_adv: advserial images     
    :param k: the number of nearest neighbours for LID estimation  
    :param batch_size: default 100
    :return: lids: LID of normal images of shape (num_examples, lid_dim)
            lids_adv: LID of advs images of shape (num_examples, lid_dim)
    """
    print("Number of layers to estimate: ", lid_dim)

    print("k: ", min(k, batch_size-1))

    # lid of a batch of query points X
    def mle_batch(data, batch, k):
        data = np.asarray(data, dtype=np.float32)
        batch = np.asarray(batch, dtype=np.float32)
        
        k = min(k, len(data)-1)

        ##########################################################################

        f = lambda v: - np.log(v/v[-1], where=v/v[-1]>0,  dtype=np.float32) # multiLID

        ##########################################################################
        
        a = cdist(batch, data)
        a = np.apply_along_axis(np.sort, axis=1, arr=a)[:,1:k+1]
        a = np.apply_along_axis(f, axis=1, arr=a)
        return a

    def estimate(i_batch):
        start = i_batch * batch_size
        end = np.minimum(len(X), (i_batch + 1) * batch_size)
        n_feed = end - start
        
        lid_batch     = np.zeros(shape=(n_feed, k, lid_dim))
        lid_batch_adv = np.zeros(shape=(n_feed, k, lid_dim))
        
        batch     = X[start:end]
        batch_adv = X_adv[start:end]

        if activation == None:
            X_act = feature_extractor(batch.to(device)).values()
        else:
            X_act = feature_extractor(args, model, batch.to(device), activation)

        X_act = [np.asarray(x.cpu().detach().numpy(), dtype=np.float32).reshape((n_feed, -1)) for x in X_act]

        if activation == None:
            X_act_adv = feature_extractor(batch_adv.to(device)).values()
        else:
            X_act_adv = feature_extractor(args, model, batch_adv.to(device), activation)
        X_act_adv = [np.asarray(x.cpu().detach().numpy(), dtype=np.float32).reshape((n_feed, -1)) for x in X_act_adv]
        
        # random clean samples
        # Maximum likelihood estimation of local intrinsic dimensionality (LID)
        for i in range(lid_dim):
            lid_batch[:, :, i]    = mle_batch(X_act[i], X_act[i], k=k).copy()
            lid_batch_adv[:,:, i] = mle_batch(X_act[i], X_act_adv[i], k=k).copy()

        return lid_batch, lid_batch_adv

    lids = []
    lids_adv = []
    n_batches = int(np.ceil(X.shape[0] / float(batch_size)))

    for i_batch in tqdm(range(n_batches)):
        lid, lid_adv = estimate(i_batch)
        lids.extend(lid)
        lids_adv.extend(lid_adv)

    lids = np.asarray(lids, dtype=np.float32)
    lids_adv = np.asarray(lids_adv, dtype=np.float32)

    return lids, lids_adv


@torch.no_grad()
def LID(args, X, X_adv, feature_extractor, model, activation=None, lid_dim=1, k=10, batch_size=100, device='cpu'):
    """
    Get the local intrinsic dimensionality of each Xi in X_adv
    estimated by k close neighbours in the random batch it lies in.
    :param X: normal images
    :param X_noisy: noisy images
    :param X_adv: advserial images     
    :param k: the number of nearest neighbours for LID estimation  
    :param batch_size: default 100
    :return: lids: LID of normal images of shape (num_examples, lid_dim)
            lids_adv: LID of advs images of shape (num_examples, lid_dim)
    """
    print("Number of layers to estimate: ", lid_dim)

    # lid of a batch of query points X
    def mle_batch(data, batch, k):
        data = np.asarray(data, dtype=np.float32)
        batch = np.asarray(batch, dtype=np.float32)

        k = min(k, len(data)-1)
        print("selected k: ", k)
            
        f = lambda v: - k / np.sum(np.log(v/v[-1]))
        a = cdist(batch, data)
        a = np.apply_along_axis(np.sort, axis=1, arr=a)[:,1:k+1]
        a = np.apply_along_axis(f, axis=1, arr=a)
        return a

    def estimate(i_batch):
        start = i_batch * batch_size
        end = np.minimum(len(X), (i_batch + 1) * batch_size)
        n_feed = end - start
        
        lid_batch = np.zeros(shape=(n_feed, lid_dim))
        lid_batch_adv = np.zeros(shape=(n_feed, lid_dim))
        
        batch     = X[start:end]
        batch_adv = X_adv[start:end]
        
        if activation == None:
            X_act = feature_extractor(batch.to(device)).values()
        else:
            X_act = feature_extractor(args, model, batch.to(device), activation)
        X_act = [np.asarray(x.cpu().detach().numpy(), dtype=np.float32).reshape((n_feed, -1)) for x in X_act]

        if activation == None:
            X_act_adv = feature_extractor(batch_adv.to(device)).values()
        else:
            X_act_adv = feature_extractor(args, model, batch_adv.to(device), activation)
        X_act_adv = [np.asarray(x.cpu().detach().numpy(), dtype=np.float32).reshape((n_feed, -1)) for x in X_act_adv] 

        
        # random clean samples
        # Maximum likelihood estimation of local intrinsic dimensionality (LID)
        for i in range(lid_dim):
            lid_batch[:, i] = mle_batch(X_act[i], X_act[i], k=k)
            lid_batch_adv[:, i] = mle_batch(X_act[i], X_act_adv[i], k=k)

        return lid_batch, lid_batch_adv

    lids = []
    lids_adv = []
    n_batches = int(np.ceil(X.shape[0] / float(batch_size)))

    for i_batch in tqdm(range(n_batches)):
        lid_batch, lid_batch_adv = estimate(i_batch)
        lids.extend(lid_batch)
        lids_adv.extend(lid_batch_adv)

    lids = np.asarray(lids, dtype=np.float32)
    lids_adv = np.asarray(lids_adv, dtype=np.float32)

    return lids, lids_adv

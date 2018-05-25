# Author: Dmytro Mishkin, 2018.
# This is simple partial re-implementation of papers 
# Fast Spectral Ranking for Similarity Search, CVPR 2018. https://arxiv.org/abs/1703.06935
# and  Efficient Diffusion on Region Manifolds: Recovering Small Objects with Compact CNN Representations, CVPR 2017 http://people.rennes.inria.fr/Ahmet.Iscen/diffusion.html


import os
import numpy as np
from scipy.io import loadmat
from scipy.sparse import csr_matrix, eye, diags
from scipy.sparse import linalg as s_linalg

def sim_kernel(dot_product):
    return np.maximum(np.power(dot_product,3),0)
 
def normalize_connection_graph(G):
    W = csr_matrix(G)
    W = W - diags(W.diagonal())
    D = np.array(1./ np.sqrt(W.sum(axis = 1)))
    D[np.isnan(D)] = 0
    D[np.isinf(D)] = 0
    D_mh = diags(D.reshape(-1))
    Wn = D_mh * W * D_mh
    return Wn

def topK_W(G, K = 100):
    sortidxs = np.argsort(-G, axis = 1)
    for i in range(G.shape[0]):
        G[i,sortidxs[i,K:]] = 0
    G = np.minimum(G, G.T)
    return G
def find_trunc_graph(qs, W, levels = 3):
    needed_idxs = []
    needed_idxs = list(np.nonzero(qs > 0)[0])
    for l in range(levels):
        idid = W.nonzero()[1]
        needed_idxs.extend(list(idid))
        needed_idxs =list(set(needed_idxs))
    return np.array(needed_idxs), W[needed_idxs,:][:,needed_idxs]

def dfs_trunk(sim, A,alpha = 0.99, QUERYKNN = 10, maxiter = 8, K = 100, tol = 1e-3):
    qsim = sim_kernel(sim).T
    sortidxs = np.argsort(-qsim, axis = 1)
    for i in range(len(qsim)):
        qsim[i,sortidxs[i,QUERYKNN:]] = 0
    qsims = sim_kernel(qsim)
    W = sim_kernel(A)
    W = csr_matrix(topK_W(W, K))
    out_ranks = []
    t =time()
    for i in range(qsims.shape[0]):
        qs =  qsims[i,:]
        tt = time() 
        w_idxs, W_trunk = find_trunc_graph(qs, W, 2);
        Wn = normalize_connection_graph(W_trunk)
        Wnn = eye(Wn.shape[0]) - alpha * Wn
        f,inf = s_linalg.minres(Wnn, qs[w_idxs], tol=tol, maxiter=maxiter)
        ranks = w_idxs[np.argsort(-f.reshape(-1))]
        missing = np.setdiff1d(np.arange(A.shape[1]), ranks)
        out_ranks.append(np.concatenate([ranks.reshape(-1,1), missing.reshape(-1,1)], axis = 0))
    print time() -t, 'qtime'
    out_ranks = np.concatenate(out_ranks, axis = 1)
    return out_ranks

def cg_diffusion(qsims, Wn, alpha = 0.99, maxiter = 10, tol = 1e-3):
    Wnn = eye(Wn.shape[0]) - alpha * Wn
    out_sims = []
    for i in range(qsims.shape[0]):
        #f,inf = s_linalg.cg(Wnn, qsims[i,:], tol=tol, maxiter=maxiter)
        f,inf = s_linalg.minres(Wnn, qsims[i,:], tol=tol, maxiter=maxiter)
        out_sims.append(f.reshape(-1,1))
    out_sims = np.concatenate(out_sims, axis = 1)
    ranks = np.argsort(-out_sims, axis = 0)
    return ranks

def fsr_rankR(qsims, Wn, alpha = 0.99, R = 2000):
    vals, vecs = s_linalg.eigsh(Wn, k = R)
    p2 = diags((1.0 - alpha) / (1.0 - alpha*vals))
    vc = csr_matrix(vecs)
    p3 =  vc.dot(p2)
    vc_norm =  (vc.multiply(vc)).sum(axis = 0)
    out_sims = []
    for i in range(qsims.shape[0]):
        qsims_sparse = csr_matrix(qsims[i:i+1,:])
        p1 =(vc.T).dot(qsims_sparse.T)
        diff_sim = csr_matrix(p3)*csr_matrix(p1)
        out_sims.append(diff_sim.todense().reshape(-1,1))
    out_sims = np.concatenate(out_sims, axis = 1)
    ranks = np.argsort(-out_sims, axis = 0)
    return ranks
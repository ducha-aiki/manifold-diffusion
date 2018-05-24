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

def cg_diffusion(qsims, Wn, alpha = 0.99, maxiter = 20, tol = 1e-6):
    Wnn = eye(Wn.shape[0]) - alpha * Wn
    out_sims = []
    for i in range(qsims.shape[0]):
        f,inf = s_linalg.cg(Wnn, qsims[i,:], tol=tol, maxiter=maxiter)
        out_sims.append(f.reshape(-1,1))
    out_sims = np.concatenate(out_sims, axis = 1)
    ranks = np.argsort(-out_sims, axis = 0)
    return ranks

def fsr_rankR(qsims, Wn, alpha = 0.99, R = 2000, do_correction = True):
    vals, vecs = s_linalg.eigsh(Wn, k = R)
    p2 = diags((1.0 - alpha) / (1.0 - alpha*vals))
    vc = csr_matrix(vecs)
    p3 =  vc.dot(p2)
    vc_norm =  (vc.multiply(vc)).sum(axis = 0)
    correct_coef = csr_matrix(1 - vc_norm).dot(vc.T).T
    out_sims = []
    for i in range(qsims.shape[0]):
        qsims_sparse = csr_matrix(qsims[i:i+1,:])
        p1 =(vc.T).dot(qsims_sparse.T)
        diff_sim = csr_matrix(p3)*csr_matrix(p1)
        if do_correction:
            diff_sim = np.array(diff_sim) - np.array(correct_coef)
        out_sims.append(diff_sim.todense().reshape(-1,1))
    out_sims = np.concatenate(out_sims, axis = 1)
    ranks = np.argsort(-out_sims, axis = 0)
    return ranks
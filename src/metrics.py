#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 14:03:39 2019

@author: ziqi
"""
import numpy as np
import math
import sklearn.metrics as metrics
from scipy import stats
from config import THRESHOLD, THRESHOLDS

def rank_metrics(ba, pred):
    # Spearman Rank Coefficient
    rho,_ = scipy.spearmanr(ba, pred)
    
    label = np.zeros_like(ba)
    label[np.where(ba<THRESHOLD)[0]] = 1
    # AUC
    auc = metrics.roc_auc_score(label, pred)
    
    # Kendall's tau correlation
    tau,_ = stats.kendalltau(ba, pred)
    
    return rho, auc, tau

def toprank_metrics(ba, pred, order):
    num = {}
    avgrank = {}
    hr = {}
    whr = {}
    ndcg = {}
    
    for t in THRESHOLDS:
        num[t], avgrank[t], hr[t], whr[t], ndcg[t] = toprank(ba, pred, t, order)
    
    return num, avgrank, hr, whr, ndcg
    
def toprank(ba, pred, beta, order):
    idxba_sort = np.argsort(ba)
    idxsc_sort = np.argsort(pred, axis=0)
    
    # get the list of index of top alpha peptides in binding affinity
    rankba = []
    rank = 0
    for i, idx in enumerate(idxba_sort):
        if i > 0 and ba[idx] == ba[idxba_sort[i-1]]:
            rankba.append([idx, rank])
        else:
            rank = rank+1
            if ba[idx] > beta:
                break
            
            rankba.append([idx, rank])
    rankba = np.array(rankba)
    
    if np.size(rankba,0)==0:
        return 0,0,0,0,0
    
    # get the list of index of top alpha peptides in prediction
    ranksc = []
    idxsc_sort = idxsc_sort[::order]
    
    # average rank
    avgrank = 0
    for i in range(np.size(rankba,0)):
        avgrank = avgrank + np.where(idxsc_sort[:,0]==rankba[i,0])[0][0] + 1
    avgrank = avgrank / np.size(rankba,0)
    
    ranksc = [[idxsc_sort[i][0], i] for i in range(np.size(rankba,0))]
    ranksc = np.array(ranksc)
    topalpha, ba_ind, sc_ind = np.intersect1d(rankba[:,0], ranksc[:,0], return_indices=True)
    
    # hit rate
    hr = np.size(topalpha, 0) / np.size(rankba, 0)
    
    # weighted hit rate
    whr = np.sum(1 / (ranksc[sc_ind, 1]+1)) / np.sum(1/np.arange(1,np.size(ranksc, 0)+1))
    
    # ndcg
    try:
        weight = [max(0, 1-math.log(ba[idx], 5e4)) for idx in ranksc[:,0]]
        # weight used in normalization
        wenorm = [max(0, 1-math.log(ba[idx], 5e4)) for idx in rankba[:,0]]
        norm = np.sum([w/math.log2(rankba[i,1]+2) for i,w in enumerate(wenorm)])
        ndcg = np.sum([w/math.log2(rankba[i,1]+2) for i,w in enumerate(weight)])/norm
    except:
        ndcg = 0.000
    return np.size(ranksc, 0), avgrank, hr, whr, ndcg
    
def cls_metrics(ba, pred, cls_tsd):
    label = np.zeros_like(ba)
    label[np.where(ba<THRESHOLD)[0]] = 1
    
    pred_label = np.zeros_like(ba)
    pred_label[np.where(pred<cls_tsd)[0]] = 1
    
    recall = metrics.recall_score(label, pred_label)
    accuracy = metrics.accuracy_score(label, pred_label)
    precision = metrics.precision_score(label, pred_label)
    f1 = metrics.f1_score(label, pred_label)
    
    return recall, accuracy, precision, f1

def evaluate(ba, pred, order, cls_tsd):
    """
    cls_tsd : classification threshold for score
    """
    metrics = np.zeros(1,7)
    num, avgrank, hr, whr, ndcg = toprank_metrics(ba, pred, order)
    metrics[0,:3] = rank_metrics(ba, pred)
    metrics[0,3:] = cls_metrics(ba, pred, cls_tsd)
    return num, avgrank, hr, whr, ndcg, metrics
    

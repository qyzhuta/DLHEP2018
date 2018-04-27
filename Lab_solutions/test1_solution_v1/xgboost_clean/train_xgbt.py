#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 20:41:12 2018

@author: qingyuzhu

xgb trainning
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import sklearn.cross_validation as cv
import itertools




def AMS(s, b):
    '''
    Approximate median significance:
        s = true positive rate
        b = false positive rate
    '''
    assert s >= 0
    assert b >= 0
    bReg = 10.
    return np.sqrt(2.0 * ((s + b + bReg) * np.log(1 + s / (b + bReg)) - s))


def get_rates(prediction, solution, weights):
    '''
    Returns the true and false positive rates.
    This assumes that:
        label 's' corresponds to 1 (int)
        label 'b' corresponds to 0 (int)
    '''
    assert prediction.size == solution.size
    assert prediction.size == weights.size

    # Compute sum of weights for true and false positives
    truePos  = sum(weights[(solution == 1) * (prediction == 1)])
    falsePos = sum(weights[(solution == 0) * (prediction == 1)])

    return truePos, falsePos

def get_train_test(fname):
    
    df=pd.read_csv(fname)
    cols = df.columns.tolist()
    X=df[cols[:-2]].values
    labels = df['Label'].values
    weights = df['Weight'].values
    
    return X, labels, weights

def train_xgbt_classfier(fname,param, num_round, folds):
    
    X, labels, weights=get_train_test(fname)
    
    kf = cv.KFold(labels.size, n_folds=folds)
    npoints  = 26
    
    all_AMS = {}
    for curr in range(npoints):
        all_AMS[curr] = []
    
    cutoffs  = np.linspace(0.05, 0.30, npoints)
    ave_ams=0.
    for train_indices, test_indices in kf:
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = labels[train_indices], labels[test_indices]
        w_train, w_test = weights[train_indices], weights[test_indices]
        
        w_train *= (sum(weights) / sum(w_train))
        w_test  *= (sum(weights) / sum(w_test))

        sum_wpos = sum(w_train[y_train == 1])
        sum_wneg = sum(w_train[y_train == 0])
        
        xgmat = xgb.DMatrix(X_train, label=y_train, missing=-999.0, weight=w_train)
        
        
        param['scale_pos_weight'] = sum_wneg/sum_wpos
        plst = param.items()#+[('eval_metric', 'ams@0.15')]
        
        watchlist = []
        
        bst = xgb.train(plst, xgmat, num_round, watchlist)
        
        xgmat_test = xgb.DMatrix(X_test, missing=-999.0)
        y_out = bst.predict(xgmat_test)
        res  = [(i, y_out[i]) for i in range(len(y_out))]
        rorder = {}
        for k, v in sorted(res, key = lambda x:-x[1]):
            rorder[k] = len(rorder) + 1

        # Explore changing threshold_ratio and compute AMS
        best_AMS = -1.
        best_thres = 0.0
        for curr, threshold_ratio in enumerate(cutoffs):
            y_pred = np.zeros(len(y_out))
            ntop = int(threshold_ratio * len(rorder))
            for k, v in res:
                if rorder[k] <= ntop:
                    y_pred[k] = 1

            truePos, falsePos = get_rates(y_pred, y_test, w_test)
            this_AMS = AMS(truePos, falsePos)
            
            all_AMS[curr].append(this_AMS)
            if this_AMS > best_AMS:
                best_AMS = this_AMS
                best_thres = threshold_ratio
        print ("Best AMS = %f at %.2f"%(best_AMS,best_thres))
        ave_ams+=best_AMS
    print ("------------------------------------------------------")
    
    return ave_ams

def save_model(fname,param, num_round, folds):
    
    X, labels, weights=get_train_test(fname)
    
    kf = cv.KFold(labels.size, n_folds=folds)
    
    for train_indices, test_indices in kf:
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = labels[train_indices], labels[test_indices]
        w_train, w_test = weights[train_indices], weights[test_indices]
        
        w_train *= (sum(weights) / sum(w_train))
        w_test  *= (sum(weights) / sum(w_test))

        sum_wpos = sum(w_train[y_train == 1])
        sum_wneg = sum(w_train[y_train == 0])
        
        xgmat = xgb.DMatrix(X_train, label=y_train, missing=-999.0, weight=w_train)
        
        
        param['scale_pos_weight'] = sum_wneg/sum_wpos
        plst = param.items()#+[('eval_metric', 'ams@0.15')]
        
        watchlist = []
        
        bst = xgb.train(plst, xgmat, num_round, watchlist)
        
        bst.save_model('best_xgb_model')
    
    
    
         
    
def main():
    # setup parameters for xgboost
    param = {}
    # use logistic regression loss, use raw prediction before logistic transformation
    # since we only need the rank
    param['objective'] = 'binary:logitraw'
    
    #param['eval_metric'] = 'auc'
    param['eval_metric'] = 'ams@0.14'
    param['silent'] = 1
    param['nthread'] = 16

    num_round = 120 # Number of boosted trees
    folds = 5 # Folds for CV
    fname='X_y'

    all_etas = [0.01]
    all_subsamples = [0.7,0.9]
    all_depth = [9,10]
    nums_rounds = [3000]

    e_s_m = list(itertools.product(all_etas,all_subsamples,all_depth,nums_rounds))
    ref=0
    e_f=0
    s_f=0
    m_f=0
    r_f=0
    for e,s,m,r in e_s_m:
        param['bst:eta'] = e
        param['bst:subsample'] = s
        param['bst:max_depth'] = m
        print ('e %.3f s %.2f m %d round %d'%(e,s,m,r))
        ave_ams=train_xgbt_classfier(fname,param,num_round,folds)
        
        if ave_ams>ref:
            ref=ave_ams
            e_f=e
            s_f=s
            m_f=m
            r_f=r
            
    #%% Save model
    param['bst:eta'] = e_f
    param['bst:subsample'] = s_f
    param['bst:max_depth'] = m_f
    
    save_model(fname,param,num_round,folds)
    
    
        
    


if __name__ == "__main__":
    main()
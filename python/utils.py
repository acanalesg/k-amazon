""" Util functions for kaggle challenges
    Created for Amazon Access Challenge Jun 14, 2013

    Arturo Canales (arturocanales@gmail.com)
"""

__author__ = 'acg'

import numpy as np
import pandas as pd
import scipy as sp
import sklearn as sk
from sklearn import (metrics, cross_validation, linear_model, preprocessing, metrics)


identity = lambda x: x


# Maybe I should use itertool.combinations
def combs(n, k):
    """
    Function returning all possible k-tuples of ints from 0 to n-1
    without repetition and having tuple[i] < tuple[i+1]
    Example: combs(3,2) --> [(0, 1), (0, 2), (1, 2)]

    Parameters
    ----------

    n : int
      Maximum integer

    k : int
      Tuples size (k-tuples)

    Returns (yields)
    -------
    Generator of k-tuples
    """
    if k == 1:
        for v in range(n):
            yield (v,)
    else:
        for v in combs(n, k - 1):
            for i in range(v[-1] + 1, n):
                yield v + (i,)



def my_cross_val_score(model, X, y, scoring, folds, random_state=123):
    """
    My cross_val_score implementation, sending to scoring function
    model, X and y (the one in sklearn send y and preds - no probs)

    Based on Paul's loop

    I'm not using sklearn.crossvalidation.cross_val_score because the sklearn function
    is sending to the scoring function  predicted class (0 or 1), instead of predicted
    class probabilities.

    Parameters
    ---------

    model :
       Model to fit (must implement fit and predic and predict_proba

    X : nparray, matrix, sparse matrix
       Features

    y : vector, series
       Target variable

    scoring : callable (function or class)
       Function use to get score of model fit

    folds : integer
       Number of folds in cross-validation

    random_state : integer (default 123)
       Seed state for splitting test and validation


    Returns (tuple)
    -------

    xcores_xc : sequence
        List of scores for each fold in crossvalidation sets

    xcores_tr : sequence
        List of scores for each fold in training sets

    """
    scores_xc, scores_tr = [], []
    for i in range(folds):
        # Split train and cv
        X_train, X_cv, y_train, y_cv = sk.cross_validation.train_test_split(
            X, y, test_size=.20, random_state=i*random_state)

        # Train
        model.fit(X_train, y_train)

        # Score
        scores_xc.append(scoring(model, X_cv, y_cv))
        scores_tr.append(scoring(model, X_train, y_train))

    return scores_xc, scores_tr


def auc_roc(model, X, y):
    """
    Use model to predict class probabilities, and then computes
    area under curve for ROC

    Parameters
    ----------
    model :
       Model to predict probs with
       (must implement predict_proba)

    X : nparray, matrix, sparse matrix
       Features

    y : vector, series
       Target variable


    Returns
    -------

    roc_auc : float
       Area under curve (ROC)
    """
    #print X.shape
    preds = model.predict_proba(X)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y, preds)
    roc_auc = metrics.auc(fpr, tpr)
    #print "AUC: %f" % (roc_auc)
    return roc_auc



def feature_selection(model, X, y, max_features, scoring, transform=identity,
                      stop_val=0.00001, kfold=3, random_state=567,
                      features_sel=[], remove_worst=5):
    """
    Select best features of X to predict y using model

    Parameters
    ----------
    model :
       Model to predict probs with
       (must implement predict_proba)

    X : nparray, matrix, sparse matrix
       Features

    y : vector, series
       Target variable

    max_features : integer
       Maximum number of features to return

    scoring : callable
       Function to evaluate the fit goodness

    transform : callable (default identity)
       Function to apply to X (if needed)

    stop_val : float
       If gain from last iter is less than stop_val, then finish the process

    kfold : integer
       Number of folds used in crossvalidation

    random_state : integer
       Random State

    features_sel : sequence (default [])
       List of pre-selected features

    remove_worst : integer
       After every iteration the N-worst features would be remove, this parameter sets
       that N


    Returns
    -------
    features_sel : sequence
       List of best features
    """
    #pre_score = 0.898
    pre_score = 0
    features_removed = []
    for l in range(max_features): # No more than 30 features
        scores_f = []
        for i in range(X.shape[1]):
            if i not in features_sel and i not in features_removed:
                f = features_sel + [i,]
                #print f
                xt = X[:,f]
                #print xt.shape
                scores, _ = my_cross_val_score(model, transform(xt), y,
                                               scoring, kfold, random_state)#*(i+1)*(l+1))
                scores_f.append((sp.mean(scores), i))
        scores_f = sorted(scores_f, reverse=True)

        print scores_f[0:2]
        if len(scores_f) == 0:
            break

        if scores_f[0][0] > pre_score:
            features_sel.append(scores_f[0][1])
        else:
            break

        # Remove worst features
        if len(scores_f) > remove_worst:
            print scores_f[-remove_worst:]
            for m in range(-remove_worst,0):
                features_removed.append(scores_f[m][1])

        #print scores_f[1]
        print features_sel
        print "--"
        if (scores_f[0][0] - pre_score) < stop_val:
            break
        pre_score = scores_f[0][0]

    return features_sel

__author__ = 'acg'

import numpy as np
import pandas as pd
import scipy as sp
import sklearn as sk
from utils import combs, my_cross_val_score, auc_roc

SEED = 6892

def load_data(filename):
    """
    Load CSV file, return separately first column as y (target)
    rest of the columns (but last -> repeated) as X (variables)
    """
    data = pd.read_csv(filename)
    y = np.array(data.ix[:, 0])
    X = np.array(data.ix[:, range(1, 8)])

    return X, y


def combine_variables(X):
    X_plus = []
    for j in range(1, 4):
        for i in combs(7, j):
            X_plus.append([hash(tuple(v)) % 800000 for v in X[:, i]])

    X = np.array(X_plus).T

    return X


def create_new_features(X):
    """
    Transform data and create new features
    """
    # Include new variables (combinations)
    print "create combs for X"
    X = combine_variables(X)
    print X.shape

    return X


def encode_factors(X, X_test):
    """
    Encode
    """
    # Use one hot encoder to codify factors (all) as dummy variables
    encoder = sk.preprocessing.OneHotEncoder()
    encoder.fit(np.vstack([X, X_test]))
    X = encoder.transform(X)  # Returns a sparse matrix (see numpy.sparse)
    X_test = encoder.transform(X_test)

    print X.shape
    print X_test.shape

    return X, X_test


def save_results(predictions, filename):
    """Given a vector of predictions, save results in CSV format."""
    fp = open(filename, "w")
    fp.write("id,ACTION\n")
    for i, p in enumerate(predictions):
            fp.write("%d,%f\n" % (i + 1, p))
    fp.close()


def pick_best_params(model, X, y):
    opt_scores = []
    for C in np.arange(5.26, 6, 0.1):
        for p in (False, ):
            #model = linear_model.LogisticRegression(C=C, penalty=p)  # the classifier we'll use
            model.C = C
            model.dual = p

            # Train and Crossvalidate
            scores, _ = my_cross_val_score(model, X, y, auc_roc, 3)
            #print str(scores)
            print "Mean AUC for p=%s, C=%f: %f" % (p, C, sp.mean(scores))
            opt_scores.append((sp.mean(scores), p, C))
    opt_scores = sorted(opt_scores, reverse=True)
    print opt_scores
    return opt_scores[0][-1]


if __name__ == '__main__':

    print "Loading train data"
    X, y = load_data("/data/kaggle/amazon_access/train.csv")

    print "Loading test data"
    X_test, _ = load_data("/data/kaggle/amazon_access/test.csv")

    print "Create new features"
    X = create_new_features(X)
    X_test = create_new_features(X_test)

    print "Encoding factors/transform data"
    X, X_test = encode_factors(X, X_test)

    # Model
    model = sk.linear_model.LogisticRegression(C=5.7, penalty="l2", dual=True)  # the classifier we'll use
    #model = sk.linear_model.LogisticRegression(penalty="l2")  # the classifier we'll use
    #model = sk.linear_model.SGDClassifier(loss="log", penalty="l2")
    #model = sk.svm.SVC(probability=True)

    print "Find best params for model (C)"
    model.C = pick_best_params(model, X, y)

    # Get cross_val scores
    print "Cross-validate"
    scores, _ = my_cross_val_score(model, X, y, auc_roc, 10, random_state=SEED*30)
    #print str(scores)
    print "Mean AUC: %f" % (sp.mean(scores))

    # Train with whole dataset
    print "Train with the whole dataset"
    model.fit(X, y)
    preds = model.predict_proba(X_test)[:, 1]

    # Save results
    print "Save results"
    save_results(preds, "new_sal.csv")

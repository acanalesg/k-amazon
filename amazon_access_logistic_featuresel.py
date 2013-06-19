__author__ = 'acg'

import numpy as np
import pandas as pd
import scipy as sp
import sklearn as sk
from utils import combs, my_cross_val_score, auc_roc, feature_selection

# Configuration
SEED = 2892
ktuples_max = 6
feature_sel_stop = 0.00001
feature_remove_worst = 10
preselected = []
#preselected = [0, 1, 5, 7]


def load_data(filename):
    """
    Load CSV file, return separately first column as y (target)
    rest of the columns (but last -> repeated) as X (variables)
    """
    data = pd.read_csv(filename)
    y = np.array(data.ix[:, 0])
    X = np.array(data.ix[:, range(1, 8)])

    return X, y


def combine_variables(X, k):
    """
    Create combinations (tuples) of variables and return then
    From size 1 (singles) to k
    """
    X_plus = []
    for j in range(1, k+1):
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
    X = combine_variables(X, ktuples_max)
    #X = Xcomb
    #X = np.hstack([X, X/2, X/3, Xcomb])
    #print X[1,:]
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


def encode_factors_single(X):
    """
    Encode
    """
    # Use one hot encoder to codify factors (all) as dummy variables
    encoder = sk.preprocessing.OneHotEncoder()
    encoder.fit(np.vstack([X]))
    X = encoder.transform(X)  # Returns a sparse matrix (see numpy.sparse)

    #print X.shape
    return X


def save_results(predictions, filename):
    """Given a vector of predictions, save results in CSV format."""
    fp = open(filename, "w")
    fp.write("id,ACTION\n")
    for i, p in enumerate(predictions):
            fp.write("%d,%f\n" % (i + 1, p))
    fp.close()


def pick_best_params(model, X, y):
    """
    Pick best params
    """
    opt_scores = []
    for C in np.arange(0.5, 4, 0.3):
        for p in (True, False):
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


    # Model
    #model = linear_model.LogisticRegression(C=5.7, penalty="l2", dual=True)  # the classifier we'll use
    #model = sk.linear_model.LogisticRegression(penalty="l2")  # the classifier we'll use
    #model = sk.linear_model.SGDClassifier(loss="log", penalty="l2")
    model = sk.linear_model.LogisticRegression(penalty="l2", dual=True)
    #model = sk.svm.SVC(probability=True)

    print "Loading train data"
    X, y = load_data("/data/kaggle/amazon_access/train.csv")

    print "Loading test data"
    X_test, _ = load_data("/data/kaggle/amazon_access/test.csv")

    print "Create new features"
    X = create_new_features(X)
    X_test = create_new_features(X_test)

    print "Select best features"
    # Think about split test and train for crossvalidating feature selection
    # Xf_train, Xf_cv, yf_train, yf_cv = sk.cross_validation.train_test_split(
    #    X, y, test_size=.20, random_state=SEED*150)
    features = feature_selection(model, X, y,
                                 max_features=30,
                                 scoring=auc_roc,
                                 transform=encode_factors_single,
                                 stop_val=feature_sel_stop,
                                 kfold=4,
                                 random_state=SEED*78,
                                 features_sel=preselected,
                                 remove_worst=feature_remove_worst)
    #features = [0, 51, 61, 43, 34, 33, 52, 29, 80, 92, 73, 64, 90, 32, 63]
    X = X[:, features]
    X_test = X_test[:, features]
    print X.shape

    print "Encoding factors/transform data"
    X, X_test = encode_factors(X, X_test)

    print "Find best params for model (C)"
    model.C = pick_best_params(model, X, y)

    # Get cross_val scores
    print "Cross-validate"
    scores, _ = my_cross_val_score(model, X, y, auc_roc, 10, random_state=SEED*31)
    #print str(scores)
    print "Mean AUC: %f" % (sp.mean(scores))

    # Train with whole dataset
    print "Train with the whole dataset"
    model.fit(X, y)
    preds = model.predict_proba(X_test)[:, 1]

    # Save results
    print "Save results"
    save_results(preds, "new_sal.csv")

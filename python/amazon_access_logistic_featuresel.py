__author__ = 'acg'
#Special thanks to Paul Duan and Miroslav Horbal

import numpy as np
import pandas as pd
import scipy as sp
import sklearn as sk
import math
from utils import combs, my_cross_val_score, auc_roc, feature_selection
import logging

# Configuration
SEED = 2892
ktuples_max = 2
feature_sel_stop = 0.00001
feature_remove_worst = 4
preselected = []


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

    # Create combinations of features, up to size k
    for j in range(1, k+1):
        for i in combs(7, j):
            X_plus.append([hash(tuple(v)) for v in X[:, i]])



    X = np.array(X_plus).T
    return X


def create_new_features(X):
    """
    Transform data and create new features
    """

    # Include new variables (combinations)
    logging.info("create combs for X size " + str(X.shape))
    X_c = combine_variables(X, ktuples_max)

    # Create vars by dividing (as integers)
    # "factors" seeking if the nearer groups (i.e Depts)
    # hold also nearer ids
    X_n = []
    for i in range(1,7):
        ix = (0, i)
        X_n.append([hash(tuple(v)) for v in X[:, ix]/10])
        X_n.append([hash(tuple(v)) for v in X[:, ix]/100])
        X_n.append([hash(tuple(v)) for v in X[:, ix]/1000])
        X_n.append([hash(tuple(v)) for v in X[:, ix]/10000])
        X_n.append([hash(tuple(v)) for v in X[:, ix]/100000])



    # To this point, all variables are categorical
    # Explore creating some quantitative features based on
    # people per Department, people per manager, ...


    X_n = np.array(X_n).T
    X = np.hstack([X_c, X_n])

    logging.debug("Dataset dimensions: " + str(X.shape))

    return X


def encode_factors(X, X_test):
    """
    Encode
    """
    # Label encoder (hash values not valid as input for onehotencoder
    labelencoder = sk.preprocessing.LabelEncoder()
    labelencoder.fit(np.vstack([X, X_test]))

    X = labelencoder.transform(X)
    X_test = labelencoder.transform(X_test)

    # Use one hot encoder to codify factors (all) as dummy variables
    encoder = sk.preprocessing.OneHotEncoder()
    encoder.fit(np.vstack([X, X_test]))
    X = encoder.transform(X)  # Returns a sparse matrix (see numpy.sparse)
    X_test = encoder.transform(X_test)

    logging.debug(X.shape)
    logging.debug(X_test.shape)

    return X, X_test


def encode_factors_single(X):
    """
    Encode
    """
    #logging.debug("Input shape: " + str(X.shape))
    # Label encoder (hash values not valid as input for onehotencoder
    labelencoder = sk.preprocessing.LabelEncoder()
    labelencoder.fit(X)

    X = labelencoder.transform(X)

    # Use one hot encoder to codify factors (all) as dummy variables
    encoder = sk.preprocessing.OneHotEncoder()
    encoder.fit(np.vstack([X]))
    X = encoder.transform(X)  # Returns a sparse matrix (see numpy.sparse)

    #logging.debug("Output shape (sparse): " + str(X.shape))
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
        for p in (True, ):
            model.C = C
            model.dual = p

            # Train and Crossvalidate
            scores, _ = my_cross_val_score(model, X, y, auc_roc, 3)
            #print str(scores)
            logging.debug("Mean AUC for p=%s, C=%f: %f" % (p, C, sp.mean(scores)))
            opt_scores.append((sp.mean(scores), p, C))
    opt_scores = sorted(opt_scores, reverse=True)
    logging.debug(opt_scores)
    return opt_scores[0][-1]

if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s',
                        )

    # Model
    #model = linear_model.LogisticRegression(C=5.7, penalty="l2", dual=True)  # the classifier we'll use
    #model = sk.linear_model.LogisticRegression(penalty="l2")  # the classifier we'll use
    #model = sk.linear_model.SGDClassifier(loss="log", penalty="l2")
    model = sk.linear_model.LogisticRegression(penalty="l2", dual=True)
    #model = sk.svm.SVC(probability=True)

    logging.info("Loading train data")
    X, y = load_data("/data/kaggle/amazon_access/train.csv")

    logging.info("Loading test data")
    X_test, _ = load_data("/data/kaggle/amazon_access/test.csv")

    logging.info("Create new features")
    X = create_new_features(X)
    X_test = create_new_features(X_test)

    logging.info("Select best features")
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

    X = X[:, features]
    X_test = X_test[:, features]
    logging.debug(X.shape)

    logging.info("Encoding factors/transform data")
    X, X_test = encode_factors(X, X_test)

    logging.info("Find best params for model (C)")
    model.C = pick_best_params(model, X, y)

    # Get cross_val scores
    logging.info("Cross-validate")
    scores, _ = my_cross_val_score(model, X, y, auc_roc, 10, random_state=SEED*31)
    #print str(scores)
    logging.info("Mean AUC: %f" % (sp.mean(scores)))

    # Train with whole dataset
    logging.info("Train with the whole dataset")
    model.fit(X, y)
    preds = model.predict_proba(X_test)[:, 1]

    # Save results
    logging.info("Save results")
    save_results(preds, "new_sal.csv")

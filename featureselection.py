import numpy as np
from sklearn.feature_selection import mutual_info_classif
from ReliefF import ReliefF
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import FCBF
import su_calculation as su
import MRMR as mr


def infogain(X, y, n_features):
    """Runs infogain feature selection on the data (X) and the target values (y) and finds
    the index of the top n_features number of features.

    Args:
        X (Numpy Array): An array containing the dataset.
        y (Numpy Array): An array consisting of the target values
        n_features (int): An integer specifying how many features should be selected.

    Returns:
        List: Returns a list containing the indices of the features that have been selected.
    """
    score = mutual_info_classif(X, y,random_state=0)
    index = list(np.argsort(list(score))[-1*n_features:])
    index.sort()
    return index

def reliefF(X, y, n_features):
    """Runs ReliefF algorithm on the data (X) and the target values (y) and finds 
    the index of the top n_features number of features.

    Args:
        X (Numpy Array): An array containing the dataset.
        y (Numpy Array): An array consisting of the target values.
        n_features (int): An integer specifying how many features should be selected.

    Returns:
        List: Returns a list containing the indices of the features that have been selected.
    """
    fs = ReliefF(n_neighbors=100, n_features_to_keep=n_features)
    fs.fit_transform(X, y)
    index = fs.top_features[:n_features]
    return index

def fcbf(X, y):
    selection = FCBF.fcbf(X, y)
    index = list(selection[0])#[-1*n_features:]
    index.sort()
    return index

def merit_calculation(X, y):
    """
    This function calculates the merit of X given class labels y, where
    merits = (k * rcf)/sqrt(k+k*(k-1)*rff)
    rcf = (1/k)*sum(su(fi,y)) for all fi in X
    rff = (1/(k*(k-1)))*sum(su(fi,fj)) for all fi and fj in X
    Input
    ----------
    X: {numpy array}, shape (n_samples, n_features)
        input data
    y: {numpy array}, shape (n_samples,)
        input class labels
    Output
    ----------
    merits: {float}
        merit of a feature subset X
    """

    n_samples, n_features = X.shape
    rff = 0
    rcf = 0
    for i in range(n_features):
        fi = X[:, i]
        rcf += su.su_calculation(fi, y)
        for j in range(n_features):
            if j > i:
                fj = X[:, j]
                rff += su.su_calculation(fi, fj)
    rff *= 2
    merits = rcf / np.sqrt(n_features + rff)
    return merits


def cfs(X, y):
   

    n_samples, n_features = X.shape
    F = []
    # M stores the merit values
    M = []
    while True:
        merit = -100000000000
        idx = -1
        for i in range(n_features):
            if i not in F:
                F.append(i)
                # calculate the merit of current selected features
                t = merit_calculation(X[:, F], y)
                if t > merit:
                    merit = t
                    idx = i
                F.pop()
        F.append(idx)
        M.append(merit)
        if len(M) > 6:
            if M[len(M)-1] <= M[len(M)-2]:
                if M[len(M)-2] <= M[len(M)-3]:
                    if M[len(M)-3] <= M[len(M)-4]:
                        if M[len(M)-4] <= M[len(M)-5]:
                            break
  
    return np.array(F)
    

def run_feature_selection(method,X,y,n_features):
    """Runs the specific ranking based feature selection method.

    Args:
        method (String): A string that refers to the feature selection method to be used.
        X (Numpy Array): An array containing the dataset.
        y (Numpy Array): An array consisting of the target values.
        n_features (int): An integer specifying how many features should be selected.

    Returns:
        List: Returns a list containing the indices of the features that have been selected.
    """
    if method == 'infogain':
        return infogain(X,y,n_features)
    elif method == 'reliefF':
        return reliefF(X,y,n_features)
    elif method == 'cfs':
        return cfs(X, y)
    elif method == 'mrmr':
        f =  mr.mrmr(X,y)
 
        return f
    elif method =='fcbf':
        return fcbf(X,y)
 


def sfs(X_train, y_train, estimator, metric): 
    sfs1 = SFS(estimator, k_features=(1,X_train.shape[1]), forward=True, floating=False, scoring=metric, cv=0)
    sfs1 = sfs1.fit(X_train, y_train)
    return sfs1.k_feature_idx_



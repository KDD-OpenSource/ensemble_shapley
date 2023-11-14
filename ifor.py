from sklearn.ensemble import IsolationForest as ifor

import numpy as np

from sklearn.metrics import roc_auc_score

from sklearn.utils import check_array




def _average_path_length(n_samples_leaf):
    """
    The average path length in a n_samples iTree, which is equal to
    the average path length of an unsuccessful BST search since the
    latter has the same structure as an isolation tree.
    Parameters
    ----------
    n_samples_leaf : array-like of shape (n_samples,)
        The number of training samples in each test sample leaf, for
        each estimators.

    Returns
    -------
    average_path_length : ndarray of shape (n_samples,)
    """

    n_samples_leaf = check_array(n_samples_leaf, ensure_2d=False)

    n_samples_leaf_shape = n_samples_leaf.shape
    n_samples_leaf = n_samples_leaf.reshape((1, -1))
    average_path_length = np.zeros(n_samples_leaf.shape)

    mask_1 = n_samples_leaf <= 1
    mask_2 = n_samples_leaf == 2
    not_mask = ~np.logical_or(mask_1, mask_2)

    average_path_length[mask_1] = 0.0
    average_path_length[mask_2] = 1.0
    average_path_length[not_mask] = (
        2.0 * (np.log(n_samples_leaf[not_mask] - 1.0) + np.euler_gamma)
        - 2.0 * (n_samples_leaf[not_mask] - 1.0) / n_samples_leaf[not_mask]
    )

    return average_path_length.reshape(n_samples_leaf_shape)


def find_average_depth(x,tx,**kwargs):
    clf=ifor(**kwargs)
    clf.fit(x)
    #sc=clf.score_samples(tx)
    sc=clf._compute_score_samples(tx.astype(np.float32),False)
    sc=np.log(sc)/np.log(2)
    sc*=-_average_path_length([clf._max_samples])
    return sc

class FB():
    def __init__(self,bag,N):
        self.bag=bag
        self.N=N
        self.which=np.random.choice(np.arange(N),bag,replace=False)
    def predict(self,X):
        assert X.shape[1]==self.N
        return np.stack([X[:,i] for i in self.which],axis=1)

    def __call__(self,X):
        return self.predict(X)

def baggin(x,tx,count=32):
    fb=FB(count,x.shape[1])
    return fb(x),fb(tx),fb.which



if __name__=="__main__":
    
    
    x=np.load("training.npz")["q"]
    f=np.load("test.npz")
    tx,ty=f["q"],f["y"]

    x,tx,which=baggin(x,tx)
    
    
    
    clf=ifor()
    clf.fit(x)
    
    features=clf.estimators_features_
    
    feat=0
    valid=[feat in feats for feats in features]
    
    
    
    tx=tx.astype(np.float32)
    
    scores=-clf.score_samples(tx)
    auc=roc_auc_score(ty,scores)
    print(auc)
    
    
    #scores=_compute_score_samples(clf,tx,True)
    #scores=clf._compute_score_samples(tx,True)
    #scores=clf._compute_chunked_score_samples(tx)
    scores=-find_average_depth(x,tx)
    auc=roc_auc_score(ty,scores)
    print(auc)
    
    exit()
    
    clf.estimators_=[zw for zw,val in zip(clf.estimators_,valid) if val]
    clf.estimators_features_=[zw for zw,val in zip(clf.estimators_features_,valid) if val]
    clf.estimators_samples_=[zw for zw,val in zip(clf.estimators_samples_,valid) if val]
    scores=-clf.score_samples(tx)
    auc=roc_auc_score(ty,scores)
    print(auc)
    
    
    
    
                                                                                                                                                                                                                                    

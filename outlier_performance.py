import numpy as np
from sklearn.metrics import roc_auc_score


f=np.load("thefile.npz")
#f=np.load("smallfile.npz")

preds=f['preds']
preds=np.mean(preds,axis=0)
print("loaded preds")

f=np.load("test.npz")
ty=f['y']
print("loaded ty")


auc=roc_auc_score(ty,preds)
print(auc)




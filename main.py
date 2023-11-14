import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

import os
import sys
from time import time

from ifor import find_average_depth

trainfile="training.npz"
testfile="test.npz"

#trainfile="playing.npz"

bag=32


indice=0
if len(sys.argv)>1:
    indice=int(sys.argv[1])


x=np.load(trainfile)["q"]
tx=np.load(testfile)["q"]


def feature_bagging(x,tx,bag=bag):
    #chooses a random subset of bag features, returns x', tx', features

    features=np.random.choice(int(x.shape[1]),bag,replace=False)

    x=np.stack([x[:,i] for i in features],axis=-1)
    tx=np.stack([tx[:,i] for i in features],axis=-1)

    return x,tx,features


def train_model(x,tx):
    #trains a model, returns the normalized predictions on tx

    pred=find_average_depth(x,tx,n_estimators=1)

    return pred,1.0






def one_model():
    xp,txp,feat=feature_bagging(x,tx,bag=bag)

    pred, maximum=train_model(xp,txp)

    return feat, pred, maximum

save_interval=5000
dex=0
os.makedirs("results",exist_ok=True)

feats,preds,maxima=[],[],[]

t0=time()

while True:
    try:
        feat,pred,maximum=one_model()
    except Exception:
        continue
    feats.append(feat)
    preds.append(pred)
    maxima.append(maximum)

    if len(preds)>=save_interval:
        filenam=None
        while filenam is None or os.path.isfile(filenam):
            filenam=f"results/{indice}_{dex}.npz"
            dex+=1
        np.savez_compressed(filenam,feats=feats,preds=preds,maxima=maxima)
        feats=[]
        preds=[]
        maxima=[]
    
    print(f"did {dex*save_interval+len(feats)} models under {indice}")
    timep=(time()-t0)/(dex*save_interval+len(feats))
    print(f"Training time/Model: {timep}, {timep*save_interval}")
















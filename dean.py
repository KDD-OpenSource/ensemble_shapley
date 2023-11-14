import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

import os
import sys
from time import time


trainfile="take.npz"
testfile="anti.npz"

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

    inp=keras.Input(x.shape[1:])
    q=inp
    
    for i in range(3):
        q=keras.layers.Dense(100,use_bias=False,activation="relu")(q)
    q=keras.layers.Dense(1,use_bias=False,activation="linear")(q)

    loss=keras.losses.mse(q,K.ones_like(q))

    model=keras.Model(inp,q)
    model.add_loss(loss)

    model.compile("adam")

    cb=[keras.callbacks.EarlyStopping(patience=5,restore_best_weights=True), keras.callbacks.TerminateOnNaN()]

    model.fit(x,None,
                epochs=100,
                batch_size=100,
                verbose=1,
                validation_split=0.2,
                callbacks=cb)

    pred=model.predict(x)
    mean=np.mean(pred)
    
    delta=np.mean((pred-mean)**2,axis=1)
    maximum=np.max(delta)
    
    ptx=model.predict(tx)
    score=np.mean((ptx-mean)**2,axis=1)
    return score/(maximum+0.0000000000001), maximum






def one_model():
    xp,txp,feat=feature_bagging(x,tx,bag=bag)

    pred, maximum=train_model(xp,txp)

    return feat, pred, maximum

save_interval=50
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
    print(f"Training time/Model: {timep}")
















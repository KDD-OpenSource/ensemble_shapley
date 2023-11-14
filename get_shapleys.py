import numpy as np
from plt import *
from tqdm import tqdm

from scipy.special import digamma

def correction(N,bag):
    return (N/(N-bag))*(digamma(N+1)-digamma(bag+1))


f=np.load("thefile.npz")


feats,preds=f['feats'],f['preds']

avg=np.mean(preds,axis=0)
#print(feats.shape,preds.shape)
#exit()

dimension=(28,28)

maximumdim=np.prod(dimension)

bag=int(feats.shape[1])

alpha=correction(maximumdim,bag)

avg=np.expand_dims(avg,axis=1)
avg=np.expand_dims(avg,axis=1)
avg=np.repeat(avg,dimension[0],axis=1)
avg=np.repeat(avg,dimension[1],axis=2)


by_feat=[[] for i in range(maximumdim)]

for feat,pred in zip(tqdm(feats),preds):
    for f in feat:
        by_feat[f].append(pred)

lens=[len(x) for x in by_feat]
print("max",max(lens))
print("min",min(lens))
print("mean",np.mean(lens))
print("median",np.median(lens))
print("zeros",lens.count(0),lens.count(0)/len(lens))

if lens.count(0)>0:
    print("found zeros")
    exit()

#by_feat=[zw if zw else np.zeros((1,6856)) for zw in by_feat]

means=[np.mean(x,axis=0) for x in by_feat]

image=np.reshape(means,list(dimension)+[-1])
image=np.transpose(image,(2,0,1))

q=alpha*(avg-image)


np.savez_compressed("shapleys.npz",q=q)







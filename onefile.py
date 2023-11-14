import os
import numpy as np
from tqdm import tqdm

pth="results/"

files=os.listdir(pth)
counts=0

print(f"found {len(files)} files")

feats,preds,maximas=[],[],[]

lens=0
for i,fil in enumerate(tqdm(files)):
    f=np.load(pth+fil)
    feats.append(f["feats"])
    preds.append(f["preds"])
    maximas.append(f["maxima"])
    lens+=len(feats[-1])
    if lens>1_000_000:break

print("concatting")
feats=np.concatenate(feats,axis=0)
preds=np.concatenate(preds,axis=0)
maximas=np.concatenate(maximas,axis=0)

print("saving")
np.savez_compressed("thefile.npz",feats=feats,preds=preds,maximas=maximas)


import numpy as np
from plt import *
from tqdm import tqdm


print("loading data")
fs=np.load("shapleys.npz")
ft=np.load("test.npz")

s=fs["q"]
print("loaded shapley values")
img=ft["q"]
y=ft["y"]
print("loaded original images")

dimension=(28,28)

dex=0
for dex in tqdm(range(len(y)),total=len(y)):
    plt.close()
    plt.figure(figsize=(10,6))
    plt.subplot(1,2,1)
    plt.imshow(np.reshape(img[dex],dimension),cmap="gray")
    plt.title("Original Image, "+("normal" if y[dex]==0 else "abnormal"))
    plt.subplot(1,2,2)
    ss=s[dex]
    #ss=np.reshape(ss,np.prod(dimension))
    #ss=np.reshape(ss,dimension[::-1])
    #ss=np.transpose(ss)
    plt.imshow(ss,cmap="hot")
    plt.title("Shapley Values")
    plt.savefig(f"fig/{dex}.png")
    plt.savefig(f"fig/{dex}.pdf")
    if not dex:plt.show()









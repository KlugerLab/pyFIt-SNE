import fitsne
import numpy as np

X = np.random.normal(0, 1, (10000, 10))
X = np.concatenate([X + 5, X - 5, X * 0.5 - 1])
Y = fitsne.FItSNE(X, fft_not_bh=True, ann_not_vptree=True)
#Y = fitsne.FItSNE(X, fft_not_bh=False, ann_not_vptree=False) #Test the barnes-hut


#import matplotlib.pyplot as plt
#plt.scatter(Y[:, 0], Y[:, 1])
#plt.show()

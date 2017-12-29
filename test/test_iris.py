from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import fitsne


iris = load_iris()
X = iris.data

Y = fitsne.FItSNE(X, fft_not_bh=True, ann_not_vptree=True)
plt.scatter(Y[:, 0], Y[:, 1], c=iris.target)
plt.show()

from sklearn.datasets import load_iris
import fitsne


iris = load_iris()
X = iris.data

Y = fitsne.FItSNE(X)
try:
    import matplotlib.pyplot as plt
    plt.scatter(Y[:, 0], Y[:, 1], c=iris.target)
    plt.show()
except ImportError:
    print("Not plotting, because matplotlib is not installed")

import mglearn
import sklearn
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

mglearn.plots.plot_knn_classification(n_neighbors=1)
plt.show()

mglearn.plots.plot_knn_classification(n_neighbors=3)
plt.show()

import mglearn
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import make_blobs
# generate dataset
#
X, y = mglearn.datasets.make_forge()

# plot dataset

mglearn.discrete_scatter(X[:,0], X[:, 1], y)
plt.legend(["Class 0", "Class 1"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
print("X.shape:", X.shape)

plt.show()

X, y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X,y,'o')
plt.ylim(-3,3)
plt.xlabel("Feature")
plt.ylabel("Target")

plt.show()

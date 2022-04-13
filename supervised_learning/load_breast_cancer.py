from sklearn.datasets import load_breast_cancer
import mglearn
import matplotlib.pyplot as plt
import numpy as np

cancer = load_breast_cancer()
print("cancer.keys():\n", cancer.keys())
print("Shape of cancer data:", cancer.data.shape)

print("Sample counts per class:\n", {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))})

print("Feature names:\n", cancer.feature_names)

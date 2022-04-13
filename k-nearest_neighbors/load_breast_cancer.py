import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import mglearn
import pandas as pd

cancer = load_breast_cancer()
print("Visualizing data from load_breast_cancer datasets")
print("Keys of breast_cancer_dataset:\n", cancer.keys())
print(cancer['DESCR'][:569] + "\n...")
print("Target names:", cancer['target_names'])
print("Feature names:", cancer['feature_names'])
print("Type of data:", type(cancer['data']))
print("Shape of data:", cancer['data'].shape)
print("First five rows of data:\n", cancer['data'][:5])
print("Shape of target:", cancer['target'].shape)
print("Target:\n", cancer['target'])

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=66)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# create dataframe from data in X_train
cancer_dataframe = pd.DataFrame(X_train, columns=cancer.feature_names)
# create a scatter matrix from the dataframe, color by y_train
pd.plotting.scatter_matrix(cancer_dataframe, c=y_train, figsize=(15,15), marker = 'o', hist_kwds={'bins':20}, s=60, alpha=.8, cmap=mglearn.cm3)

plt.show()

training_accuracy = []
test_accuracy = []
# try n_neighbors from 1 to 10

neighbors_settings = range(1,11)

for n_neighbors in neighbors_settings:
    #build the model
    clf = KNeighborsClassifier(n_neighbors=n_neighbors) 
    clf.fit(X_train, y_train)
    # record training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    # record generalization accuracy
    test_accuracy.append(clf.score(X_test,y_test))

plt.plot(neighbors_settings, training_accuracy, label="training_accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test_accuracy")
plt.ylabel("accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()

mglearn.plots.plot_knn_regression(n_neighbors=1)
plt.show()
mglearn.plots.plot_knn_regression(n_neighbors=3)
plt.show()

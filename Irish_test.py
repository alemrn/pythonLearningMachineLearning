from sklearn.datasets import load_iris


iris_dataset = load_iris()
print("Keys of iris_dataset:\n" , iris_dataset.keys())

print(iris_dataset['DESCR'][:5] + "\n ...")

# The value of the key target_name is an array of strings, 
# containing the spacies of flower that we want to predict:

print (iris_dataset['target_names'])


print(iris_dataset['feature_names'])

print(type(iris_dataset['data']))

print("shape of data:", iris_dataset['data'].shape)

print("First five rows of data:\n", iris_dataset['data'][:5])

print()
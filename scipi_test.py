import numpy as np
from scipy import sparse

eye = np.eye(4)
print("Numpy array: \n", eye)

sparse_matrix=sparse.csr_matrix(eye)
print("\n SciPy sparse CSR matrix\n", sparse_matrix)

data=np.ones(4)
row_indices=np.arange(4)
col_indices=np.arange(4)
eye_coo =sparse.coo_matrix((data, (row_indices, col_indices)))
print("COO representation:\n", eye_coo)
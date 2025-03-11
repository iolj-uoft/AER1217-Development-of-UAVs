import numpy as np

# Create a 4x4 matrix
matrix = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12],
                   [13, 14, 15, 16]])

# Using :3, :3 to select the first three rows and columns
submatrix1 = matrix[:3, :3]

# Incorrect usage of [:3][:3]
# This will first select the first three rows, and then select the first three rows again from the resulting array
submatrix2 = matrix[:2][:2]

print("Original matrix:")
print(matrix)
print("\nSubmatrix using :3, :3:")
print(submatrix1)
print("\nSubmatrix using [:3][:3]:")
print(submatrix2)

# Ali Alzurufi
# Professor Lauren
# Date: September 10 2023
# MCS 5223: Text Mining and Data Analytics

""" Description: This program will allow the user to input values for a 2 x 2 matrix and
will return the user's matrix, the transpose of the matrix, the eiganvalues, and the eigenvectors """


import numpy as np


# Class for a vector containing methods for computing the dot product and the magnitude
class Vector:
    def __init__(self, values):
        self.values = np.array(values)

    def dot(self, other):
        return np.dot(self.values, other)

    def magnitude(self):
        return np.linalg.norm(self.values)


# Class for a matrix containing methods for mulitiplying two matrices and computing the transpose
class Matrix:
    def __init__(self, values):
        self.values = np.array(values).reshape(2, 2)

    def multiply(self, other):
        return np.multiply(self.values, other)

    def transpose(self):
        return np.transpose(self.values)


# Function for computing the eigenvalues and eigenvectors of a 2 x 2 matrix
def find_eigen(matrix):
    matrix = np.array(matrix).reshape(2, 2)

    evals, evects = np.linalg.eig(matrix)

    return evals, evects


# Main function takes a user input for a 2 x 2 matrix and display the user's matrix, the tranpose, the eigenvalues, and the eigenvectors
def main():
    user_input = input("Enter your 2 x 2 matrix values separated by spaces: ").split()
    print()
    user_matrix = list(map(int, user_input))

    matrix = Matrix(user_matrix)

    print(f"Your 2 x 2 matrix: \n{matrix.values} \n")

    print(f"Transpose: \n{matrix.transpose()} \n")

    eigenvalues, eigenvectors = find_eigen(matrix.values)

    print(f"Eigenvalues: \n{eigenvalues} \n")

    print(f"Eigenvectors: \n{eigenvectors}")


main()

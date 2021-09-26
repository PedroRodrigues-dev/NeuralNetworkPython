import random


class Matrix:
    """Class to work with matrix"""

    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols

        self.data = []

        for i in range(rows):
            arr = []
            for j in range(cols):
                arr.append(0)
            self.data.append(arr)

    """convert array to matrix"""

    @staticmethod
    def arrayToMatrix(arr):
        matrix = Matrix(len(arr), 1)
        for i in range(matrix.rows):
            for j in range(matrix.cols):
                matrix.data[i][j] = arr[i]
        return matrix

    """convert matrix to array"""

    @staticmethod
    def matrixToArray(matrix):
        arr = []
        for i in range(matrix.rows):
            for j in range(matrix.cols):
                arr.append(matrix.data[i][j])
        return arr

    """insert random values ​​into the matrix"""

    def randomize(self):
        for i in range(self.rows):
            for j in range(self.cols):
                self.data[i][j] = random.random() * 2 - 1

    @staticmethod
    def transpose(A):
        matrix = Matrix(A.cols, A.rows)

        for i in range(A.cols):
            for j in range(A.rows):
                matrix.data[i][j] = A.data[j][i]

        return matrix

    """Operations"""

    """Matrix x Scalar"""

    @staticmethod
    def scalar_multiply(A, scalar):
        matrix = Matrix(A.rows, A.cols)

        for i in range(A.rows):
            for j in range(A.cols):
                matrix.data[i][j] = A.data[i][j] * scalar

        return matrix

    """Hadamard between matrices"""

    @staticmethod
    def hadamard(A, B):
        matrix = Matrix(A.rows, A.cols)

        for i in range(A.rows):
            for j in range(B.cols):
                matrix.data[i][j] = A.data[i][j] * B.data[i][j]

        return matrix

    """Sum between matrices"""

    @staticmethod
    def add(A, B):
        matrix = Matrix(A.rows, A.cols)

        for i in range(A.rows):
            for j in range(B.cols):
                matrix.data[i][j] = A.data[i][j] + B.data[i][j]

        return matrix

    """Subtraction between matrices"""

    @staticmethod
    def subtract(A, B):
        matrix = Matrix(A.rows, A.cols)

        for i in range(A.rows):
            for j in range(B.cols):
                matrix.data[i][j] = A.data[i][j] - B.data[i][j]

        return matrix

    """Product between matrices"""

    @staticmethod
    def multiply(A, B):
        matrix = Matrix(A.rows, B.cols)

        for i in range(A.rows):
            for j in range(B.cols):
                sum = 0
                for k in range(A.cols):
                    elm1 = A.data[i][k]
                    elm2 = B.data[k][j]
                    sum += elm1 * elm2
                    matrix.data[i][j] = sum

        return matrix

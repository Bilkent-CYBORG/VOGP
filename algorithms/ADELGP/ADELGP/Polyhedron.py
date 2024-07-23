import numpy as np
import cvxpy as cp


class Polyhedron:
    # TODO: Change to hyperrectangle class

    def __init__(self, A=None, b=None):
        """
        Polyhedron in the format P = {x: Ax > b}
        :param A:
        :param b:
        """
        self.A = A
        self.b = b

        if A is None:
            self.A = None
            self.b = None
            self.lower = None
            self.upper = None

        if self.b is not None:
            self.lower = self.b[0:2]
            self.upper = -self.b[2:4]

    def diameter(self):
        return np.linalg.norm(self.upper - self.lower)

    def __str__(self):
        if self.A is None:
            return "empty"
        else:
            return "A \n" + np.array2string(self.A) + "\nb \n" + np.array2string(self.b)

    def __eq__(self, other):
        return self.A == other.A and self.b == other.b

import numpy as np
import itertools
import matplotlib.pyplot as plt
"""
Represents a hypercube context region.
"""


class Hyperrectangle:
    def __init__(self, lower, upper):
        self.upper = upper
        self.lower = lower
        
        self.dim = len(self.upper)
        self.diameter = np.linalg.norm(np.array(upper)-np.array(lower))

        self.A = self.setA()
        self.b = self.setb()

    def setA(self):
        A = np.vstack((np.eye(self.dim), -np.eye(self.dim)))
        return A

    def setb(self):
        b = np.hstack((np.array(self.lower), -np.array(self.upper)))
        return b

    def intersect(self, rect):
        lower_new = []
        upper_new = []

        if self.check_intersect(rect):  # if the two rectangles overlap
            for l1, l2 in zip(self.lower, rect.get_lower()):
                lower_new.append(max(l1, l2))

            for u1, u2 in zip(self.upper, rect.get_upper()):
                upper_new.append(min(u1, u2))

            return Hyperrectangle(lower_new, upper_new)
        else:
            # if there is no intersection,then use the new hyperrectangle
            # return Hyperrectangle(rect.get_lower(), rect.get_upper())
            for l1, l2 in zip(self.lower, rect.get_lower()):
                lower_new.append(min(l1, l2))

            for u1, u2 in zip(self.upper, rect.get_upper()):
                upper_new.append(max(u1, u2))

            return Hyperrectangle(lower_new, upper_new)

    def get_lower(self):
        return np.array(self.lower)

    def get_upper(self):
        return np.array(self.upper)

    def __str__(self):
        return "Upper: " + str(np.round(self.upper, 4)) + ", Lower: " + str(np.round(self.lower, 4))

    def get_vertices(self):
        a = [[l1, l2] for l1, l2 in zip(self.lower, self.upper)]
        vertex_list = [element for element in itertools.product(*a)]
        return np.array(vertex_list)

    def check_intersect(self, rect):
        for i in range(self.dim):
            if self.lower[i] >= rect.upper[i] or self.upper[i] <= rect.lower[i]:
                return False
        
        return True

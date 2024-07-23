from __future__ import annotations

from itertools import product
from typing import Optional, List

import numpy as np

from algorithms.ADELGP.ADELGP.Hyperrectangle import Hyperrectangle


class DesignPoint:
    def __init__(self, x: np.ndarray, R: Hyperrectangle, design_index: int):
        self.x = x
        self.R = R  # The confidence region (Hyperrectangle)
        self.design_index = design_index

        self.d = len(x)
        self.m = self.R.dim

    def __eq__(self, other):
        return (self.x == other.x).all()

    def __str__(self):
        name = "\nDesign Point: x " + str(np.round(self.x, 2)) +\
               "\nHyperrectangle" + str(self.R)
        return name

class DesignPointAD(DesignPoint):
    def __init__(
        self, x: np.ndarray, R: Hyperrectangle, design_index: int,
        depth: Optional[int] = None, cell_bound: List[List] = None,
        parent_x: Optional[DesignPointAD] = None
    ):
        super().__init__(x, R, design_index)

        self.depth = depth
        self.cell_bound = cell_bound
        if parent_x:
            parent_x.parent_x = None
        self.parent_x = parent_x

    def get_child_list(self):  # returns a list of children design_points
        options = []
        for dim_i in range(self.d):
            options.append([
                [
                    self.cell_bound[dim_i][0],
                    (self.cell_bound[dim_i][0] + self.cell_bound[dim_i][1])/2
                ],
                [
                    (self.cell_bound[dim_i][0] + self.cell_bound[dim_i][1])/2,
                    self.cell_bound[dim_i][1]
                ]
            ])
        new_bounds = list(map(list, product(*options)))

        chld_cnt = 2**self.d
        list_children = []
        for ind, bound in enumerate(new_bounds, 1):
            x = np.array(bound, dtype=float).mean(axis=1)
            child = DesignPointAD(
                x=x, R=self.R, design_index=(self.design_index-1)*chld_cnt+ind, depth=self.depth+1,
                cell_bound=bound, parent_x=self
            )
            list_children.append(child)  # TODO: fix design_index here

        return list_children

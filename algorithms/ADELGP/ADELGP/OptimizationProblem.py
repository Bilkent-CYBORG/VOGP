from typing import List, Union

import numpy as np

from algorithms.ADELGP.ADELGP.DesignPoint import DesignPoint
from utils.dataset import Dataset, ContinuousDataset


class OptimizationProblem:
    def __init__(self, dataset: Union[Dataset, ContinuousDataset], obs_noise_var=0):
        self.dataset = dataset

        if isinstance(dataset, ContinuousDataset):
            self.func = dataset.evaluate
        else:
            self.func = None
            self.x = dataset.in_data
            self.y = dataset.out_data
            self.cardinality = len(self.x)

        self.obs_noise_var = obs_noise_var
        self.obs_noise_std = np.sqrt(self.obs_noise_var)

    def __call__(self, points: Union[List[DesignPoint], DesignPoint], remove_noise=False):
        if isinstance(points, DesignPoint):
            points = [points]

        if self.func:
            x_val = np.vstack([point.x for point in points])
            y_val = self.func(x_val)
        else:
            indices = [point.design_index for point in points]
            y_val = self.y[indices]
        if not remove_noise:  # Noise is independent for each objective
            y_val += self.obs_noise_std * np.random.randn(*y_val.shape)
        
        return y_val

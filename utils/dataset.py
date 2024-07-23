import os
from math import pi

import torch

import numpy as np
import gpytorch.kernels
from sklearn.preprocessing import StandardScaler, MinMaxScaler

try:
    from utils.utils import (
        get_delta, get_cone_params, get_preference_cone,
        get_alpha_vec, compute_ice_cream_params, generate_domain_pts
    )
except:
    from utils import (
        get_delta, get_cone_params, get_preference_cone,
        get_alpha_vec, compute_ice_cream_params, generate_domain_pts
    )

### CONTINUOUS DATASETS ###

class ContinuousDataset:
    def __init__(self, cone_degree):
        self.cone_degree = cone_degree
        if isinstance(self.cone_degree, int):
            self.W, self.alpha_vec = get_cone_params(self.cone_degree, self.out_dim)
        elif self.cone_degree[0] == "P1":
            self.W = get_preference_cone(self.out_dim)
            self.alpha_vec = get_alpha_vec(self.W)
        elif self.cone_degree[0] == "theta":
            self.W, self.alpha_vec = get_cone_params(self.cone_degree[1], dim=self.out_dim)
        elif self.cone_degree[0] == "W":
            self.W = np.array(self.cone_degree[1])
            self.alpha_vec = get_alpha_vec(self.W)
        elif self.cone_degree[0] == "ice_cream":
            self.W, self.alpha_vec = compute_ice_cream_params(self.cone_degree[1])
        elif self.cone_degree[0] == "A":
            raise NotImplementedError  # get_preference_cone with sepcific A

        self.model_kernel = None
    
    def evaluate(self, points):
        raise NotImplementedError

class BraninCurrin(ContinuousDataset):
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    in_dim = 2
    out_dim = 2
    domain_discretization_each_dim = 33

    def __init__(self, cone_degree):
        super().__init__(cone_degree)

        self.model_kernel = gpytorch.kernels.RBFKernel
        self.max_discretization_depth = 5
    
    def _branin(self, X):
        x_0 = 15 * X[..., 0] - 5
        x_1 = 15 * X[..., 1]
        X = np.stack([x_0, x_1], axis=1)

        t1 = (
            X[..., 1]
            - 5.1 / (4 * np.pi**2) * X[..., 0] ** 2
            + 5 / np.pi * X[..., 0]
            - 6
        )
        t2 = 10 * (1 - 1 / (8 * np.pi)) * np.cos(X[..., 0])
        return t1**2 + t2 + 10

    def _currin(self, X):
        x_0 = X[..., 0]
        x_1 = X[..., 1]
        x_1[x_1 == 0] += 1e-9
        factor1 = 1 - np.exp(-1 / (2 * x_1))
        numer = 2300 * np.power(x_0, 3) + 1900 * np.power(x_0, 2) + 2092 * x_0 + 60
        denom = 100 * np.power(x_0, 3) + 500 * np.power(x_0, 2) + 4 * x_0 + 20
        return factor1 * numer / denom

    def evaluate(self, points):
        branin = self._branin(points)
        currin = self._currin(points)

        # Normalize the results
        branin = (branin - 54.3669) / 51.3086
        currin = (currin - 7.5926) / 2.6496
        
        Y = np.stack([-branin, -currin], axis=1)
        return Y

class ZDT3(ContinuousDataset):
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    in_dim = 2
    out_dim = 2
    domain_discretization_each_dim = 33

    def __init__(self, cone_degree):
        super().__init__(cone_degree)

        self.model_kernel = gpytorch.kernels.RBFKernel
        self.max_discretization_depth = 5

    def evaluate_true(self, X):
        f_0 = X[..., 0]
        g = self._g(X=X)
        f_1 = 1 - (f_0 / g).sqrt() - f_0 / g * torch.sin(10 * pi * f_0)
        return torch.stack([f_0, f_1], dim=-1)

    @staticmethod
    def _g(X):
        return 1 + 9 * X[..., 1:].mean(dim=-1)

    def evaluate(self, points):
        Y = -self.evaluate_true(torch.from_numpy(points)).cpu().detach().numpy()
        return Y


### DISCRETE DATASETS ###

DATASET_SIZES = {
    "BC500": 500,
    "SnAr": 2000,
    "VehicleSafety": 500,
    "Lactose": 250,
}

class Dataset:
    def __init__(self, cone_degree, data_process=True):
        if data_process:
            # Standardize
            input_scaler = MinMaxScaler()
            self.in_data = input_scaler.fit_transform(self.in_data)

            output_scaler = StandardScaler(with_mean=True, with_std=True)
            self.out_data = output_scaler.fit_transform(self.out_data)
        
        self.in_dim = len(self.in_data[0])
        self.out_dim = len(self.out_data[0])

        self.cone_degree = cone_degree
        if isinstance(self.cone_degree, int):
            self.W, self.alpha_vec = get_cone_params(self.cone_degree, self.out_dim)
        elif self.cone_degree[0] == "P1":
            self.W = get_preference_cone(self.out_dim)
            self.alpha_vec = get_alpha_vec(self.W)
        elif self.cone_degree[0] == "theta":
            self.W, self.alpha_vec = get_cone_params(self.cone_degree[1], dim=self.out_dim)
        elif self.cone_degree[0] == "W":
            self.W = np.array(self.cone_degree[1])
            self.alpha_vec = get_alpha_vec(self.W)
        elif self.cone_degree[0] == "ice_cream":
            self.W, self.alpha_vec = compute_ice_cream_params(self.cone_degree[1])
        elif self.cone_degree[0] == "A":
            raise NotImplementedError  # get_preference_cone with sepcific A

        self.pareto_indices = None
        self.pareto = None
        self.delta = None

        self.f1label=r'$f_1$'
        self.f2label=r'$f_2$'
    
    def set_pareto_indices(self):
        self.find_pareto()

    def find_pareto(self):
        """
        Find the indices of Pareto designs (rows of out_data)
        :param mu: An (n_points, D) array
        :param W: (n_constraint,D) ndarray
        :param alpha_vec: (n_constraint,1) ndarray of alphas of W
        :return: An array of indices of pareto-efficient points.
        """
        out_data = self.out_data.copy()

        out_data = out_data @ self.W.T
        
        is_efficient = np.arange(out_data.shape[0])
        
        next_point_index = 0  # Next index in the is_efficient array to search for
        while next_point_index < len(out_data):
            nondominated_point_mask = np.zeros(out_data.shape[0], dtype=bool)
            vj = out_data[next_point_index].reshape(-1,1)
            for i in range(len(out_data)):
                vi = out_data[i].reshape(-1, 1)
                nondominated_point_mask[i] = not ((vj - vi) >= 0).all()

            nondominated_point_mask[next_point_index] = True
            is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
            out_data = out_data[nondominated_point_mask]
            next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
        
        self.pareto_indices = is_efficient

    def get_params(self):
        if self.delta is None:
            self.delta = get_delta(self.out_data, self.W, self.alpha_vec)
        return self.delta, self.pareto_indices

class ContinuousDatasetWrapper(Dataset):
    def __init__(self, cont_dataset: ContinuousDataset, manual_discretization=None):
        discretization = cont_dataset.domain_discretization_each_dim
        if manual_discretization is not None:
            discretization = manual_discretization
        pts = generate_domain_pts(cont_dataset.bounds, discretization)

        self.in_data = pts
        self.out_data = cont_dataset.evaluate(pts)
        if hasattr(cont_dataset, "_ref_point"):
            self._ref_point = cont_dataset._ref_point

        super().__init__(cont_dataset.cone_degree, data_process=False)

        self.model_kernel = cont_dataset.model_kernel

        self.set_pareto_indices()
        self.pareto = self.in_data[self.pareto_indices]

class BC500(Dataset):
    def __init__(self, cone_degree):
        data = np.load(
            os.path.join('data', 'bc500.npy'), allow_pickle=True
        )
        self.in_data = data[:, :2]
        self.out_data = data[:, 2:]

        super().__init__(cone_degree)

        self.model_kernel = gpytorch.kernels.RBFKernel

        self.set_pareto_indices()
        self.pareto = self.in_data[self.pareto_indices]

class SnAr(Dataset):
    def __init__(self, cone_degree):
        data = np.load(
            os.path.join('data', 'SnAr.npy'), allow_pickle=True
        )
        self.in_data = data[:, :4]
        self.out_data = data[:, 4:]

        super().__init__(cone_degree)

        self.model_kernel = gpytorch.kernels.RBFKernel

        self.set_pareto_indices()
        self.pareto = self.in_data[self.pareto_indices]

class VehicleSafety(Dataset):
    def __init__(self, cone_degree):
        data = np.load(
            os.path.join('data', 'VehicleSafety.npy'), allow_pickle=True
        )
        self.in_data = data[:, :5]
        self.out_data = data[:, 5:]

        super().__init__(cone_degree)

        self.model_kernel = gpytorch.kernels.RBFKernel

        self.set_pareto_indices()
        self.pareto = self.in_data[self.pareto_indices]

class Lactose(Dataset):
    def __init__(self, cone_degree):
        data = np.load(
            os.path.join('data', 'Lactose.npy'), allow_pickle=True
        )
        self.in_data = data[:, :2]
        self.out_data = data[:, 2:]

        super().__init__(cone_degree)

        self.model_kernel = gpytorch.kernels.RBFKernel

        self.set_pareto_indices()
        self.pareto = self.in_data[self.pareto_indices]

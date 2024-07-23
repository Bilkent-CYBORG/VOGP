import logging

import numpy as np

import torch
import gpytorch
from botorch.fit import fit_gpytorch_model

torch.set_default_dtype(torch.float64)


class MultitaskExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, input_X, input_Y, likelihood, kernel, constant_mean):
        if not isinstance(input_X, torch.Tensor):
            input_X = torch.tensor(input_X, dtype=torch.float64)
        if not isinstance(input_Y, torch.Tensor):
            input_Y = torch.tensor(input_Y, dtype=torch.float64)

        input_dim = input_X.shape[-1]
        output_dim = input_Y.shape[-1]

        super().__init__(input_X, input_Y, likelihood)

        if constant_mean:
            self.mean_module = gpytorch.means.MultitaskMean(
                gpytorch.means.ConstantMean(), num_tasks=output_dim
            )
        else:
            self.mean_module = gpytorch.means.MultitaskMean(
                gpytorch.means.ZeroMean(), num_tasks=output_dim
            )

        # self.base_covar_module = gpytorch.kernels.ScaleKernel(kernel(ard_num_dims=input_dim))
        self.base_covar_module = kernel(
            ard_num_dims=input_dim,
            # lengthscale_prior=gpytorch.priors.GammaPrior(3.0, 6.0)
        )  # Scale kernel is unnecessary
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            self.base_covar_module, num_tasks=output_dim, rank=output_dim
        )

        self.mean_module.requires_grad_(True)
        self.covar_module.requires_grad_(True)
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

class GPModel:
    def __init__(self, input_dim, output_dim, noise_var, kernel, constant_mean=False):
        super().__init__()

        self.device = 'cpu'

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.constant_mean = constant_mean

        # Data containers.
        self.clear_data()

        # Set up likelihood
        self.noise_var = torch.tensor(noise_var)
        if self.noise_var.dim() > 1:
            self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                num_tasks=self.output_dim,
                rank=len(self.noise_var),
                noise_constraint=gpytorch.constraints.GreaterThan(1e-10),
                has_global_noise=False
            ).to(self.device)
            self.likelihood.task_noise_covar = self.noise_var
        else:
            self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                num_tasks=self.output_dim,
                rank=0,
                noise_constraint=gpytorch.constraints.GreaterThan(1e-10),
                has_task_noise=False
            ).to(self.device)
            self.likelihood.noise = self.noise_var
        self.likelihood.requires_grad_(False)

        self.kernel = kernel
        self.model = None

    def to_tensor(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float64).to(self.device)
        
        return data

    def add_sample(self, X_t, Y_t):
        # Last column of X_t are sample space indices.
        X_t = self.to_tensor(X_t[..., :self.input_dim])
        Y_t = self.to_tensor(Y_t)

        self.X_T = torch.cat([self.X_T, X_t], 0)
        self.Y_T = torch.cat([self.Y_T, Y_t], 0)

    def clear_data(self):
        self.X_T = torch.empty((0, self.input_dim)).to(self.device)
        self.Y_T = torch.empty((0, self.output_dim)).to(self.device)

    def update(self):
        # Create the model with new data and sparsity setting.
        if self.model == None:
            self.model = MultitaskExactGPModel(
                self.X_T, self.Y_T, self.likelihood, self.kernel, constant_mean=self.constant_mean
            ).to(self.device)

            # lengthscales = torch.tensor([1.0, 5.0], requires_grad=False)
            # multitask_matrix = np.array([[1, 0], [0, 3]])  # Diagonal ones will be added.
            # evalues, evectors = np.linalg.eig(multitask_matrix)
            # assert (evalues >= 0).all()
            # sqrt_matrix = evectors * np.sqrt(evalues) @ np.linalg.inv(evectors)

            # self.model.covar_module.data_covar_module.lengthscale = lengthscales
            # self.model.covar_module.data_covar_module.raw_lengthscale.requires_grad_(False)

            # self.model.covar_module.task_covar_module.var = torch.ones(2, requires_grad=False)
            # self.model.covar_module.task_covar_module.raw_var.requires_grad_(False)

            # self.model.covar_module.task_covar_module.covar_factor = torch.nn.Parameter(
            #     torch.tensor(sqrt_matrix, requires_grad=False), requires_grad=False
            # )

            # print("MODEL PARAMETERS")
            # for name, param in self.model.named_parameters():
            #     if param.requires_grad:
            #         print(f"{name}, {param.data}")
        else:
            self.model.set_train_data(self.X_T, self.Y_T, strict=False)

        self.model.eval()
        self.likelihood.eval()

    def train(self):
        self.model.train()
        self.likelihood.train()

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        print("Training started.")
        fit_gpytorch_model(mll)
        print("Training done.")

        self.model.eval()
        self.likelihood.eval()

    def predict(self, test_X):
        # test_X = self.input_scaler.transform(test_X)
        # Last column of X_t are sample space indices.
        test_X = self.to_tensor(test_X[:, :self.input_dim])

        test_X = test_X[:, None, :]  # Prepare for batch inference

        with torch.no_grad(), torch.autograd.set_detect_anomaly(True):
            res = self.model(test_X)

            means = res.mean.squeeze().cpu().numpy()  # Squeeze the sample dimension
            # Make sure covariance matrix is symmetric, for inverse matrix calculation.
            variances = (res.covariance_matrix + res.covariance_matrix.transpose(1, 2)) / 2
            variances = variances.cpu().numpy()
        
        return means, variances

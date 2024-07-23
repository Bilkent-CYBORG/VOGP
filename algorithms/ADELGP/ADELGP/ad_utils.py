import gpytorch
import numpy as np

def check_kernel_type(kernel):
    if isinstance(kernel, gpytorch.kernels.RBFKernel):
        return "RBF"
    elif isinstance(kernel, gpytorch.kernels.MaternKernel):
        return "Matern"
    elif isinstance(kernel, gpytorch.kernels.MultitaskKernel):
        return check_kernel_type(kernel.data_covar_module)
    elif hasattr(kernel, 'base_kernel'):
        return check_kernel_type(kernel.base_kernel)
    elif hasattr(kernel, 'kernels'):
        for sub_kernel in kernel.kernels:
            result = check_kernel_type(sub_kernel)
            if result in ["RBF", "Matern"]:
                return result
    return "Unknown"

def calculate_vh(
    point_depth, gp, m, d, delta=0.05, rho=0.5, alpha=1, N=4, depth_offset=0
):
    cov_module = gp.model.covar_module
    lengthscales = cov_module.data_covar_module.lengthscale.detach().cpu().squeeze().numpy()
    variances = cov_module.task_covar_module.var.detach().cpu().squeeze().numpy()
    Vh = np.zeros([m, 1])
    depth = point_depth + depth_offset
    diam_x = np.sqrt(d)
    v1 = 0.5 * np.sqrt(d)

    for i in range(m):
        kernel_type = check_kernel_type(cov_module)
        if kernel_type == 'RBF':
            Cki = np.sqrt(variances[i]) / lengthscales[i]
        else:
            raise ValueError

        C1 = np.power((diam_x + 1) * diam_x / 2, d) * np.power(Cki, 1 / alpha)
        C2 = 2 * np.log(2 * np.power(C1 , 2) * np.power(np.pi, 2) / 6)
        C3 = 1. + 2.7 * np.sqrt(2 * d * alpha * np.log(2))

        term1 = Cki * np.power(v1 * np.power(rho, depth), alpha)
        term2 = np.log(2 * np.power(depth + 1, 2) * np.power(np.pi, 2) * m / (6 * delta))
        term3 = depth * np.log(N)
        term4 = np.maximum(0, -4 * d / alpha * np.log(term1))

        Vh[i] = 4 * term1 * (np.sqrt(C2 + 2 * term2 + term3 + term4) + C3)
    return Vh

import os
import shutil
import pickle
import itertools
from pathlib import Path
from fractions import Fraction
from concurrent.futures import ProcessPoolExecutor as Pool

import torch
import numpy as np
import cvxpy as cp
from scipy.optimize import minimize
from sklearn.metrics.pairwise import euclidean_distances

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True


def read_sorted_results(alg_path, sort=True):
    result_file_path = os.path.join(alg_path, "results.pkl")
    with open(result_file_path, "rb") as results_file:
        results_list = pickle.load(results_file)
    
    if sort:
        # results_list.sort(key=lambda x:x["batch_size"])
        results_list.sort(key=lambda x:x["conf_contraction"])
        results_list.sort(key=lambda x:x["eps"], reverse=True)
        results_list.sort(key=lambda x:x["cone_degree"])
        results_list.sort(key=lambda x:x["dataset_name"])

    return results_list


def overwrite_makedirs(path):
    if os.path.exists(path):
        shutil.rmtree(path)

    os.makedirs(path)


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_closest_indices_from_points(pts_to_find, pts_to_check):
    if len(pts_to_find) == 0 or len(pts_to_check) == 0:
        return []

    distances = euclidean_distances(pts_to_find, pts_to_check, squared=True)
    x_inds = np.argmin(distances, axis=1)
    return x_inds.astype(int)


def get_max_variance_point(model, candidates):
    largest = 0
    to_observe_ind = 0
    _, covs = model.predict(candidates)
    for x_i in range(len(candidates)):
        diameter = np.linalg.norm(np.diag(covs[x_i]))
        if diameter > largest:
            largest = diameter
            to_observe_ind = x_i
    
    return to_observe_ind


def save_new_result(output_folder_path, experiment_res_dict):
    res_pickle_path = os.path.join(output_folder_path, "results.pkl")
    res_pickle_tmp_path = os.path.join(output_folder_path, "results_tmp.pkl")

                # Create a result file if it doesn't exist
    if not os.path.exists(res_pickle_path):
        with open(res_pickle_path, "wb") as results_file:
            pickle.dump([], results_file)
                # Load the results until now
    with open(res_pickle_path, "rb") as results_file:
        results_list = pickle.load(results_file)
                # Add new experiment results
    results_list.append(experiment_res_dict)
                # Rename current result file as tmp in case of an error
    os.rename(res_pickle_path, res_pickle_tmp_path)
                # Write new results
    with open(res_pickle_path, "wb") as results_file:
        pickle.dump(results_list, results_file)
                # Remove the tmp one if the new one is successfuly written
    os.remove(res_pickle_tmp_path)


def get_noisy_evaluations_chol(means, cholesky_cov, W=None):
    """Used for vectorized multivariate normal sampling."""
    n, d = means.shape[0], len(cholesky_cov)
    X = np.random.normal(size=(n, d))
    complicated_X = X.dot(cholesky_cov)
    if W is not None:
        complicated_X = complicated_X @ W.T
    
    noisy_samples = complicated_X + means
    
    return noisy_samples


def get_preference_cone(m):

    # Always has its own randomness.
    rng = np.random.default_rng(m)

    A = dict()
    for i in range(m):
        other_i = list(range(m))
        other_i.remove(i)
        A[i] = dict(zip(other_i, rng.random(m-1).tolist()))

    block_list = list()
    for e in range(m):
        diagonal = np.diag(np.array(list(A[e].values()), dtype=float))
        if e == 0:
            block = np.concatenate((np.ones((m-1, 1)), diagonal), axis=1)
        elif e == m - 1:
            block = np.concatenate((diagonal, np.ones((m-1, 1))), axis=1)
        else:
            first_half = diagonal[:, :e]
            second_half = diagonal[:, e:]
            block = np.concatenate((first_half, np.ones((m-1,1)), second_half), axis=-1)
        block_list.append(block)
    W = np.concatenate(block_list, axis=0)

    # Calculate L2 norm for each row
    row_norms = np.linalg.norm(W, axis=1, keepdims=True)
    # Prevent division by zero by replacing zero norms with ones
    row_norms[row_norms == 0] = 1
    # Normalize each row
    W = W / row_norms

    return W


def get_2d_w(cone_angle):
    angle_radian = (cone_angle/180) * np.pi
    if cone_angle <= 90:
        W_1 = np.array([-np.tan(np.pi/4-angle_radian/2), 1])
        W_2 = np.array([+np.tan(np.pi/4+angle_radian/2), -1])
    else:
        W_1 = np.array([-np.tan(np.pi/4-angle_radian/2), 1])
        W_2 = np.array([-np.tan(np.pi/4+angle_radian/2), 1])
    W_1 = W_1/np.linalg.norm(W_1)
    W_2 = W_2/np.linalg.norm(W_2)
    W = np.vstack((W_1, W_2))

    return W


def get_cone_params(degree, dim=2):
    if dim == 3 and degree in [45, 60]:
        W = np.array([
            [1.0, -2, 4],
            [4, 1.0, -2],
            [-2, 4, 1.0],
        ])
        norm = np.linalg.norm(W[0])
        W /= norm
        alpha_vec = np.array([[0.87831007], [0.87831007], [0.87831007]])
    elif dim == 3 and degree in [120, 135]:
        W = np.array([
            [1, 0.4, 1.6],
            [1.6, 1, 0.4],
            [0.4, 1.6, 1],
        ])
        norm = np.linalg.norm(W[0])
        W /= norm
        alpha_vec = np.ones((dim, 1))
    elif degree == 45:
        W = get_2d_w(degree)
        alpha_vec = np.array([[0.70710678], [0.70710678]])
    elif degree == 60:
        W = get_2d_w(degree)
        alpha_vec = np.array([[0.8660254], [0.8660254]])
    elif degree == 90:
        W = np.eye(dim)
        alpha_vec = np.ones((dim, 1))
    elif dim == 2 and degree > 90:
        W = get_2d_w(degree)
        alpha_vec = np.array([[1], [1]])
    elif dim == 2 and degree <= 90:
        W = get_2d_w(degree)
        alpha_vec = get_alpha_vec(W)
    else:
        raise ValueError

    return W, alpha_vec


def compute_ice_cream_params(K, theta=60):
    delta_angle = 2 * np.pi / K
    theta_rad = np.pi/2 - np.radians(theta)

    radius = np.tan(theta_rad)
    W = []
    for i in range(K):
        angle = i * delta_angle
        rotated_ny = radius * np.sin(angle)
        rotated_nx = radius * np.cos(angle)
        W.append([rotated_nx, rotated_ny, 1])
    W = np.array(W)
    
    rot_axis = np.array([-1/np.sqrt(2), 1/np.sqrt(2), 0])
    rot_rad = np.pi/4
    C = np.array([
        [0, -rot_axis[2], rot_axis[1]],
        [rot_axis[2], 0, -rot_axis[0]],
        [-rot_axis[1], rot_axis[0], 0],
    ])

    r = np.eye(3) + C*np.sin(rot_rad) + (C @ C)*(1 - np.cos(rot_rad))
    W = (r @ W.T).T

    # Normalize half plane normal vectors
    for i in range(K):
        W[i] = W[i] / np.linalg.norm(W[i])

    alpha_vec = get_alpha_vec(W)
    return W, alpha_vec


def compute_ustar_scipy(W):
    """
    Given a matrix W that corresponds to a polyhedral ordering cone, this function 
    computes the ordering difficulty of the cone and u^* direction of the cone.
    
    The function solves an optimization problem where the objective is to minimize the
    Euclidean norm of `z`, subject to the constraint that W @ z >= 1 for each row
    of W. 
    
    Parameters
    ----------
    W : numpy.ndarray
        A numpy array representing the matrix that defines the half-spaces of the
        polyhedral cone. Each row of W corresponds to a linear constraint on `z`.
    
    Returns
    -------
    u_star of cone : numpy.ndarray
        The normalized optimized vector `z`.
    d(1) of cone : float
        The Euclidean norm of the optimized `z`.
        
    Prints
    ------
        
    d(1) of cone : float
        The Euclidean norm of the optimized `z`.
        
    Are constraints obeyed : bool
        A boolean flag indicating whether all cone constraints are satisfied. 
        This corresponds to the unit sphere to be inside of the cone.
        
    If the constraints are not obeyed, it also prints:
    
    Distance to cone hyperplanes : numpy.ndarray
        The distances from the optimized `z` to the hyperplanes defined by W.
    
    Notes
    -----
    The optimization problem is solved using the Sequential Least SQuares Programming (SLSQP)
    method provided by scipy.optimize.minimize. The initial guess is a vector of ones, and
    the optimization runs with a very high maximum iteration limit and tight function tolerance
    to ensure convergence.
    
    Examples
    --------
    >>> W = W = np.sqrt(21)*np.array([[1, -2, 4], [4, 1, -2], [-2, 4, 1]])
    >>> u_star_optimized,d_1 = compute_ustar_scipy(W)
    """
    
    m = W.shape[1]  # dimension of the objective space.
    n = W.shape[0] 
    def objective(z):
        return np.linalg.norm(z)        
    def constraint_func(z):
        constraints = []
        constraint = W @ (z ) - np.ones((n,))
        constraints.extend(constraint)
        return np.array(constraints) 
    z_init = np.ones(m)  # Initial guess
    cons = [{'type': 'ineq', 'fun': lambda z: constraint_func(z)}] # Constraints 
    # Solving the problem
    res = minimize(objective, z_init, method='SLSQP', constraints=cons,
                   options={'maxiter': 1000000, 'ftol': 1e-30})
    norm= np.linalg.norm(res.x)
    construe = np.all(constraint_func(res.x)+1e-14)
    print(f"Optimized d(1) was found to be {norm}")
    print(f"Optimized u_star was found to be {res.x/norm}")
    print(f"Are constraints obeyed: {construe}")
    if not construe: 
        print(f"Distance to cone hyperplanes: {constraint_func(res.x)}")
    return res.x/norm, norm


def get_alpha(rind, W):
    """
    Compute alpha_rind for row rind of W 
    :param rind: row index
    :param W: (n_constraint,D) ndarray
    :return: alpha_rind.
    """
    m = W.shape[0]+1 #number of constraints
    D = W.shape[1]
    f = -W[rind,:]
    A = []
    b = []
    c = []
    d = []
    for i in range(W.shape[0]):
        A.append(np.zeros((1, D)))
        b.append(np.zeros(1))
        c.append(W[i,:])
        d.append(np.zeros(1))
    
    A.append(np.eye(D))
    b.append(np.zeros(D))
    c.append(np.zeros(D))
    d.append(np.ones(1))

    # Define and solve the CVXPY problem.
    x = cp.Variable(D)
    # We use cp.SOC(t, x) to create the SOC constraint ||x||_2 <= t.
    soc_constraints = [
        cp.SOC(c[i].T @ x + d[i], A[i] @ x + b[i]) for i in range(m)
    ]
    prob = cp.Problem(cp.Minimize(f.T@x),
                  soc_constraints)
    prob.solve(solver="ECOS")

    """
    # Print result.
    print("The optimal value is", -prob.value)
    print("A solution x is")
    print(x.value)
    for i in range(m):
        print("SOC constraint %i dual variable solution" % i)
        print(soc_constraints[i].dual_value)
    """    
        
    return -prob.value   


def get_alpha_vec(W):
    """
    Compute alpha_vec for W 
    :param W: an (n_constraint,D) ndarray
    :return: alpha_vec, an (n_constraint,1) ndarray
    """    
    alpha_vec = np.zeros((W.shape[0],1))
    for i in range(W.shape[0]):
        alpha_vec[i] = get_alpha(i, W)
    return alpha_vec


def get_bigmij(vi, vj, W):
    """
    Compute M(i,j) for designs i and j 
    :param vi, vj: (D,1) ndarrays
    :param W: (n_constraint,D) ndarray
    :return: M(i,j).
    """
    D = W.shape[1]
    P = 2*np.eye(D)
    q = (-2*(vj-vi)).ravel()
    G = -W
    h = -np.array([np.max([0,np.dot(row, vj-vi)[0]]) for row in W])
    # h = -np.array([
    #     np.max([0,np.dot(W[0,:], vj-vi)[0]]),
    #     np.max([0,np.dot(W[1,:], vj-vi)[0]])
    # ])

    # Define and solve the CVXPY problem.
    x = cp.Variable(D)
    prob = cp.Problem(
        cp.Minimize((1/2)*cp.quad_form(x, P) + q.T @ x),
        [G @ x <= h]
        )
    #A @ x == b    
    prob.solve()
    bigmij = np.sqrt(prob.value + np.dot((vj-vi).T, vj-vi)).ravel()

    # Print result.
    #print("\nThe optimal value is", prob.value)
    #print("A solution x is")
    #print(x.value)
    #print("A dual solution corresponding to the inequality constraints is")
    #print(prob.constraints[0].dual_value)
    #print("M(i,j) is", bigmij)
    return bigmij


def get_smallmij(vi, vj, W, alpha_vec):
    """
    Compute m(i,j) for designs i and j 
    :param vi, vj: (D,1) ndarrays
    :param W: (n_constraint,D) ndarray
    :param alpha_vec: (n_constraint,1) ndarray of alphas of W
    :return: m(i,j).
    """
    prod = np.matmul(W, vj - vi)
    prod[prod<0] = 0
    smallmij = (prod/alpha_vec).min()
    
    return smallmij


def get_delta(mu, W, alpha_vec):
    """
    Computes Delta^*_i for each i in [n.points]
    :param mu: An (n_points, D) array
    :param W: (n_constraint,D) ndarray
    :param alpha_vec: (n_constraint,1) ndarray of alphas of W
    :return: An (n_points, D) array of Delta^*_i for each i in [n.points]
    """
    n = mu.shape[0]
    Delta = np.zeros(n)
    for i in range(n):
        for j in range(n):
            vi = mu[i,:].reshape(-1,1)
            vj = mu[j,:].reshape(-1,1)
            mij = get_smallmij(vi, vj, W, alpha_vec)
            if mij>Delta[i]:
                Delta[i] = mij
    
    return Delta.reshape(-1,1)


def intersection(lst1, lst2):
    # Use of hybrid method
    temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3


def get_uncovered_set(p_opt_miss, p_opt_hat, mu, eps, W):
    """
    Check if vi is eps covered by vj for cone matrix W
    :param p_opt_hat: ndarray of indices of designs in returned Pareto set
    :param p_opt_miss: ndarray of indices of Pareto optimal points not in p_opt_hat
    :mu: An (n_points,D) mean reward matrix
    :param eps: float
    :param W: An (n_constraint,D) ndarray
    :return: ndarray of indices of points in p_opt_miss that are not epsilon covered
    """
    uncovered_set = []
    
    for i in p_opt_miss:
        for j in p_opt_hat:
            if is_covered(mu[i,:].reshape(-1,1), mu[j,:].reshape(-1,1), eps, W):
                break
        else:
            uncovered_set.append(i)
        
    return uncovered_set


def is_covered_SOCP(vi, vj, eps, W):
    """
    Check if vi is eps covered by vj for cone matrix W 
    :param vi, vj: (D,1) ndarrays
    :param W: An (n_constraint,D) ndarray
    :param eps: float
    :return: Boolean.
    """    
    m = 2*W.shape[0]+1 # number of constraints
    D = W.shape[1]
    f = np.zeros(D)
    A = []
    b = []
    c = []
    d = []

    # 0 <= W[i].T @ x, make sure x is in cone
    for i in range(W.shape[0]):
        A.append(np.zeros((1, D)))
        b.append(np.zeros(1))
        c.append(W[i,:])
        d.append(np.zeros(1))
    
    # ||x + (vi-vj)||_2 <= eps
    A.append(np.eye(D))
    b.append((vi-vj).ravel())
    c.append(np.zeros(D))
    d.append(eps*np.ones(1))

    # 0 <= W[i].T @ (x + (vi-vj))
    for i in range(W.shape[0]):
        A.append(np.zeros((1, D)))
        b.append(np.zeros(1))
        c.append(W[i,:])
        d.append(np.dot(W[i,:],(vi-vj)))
        
    # Define and solve the CVXPY problem.
    x = cp.Variable(D)
    # We use cp.SOC(t, x) to create the SOC constraint ||x||_2 <= t.
    soc_constraints = [
          cp.SOC(c[i].T @ x + d[i], A[i] @ x + b[i]) for i in range(m)
    ]
    prob = cp.Problem(cp.Minimize(f.T@x),
                  soc_constraints)
    prob.solve(solver="ECOS")

    """
    # Print result.
    print("The optimal value is", prob.value)
    print("A solution x is")
    print(x.value)
    print(x.value is not None)
    for i in range(m):
        print("SOC constraint %i dual variable solution" % i)
        print(soc_constraints[i].dual_value)
    """     
    return x.value is not None


def is_covered(vi, vj, eps, W):
    """
    Check if vi is eps covered by vj for cone matrix W 
    :param vi, vj: (D,1) ndarrays
    :param W: An (n_constraint,D) ndarray
    :param eps: float
    :return: Boolean.
    """
    # if np.dot((vi-vj).T, vi-vj) <= eps**2:
    #     return True
    return is_covered_SOCP(vi, vj, eps, W)


def generate_domain_pts(bounds, discretization):
    dim_pts_list = [
        np.linspace(
            bounds[i][0], bounds[i][1], discretization
        )
        for i in range(len(bounds))
    ]
    pts = np.array(list(itertools.product(*dim_pts_list)))

    return pts


def normalize(X, bounds):
    r"""Min-max normalize X w.r.t. the provided bounds.
    """
    bounds = np.array(bounds)
    return (X - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])


def unnormalize(X, bounds):
    r"""Un-normalizes X w.r.t. the provided bounds.
    """
    bounds = np.array(bounds)
    return X * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]


def hyperrectangle_get_vertices(lower: np.ndarray, upper: np.ndarray):
    a = [[l1, l2] for l1, l2 in zip(lower, upper)]
    vertex_list = [element for element in itertools.product(*a)]
    return np.array(vertex_list)


def hyperrectangle_get_region_matrix(lower: np.ndarray, upper: np.ndarray):
    dim = len(lower)
    region_matrix = np.vstack((np.eye(dim), -np.eye(dim)))
    region_boundary = np.hstack((np.array(lower), -np.array(upper)))

    return region_matrix, region_boundary


def get_pareto_set(mu, W, alpha_vec):
    """
    Find the indices of Pareto designs (rows of mu)
    :param mu: An (n_points, D) array
    :param W: (n_constraint,D) ndarray
    :param alpha_vec: (n_constraint,1) ndarray of alphas of W
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    def is_nondominated(i):
        vi = mu[i].reshape(-1, 1)
        return not ((vj - vi) >= 0).all()

    mu = mu @ W.T

    is_efficient = np.arange(mu.shape[0])
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(mu):
        nondominated_point_mask = np.zeros(mu.shape[0], dtype=bool)
        vj = mu[next_point_index].reshape(-1,1)
        
        for i in range(len(mu)):
            nondominated_point_mask[i] = is_nondominated(i)
        # with Pool(max_workers=8) as pool:
        #     results = pool.map(
        #         is_nondominated,
        #         range(len(mu)),
        #     )
        # nondominated_point_mask = np.array(list(results), dtype=bool)
        
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        mu = mu[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1

    return is_efficient

import logging
from itertools import product
from functools import partial
from concurrent.futures import ProcessPoolExecutor as Pool

import torch
import gpytorch
import numpy as np

import utils.dataset
from utils.dataset import ContinuousDataset
from utils.seed import SEED
from utils.utils import (
    set_seed, save_new_result, get_pareto_set, generate_domain_pts, normalize
)

from algorithms.ADELGP.ADELGP.Polyhedron import Polyhedron
from algorithms.ADELGP.ADELGP.VectorEpsilonPAL import VectorEpsilonPAL
from algorithms.ADELGP.ADELGP.OptimizationProblem import OptimizationProblem


def simulate_once(
    i, dataset_name, cone_degree, noise_var, epsilon, delta, conf_contraction, alg_config
):
    print(f"Starting {i}.")
    set_seed(SEED + i + 1)
    
    batched = False

    dataset_cls = getattr(utils.dataset, dataset_name)
    dataset = dataset_cls(cone_degree)

    is_adaptive = isinstance(dataset, ContinuousDataset)

    problem_model = OptimizationProblem(dataset, obs_noise_var=noise_var)

    A = dataset.W
    b = np.zeros((len(dataset.W),))
    C = Polyhedron(A = A, b = b)

    alg = VectorEpsilonPAL(
        problem_model=problem_model, cone=C, epsilon=epsilon, delta=delta,
        conf_contraction=conf_contraction, is_adaptive=is_adaptive,
        unknown_params=alg_config["unknown_params"], init_design_cnt=alg_config["init_design_cnt"],
        rkhs_bound=alg_config.get("rkhs_bound"),
        maxiter=None, batch_size=batched
    )

    round_results = []
    while not alg.finished():
        alg.run_one_step()

        if alg_config["save_all_round_results"] or alg.finished():
            if is_adaptive:
                pts = generate_domain_pts(
                    dataset.bounds,
                    dataset.domain_discretization_each_dim
                )

                means, _ = alg.gp.predict(normalize(pts, dataset.bounds))

                pareto_indices = get_pareto_set(means, dataset.W, dataset.alpha_vec)
                pareto_points = pts[pareto_indices].tolist()
            else:
                pareto_point_ids = [pareto_pt.design_index for pareto_pt in alg.P]
                pareto_points = problem_model.x[pareto_point_ids].tolist()

            round_results.append([alg.sample_count, pareto_points])

    return round_results


def run_adelgp(
    datasets_and_workers, cone_degrees, noise_var, delta, epsilons, conf_contractions,
    iteration, output_folder_path, alg_config
):
    # dset, eps, cone, conf, batch
    for dataset_name, dataset_worker in datasets_and_workers:
        for eps in epsilons:
            alg_independent_params = product(cone_degrees, conf_contractions)

            for cone_degree, conf_contraction in alg_independent_params:
                simulate_part = partial(
                    simulate_once,
                    dataset_name=dataset_name,
                    cone_degree=cone_degree,
                    delta=delta,
                    noise_var=noise_var,
                    epsilon=eps,
                    conf_contraction=conf_contraction,
                    alg_config=alg_config
                )

                # results = []
                # for iter_i in range(iteration):
                #     results.append(simulate_part(iter_i))
                with Pool(max_workers=dataset_worker) as pool:
                    results = pool.map(
                        simulate_part,
                        range(iteration),
                    )
                results = list(results)
                
                experiment_res_dict = {
                    "dataset_name": dataset_name,
                    "cone_degree": cone_degree,
                    "ice_cream_k": alg_config.get("ice_cream_k", None),
                    "alg": "ADELGP",
                    "delta": delta,
                    "noise_var": noise_var,
                    "eps": eps,
                    "disc": -1,
                    "conf_contraction": conf_contraction,
                    "batch_size": 1,
                    "results": results
                }

                save_new_result(output_folder_path, experiment_res_dict)

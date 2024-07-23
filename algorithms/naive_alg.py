import logging

import numpy as np

import utils.dataset
from utils.utils import get_pareto_set, save_new_result

from utils.seed import SEED


def naive_with_sample_count(
    dataset_name, cone_angle, noise_var, delta, conf_cont, sample_eps, nrun, output_folder_path,
    alg_config
):
    dataset_cls = getattr(utils.dataset, dataset_name)
    dataset = dataset_cls(cone_angle)

    W = dataset.W
    alpha_vec = dataset.alpha_vec

    mu = dataset.out_data

    D = mu.shape[1]
    K = mu.shape[0]

    sigma = np.sqrt(noise_var)

    # Simulation model
    smp_cnt = sample_eps[:, 0]
    eps_vals = sample_eps[:, 1]
    # eps_vals = [0.09, 0.07, 0.05, 0.03, 0.01, 0.1]

    if conf_cont != -1:
        c = 1 + np.sqrt(2)  # Any c>0 should suffice according to Lemma B.12, keep it as original.
        beta = 1.0 if cone_angle >= 90 else (1.0 / np.sin((cone_angle/180) * np.pi))
        eval_num = np.maximum(np.ceil(
            4
            * ((c*sigma*beta/eps_vals)**2)
            * np.log(4*D/(2*delta/(K*(K-1))))
            / (conf_cont ** 2)
        ).astype(int), 1)
    else:
        eval_num = np.ceil(smp_cnt / len(mu)).astype(int)

    np.random.seed(SEED)

    nsample = eval_num.shape[0]
    noisemat = sigma * np.random.randn(nrun, nsample, K, D)

    # Compute epsilon independent results
    noisemat_interval = noisemat * np.sqrt(
        np.hstack((eval_num[0], np.diff(eval_num)))
    )[np.newaxis,:,np.newaxis,np.newaxis]
    noisemat_avg = np.cumsum(noisemat_interval,axis=1)/eval_num[np.newaxis,:,np.newaxis,np.newaxis]

    mu_hat = mu[np.newaxis,np.newaxis,:,:] + noisemat_avg

    for j in range(eval_num.shape[0]):
        print(f"New evaluation ({j}).")
        results = []
        for i in range(nrun):
            print(f"Run number ({i}).")
            p_opt_hat = get_pareto_set(mu_hat[i,j,:,:], W, alpha_vec)
            pareto_points = dataset.in_data[p_opt_hat].tolist()
            results.append([[eval_num[j] * len(mu), pareto_points]])
        print("")

        experiment_res_dict = {
            "dataset_name": dataset_name,
            "cone_degree": cone_angle,
            "alg": "Naive",
            "delta": delta,
            "noise_var": sigma*sigma,
            "eps": eps_vals[j],
            "disc": -1,
            "conf_contraction": conf_cont,
            "batch_size": -1,
            "results": results
        }

        save_new_result(output_folder_path, experiment_res_dict)


# alg_id, gp_dict, datasets_and_workers, cone_degrees, noise_var, delta, epsilons, iteration,
# conf_contractions, output_folder_path
def run_naive_elimination(
    naive_config, datasets, cone_degrees, delta, epsilons, noise_var, iteration,
    conf_factors, output_folder_path
):
    pass
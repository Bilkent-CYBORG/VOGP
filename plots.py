import os

import torch
from botorch.utils.multi_objective.hypervolume import Hypervolume
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = (7, 3)

import utils.dataset
from utils.seed import SEED
from utils.utils import read_sorted_results, get_closest_indices_from_points


def create_visualization_dir(exp_path):
    visualization_path = os.path.join(exp_path, "vis")
    os.makedirs(visualization_path, exist_ok=True)
    return visualization_path

def adelgp_exp_plot(exp_path, name_exp, exp_start_id, exp_end_id):
    visualization_path = create_visualization_dir(exp_path)

    algorithm_names = sorted(
        [
            subpath
            for subpath in os.listdir(exp_path)
            if os.path.isdir(os.path.join(exp_path, subpath))
        ]
    )

    cone_results = []
    cone_degrees = dict()

    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams.update({'font.size': 14})
    plt.style.use('seaborn-v0_8-whitegrid')

    for alg_name in algorithm_names:
        if "exp" not in alg_name or "Naive" in alg_name:
            continue
        alg_num = int(alg_name.split('-')[0][3:])
        if not (alg_num >= exp_start_id and alg_num <= exp_end_id):
            continue
        # Load results file
        alg_path = os.path.join(exp_path, alg_name)
        try:
            results_list = read_sorted_results(alg_path)
        except:
            continue
    
        for exp_dict in results_list:
            c_d = exp_dict["cone_degree"]
            if isinstance(c_d, list):
                assert c_d[0] == "theta", "Cone type is not supported right now."
                c_d = c_d[1]

            if c_d not in cone_degrees:
                cone_degrees[c_d] = len(cone_results)
                cone_results.append([])
            
            cone_results[cone_degrees[c_d]].append([alg_name, exp_dict])
    
    all_evaluated_results = []
    for cone_degree, cone_i in cone_degrees.items():
        print(f"Cone degree {cone_degree}")
        all_evaluated_results.append([])

        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        
        cone_degree_results = cone_results[cone_i]
        for alg_name, cone_degree_result in cone_degree_results:
            print(f"  - Alg. {alg_name}")
            all_evaluated_results[-1].append([])
            
            loghvd = []
            loghvd_std = []

            round_count = int(np.median(list(map(len, cone_degree_result["results"]))))
            results, results_std = adelgp_exp_plot_evaluate_experiment(cone_degree_result)
            for round_idx in range(round_count):
                all_evaluated_results[-1][-1].append([results[round_idx], results_std[round_idx]])

                loghvd.append(results[round_idx])
                loghvd_std.append(results_std[round_idx])
            
            loghvd = np.array(loghvd)
            loghvd_std = np.array(loghvd_std)

            ax.fill_between(
                np.arange(round_count), loghvd-loghvd_std/2, loghvd+loghvd_std/2, alpha = 0.25
            )
            ax.plot(np.arange(round_count), loghvd, "-o", label=alg_name)
            ax.set_xlabel("Round")
            ax.set_ylabel(r"$log(d_{HV})$")
        
        ax.legend(loc="lower left")
        fig.tight_layout()
        plt.savefig(os.path.join(visualization_path, f'{name_exp}_loghvd_{cone_degree}.png'))

def adelgp_exp_plot_evaluate_experiment(exp_dict: dict):
    dataset_name = exp_dict["dataset_name"]
    cone_degree = exp_dict["cone_degree"]

    dataset_cls = getattr(utils.dataset, dataset_name)
    dataset = dataset_cls(cone_degree)
    is_continuous = isinstance(dataset, utils.dataset.ContinuousDataset)
    if is_continuous:
        dataset = utils.dataset.ContinuousDatasetWrapper(dataset, manual_discretization=55)
    
    _, true_pareto_indices = dataset.get_params()

    W_CONE, _ = dataset.W, dataset.alpha_vec

    transformed_out_data = dataset.out_data @ W_CONE.T

    hv_ref_pt = torch.tensor(np.min(transformed_out_data, axis=0))
    
    hypervolume_instance = Hypervolume(hv_ref_pt)

    hypervol_true = hypervolume_instance.compute(
        torch.tensor(transformed_out_data[true_pareto_indices])
    )

    round_count = int(np.median(list(map(len, exp_dict["results"]))))

    result_sum = np.full((round_count, len(exp_dict["results"])), np.nan)

    for round_idx in range(round_count):
        print(f"    - Round {round_idx}")
        for res_i, iter_result in enumerate(exp_dict["results"]):
            # Calculate for only the round_idx'th round
            if round_idx >= len(iter_result):
                continue

            _, pred_pareto_pts = iter_result[round_idx]
            pred_pareto_pts = np.array(pred_pareto_pts).reshape(-1, dataset.in_dim)
            pred_pareto_indices = get_closest_indices_from_points(pred_pareto_pts, dataset.in_data)

            hypervol_disc = np.log(hypervol_true - hypervolume_instance.compute(
                torch.tensor(transformed_out_data[pred_pareto_indices])
            ))

            result_sum[round_idx, res_i] = hypervol_disc

    result = np.nanmean(result_sum, axis=1)
    result_std = np.nanstd(result_sum, axis=1)

    return result, result_std


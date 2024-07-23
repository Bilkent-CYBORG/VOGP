import os
import argparse
from glob import glob
from pathlib import Path

import numpy as np
import torch
from botorch.utils.multi_objective.hypervolume import Hypervolume

import utils.dataset
from utils.utils import (
    get_uncovered_set, read_sorted_results, get_closest_indices_from_points
)


def evaluate_experiment(exp_dict: dict, round_idx=-1):
    dataset_name = exp_dict["dataset_name"]
    cone_degree = exp_dict["cone_degree"]
    cover_eps = exp_dict["eps"]

    dataset_cls = getattr(utils.dataset, dataset_name)
    dataset = dataset_cls(cone_degree)

    is_continuous = isinstance(dataset, utils.dataset.ContinuousDataset)
    if is_continuous:
        dataset = utils.dataset.ContinuousDatasetWrapper(dataset, manual_discretization=55)

    delta_cone, true_pareto_indices = dataset.get_params()

    W_CONE, _ = dataset.W, dataset.alpha_vec

    if is_continuous:
        transformed_out_data = dataset.out_data @ W_CONE.T

        hv_ref_pt = torch.tensor(np.min(transformed_out_data, axis=0))
        hypervolume_instance = Hypervolume(hv_ref_pt)
        
        hypervol_true = hypervolume_instance.compute(
            torch.tensor(transformed_out_data[true_pareto_indices])
        )

    metric_key = 'F1E' if not is_continuous else 'HVD'
    result_keys = [
        metric_key,
        'SC',
    ]
    result_sum = np.full((len(exp_dict["results"]), len(result_keys)), np.nan)
    for res_i, iter_result in enumerate(exp_dict["results"]):
        # Calculate for only the round_idx'th round
        if round_idx >= len(iter_result):
            continue

        samples, pred_pareto_pts = iter_result[round_idx]
        pred_pareto_pts = np.array(pred_pareto_pts).reshape(-1, dataset.in_dim)
        pred_pareto_indices = get_closest_indices_from_points(pred_pareto_pts, dataset.in_data)

        if is_continuous:
            hypervol_disc = np.log(hypervol_true - hypervolume_instance.compute(
                torch.tensor(transformed_out_data[pred_pareto_indices])
            ))
            metric = hypervol_disc
        else:
            pred_set = set(pred_pareto_indices)
            gt_set = set(true_pareto_indices)

            indices_of_missed_pareto = list(gt_set - pred_set)

            # Returns non-covered pareto indices that are missed
            uncovered_missed_pareto_indices = get_uncovered_set(
                indices_of_missed_pareto, pred_pareto_indices, dataset.out_data,
                np.linalg.norm(cover_eps), W_CONE
            )

            true_eps = np.sum(delta_cone[pred_pareto_indices] <= np.min(cover_eps), axis=0)[0]

            tp_eps = true_eps
            fp_eps = len(pred_set) - true_eps
            f1_eps = (2 * tp_eps) / (2*tp_eps + fp_eps + len(uncovered_missed_pareto_indices))
            metric = f1_eps

        result_sum[res_i] = [
            metric,
            samples,
        ]

    result = np.nanmean(result_sum, axis=0)
    result_std = np.nanstd(result_sum, axis=0)

    result_dict = dict(zip(result_keys, np.around(result, 2).tolist()))
    result_std_dict = dict(zip(
        list(map(lambda x: x+' Std', result_keys)),
        np.around(result_std, 2).tolist()
    ))

    return result_dict, result_std_dict


if __name__ == "__main__":
    exp_path = None
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=Path, required=True, default=None)
    args = parser.parse_args()

    exp_path = os.path.join("outputs", args.exp_name)

    algorithm_names = sorted(
        [
            subpath
            for subpath in os.listdir(exp_path)
            if os.path.isdir(os.path.join(exp_path, subpath))
        ],
        key=lambda x: x.split('-')[0]
    )
    algorithm_names.sort(key=lambda x: len(x.split('-')[0]))
    
    for alg_name in algorithm_names:
        if '-' not in alg_name:
            continue

        alg_num = int(alg_name.split('-')[0][3:])

        alg_text = alg_name.split('-')[-1]

        # Load results file
        alg_path = os.path.join(exp_path, alg_name)
        try:
            results_list = read_sorted_results(alg_path)
        except:
            continue

        print(
            "---   "
            f"Algorithm ID: {alg_name}"
            f", Iteration count: {len(results_list[0]['results'])}"
            "   ---"
        )

        # Evaluate each config
        for exp_dict in results_list:
            result, result_std = evaluate_experiment(exp_dict)
            
            for (k, v), std_v in zip(result.items(), result_std.values()):
                if k == "SC":
                    result[k] = f"{v:05.2f} ± {std_v:04.2f}"
                else:
                    result[k] = f"{v:06.2f} ± {std_v:05.2f}"

            print(
                f"D.set: {exp_dict['dataset_name']:<16}"
                f"Cone: {str(exp_dict['cone_degree']):<20}"
                f"Eps.: {exp_dict['eps']:<6}",
                f"Cont.: {exp_dict['conf_contraction']:<4}",
                f"B.S.: {exp_dict['batch_size']:<4}",
                result
            )
        print()

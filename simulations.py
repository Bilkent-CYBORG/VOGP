import os
import time
import shutil
import logging
import argparse
import tracemalloc
from pathlib import Path
from datetime import datetime
from itertools import product

from multiprocessing import set_start_method

import numpy as np
import oyaml as yaml

from utils.seed import SEED
from utils.config import Config
from utils.dataset import DATASET_SIZES
from utils.utils import set_seed, overwrite_makedirs, read_sorted_results

from algorithms.naive_alg import naive_with_sample_count
from algorithms.ADELGP.adelgp import run_adelgp


def sample_means_to_compare(compare_experiment_id):
    os.listdir(experiment_folder)
    compare_experiment_name = [
        exp_name
        for exp_name in os.listdir(experiment_folder)
        if f'exp{compare_experiment_id}-' in exp_name
    ][0]
    compare_results_list = read_sorted_results(
        os.path.join(experiment_folder, compare_experiment_name), sort=False
    )
    sample_counts = []
    normalizations = []
    for exp_dict in compare_results_list:
        sample_counts.append(list(zip(*[exp_res[-1] for exp_res in exp_dict["results"]]))[0])
        normalizations.append(DATASET_SIZES.get(exp_dict["dataset_name"], -1))
    return sample_counts, normalizations

def parse_and_run_experiment(exp_id, exp_d):
    tracemalloc.start()
    start = time.time()

    config = Config(exp_d)
    alg_config = config.get(config.algorithm, default=Config({}))

    datasets = config.datasets_and_workers
    output_folder_path = os.path.join(experiment_folder, f'exp{exp_id}-' + config.algorithm)
    if config.algorithm == "Naive":
        overwrite_makedirs(output_folder_path)

        datasets = [dataset[0] for dataset in datasets]

        fixed_conf = alg_config.get('fixed_conf') is not None
        if fixed_conf:
            samples = np.zeros((len(datasets) * len(config.cone_degrees), len(config.epsilons)))
            conf_conts = config.conf_contractions
        else:
            conf_conts = [-1]  # Sample counts from outside

        if not fixed_conf and alg_config.get("samples") is not None:
            samples = alg_config.samples
        elif not fixed_conf:
            # Get samples from compared experiment
            sample_counts, normalizations = sample_means_to_compare(
                alg_config.compare_experiment_id
            )
            sample_means = [np.mean(exp_sample_counts) for exp_sample_counts in sample_counts]
            sample_means = np.array(sample_means)
            normalizations = np.array(normalizations)
            sample_means = (np.ceil(sample_means / normalizations) * normalizations).astype(int)
            samples = np.array(sample_means).reshape(len(datasets), len(config.epsilons), -1)
            samples = samples.transpose((0, 2, 1)).reshape(-1, len(config.epsilons))

        # Shape: (len_configurations, len_epsilons, 2(sample size, epsilon))
        samples_with_eps = np.concatenate(
            (
                samples.reshape(-1, 1),
                np.repeat(
                    config.epsilons, samples.shape[0]
                ).reshape(samples.shape[1], -1).T.reshape(-1, 1)
            ), axis=1
        ).reshape(-1, len(config.epsilons), 2)

        dset_and_angle = list(product(datasets, config.cone_degrees))

        assert(len(dset_and_angle) == len(samples_with_eps))

        for ((dataset_name, cone_angle), sample_with_eps) in zip(dset_and_angle, samples_with_eps):
            for conf_cont in conf_conts:
                naive_with_sample_count(
                    dataset_name, cone_angle, config.noise_var, config.delta, conf_cont,
                    sample_with_eps, config.iteration, output_folder_path, alg_config.dict
                )
    elif config.algorithm == "ADELGP":
        overwrite_makedirs(output_folder_path)

        run_adelgp(
            datasets, config.cone_degrees, config.noise_var, config.delta,
            config.epsilons, config.conf_contractions, config.iteration,
            output_folder_path, alg_config.dict
        )
    else:
        raise NotImplementedError

    end = time.time()

    peak_mem = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    with open(os.path.join(experiment_folder, "times.txt"), 'a') as f:
        print(f"Experiment ID={exp_id} done in {end - start:.2f} seconds.", file=f)
    with open(os.path.join(experiment_folder, "peak_memories.txt"), 'a') as f:
        print(f"Experiment ID={exp_id} done in {peak_mem} bytes.", file=f)


if __name__ == "__main__":
    set_start_method("spawn")

    # Disable warnings, especially for CVXPY
    import warnings
    warnings.filterwarnings("ignore")

    # Set up logging level
    logging.basicConfig(level=logging.DEBUG)

    # Set seed
    set_seed(SEED)

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_file', type=Path, required=True)
    args = parser.parse_args()

    # Read experiment config
    experiment_file: Path = args.experiment_file
    with open(experiment_file, "r") as f:
        config = yaml.safe_load(f)
    
    # Continue experiment
    if config["experiment_name"] != "":
        if config["experiment_ids"] == 1:
            raise Exception("Check start ID, it overwrites whole experiment.")
        experiment_name = config["experiment_name"]
    else:  # New experiment
        experiment_name = datetime.now().strftime("%m_%d_%Y__%H_%M_%S")
        experiment_name += '__'
        experiment_name += '__'.join(experiment_file.with_suffix("").as_posix().split('/')[1:])
    experiment_folder = os.path.join("outputs", experiment_name)

    # Copy experiment config, overwrite if exists
    os.makedirs(experiment_folder, exist_ok=True)
    copy_experiment_file = os.path.join(experiment_folder, "experiment.yaml")
    shutil.copy(src=experiment_file, dst=copy_experiment_file)

    # Which experiments to run
    experiment_ids = config["experiment_ids"]
    if isinstance(experiment_ids, int):
        experiment_ids = range(experiment_ids, config["num_experiments"]+1)

    # Run experiments
    for i in experiment_ids:
        parse_and_run_experiment(i, config[f"experiment{i}"])

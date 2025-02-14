import argparse
import logging
import os
from pathlib import Path
from typing import List, Callable

import keras
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import ops as k_ops
from resolv_ml.utilities.bijectors.power_transform import BoxCox
from resolv_mir.note_sequence.attributes import ATTRIBUTE_FN_MAP
from scipy.stats import norm, boxcox, kurtosis

import utilities


def test_power_transform(args):
    for attribute in args.attributes:
        logging.info(f"Evaluating power transform for attribute '{attribute}'...")
        main_output_path = Path(args.histogram_output_path) / attribute
        if not main_output_path.exists():
            # Load dataset
            attribute_data = utilities.load_flat_dataset(dataset_path=args.dataset_path,
                                                         sequence_length=args.sequence_length,
                                                         attribute=attribute,
                                                         batch_size=args.batch_size,
                                                         parse_sequence_feature=False)[:, 0]
            if args.remove_outlier:
                outlier_mask = np.where(np.logical_not(np.isclose(attribute_data, 0)))
                attribute_data = attribute_data[outlier_mask]
            shifts_grid = [0.] if not args.shift_grid_search \
                else keras.ops.arange(args.shift_min, args.shift_max, args.shift_step)
            normality_metrics = []
            for s in shifts_grid:
                output_path = main_output_path / f"shift_{s:.3f}"
                output_path.mkdir(parents=True, exist_ok=True)
                numpy_output_path = output_path / "numpy"
                numpy_output_path.mkdir(parents=True, exist_ok=True)
                # add epsilon to shift in order to avoid zero input values for BoxCox
                s = 1e-5 if not s else s
                shifted_attribute_data = attribute_data + s
                # Compute best power parameter for BoxCox using log-likelihood maximization
                _, llm_lmbda = boxcox(shifted_attribute_data, lmbda=None)
                logging.info(f"LLM power transform best power value for attribute '{attribute}' shifted by {s:.5f} "
                             f"is: {llm_lmbda:.10f}'")
                # Create PowerTransform bijector
                power_transform_bij = BoxCox(
                    power_init_value=llm_lmbda,
                    power_trainable=False,
                    shift_trainable=False
                )
                power_transform_bij.build(attribute_data.shape)
                # Compute PowerTransform
                pt_out = power_transform_bij.inverse(shifted_attribute_data)
                pt_out_norm = (pt_out - k_ops.mean(pt_out)) / k_ops.std(pt_out)
                pt_out_norm = pt_out_norm.numpy()
                # Compute Kurtosis
                kurt = kurtosis(pt_out_norm)
                logging.info(f"Kurtosis index is {kurt:.10f}.")
                # Compute Negentropy Naive
                negentropy_naive = negentropy_approx_naive(pt_out_norm)
                logging.info(f"Negentropy naive index is {negentropy_naive:.10f}.")
                # Compute Negentropy exp
                negentropy_exp = negentropy_approx_fn(pt_out_norm, lambda u: -np.exp(-(u ** 2 / 2)))
                logging.info(f"Negentropy exp index is {negentropy_exp:.10f}.")
                # Compute Negentropy cosh
                negentropy_cosh = negentropy_approx_fn(pt_out_norm, lambda u: np.log(np.cosh(u)))
                logging.info(f"Negentropy cosh index is {negentropy_cosh:.10f}.")
                # Save metrics to global array
                normality_metrics.append([kurt, negentropy_naive, negentropy_exp, negentropy_cosh])
                # Plot histogram of the output distributions
                logging.info(f"Plotting output distribution histogram...")
                plot_distributions(
                    pt_data=pt_out_norm,
                    original_data=attribute_data,
                    output_path=output_path,
                    power=llm_lmbda,
                    shift=s,
                    attribute=attribute,
                    histogram_bins=args.histogram_bins,
                    x_lim=args.x_lim,
                )
                # Save output distributions
                logging.info(f"Saving output distributions to numpy file...")
                numpy_pt_out_filename = f'pt_out_norm_{attribute}_power_{llm_lmbda:.2f}_shift_{s:.3f}.npy'
                np.save(numpy_output_path / numpy_pt_out_filename, pt_out_norm)
            normality_metrics = np.array(normality_metrics)
            min_kurt_idx = np.argmin(np.abs(normality_metrics[:, 0]))
            min_kurt = normality_metrics[min_kurt_idx, 0]
            logging.info(f"Minimum Kurtosis index is {min_kurt:.10f} obtained with data shifted by "
                         f"{shifts_grid[min_kurt_idx]:.5f}.")
            min_negentropy_naive_idx = np.argmin(np.abs(normality_metrics[:, 1]))
            min_negentropy_naive = normality_metrics[min_negentropy_naive_idx, 1]
            logging.info(f"Minimum Negentropy naive index is {min_negentropy_naive:.10f} obtained with data shifted by "
                         f"{shifts_grid[min_negentropy_naive_idx]:.5f}.")
            min_negentropy_exp_idx = np.argmin(np.abs(normality_metrics[:, 2]))
            min_negentropy_exp = normality_metrics[min_negentropy_exp_idx, 2]
            logging.info(f"Minimum Negentropy exp index is {min_negentropy_exp:.10f} obtained with data shifted by "
                         f"{shifts_grid[min_negentropy_exp_idx]:.5f}.")
            min_negentropy_cosh_idx = np.argmin(np.abs(normality_metrics[:, 3]))
            min_negentropy_cosh = normality_metrics[min_negentropy_cosh_idx, 3]
            logging.info(f"Minimum Negentropy cosh index is {min_negentropy_cosh:.10f} obtained with data shifted by "
                         f"{shifts_grid[min_negentropy_cosh_idx]:.5f}.")
        else:
            logging.info(f"Power transform distribution for attribute '{attribute}' already exists. "
                         f"Remove the folder {main_output_path} to override it.")


def negentropy_approx_naive(x):
    return (1 / 12) * np.mean(x ** 3) ** 2 + (1 / 48) * kurtosis(x) ** 2


def negentropy_approx_fn(x, fn: Callable):
    gaussian_data = np.random.normal(0, 1, x.shape[0])
    negentropy = (np.mean(fn(x)) - np.mean(fn(gaussian_data))) ** 2
    return negentropy


@mpl.rc_context({'text.usetex': True, 'font.family': 'serif', 'font.size': 20, 'font.serif': 'Computer Modern Roman',
                 'lines.linewidth': 1.5})
def plot_distributions(pt_data,
                       original_data,
                       output_path: Path,
                       power: float,
                       shift: float,
                       attribute: str,
                       x_lim: float,
                       histogram_bins: List[int]):
    histograms_output_path = output_path / "histograms"
    histograms_output_path.mkdir(parents=True, exist_ok=True)
    x = np.linspace(-4, 4, pt_data.shape[0])
    standard_gauss = norm.pdf(x, 0, 1)
    for n_bins in histogram_bins:
        filename = (f'{str(histograms_output_path)}/histogram_{attribute}_power_{power:.2f}_shift_{shift:.3f}'
                    f'_bins_{n_bins}.png')
        # Create subplots
        fig, axes = plt.subplots(1, 2, sharey=True, figsize=(5, 5))
        # Original distribution histogram
        counts, bins = np.histogram(original_data, bins=n_bins)
        weights = (counts / np.max(counts)) * 0.45
        axes[0].hist(bins[:-1], bins=n_bins, weights=weights, color='C1')
        axes[0].set_xlabel('$a$')
        axes[0].set_axisbelow(True)
        axes[0].yaxis.grid(linestyle=':')
        if x_lim >= 0:
            axes[0].set_xlim(right=x_lim)
        # Power transform histogram
        counts, bins = np.histogram(pt_data, bins=n_bins)
        weights = (counts / np.max(counts)) * 0.45
        axes[1].hist(bins[:-1], bins=n_bins, weights=weights, color='C0')
        axes[1].plot(x, standard_gauss, '-', color='#fd5656', linewidth=2)
        axes[1].set_xlabel('$T_\lambda(a)$')
        axes[1].set_axisbelow(True)
        axes[1].yaxis.grid(linestyle=':')
        plt.tight_layout()
        plt.savefig(filename, format='png', dpi=300)
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate power transforms for specified attributes contained in the given SequenceExample "
                    "dataset."
    )
    parser.add_argument('--dataset-path', required=True, help='Path to the dataset containing SequenceExample with the '
                                                              'attributes to evaluate saved in the context.')
    parser.add_argument('--sequence-length', help='Length of the sequences in the dataset.', required=True, type=int)
    parser.add_argument('--attributes', nargs='+', help='Attributes to evaluate.', required=True)
    parser.add_argument('--histogram-output-path', help='Path where the histograms will be saved.', required=True)
    parser.add_argument('--histogram-bins', help='Number of bins for the histogram.', nargs='+', default=[60],
                        required=False, type=int)
    parser.add_argument('--batch-size', help='Batch size.', required=False, default=64, type=int,
                        choices=[32, 64, 128, 256, 512])
    parser.add_argument('--overlap-gauss-orig', help='Length of the sequences in the dataset.', action="store_true")
    parser.add_argument('--remove-outlier', help='Remove outliers from the datasets.', action="store_true")
    parser.add_argument('--shift-grid-search', help='Do a grid search for the power transform shift parameter.',
                        action="store_true")
    parser.add_argument('--shift-min', help='Start value for the grid search range of the shift parameter for the '
                                            'BoxCox power transform.',
                        default=0., required=False, type=float)
    parser.add_argument('--shift-max', help='Stop value for the grid search range of the shift parameter for the '
                                            'BoxCox power transform.',
                        default=3., required=False, type=float)
    parser.add_argument('--shift-step', help='Increment step for the grid search range of the shift parameter for the '
                                             'BoxCox power transform.',
                        default=0.25, required=False, type=float)
    parser.add_argument('--x-lim', help='X axis limit value for original distribution histogram.',
                        default=-1, required=False, type=float)
    parser.add_argument('--seed', help='Seed for random initializers.', required=False, type=int)
    parser.add_argument('--logging-level', help='Set the logging level.', default="INFO", required=False,
                        choices=["CRITICAL", "ERROR", "WARNING", "INFO"])
    os.environ["KERAS_BACKEND"] = "tensorflow"
    vargs = parser.parse_args()
    if vargs.attributes[0] == "all":
        vargs.attributes = ATTRIBUTE_FN_MAP.keys()
    if vargs.seed:
        keras.utils.set_random_seed(vargs.seed)
        np.random.seed(vargs.seed)
        tf.config.experimental.enable_op_determinism()
    logging.getLogger().setLevel(vargs.logging_level)
    test_power_transform(vargs)

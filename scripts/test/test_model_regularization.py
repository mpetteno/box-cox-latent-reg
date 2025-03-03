import logging
import os
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import scipy
import tensorflow as tf

import metrics
import utilities


def test_model_regularization(args):
    for attribute in args.attributes:
        output_dir = Path(args.output_path) / Path(args.conditional_model_path).stem / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        utilities.set_log_handler(output_dir=output_dir.parent, log_level=getattr(logging, vargs.logging_level))
        dataset = utilities.load_dataset(dataset_path=args.test_dataset_path,
                                         sequence_length=args.sequence_length,
                                         attribute=attribute,
                                         batch_size=args.batch_size,
                                         shift=keras.backend.epsilon(),
                                         parse_sequence_feature=True)
        logging.info("-----------------------------------------------------------------------------------------------")
        logging.info("Encoding-decoding sequences with conditional model...")
        conditional_model = keras.saving.load_model(args.conditional_model_path, compile=False)
        conditional_model.trainable = False
        conditional_model.compile(run_eagerly=True)
        (cond_decoded_sequences, cond_latent_codes, input_sequences, input_sequences_attributes,
         _) = conditional_model.predict(
            dataset, steps=args.dataset_cardinality // args.batch_size
        )
        input_sequences_attributes = input_sequences_attributes.squeeze()
        max_input_sequences_attributes = np.max(input_sequences_attributes)
        min_input_sequences_attributes = np.min(input_sequences_attributes)
        if not args.non_regularized_dimension:
            correlation_matrix = np.corrcoef(cond_latent_codes, rowvar=False)
            non_regularized_dimension = np.argmin(np.abs(correlation_matrix[0, :]))
            logging.info(f"Setting non regularized dimension to {non_regularized_dimension}. It is the least "
                         f"correlated dimension with the regularized dimension {args.regularized_dimension}. "
                         f"Correlation coefficient is: {correlation_matrix[0, non_regularized_dimension]}")
        else:
            non_regularized_dimension = args.non_regularized_dimension
        reg_dim_data = cond_latent_codes[:, args.regularized_dimension]
        non_reg_dim_data = cond_latent_codes[:, non_regularized_dimension]
        logging.info("-----------------------------------------------------------------------------------------------")
        logging.info("Encoding-decoding sequences with unconditional model...")
        unconditional_model = keras.saving.load_model(args.unconditional_model_path, compile=False)
        unconditional_model.trainable = False
        unconditional_model.compile(run_eagerly=True)
        uncond_decoded_sequences, uncond_latent_codes, _, _, _ = unconditional_model.predict(
            dataset, steps=args.dataset_cardinality // args.batch_size
        )
        logging.info("-----------------------------------------------------------------------------------------------")
        logging.info("Correlation coefficients for encoded sequences...")
        enc_reg_pearson_coefficient = scipy.stats.pearsonr(input_sequences_attributes, reg_dim_data)
        logging.info(f"Pearson coefficient of input sequences attributes with regularized dimension "
                     f"{args.regularized_dimension} is: {enc_reg_pearson_coefficient.statistic:.6f}")
        enc_reg_spearman_coefficient = scipy.stats.spearmanr(input_sequences_attributes, reg_dim_data)
        logging.info(f"Spearman coefficient of input sequences attributes with regularized dimension "
                     f"{args.regularized_dimension} is: {enc_reg_spearman_coefficient.statistic:.6f}")
        enc_non_reg_pearson_coefficient = scipy.stats.pearsonr(input_sequences_attributes, non_reg_dim_data)
        logging.info(f"Pearson coefficient of input sequences attribute with non-regularized dimension "
                     f"{non_regularized_dimension} is: {enc_non_reg_pearson_coefficient.statistic:.6f}")
        enc_non_reg_spearman_coefficient = scipy.stats.spearmanr(input_sequences_attributes, non_reg_dim_data)
        logging.info(f"Spearman coefficient of input sequences attributes with non-regularized dimension "
                     f"{non_regularized_dimension} is: {enc_non_reg_spearman_coefficient.statistic:.6f}")
        logging.info("-----------------------------------------------------------------------------------------------")
        z_i_cond_latents = cond_latent_codes[:, args.regularized_dimension]
        metric_names = {
            'kl_divergence': 'KL Divergence',
            'hellinger_distance': 'Hellinger Distance',
            'overlap_coefficient': 'Overlap Coefficient',
            'total_variation': 'Total Variation Distance',
            'jensen_shannon': 'Jensen-Shannon Divergence',
            'wasserstein': 'Wasserstein Distance',
            'mmd_rbf': 'MMD (RBF Kernel)',
            'mmd_linear': 'MMD (Linear Kernel)',
            'mmd_polynomial': 'MMD (Polynomial Kernel)'
        }
        logging.info("Regularized dimension metrics for conditional encoded sequences...")
        z_i_cond_metrics = metrics.plot_all_metrics(z_i_cond_latents, methods=['histogram', 'kde'])
        logging.info("Distribution Metrics:")
        logging.info("-" * 40)
        for key, name in metric_names.items():
            logging.info(f"{name:30s}: {z_i_cond_metrics[key]:.6f}")
        logging.info("-----------------------------------------------------------------------------------------------")
        logging.info("Regularized dimension metrics for unconditional encoded sequences...")
        z_i_uncond_latents = uncond_latent_codes[:, args.regularized_dimension]
        z_i_uncond_metrics = metrics.plot_all_metrics(z_i_uncond_latents, methods=['histogram', 'kde'])
        logging.info("Distribution Metrics:")
        logging.info("-" * 40)
        for key, name in metric_names.items():
            logging.info(f"{name:30s}: {z_i_uncond_metrics[key]:.6f}")
        # -----------------------------------------------------------------------------------------------
        regularization_scatter_plot(
            output_path=str(output_dir / "encoded_sequences_reg_latent_space.png"),
            title="",
            reg_dim_data=reg_dim_data,
            non_reg_dim_data=non_reg_dim_data,
            attributes=input_sequences_attributes,
            colorbar=True,
            vmax=max_input_sequences_attributes if not args.plot_power_norm else None,
            vmin=min_input_sequences_attributes if not args.plot_power_norm else None,
            norm=colors.PowerNorm(
                gamma=args.plot_power_norm,
                vmax=max_input_sequences_attributes,
                vmin=min_input_sequences_attributes,
            ) if args.plot_power_norm else None
        )
        logging.info("-----------------------------------------------------------------------------------------------")
        logging.info("Correlation coefficients for decoded sequences...")
        decoded_sequences_attrs, hold_note_start_seq_count = utilities.compute_sequences_attributes(
            uncond_decoded_sequences, attribute, args.sequence_length)
        logging.info(f"Decoded {hold_note_start_seq_count} sequences that start with an hold note token "
                     f"({hold_note_start_seq_count * 100 / args.dataset_cardinality:.2f}%).")
        dec_reg_pearson_coefficient = scipy.stats.pearsonr(decoded_sequences_attrs, reg_dim_data)
        logging.info(f"Pearson coefficient of decoded sequences attributes with regularized dimension "
                     f"{args.regularized_dimension} is: {dec_reg_pearson_coefficient.statistic:.6f}")
        dec_reg_spearman_coefficient = scipy.stats.spearmanr(decoded_sequences_attrs, reg_dim_data)
        logging.info(f"Spearman coefficient of decoded sequences attributes with regularized dimension "
                     f"{args.regularized_dimension} is: {dec_reg_spearman_coefficient.statistic:.6f}")
        dec_non_reg_pearson_coefficient = scipy.stats.pearsonr(decoded_sequences_attrs, non_reg_dim_data)
        logging.info(f"Pearson coefficient of decoded sequences attribute with non-regularized dimension "
                     f"{non_regularized_dimension} is: {dec_non_reg_pearson_coefficient.statistic:.6f}")
        dec_non_reg_spearman_coefficient = scipy.stats.spearmanr(decoded_sequences_attrs, non_reg_dim_data)
        logging.info(f"Spearman coefficient of decoded sequences attributes with non-regularized dimension "
                     f"{non_regularized_dimension} is: {dec_non_reg_spearman_coefficient.statistic:.6f}")
        # -----------------------------------------------------------------------------------------------
        regularization_scatter_plot(
            output_path=str(output_dir / "decoded_sequences_reg_latent_space.png"),
            title="",
            reg_dim_data=reg_dim_data,
            non_reg_dim_data=non_reg_dim_data,
            attributes=decoded_sequences_attrs,
            colorbar=True,
            vmax=max_input_sequences_attributes if not args.plot_power_norm else None,
            vmin=min_input_sequences_attributes if not args.plot_power_norm else None,
            norm=colors.PowerNorm(
                gamma=args.plot_power_norm,
                vmax=max_input_sequences_attributes,
                vmin=min_input_sequences_attributes,
            ) if args.plot_power_norm else None
        )


def regularization_scatter_plot(output_path: str,
                                title: str,
                                reg_dim_data,
                                non_reg_dim_data,
                                attributes,
                                colorbar: bool = True,
                                vmin: float = None,
                                vmax: float = None,
                                norm: colors.FuncNorm = None):
    with plt.rc_context(utilities.get_matplotlib_context()):
        plt.scatter(x=non_reg_dim_data,
                    y=reg_dim_data,
                    c=attributes,
                    norm=norm,
                    vmin=vmin,
                    vmax=vmax,
                    cmap='viridis',
                    alpha=0.8,
                    edgecolors='none')
        if colorbar:
            plt.colorbar()
        plt.xlabel(r'$z_{\ell}$')
        plt.ylabel('$z_i$')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_path, format='png', dpi=300)
        plt.close()


if __name__ == '__main__':
    parser = utilities.get_arg_parser("")
    parser.add_argument('--plot-power-norm', help='Exponent for the color bar\'s power normalization.',
                        required=False, type=float, default=0.0)
    parser.add_argument('--non-regularized-dimension', help='Index of the latent code non regularized dimension.',
                        required=False, type=int)
    os.environ["KERAS_BACKEND"] = "tensorflow"
    vargs = parser.parse_args()
    if vargs.seed:
        keras.utils.set_random_seed(vargs.seed)
        tf.config.experimental.enable_op_determinism()
    test_model_regularization(vargs)

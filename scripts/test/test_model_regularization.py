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
        # ------------------------------------------------------------------------------------------------------------
        logging.info("Encoding-decoding sequences with conditional model...")
        conditional_model = keras.saving.load_model(args.conditional_model_path, compile=False)
        conditional_model.trainable = False
        conditional_model.compile(run_eagerly=True)
        (cond_decoded_sequences, cond_latent_codes, input_sequences, input_sequences_attributes,
            _) = conditional_model.predict(
                dataset, steps=args.dataset_cardinality//args.batch_size
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
        # ------------------------------------------------------------------------------------------------------------
        logging.info("Encoding-decoding sequences with unconditional model...")
        unconditional_model = keras.saving.load_model(args.unconditional_model_path, compile=False)
        unconditional_model.trainable = False
        unconditional_model.compile(run_eagerly=True)
        uncond_decoded_sequences, uncond_latent_codes, _, _, _ = unconditional_model.predict(
            dataset, steps=args.dataset_cardinality//args.batch_size
        )
        # ------------------------------------------------------------------------------------------------------------
        logging.info("Regularization for encoded sequences...")
        enc_reg_pearson_coefficient = scipy.stats.pearsonr(input_sequences_attributes, reg_dim_data)
        logging.info(f"Pearson coefficient of input sequences attributes with regularized dimension "
                     f"{args.regularized_dimension} is: {enc_reg_pearson_coefficient.statistic}")
        enc_reg_spearman_coefficient = scipy.stats.spearmanr(input_sequences_attributes, reg_dim_data)
        logging.info(f"Spearman coefficient of input sequences attributes with regularized dimension "
                     f"{args.regularized_dimension} is: {enc_reg_spearman_coefficient.statistic}")
        enc_non_reg_pearson_coefficient = scipy.stats.pearsonr(input_sequences_attributes, non_reg_dim_data)
        logging.info(f"Pearson coefficient of input sequences attribute with non-regularized dimension "
                     f"{non_regularized_dimension} is: {enc_non_reg_pearson_coefficient.statistic}")
        enc_non_reg_spearman_coefficient = scipy.stats.spearmanr(input_sequences_attributes, non_reg_dim_data)
        logging.info(f"Spearman coefficient of input sequences attributes with non-regularized dimension "
                     f"{non_regularized_dimension} is: {enc_non_reg_spearman_coefficient.statistic}")
        mu_cond, sigma_cond = np.mean(cond_latent_codes, axis=0), np.cov(cond_latent_codes, rowvar=False)
        mu_uncond, sigma_uncond = np.mean(uncond_latent_codes, axis=0), np.cov(uncond_latent_codes, rowvar=False)
        mu_prior, sigma_prior = np.zeros(conditional_model._z_size), np.identity(conditional_model._z_size)
        z_fid = metrics.compute_gaussian_fid(mu1=mu_cond, sigma1=sigma_cond, mu2=mu_uncond, sigma2=sigma_uncond)
        logging.info(f"FD between conditional and unconditional latent spaces is {z_fid}")
        z_fid_prior = metrics.compute_gaussian_fid(mu1=mu_cond, sigma1=sigma_cond, mu2=mu_prior, sigma2=sigma_prior)
        logging.info(f"FD between conditional and prior latent spaces is {z_fid_prior}")
        z_mmd_rbf = metrics.compute_mmd_rbf(cond_latent_codes, uncond_latent_codes)
        logging.info(f"MMD RBF between conditional and unconditional latent spaces is {z_mmd_rbf}")
        z_mmd_poly = metrics.compute_mmd_polynomial(cond_latent_codes, uncond_latent_codes)
        logging.info(f"MMD polynomial between conditional and unconditional latent spaces is {z_mmd_poly}")
        regularization_scatter_plot(
            output_path=str(output_dir/"encoded_sequences_reg_latent_space.png"),
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
        # ------------------------------------------------------------------------------------------------------------
        logging.info("Regularization for decoded sequences...")
        decoded_sequences_attrs, hold_note_start_seq_count = utilities.compute_sequences_attributes(
            uncond_decoded_sequences, attribute, args.sequence_length)
        logging.info(f"Decoded {hold_note_start_seq_count} sequences that start with an hold note token "
                     f"({hold_note_start_seq_count*100/args.dataset_cardinality:.2f}%).")
        dec_reg_pearson_coefficient = scipy.stats.pearsonr(decoded_sequences_attrs, reg_dim_data)
        logging.info(f"Pearson coefficient of decoded sequences attributes with regularized dimension "
                     f"{args.regularized_dimension} is: {dec_reg_pearson_coefficient.statistic}")
        dec_reg_spearman_coefficient = scipy.stats.spearmanr(decoded_sequences_attrs, reg_dim_data)
        logging.info(f"Spearman coefficient of decoded sequences attributes with regularized dimension "
                     f"{args.regularized_dimension} is: {dec_reg_spearman_coefficient.statistic}")
        dec_non_reg_pearson_coefficient = scipy.stats.pearsonr(decoded_sequences_attrs, non_reg_dim_data)
        logging.info(f"Pearson coefficient of decoded sequences attribute with non-regularized dimension "
                     f"{non_regularized_dimension} is: {dec_non_reg_pearson_coefficient.statistic}")
        dec_non_reg_spearman_coefficient = scipy.stats.spearmanr(decoded_sequences_attrs, non_reg_dim_data)
        logging.info(f"Spearman coefficient of decoded sequences attributes with non-regularized dimension "
                     f"{non_regularized_dimension} is: {dec_non_reg_spearman_coefficient.statistic}")
        regularization_scatter_plot(
            output_path=str(output_dir/"decoded_sequences_reg_latent_space.png"),
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

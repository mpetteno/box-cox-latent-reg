import logging
import os
from pathlib import Path

import keras
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import tensorflow as tf

import utilities


def test_model_regularization(args):
    output_dir = Path(args.output_path) / Path(args.model_path).stem / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    for attribute in args.attributes:
        dataset = utilities.load_dataset(dataset_path=args.test_dataset_path,
                                         sequence_length=args.sequence_length,
                                         attribute=attribute,
                                         batch_size=args.batch_size,
                                         shift=keras.backend.epsilon(),
                                         parse_sequence_feature=True)
        model = keras.saving.load_model(args.model_path, compile=False)
        model.trainable = False
        model.compile(run_eagerly=True)
        (decoded_sequences, latent_codes, input_sequences, input_sequences_attributes,
            batch_norm_sequences_attributes) = model.predict(dataset, steps=args.dataset_cardinality//args.batch_size)
        max_input_sequences_attributes = np.max(input_sequences_attributes, axis=0)
        min_input_sequences_attributes = np.min(input_sequences_attributes, axis=0)

        if not args.non_regularized_dimension:
            correlation_matrix = np.corrcoef(latent_codes, rowvar=False)
            non_regularized_dimension = np.argmin(np.abs(correlation_matrix[0, :]))
            logging.info(f"Setting non regularized dimension to {non_regularized_dimension}. It is the least "
                         f"correlated dimension with the regularized dimension {args.regularized_dimension}. "
                         f"Correlation coefficient is: {correlation_matrix[0, non_regularized_dimension]}")
        else:
            non_regularized_dimension = args.non_regularized_dimension

        reg_dim_data = latent_codes[:, args.regularized_dimension]
        non_reg_dim_data = latent_codes[:, non_regularized_dimension]

        # Regularization for encoded sequences
        enc_reg_corr_mat = np.corrcoef(reg_dim_data, input_sequences_attributes[:, 0])
        enc_non_reg_corr_mat = np.corrcoef(non_reg_dim_data, input_sequences_attributes[:, 0])
        regularization_scatter_plot(
            output_path=str(output_dir/"encoded_sequences_reg_latent_space.png"),
            title="",
            reg_dim_data=reg_dim_data,
            non_reg_dim_data=non_reg_dim_data,
            attributes=input_sequences_attributes[:, 0],
            colorbar=True,
            vmax=max_input_sequences_attributes if not args.plot_power_norm else None,
            vmin=min_input_sequences_attributes if not args.plot_power_norm else None,
            norm=colors.PowerNorm(
                gamma=args.plot_power_norm,
                vmax=max_input_sequences_attributes,
                vmin=min_input_sequences_attributes,
            ) if args.plot_power_norm else None
        )

        # Regularization for generated sequences
        decoded_sequences_attrs, hold_note_start_seq_count = utilities.compute_sequences_attributes(
            decoded_sequences, attribute, args.sequence_length)
        gen_reg_corr_mat = np.corrcoef(reg_dim_data, decoded_sequences_attrs)
        gen_non_reg_corr_mat = np.corrcoef(non_reg_dim_data, decoded_sequences_attrs)
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

        # Logging
        logging.info(f"Correlation coefficient of input sequences attributes with regularized dimension "
                     f"{args.regularized_dimension} is: {enc_reg_corr_mat[0, 1]}")
        logging.info(f"Correlation coefficient of input sequences attribute with non-regularized dimension "
                     f"{non_regularized_dimension} is: {enc_non_reg_corr_mat[0, 1]}")
        logging.info(f"Correlation coefficient of generated sequences attributes with regularized dimension "
                     f"{args.regularized_dimension} is: {gen_reg_corr_mat[0, 1]}")
        logging.info(f"Correlation coefficient of generated sequences attribute with non-regularized dimension "
                     f"{non_regularized_dimension} is: {gen_non_reg_corr_mat[0, 1]}")
        logging.info(f"Decoded {hold_note_start_seq_count} sequences that start with an hold note token "
                     f"({hold_note_start_seq_count*100/args.dataset_cardinality:.2f}%).")

@mpl.rc_context({'text.usetex': True, 'font.family': 'serif', 'font.size': 20, 'font.serif': 'Computer Modern Roman',
                 'lines.linewidth': 1.5})
def regularization_scatter_plot(output_path: str,
                                title: str,
                                reg_dim_data,
                                non_reg_dim_data,
                                attributes,
                                colorbar: bool = True,
                                vmin: float = None,
                                vmax: float = None,
                                norm: colors.FuncNorm = None):
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
    logging.getLogger().setLevel(vargs.logging_level)
    test_model_regularization(vargs)

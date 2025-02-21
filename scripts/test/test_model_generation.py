import logging
import os
import random
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
import scipy
import tensorflow as tf
from resolv_mir.note_sequence.io import midi_io
from resolv_pipelines.data.representation.mir import PitchSequenceRepresentation

import utilities


def test_model_generation(args):
    for attribute in args.attributes:
        output_dir = Path(args.output_path) / Path(args.conditional_model_path).stem / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        utilities.set_log_handler(output_dir=output_dir.parent, log_level=getattr(logging, vargs.logging_level))
        conditional_model = keras.saving.load_model(args.conditional_model_path, compile=False)
        conditional_model.compile(run_eagerly=True)
        conditional_model.trainable = False
        # Get normalizing flow if model uses PT regularization
        normalizing_flow_ar_layer = conditional_model._regularizers.get("nf_ar", None)
        if normalizing_flow_ar_layer:
            normalizing_flow = normalizing_flow_ar_layer._normalizing_flow
            normalizing_flow._add_loss = False
            normalizing_flow._bijectors[1]._training = False
        else:
            normalizing_flow = None
        # ------------------------------------------------------------------------------------------------------------
        logging.info("Generating sequences...")
        latent_codes = conditional_model.get_latent_codes(keras.ops.convert_to_tensor(args.dataset_cardinality)).numpy()
        decoder_inputs = keras.ops.convert_to_tensor(args.sequence_length, dtype="int32")
        if args.control_reg_dim:
            latent_codes[:, args.regularized_dimension] = keras.ops.linspace(
                start=args.latent_min_val, stop=args.latent_max_val, num=args.dataset_cardinality
            )
        generated_sequences = conditional_model.decode(inputs=(latent_codes, decoder_inputs))
        generated_sequences_attrs, hold_note_start_seq_count = utilities.compute_sequences_attributes(
            generated_sequences.numpy(), attribute, args.sequence_length)
        # ------------------------------------------------------------------------------------------------------------
        logging.info("Plot generated sequences attributes histogram...")
        filename = f'{str(output_dir)}/histogram_generated_{attribute}_{args.histogram_bins}_bins.png'
        logging.info(f"Plotting generated histogram with {args.histogram_bins} bins for attribute {attribute}...")
        with plt.rc_context(utilities.get_matplotlib_context()):
            plt.hist(generated_sequences_attrs, bins=args.histogram_bins, density=True, stacked=True, color='#4c92c3',
                     alpha=0.7)
            plt.grid(linestyle=':')
            plt.savefig(filename, format='png', dpi=300)
            plt.close()
        # ------------------------------------------------------------------------------------------------------------
        logging.info("Computing coefficients and plotting graph with the best linear fitting model...")
        reg_dim_data = latent_codes[:, args.regularized_dimension]
        if conditional_model._attribute_processing_layer:
            bn_layer = conditional_model._attribute_processing_layer
            shift = bn_layer.moving_mean
            scale = np.sqrt(bn_layer.moving_variance) + bn_layer.epsilon
            reg_dim_data = reg_dim_data * scale + shift
        if normalizing_flow:
            reg_dim_data = normalizing_flow(inputs=reg_dim_data, training=False)
        pearson_coefficient = scipy.stats.pearsonr(generated_sequences_attrs, reg_dim_data)
        spearman_coefficient = scipy.stats.spearmanr(generated_sequences_attrs, reg_dim_data)
        slope, intercept = plot_reg_dim_vs_attribute(output_path=str(output_dir / 'reg_dim_vs_attribute.png'),
                                                     reg_dim_data=reg_dim_data,
                                                     attribute_data=generated_sequences_attrs)
        logging.info(f"Best linear model fit parameters. Slope: {slope}, Intercept: {intercept}")
        logging.info(f"Pearson coefficient {pearson_coefficient.statistic}.")
        logging.info(f"Spearman coefficient {spearman_coefficient.statistic}.")
        logging.info(f"Generated {hold_note_start_seq_count} sequences that start with an hold note token "
                     f"({hold_note_start_seq_count * 100 / args.dataset_cardinality:.2f}%).")
        # ------------------------------------------------------------------------------------------------------------
        logging.info("Converting generated sequences to MIDI and save to disk...")
        representation = PitchSequenceRepresentation(args.sequence_length)
        seq_to_save_count = min(args.dataset_cardinality, args.num_midi_to_save)
        random_idxes = [random.randint(0, args.dataset_cardinality) for _ in range(seq_to_save_count)]
        for idx, generated_sequence in enumerate(keras.ops.take(generated_sequences, indices=random_idxes, axis=0)):
            generated_note_sequence = representation.to_canonical_format(generated_sequence, attributes=None)
            filename = f"midi/{attribute}_{latent_codes[random_idxes[idx], args.regularized_dimension]:.2f}.midi"
            midi_io.note_sequence_to_midi_file(generated_note_sequence,
                                               Path(args.output_path) / Path(args.model_path).stem / filename)
        # ------------------------------------------------------------------------------------------------------------
        # if attribute regularization is carried out by a normalizing flow, compute the minimum and maximum mapped
        # latent values
        if normalizing_flow:
            attribute_values = utilities.load_flat_dataset(dataset_path=args.test_dataset_path,
                                                           sequence_length=args.sequence_length,
                                                           attribute=attribute,
                                                           batch_size=args.batch_size,
                                                           parse_sequence_feature=False)
            attr_min_val = np.min(attribute_values)
            nf_min_val = normalizing_flow(inputs=keras.ops.convert_to_tensor([[attr_min_val]]), inverse=True).numpy()
            logging.info(f"Minimum attribute value in dataset is: {attr_min_val}. It is mapped to "
                         f"{nf_min_val[0][0]} in the latent regularized dimension.")
            attr_max_val = np.max(attribute_values)
            nf_max_val = normalizing_flow(inputs=keras.ops.convert_to_tensor([[attr_max_val]]), inverse=True).numpy()
            logging.info(f"Maximum attribute value in dataset is: {attr_max_val}. It is mapped to "
                         f"{nf_max_val[0][0]} in the latent regularized dimension.")


def plot_reg_dim_vs_attribute(output_path: str,
                              reg_dim_data,
                              attribute_data):
    with plt.rc_context(utilities.get_matplotlib_context()):
        slope, intercept = np.polyfit(reg_dim_data, attribute_data, 1)
        plt.scatter(reg_dim_data, attribute_data, color='C0', s=5, alpha=0.35, edgecolors='none')
        plt.plot(reg_dim_data, slope * reg_dim_data + intercept, color='#fd5656')
        plt.xlabel(r'$z_i$')
        plt.ylabel(r'$a$')
        plt.tight_layout()
        plt.savefig(output_path, format='png', dpi=300)
        plt.close()
        return slope, intercept


if __name__ == '__main__':
    parser = utilities.get_arg_parser("")
    parser.add_argument('--control-reg-dim', action="store_true",
                        help='Control the regularized latent dimension of the sampled latent codes using the min and '
                             'max values provided in `--latent-min-val` and `--latent-max-val.`')
    parser.add_argument('--latent-min-val', help='Minimum value for manipulation of the regularized latent dimension.',
                        default=-4, required=False, type=float)
    parser.add_argument('--latent-max-val', help='Maximum value for manipulation of the regularized latent dimension.',
                        default=4, required=False, type=float)
    parser.add_argument('--num-midi-to-save', help='Number of generated sequences to save as MIDI file. '
                                                   'The N sequences will be chosen randomly.', required=True, type=int)
    parser.add_argument('--histogram-bins', help='Number of bins for the histogram.', default=120, required=False,
                        type=int)
    os.environ["KERAS_BACKEND"] = "tensorflow"
    vargs = parser.parse_args()
    if vargs.seed:
        keras.utils.set_random_seed(vargs.seed)
        tf.config.experimental.enable_op_determinism()
    test_model_generation(vargs)

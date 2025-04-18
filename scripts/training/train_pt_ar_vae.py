"""
Usage example:

    python ./scripts/training/train_pt_ar_vae.py \
        --model-config-path=./scripts/training/config/model.json \
        --trainer-config-path=./scripts/training/config/trainer.json \
        --train-dataset-config-path=./scripts/training/config/train_dataset.json \
        --val-dataset-config-path=./scripts/training/config/val_dataset.json \
        --attribute="contour" \
        --reg-dim=0 \
        --power=0.0 \
        --shift=0.0 \
        --gpus=0

"""
import json
import logging

import keras
from resolv_ml.models.dlvm.normalizing_flows.base import NormalizingFlow
from resolv_ml.training.callbacks import LearningRateLoggerCallback
from resolv_ml.utilities.bijectors import BatchNormalization, BoxCox
from resolv_ml.utilities.regularizers.attribute import NormalizingFlowAttributeRegularizer
from resolv_ml.utilities.schedulers import get_scheduler

from scripts.training import utilities

if __name__ == '__main__':
    arg_parser = utilities.get_arg_parser(description="Train AR-VAE model with PT attribute regularization.")
    arg_parser.add_argument('--attribute', help='Attribute to regularize.', required=True)
    arg_parser.add_argument('--reg-dim', help='Latent code regularization dimension.', default=0, type=int)
    arg_parser.add_argument('--power', help='Initial value for the power transform\'s power parameter.',
                            default=0.0, type=float)
    arg_parser.add_argument('--shift', help='Initial value for the power transform\'s shift parameter.',
                            default=0.0, type=float)
    args = arg_parser.parse_args()

    logging.getLogger().setLevel(args.logging_level)

    strategy = utilities.get_distributed_strategy(args.gpus, args.gpu_memory_growth)
    with strategy.scope():
        train_data, val_data, input_shape = utilities.load_datasets(
            train_dataset_config_path=args.train_dataset_config_path,
            val_dataset_config_path=args.val_dataset_config_path,
            trainer_config_path=args.trainer_config_path,
            attribute=args.attribute
        )

        with open(args.model_config_path) as file:
            model_config = json.load(file)
            schedulers_config = model_config["schedulers"]

        with open(args.trainer_config_path) as file:
            fit_config = json.load(file)["fit"]

        vae = utilities.get_model(
            model_config_path=args.model_config_path,
            trainer_config_path=args.trainer_config_path,
            hierarchical_decoder=args.hierarchical_decoder,
            attribute_regularizers={
                "nf_ar": NormalizingFlowAttributeRegularizer(
                    normalizing_flow=NormalizingFlow(
                        bijectors=[
                            BoxCox(
                                power_init_value=args.power,
                                shift_init_value=args.shift if args.shift else 1e-5,
                                power_trainable=False,
                                shift_trainable=False
                            ),
                            BatchNormalization(scale=False, center=False)
                        ],
                        add_loss=False
                    ),
                    reg_weight_scheduler=get_scheduler(
                        schedule_type=schedulers_config["attr_reg_gamma"]["type"],
                        schedule_config=schedulers_config["attr_reg_gamma"]["config"]
                    ),
                    loss_fn=keras.losses.MeanAbsoluteError(),
                    regularization_dimension=args.reg_dim
                )
            }
        )
        vae.build(input_shape)
        trainer = utilities.get_trainer(model=vae, trainer_config_path=args.trainer_config_path)
        history = trainer.train(
            train_data=train_data[0],
            train_data_cardinality=train_data[1],
            validation_data=val_data[0],
            validation_data_cardinality=val_data[1],
            custom_callbacks=[LearningRateLoggerCallback()]
        )

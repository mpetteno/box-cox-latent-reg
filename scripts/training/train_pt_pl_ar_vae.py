"""
Usage example:

    python ./scripts/training/train_pt_pl_ar_vae.py \
        --model-config-path=./scripts/training/config/model.json \
        --trainer-config-path=./scripts/training/config/trainer.json \
        --train-dataset-config-path=./scripts/training/config/train_dataset.json \
        --val-dataset-config-path=./scripts/training/config/val_dataset.json \
        --attribute="contour" \
        --reg-dim=0 \
        --power=0.0 \
        --shift=0.0 \
        --scale-factor=1.0 \
        --gpus=0

"""
import json
import logging

import keras
from resolv_ml.models.dlvm.normalizing_flows.base import NormalizingFlow
from resolv_ml.training.callbacks import LearningRateLoggerCallback
from resolv_ml.utilities.bijectors import BatchNormalization, BoxCox
from resolv_ml.utilities.regularizers.attribute import SignAttributeRegularizer
from resolv_ml.utilities.schedulers import get_scheduler, Scheduler
from tensorflow_probability import distributions as tfd

from scripts.training import utilities


@keras.saving.register_keras_serializable(package="AttributeRegularizer", name="PTPLAttributeRegularizer")
class PTPLAttributeRegularizer(SignAttributeRegularizer):

    def __init__(self,
                 normalizing_flow: NormalizingFlow,
                 weight_scheduler: Scheduler = None,
                 loss_fn: keras.losses.Loss = keras.losses.MeanAbsoluteError(),
                 regularization_dimension: int = 0,
                 scale_factor: float = 1.0,
                 name: str = "sign_attr_reg",
                 **kwargs):
        super(PTPLAttributeRegularizer, self).__init__(
            weight_scheduler=weight_scheduler,
            loss_fn=loss_fn,
            regularization_dimension=regularization_dimension,
            scale_factor=scale_factor,
            name=name,
            **kwargs
        )
        self._normalizing_flow = normalizing_flow

    def build(self, input_shape):
        super().build(input_shape)
        attribute_input_shape = input_shape[1]
        self._normalizing_flow.build(attribute_input_shape)

    def _compute_attribute_regularization_loss(self,
                                               latent_codes,
                                               attributes,
                                               prior: tfd.Distribution,
                                               posterior: tfd.Distribution,
                                               current_step=None,
                                               training: bool = False,
                                               evaluate: bool = False):
        nf_base_dist = tfd.MultivariateNormalDiag(
            loc=posterior.loc[:, self._regularization_dimension],
            scale_diag=posterior.scale.diag[:, self._regularization_dimension]
        )
        transformed_attributes = self._normalizing_flow(attributes,
                                                        base_distribution=nf_base_dist,
                                                        current_step=current_step,
                                                        inverse=True,
                                                        training=training)
        return super()._compute_attribute_regularization_loss(
            latent_codes=latent_codes,
            attributes=transformed_attributes,
            prior=prior,
            posterior=posterior,
            current_step=current_step,
            training=training,
            evaluate=evaluate
        )

    def get_config(self):
        base_config = super().get_config()
        config = {
            "normalizing_flow": keras.saving.serialize_keras_object(self._normalizing_flow)
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        normalizing_flow = keras.saving.deserialize_keras_object(config.pop("normalizing_flow"))
        loss_fn = keras.saving.deserialize_keras_object(config.pop("loss_fn"))
        return cls(loss_fn=loss_fn, normalizing_flow=normalizing_flow, **config)


if __name__ == '__main__':
    arg_parser = utilities.get_arg_parser(description="Train AR-VAE model with PT+PL attribute regularization.")
    arg_parser.add_argument('--attribute', help='Attribute to regularize.', required=True)
    arg_parser.add_argument('--reg-dim', help='Latent code regularization dimension.', default=0, type=int)
    arg_parser.add_argument('--power', help='Initial value for the power transform\'s power parameter.',
                            default=0.0, type=float)
    arg_parser.add_argument('--shift', help='Initial value for the power transform\'s shift parameter.',
                            default=0.0, type=float)
    arg_parser.add_argument('--scale-factor', help='Scale factor for tanh in PL regularization loss.', default=1.0,
                            type=float)
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
                "pt_pl_ar": PTPLAttributeRegularizer(
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
                    weight_scheduler=get_scheduler(
                        schedule_type=schedulers_config["attr_reg_gamma"]["type"],
                        schedule_config=schedulers_config["attr_reg_gamma"]["config"]
                    ),
                    loss_fn=keras.losses.MeanAbsoluteError(),
                    regularization_dimension=args.reg_dim,
                    scale_factor=args.scale_factor
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

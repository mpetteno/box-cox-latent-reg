import keras
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import boxcox

from resolv_ml.utilities.bijectors import BatchNormalization, BoxCox

np.random.seed(0)
tf.random.set_seed(0)

tfd = tfp.distributions
tfb = tfp.bijectors

plt.rcParams['figure.figsize'] = (10, 6)

n_samples = 99840
batch_size = 512

# Data exponential generation
data_dist = tfd.Sample(tfd.Exponential(1.), sample_shape=n_samples)
data = data_dist.sample()
plt.hist(data.numpy(), bins=120, density=True, alpha=0.6, color='blue')
plt.title("Target Distribution")
plt.show()

# Data gaussian target
z_dist = tfd.Normal(loc=0., scale=1.)
z_samples = z_dist.sample(n_samples)
plt.hist(z_samples.numpy(), bins=120, density=True, alpha=0.6, color='red')
plt.title("Base Distribution")
plt.show()


class NFModel(keras.Model):

    def __init__(self, bijectors: tfb.Bijector):
        super().__init__()
        self._bijectors = bijectors

    def build(self, input_shape):
        for bij in self._bijectors:
            bij.build(input_shape + (1,))
        self._bijectors_chain = tfb.Chain(self._bijectors)
        self.nf = tfd.TransformedDistribution(
            tfd.MultivariateNormalDiag(loc=tf.zeros(input_shape), scale_diag=tf.ones(input_shape)),
            bijector=self._bijectors_chain
        )

    def call(self, inputs, training=None, mask=None):
        inputs = tf.expand_dims(inputs, axis=-1)
        outputs = self.nf.bijector.inverse(inputs)
        log_likelihood = self.nf.log_prob(inputs)
        negative_log_likelihood = -tf.reduce_mean(log_likelihood)
        self.add_loss(negative_log_likelihood)
        return outputs


class EarlyStoppingOnIncrease(keras.callbacks.Callback):
    def __init__(self, monitor='val_loss'):
        super(EarlyStoppingOnIncrease, self).__init__()
        self.monitor = monitor
        self.best_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get(self.monitor)
        if current_loss is None:
            return

        if current_loss < self.best_loss:
            self.best_loss = current_loss
        else:
            print(f'\nEpoch {epoch + 1}: early stopping as {self.monitor} has increased')
            self.model.stop_training = True


# Plot negative log-likelihood fn as function of lambda
lambdas = np.arange(0., 1., 0.01)
ng_lls = []
for lmbda in lambdas:
    static_model = NFModel([BoxCox(power_init_value=lmbda,
                                   shift_init_value=0.,
                                   power_trainable=False,
                                   shift_trainable=False)])
    static_model.build(input_shape=(n_samples,))
    neg_ll = -tf.reduce_mean(static_model.nf.log_prob(data)).numpy()
    ng_lls.append(neg_ll)
print(f"Minimum lambda log-likelihood is {lambdas[np.argmin(ng_lls)]}")
plt.plot(lambdas, ng_lls)
plt.title("Negative Log Likelihood function of lambda")
plt.show()

# Train a model
model = NFModel(bijectors=[BoxCox(power_init_value=1., shift_init_value=0., power_trainable=True, shift_trainable=True),
                           BatchNormalization(center=False, scale=False)])
model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-3,
            decay_steps=195,
            decay_rate=0.98
        )
    ),
    run_eagerly=True
)
result = model.fit(x=data, y=np.zeros(n_samples), epochs=3, batch_size=batch_size,
                   callbacks=[EarlyStoppingOnIncrease(monitor='loss')])
plt.plot(result.history["loss"])
plt.title("Training loss")
plt.show()

# Check Scipy Brent method optimizer value
tr_data, llm_lambda = boxcox(data, lmbda=None)
plt.hist(tr_data, bins=120, density=True, alpha=0.6, color='green')
plt.title("Scipy transformed inputs")
plt.show()
print(f"Scipy MLE power is: {llm_lambda}")

# Predict dataset
output = model.predict(data, batch_size=batch_size)
plt.hist(output, bins=120, density=True, alpha=0.6, color='green')
plt.title("SGD transformed inputs")
plt.show()

print(f"Trained Power: {model.nf.bijector.bijectors[0].power.value.numpy()}")
print(f"Trained Shift: {model.nf.bijector.bijectors[0].shift.value.numpy()}")
z_samples = model.nf.sample(n_samples / 512)
plt.hist(tf.reshape(z_samples, [-1]).numpy(), bins=120, density=True, alpha=0.6, color='green')
plt.title("SGD samples")
plt.show()

trans_z_samples = model.nf.bijector.inverse(z_samples)
plt.hist(tf.reshape(trans_z_samples, [-1]).numpy(), bins=120, density=True, alpha=0.6, color='green')
plt.title("SGD inverse samples")
plt.show()

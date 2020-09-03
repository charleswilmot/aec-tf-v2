import tensorflow as tf
from tensorflow import keras
import numpy as np
from custom_layers import custom_objects


def divide_no_nan(a, b, default=0.0):
    return np.divide(a, b, out=np.full_like(a, fill_value=default), where=b!=0)


class Agent(object):
    def __init__(self,
            policy_learning_rate, critic_learning_rate, encoder_learning_rate,
            exploration, n_simulations, scales, pathways):
        self.policy_learning_rate = policy_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.encoder_learning_rate = encoder_learning_rate
        self.policy_optimizer = {}
        self.critic_optimizer = {}
        self.encoder_optimizer = {}
        self.pathways = pathways
        self.scales = scales
        self.models = {}
        for pathway in pathways:
            #   POLICY
            self.models[pathway.name] = {}
            self.models[pathway.name]["policy_model"] = \
                keras.models.model_from_yaml(
                    pathway.policy_model_arch.pretty(resolve=True),
                    custom_objects=custom_objects
                )
            self.policy_optimizer[pathway.name] = keras.optimizers.Adam(self.policy_learning_rate)
            #   CRITIC
            self.models[pathway.name]["critic_model"] = \
                keras.models.model_from_yaml(
                    pathway.critic_model_arch.pretty(resolve=True),
                    custom_objects=custom_objects
                )
            self.critic_optimizer[pathway.name] = keras.optimizers.Adam(self.critic_learning_rate)
            #   ENCODERS / DECODERS
            self.models[pathway.name]["encoder_models"] = {}
            self.models[pathway.name]["decoder_models"] = {}
            for scale in scales:
                self.models[pathway.name]["encoder_models"][scale.name] = \
                    keras.models.model_from_yaml(
                        pathway.encoder_model_arch.pretty(resolve=True),
                        custom_objects=custom_objects
                    )
                self.models[pathway.name]["decoder_models"][scale.name] = \
                    keras.models.model_from_yaml(
                        pathway.decoder_model_arch.pretty(resolve=True),
                        custom_objects=custom_objects
                    )
            self.encoder_optimizer[pathway.name] = keras.optimizers.Adam(self.encoder_learning_rate)
        #   EXPLORATION NOISE
        self.exploration_params = exploration
        self.exploration_stddev = tf.Variable(exploration.stddev, dtype=tf.float32)
        self.exploration_prob = exploration.prob
        self.success_rate = None
        self.n_simulations = n_simulations
        ###

    def save_weights(self, path):
        for pathway_name in self.models:
            self.models[pathway_name]["policy_model"].save_weights(path + "/policy_model_{}".format(pathway_name))
            self.models[pathway_name]["critic_model"].save_weights(path + "/critic_model_{}".format(pathway_name))
            for scale_name in self.models[pathway_name]["encoder_models"]:
                self.models[pathway_name]["encoder_models"][scale_name].save_weights(path + "/encoder_model_{}_{}".format(pathway_name, scale_name))
                self.models[pathway_name]["decoder_models"][scale_name].save_weights(path + "/decoder_model_{}_{}".format(pathway_name, scale_name))

    def load_weights(self, path):
        for pathway_name in self.models:
            self.models[pathway_name]["policy_model"].load_weights(path + "/policy_model_{}".format(pathway_name))
            self.models[pathway_name]["critic_model"].load_weights(path + "/critic_model_{}".format(pathway_name))
            for scale_name in self.models[pathway_name]["encoder_models"]:
                self.models[pathway_name]["encoder_models"][scale_name].load_weights(path + "/encoder_model_{}_{}".format(pathway_name, scale_name))
                self.models[pathway_name]["decoder_models"][scale_name].load_weights(path + "/decoder_model_{}_{}".format(pathway_name, scale_name))

    @tf.function
    def create_all_variables(self, fake_frame_by_scale_pavro, fake_frame_by_scale_magno):
        # PAVRO
        fake_encodings_by_scale_pavro = {
            scale_name: self.models["pavro"]["encoder_models"][scale_name](frame)
            for scale_name, frame in fake_frame_by_scale_pavro.items()
        }
        fake_decodings_by_scale_pavro = {
            scale_name: self.models["pavro"]["decoder_models"][scale_name](frame)
            for scale_name, frame in fake_encodings_by_scale_pavro.items()
        }
        fake_actions_pavro = self.models["pavro"]["policy_model"](fake_encodings_by_scale_pavro)
        fake_return_estimate = self.models["pavro"]["critic_model"]((fake_encodings_by_scale_pavro, fake_actions_pavro))
        # MOGNO
        fake_encodings_by_scale_magno = {
            scale_name: self.models["magno"]["encoder_models"][scale_name](frame)
            for scale_name, frame in fake_frame_by_scale_magno.items()
        }
        fake_decodings_by_scale_magno = {
            scale_name: self.models["magno"]["decoder_models"][scale_name](frame)
            for scale_name, frame in fake_encodings_by_scale_magno.items()
        }
        fake_actions_magno = self.models["magno"]["policy_model"](fake_encodings_by_scale_magno)
        fake_return_estimate = self.models["magno"]["critic_model"]((fake_encodings_by_scale_magno, fake_actions_magno))

        targets = np.zeros(shape=(len(fake_actions_magno), 1))

        with tf.GradientTape() as tape:
            total_loss = tf.reduce_sum(self.get_encoder_loss(fake_frame_by_scale_pavro, "pavro"))
            variables = sum([model.variables for model in self.models["pavro"]["encoder_models"].values()] + \
                   [model.variables for model in self.models["pavro"]["decoder_models"].values()], [])
            grads = tape.gradient(total_loss, variables)
            self.encoder_optimizer["pavro"].apply_gradients(zip(grads, variables))

        with tf.GradientTape() as tape:
            return_estimates = self.get_return_estimates(fake_frame_by_scale_pavro, fake_actions_pavro, "pavro")
            loss_critic = keras.losses.Huber()(return_estimates, tf.stop_gradient(targets))
            variables = self.models["pavro"]["critic_model"].variables
            grads = tape.gradient(loss_critic, variables)
            self.critic_optimizer["pavro"].apply_gradients(zip(grads, variables))

        with tf.GradientTape() as tape:
            actions = self.get_actions(fake_frame_by_scale_pavro, "pavro", exploration=False)
            return_estimates = self.get_return_estimates(fake_frame_by_scale_pavro, actions, "pavro")
            loss_policy = - tf.reduce_sum(return_estimates)
            variables = self.models["pavro"]["policy_model"].variables
            grads = tape.gradient(loss_policy, variables)
            self.policy_optimizer["pavro"].apply_gradients(zip(grads, variables))

        with tf.GradientTape() as tape:
            total_loss = tf.reduce_sum(self.get_encoder_loss(fake_frame_by_scale_magno, "magno"))
            variables = sum([model.variables for model in self.models["magno"]["encoder_models"].values()] + \
                   [model.variables for model in self.models["magno"]["decoder_models"].values()], [])
            grads = tape.gradient(total_loss, variables)
            self.encoder_optimizer["magno"].apply_gradients(zip(grads, variables))

        with tf.GradientTape() as tape:
            return_estimates = self.get_return_estimates(fake_frame_by_scale_magno, fake_actions_magno, "magno")
            loss_critic = keras.losses.Huber()(return_estimates, tf.stop_gradient(targets))
            variables = self.models["magno"]["critic_model"].variables
            grads = tape.gradient(loss_critic, variables)
            self.critic_optimizer["magno"].apply_gradients(zip(grads, variables))

        with tf.GradientTape() as tape:
            actions = self.get_actions(fake_frame_by_scale_magno, "magno", exploration=False)
            return_estimates = self.get_return_estimates(fake_frame_by_scale_magno, actions, "magno")
            loss_policy = - tf.reduce_sum(return_estimates)
            variables = self.models["magno"]["policy_model"].variables
            grads = tape.gradient(loss_policy, variables)
            self.policy_optimizer["magno"].apply_gradients(zip(grads, variables))

    @tf.function
    def get_encodings(self, frame_by_scale, pathway_name):
        return {
            scale_name: self.models[pathway_name]["encoder_models"][scale_name](frame)
            for scale_name, frame in frame_by_scale.items()
        }

    @tf.function
    def get_reconstructions(self, frame_by_scale, pathway_name):
        encodings_by_scale = self.get_encodings(frame_by_scale, pathway_name)
        return {
            scale_name: self.models[pathway_name]["decoder_models"][scale_name](encoding)
            for scale_name, encoding in encodings_by_scale.items()
        }

    @tf.function
    def get_return_estimates(self, frame_by_scale, actions, pathway_name):
        encodings_by_scale = self.get_encodings(frame_by_scale, pathway_name)
        return self.models[pathway_name]["critic_model"]((encodings_by_scale, actions))

    @tf.function
    def get_actions(self, frame_by_scale, pathway_name, exploration=False):
        encodings_by_scale = self.get_encodings(frame_by_scale, pathway_name)
        pure_actions = self.models[pathway_name]["policy_model"](encodings_by_scale)
        if exploration:
            noises = tf.random.truncated_normal(
                shape=tf.shape(pure_actions),
                stddev=self.exploration_stddev,
            ) * tf.cast(tf.random.uniform(shape=tf.shape(pure_actions)) < self.exploration_prob, tf.float32)
            noisy_actions = tf.clip_by_value(
                pure_actions + noises,
                clip_value_min=-1,
                clip_value_max=1
            )
            return pure_actions, noisy_actions
        else:
            return pure_actions

    @tf.function
    def get_encoder_loss_by_scale(self, frame_by_scale, pathway_name):
        encoder_sizes = {
            pathway.name: list(pathway.encoder_model_arch.config.layers[0].config.kernel_size)
            for pathway in self.pathways
        }
        encoder_strides = {
            pathway.name: list(pathway.encoder_model_arch.config.layers[0].config.strides)
            for pathway in self.pathways
        }
        reconstructions_by_scale = self.get_reconstructions(frame_by_scale, pathway_name)
        patches_by_scale = {
            scale_name: tf.image.extract_patches(
                frame,
                sizes=[1] + encoder_sizes[pathway_name] + [1],
                strides=[1] + encoder_strides[pathway_name] + [1],
                rates=[1, 1, 1, 1],
                padding='VALID',
            ) for scale_name, frame in frame_by_scale.items()
        }
        return {
            scale_name: tf.reduce_mean(
                (patches_by_scale[scale_name] - reconstructions_by_scale[scale_name]) ** 2,
                axis=[1, 2, 3],
            ) for scale_name in patches_by_scale
        }

    @tf.function
    def get_encoder_loss(self, frame_by_scale, pathway_name):
        loss_by_scale = self.get_encoder_loss_by_scale(frame_by_scale, pathway_name)
        return sum(loss_by_scale.values())

    @tf.function
    def train_encoders(self, frame_by_scale, pathway_name):
        with tf.GradientTape() as tape:
            total_loss = tf.reduce_sum(self.get_encoder_loss(frame_by_scale, pathway_name))
            variables = sum([model.variables for model in self.models[pathway_name]["encoder_models"].values()] + \
                   [model.variables for model in self.models[pathway_name]["decoder_models"].values()], [])
            grads = tape.gradient(total_loss, variables)
            self.encoder_optimizer[pathway_name].apply_gradients(zip(grads, variables))
        return total_loss

    @tf.function
    def train_critic(self, frame_by_scale, actions, targets, pathway_name):
        with tf.GradientTape() as tape:
            return_estimates = self.get_return_estimates(frame_by_scale, actions, pathway_name)
            loss_critic = keras.losses.Huber()(return_estimates, tf.stop_gradient(targets))
            variables = self.models[pathway_name]["critic_model"].variables
            grads = tape.gradient(loss_critic, variables)
            self.critic_optimizer[pathway_name].apply_gradients(zip(grads, variables))
        return loss_critic

    @tf.function
    def train_policy(self, frame_by_scale, pathway_name):
        with tf.GradientTape() as tape:
            actions = self.get_actions(frame_by_scale, pathway_name, exploration=False)
            return_estimates = self.get_return_estimates(frame_by_scale, actions, pathway_name)
            loss_policy = - tf.reduce_sum(return_estimates)
            variables = self.models[pathway_name]["policy_model"].variables
            grads = tape.gradient(loss_policy, variables)
            self.policy_optimizer[pathway_name].apply_gradients(zip(grads, variables))
        return loss_policy

    @tf.function
    def train(self, frame_by_scale, actions, targets, pathway_name,
            encoders=True, critic=True, policy=True):
        losses = {}
        if encoders:
            encoders_loss = self.train_encoders(frame_by_scale, pathway_name)
            losses["encoders"] = encoders_loss
        if critic:
            critic_loss = self.train_critic(frame_by_scale, actions, targets, pathway_name)
            losses["critic"] = critic_loss
        if policy:
            policy_loss = self.train_policy(frame_by_scale, pathway_name)
            losses["policy"] = policy_loss
        return losses

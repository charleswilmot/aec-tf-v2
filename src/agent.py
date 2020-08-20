import tensorflow as tf
from tensorflow import keras
import numpy as np
from custom_layers import custom_objects


def divide_no_nan(a, b, default=0.0):
    return np.divide(a, b, out=np.full_like(a, fill_value=default), where=b!=0)


class Agent(object):
    def __init__(self,
            policy_learning_rate, policy_model_arch,
            critic_learning_rate, critic_model_arch,
            encoder_learning_rate, exploration, n_simulations, action_scaling):
        self.policy_learning_rate = policy_learning_rate
        self.policy_optimizer = keras.optimizers.Adam(self.policy_learning_rate)
        self.critic_learning_rate = critic_learning_rate
        self.critic_optimizer = keras.optimizers.Adam(self.critic_learning_rate)
        self.encoder_learning_rate = encoder_learning_rate
        self.encoder_optimizer = keras.optimizers.Adam(self.encoder_learning_rate)
        self.models = {}
        for pathway in pathways:
            #   POLICY
            self.models[pathway.name] = {}
            self.models[pathway.name]["policy_model"] = \
                keras.models.model_from_yaml(
                    policy_model_arch.pretty(resolve=True),
                    custom_objects=custom_objects
                )
            #   CRITIC
            self.models[pathway.name]["critic_model"] = \
                keras.models.model_from_yaml(
                    critic_model_arch.pretty(resolve=True),
                    custom_objects=custom_objects
                )
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
        #   EXPLORATION NOISE
        self.exploration_params = exploration
        self.exploration_stddev = tf.Variable(exploration.stddev, dtype=tf.float32)
        self.exploration_n = exploration.n
        self.success_rate = None
        self.autotune_scale = exploration.autotune_scale
        self.success_rate_estimator_speed = exploration.success_rate_estimator_speed
        self.n_simulations = n_simulations
        if self.n_simulations != 1:
            self.stddev_coefs_step = self.autotune_scale ** -(2 / (self.n_simulations - 1))
            self.histogram_step = self.stddev_coefs_step ** 2
            self.stddev_coefs = self.stddev_coefs_step ** np.arange(
                -(self.n_simulations - 1) / 2,
                1 + (self.n_simulations - 1) / 2,
                1
            )
            self.bins = self.histogram_step ** np.arange(
                np.floor(np.log(0.0001) / np.log(self.histogram_step)),
                np.ceil(np.log(2) / np.log(self.histogram_step))
            )
            self.mean_reward_sum = np.zeros(len(self.bins) + 1)
            self.mean_reward_count = np.zeros(len(self.bins) + 1)
        ###

    def save_weights(self, path):
        self.policy_model.save_weights(path + "/policy_model")
        self.critic_model.save_weights(path + "/critic_model")
        self.encoder_model.save_weights(path + "/encoder_model")

    def load_weights(self, path):
        self.policy_model.load_weights(path + "/policy_model")
        self.critic_model.load_weights(path + "/critic_model")
        self.encoder_model.load_weights(path + "/encoder_model")

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
        return self.models[pathway_name]["critic_model"](encodings_by_scale, actions)

    @tf.function
    def get_actions(self, frame_by_scale, pathway_name, exploration=False):
        encodings_by_scale = self.get_encodings(frame_by_scale, pathway_name)
        pure_actions = self.models[pathway_name]["policy_model"](encodings_by_scale)
        if exploration:
            noises = tf.random.truncated_normal(
                shape=tf.shape(pure_actions),
                stddev=self.exploration_stddev,
            )
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
        reconstructions_by_scale = self.get_reconstructions(frame_by_scale, pathway_name)
        patches_by_scale = {
            scale_name: tf.extract_image_patches(
                frame,
                sizes=[1, 8, 8, 1],
                strides=[1, 4, 4, 1],
                rates=[1, 1, 1, 1],
                padding='VALID',
            ) for scale_name, frame in frame_by_scale.items()
        }
        raise NotImplementedError("Warning: error here")
        return {
            scale_name: tf.reduce_mean(
                (patches_by_scale[scale_name] - reconstructions_by_scale[scale_name]) ** 2,
                axis=[1, 2, 3, 4],
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
            vars = [model.variables for model in self.models[pathway_name]["encoder_models"].values()] + \
                   [model.variables for model in self.models[pathway_name]["decoder_models"].values()]
            grads = tape.gradient(total_loss, vars)
            self.encoder_optimizer.apply_gradients(zip(grads, vars))
        return total_loss

    @tf.function
    def train_critics(self, frame_by_scale, actions, targets, pathway_name):
        with tf.GradientTape() as tape:
            return_estimates = self.get_return_estimates(frame_by_scale, actions, pathway_name)
            loss_critic = keras.losses.Huber()(return_estimates, tf.stop_gradient(targets))
            vars = self.models[pathway_name]["critic_model"].variables
            grads = tape.gradient(total_loss, vars)
            self.critic_optimizer.apply_gradients(zip(grads, vars))
        return loss_critic

    @tf.function
    def train_policies(self, frame_by_scale, pathway_name):
        with tf.GradientTape() as tape:
            actions = self.get_actions(frame_by_scale, pathway_name, exploration=False)
            return_estimates = self.get_return_estimates(frame_by_scale, actions, pathway_name)
            loss_policy = - tf.reduce_sum(return_estimates)
            vars = self.models[pathway_name]["policy_model"].variables
            grads = tape.gradient(loss_policy, vars)
            self.policy_optimizer.apply_gradients(zip(grads, vars))
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

    def register_total_reward(self, rewards):
        stddevs = self.stddev_coefs * self.exploration_stddev
        current_bins = np.digitize(stddevs, self.bins)
        c = self.success_rate_estimator_speed
        print('current_bins', current_bins)
        print('rewards', rewards)
        for bin, reward in zip(current_bins, rewards):
            self.mean_reward_sum[bin] = reward + (1 - c) * self.mean_reward_sum[bin]
            self.mean_reward_count[bin] = 1 + (1 - c) * self.mean_reward_count[bin]
        mean_reward = divide_no_nan(self.mean_reward_sum, self.mean_reward_count, default=-2.0)
        filtered_mean_reward = np.convolve(
            mean_reward,
            self.n_simulations // 4,
            mode='same'
        )
        index = np.argmax(filtered_mean_reward)
        print('filtered_mean_reward', filtered_mean_reward)
        print('index', index)
        if index == 0:
            best_std = self.bins[0]
        elif index == len(self.bins):
            best_std = self.bins[-1]
        else:
            best_std = 0.5 * (self.bins[index - 1] + self.bins[index])
        best_std = c * min(best_std, 1.0) + (1 - c) * self.exploration_stddev.numpy()
        self.exploration_stddev.assign(best_std)

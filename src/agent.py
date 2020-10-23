import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
import numpy as np
from custom_layers import custom_objects
from omegaconf import OmegaConf


def divide_no_nan(a, b, default=0.0):
    return np.divide(a, b, out=np.full_like(a, fill_value=default), where=b!=0)


def to_model(conf):
    return keras.models.model_from_yaml(
        OmegaConf.to_yaml(conf, resolve=True),
        custom_objects=custom_objects)


class Agent(object):
    def __init__(self,
            critic_learning_rate, encoder_learning_rate, hubber_delta,
            actions_neighbourhood_size, exploration, n_simulations, scales, pathways):
        self.critic_learning_rate = critic_learning_rate
        self.encoder_learning_rate = encoder_learning_rate
        self.critic_optimizer = keras.optimizers.Adam(self.critic_learning_rate)
        self.encoder_optimizer = keras.optimizers.Adam(self.encoder_learning_rate)
        self.hubber_delta = float(hubber_delta)
        self.pathways = pathways
        self.scales = scales
        self.exploration_prob = exploration.prob
        self.exploration_temperature = exploration.temperature
        self.n_simulations = n_simulations
        self.actions_neighbourhood_size = actions_neighbourhood_size
        self.models = {}
        self.tilt_action_set = np.array(list(pathways.magno.actions.tilt))
        self.pan_action_set = np.array(list(pathways.magno.actions.pan))
        self.vergence_action_set = np.array(list(pathways.pavro.actions.vergence))
        self.cyclo_action_set = np.array(list(pathways.pavro.actions.cyclo))
        self.n_actions = {
            "tilt": len(self.tilt_action_set),
            "pan": len(self.pan_action_set),
            "vergence": len(self.vergence_action_set),
            "cyclo": len(self.cyclo_action_set),
        }
        for pathway, pathway_conf in pathways.items():
            self.models[pathway] = {}
            self.models[pathway]["critic_models"] = {}
            self.models[pathway]["encoder_models"] = {}
            self.models[pathway]["decoder_models"] = {}
            #   CRITIC
            for joint_name in pathway_conf.actions:
                self.models[pathway]["critic_models"][joint_name] = \
                    to_model(pathway_conf["{}_critic_model_arch".format(joint_name)])
            #   ENCODERS / DECODERS
            for scale, scale_conf in scales.description.items():
                self.models[pathway]["encoder_models"][scale] = \
                    to_model(pathway_conf.encoder_model_arch)
                self.models[pathway]["decoder_models"][scale] = \
                    to_model(pathway_conf.decoder_model_arch)

    def save_weights(self, path):
        for pathway_name, pathway_models in self.models.items():
            for models_name, models in pathway_models.items():
                for model_name, model in models.items():
                    model.save_weights(path + "/{}_{}_{}".format(pathway_name, models_name, model_name))

    def load_weights(self, path):
        for pathway_name, pathway_models in self.models.items():
            for models_name, models in pathway_models.items():
                for model_name, model in models.items():
                    model.load_weights(path + "/{}_{}_{}".format(pathway_name, models_name, model_name))

    @tf.function
    def get_encodings(self, frame_by_scale, pathway_name):
        return {
            scale_name: self.models[pathway_name]["encoder_models"][scale_name](frame)
            for scale_name, frame in frame_by_scale.items()
        }

    @tf.function
    def get_reconstructions(self, encodings_by_scale, pathway_name):
        return {
            scale_name: self.models[pathway_name]["decoder_models"][scale_name](encoding)
            for scale_name, encoding in encodings_by_scale.items()
        }

    @tf.function
    def get_return_estimates(self, encodings_by_scale, pathway_name, joint_name):
        return self.models[pathway_name]["critic_models"][joint_name](encodings_by_scale)

    @tf.function
    def get_actions_indices(self, return_estimates, exploration=False):
        max_indices = tf.argmax(return_estimates, axis=-1, output_type=tf.int32)
        if exploration:
            softmax = tf.nn.softmax(return_estimates / self.exploration_temperature, axis=-1)
            softmax_indices = tfp.distributions.Categorical(probs=softmax).sample()
            condition = tf.random.uniform(shape=tf.shape(max_indices)) > self.exploration_prob
            return tf.where(condition, max_indices, softmax_indices)
        else:
            return max_indices

    @tf.function
    def get_encoder_loss_by_scale(self, frame_by_scale, pathway_name):
        encoder_sizes = {
            pathway: list(pathway_conf.encoder_model_arch.config.layers[0].config.kernel_size)
            for pathway, pathway_conf in self.pathways.items()
        }
        encoder_strides = {
            pathway: list(pathway_conf.encoder_model_arch.config.layers[0].config.strides)
            for pathway, pathway_conf in self.pathways.items()
        }
        encodings_by_scale = self.get_encodings(frame_by_scale, pathway_name)
        reconstructions_by_scale = self.get_reconstructions(encodings_by_scale, pathway_name)
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
            self.encoder_optimizer.apply_gradients(zip(grads, variables))
        return total_loss

    @tf.function
    def train_critic(self, frame_by_scale, actions_indices, targets, pathway_name, joint_name):
        with tf.GradientTape() as tape:
            encodings_by_scale = self.get_encodings(frame_by_scale, pathway_name)
            return_estimates = self.get_return_estimates(encodings_by_scale, pathway_name, joint_name)
            # return_estimates # [BS, n_actions]
            # targets          # [BS, ]
            # actions_indices  # [BS, ]
            # loss_filter = tf.math.exp(-(tf.cast(
            #     tf.reshape(actions_indices, (-1, 1)) -            # [BS, 1]
            #     tf.range(self.n_actions[joint_name])[tf.newaxis], # [1, NA]
            #     tf.float32)) ** 2 / self.actions_neighbourhood_size    # [BS, NA]
            # )
            # loss_critic = tf.reduce_sum((return_estimates - targets[:, tf.newaxis]) ** 2 * loss_filter)
            batch_size = tf.shape(actions_indices)[0]
            indices = tf.stack([tf.range(batch_size), actions_indices], axis=-1)
            return_estimates = tf.gather_nd(return_estimates, indices)
            loss_critic = keras.losses.Huber(delta=self.hubber_delta)(return_estimates, targets)
            variables = self.models[pathway_name]["critic_models"][joint_name].variables
            grads = tape.gradient(loss_critic, variables)
            self.critic_optimizer.apply_gradients(zip(grads, variables))
        return loss_critic

    @tf.function
    def train(self,
            pavro_frame_by_scale,
            magno_frame_by_scale,
            tilt_actions_indices,
            pan_actions_indices,
            vergence_actions_indices,
            cyclo_actions_indices,
            magno_targets,
            pavro_targets,
            encoders=True, critic=True):
        losses = {}
        if encoders:
            losses["pavro_encoders"] = self.train_encoders(pavro_frame_by_scale, "pavro")
            losses["magno_encoders"] = self.train_encoders(magno_frame_by_scale, "magno")
        if critic:
            losses["critic_tilt"] = self.train_critic(magno_frame_by_scale, tilt_actions_indices, magno_targets, "magno", "tilt")
            losses["critic_pan"] = self.train_critic(magno_frame_by_scale, pan_actions_indices, magno_targets, "magno", "pan")
            losses["critic_vergence"] = self.train_critic(pavro_frame_by_scale, vergence_actions_indices, pavro_targets, "pavro", "vergence")
            losses["critic_cyclo"] = self.train_critic(pavro_frame_by_scale, cyclo_actions_indices, pavro_targets, "pavro", "cyclo")
        return losses

    @tf.function
    def __call__(self, pavro_vision, magno_vision):
        magno_encodings_by_scale = self.get_encodings(magno_vision, "magno")
        pavro_encodings_by_scale = self.get_encodings(pavro_vision, "pavro")
        tilt_return_estimates = self.get_return_estimates(magno_encodings_by_scale, "magno", "tilt")
        pan_return_estimates = self.get_return_estimates(magno_encodings_by_scale, "magno", "pan")
        vergence_return_estimates = self.get_return_estimates(pavro_encodings_by_scale, "pavro", "vergence")
        cyclo_return_estimates = self.get_return_estimates(pavro_encodings_by_scale, "pavro", "cyclo")
        return {
            "pavro_recerr": self.get_encoder_loss(pavro_vision, "pavro"),
            "magno_recerr": self.get_encoder_loss(magno_vision, "magno"),
            "pavro_encodings_by_scale": pavro_encodings_by_scale,
            "magno_encodings_by_scale": magno_encodings_by_scale,
            "tilt_return_estimates": tilt_return_estimates,
            "pan_return_estimates": pan_return_estimates,
            "vergence_return_estimates": vergence_return_estimates,
            "cyclo_return_estimates": cyclo_return_estimates,
            "tilt_actions_indices": self.get_actions_indices(tilt_return_estimates, exploration=False),
            "pan_actions_indices": self.get_actions_indices(pan_return_estimates, exploration=False),
            "vergence_actions_indices": self.get_actions_indices(vergence_return_estimates, exploration=False),
            "cyclo_actions_indices": self.get_actions_indices(cyclo_return_estimates, exploration=False),
            "tilt_noisy_actions_indices": self.get_actions_indices(tilt_return_estimates, exploration=True),
            "pan_noisy_actions_indices": self.get_actions_indices(pan_return_estimates, exploration=True),
            "vergence_noisy_actions_indices": self.get_actions_indices(vergence_return_estimates, exploration=True),
            "cyclo_noisy_actions_indices": self.get_actions_indices(cyclo_return_estimates, exploration=True),
        }

    def create_all_variables(self, fake_frame_by_scale_pavro, fake_frame_by_scale_magno):
        pavro_encodings = {
            scale_name: self.models["pavro"]["encoder_models"][scale_name](frame)
            for scale_name, frame in fake_frame_by_scale_pavro.items()
        }
        pavro_decodings = {
            scale_name: self.models["pavro"]["decoder_models"][scale_name](encoding)
            for scale_name, encoding in pavro_encodings.items()
        }
        vergence_values = self.models["pavro"]["critic_models"]["vergence"](pavro_encodings)
        cyclo_values = self.models["pavro"]["critic_models"]["cyclo"](pavro_encodings)

        magno_encodings = {
            scale_name: self.models["magno"]["encoder_models"][scale_name](frame)
            for scale_name, frame in fake_frame_by_scale_magno.items()
        }
        magno_decodings = {
            scale_name: self.models["magno"]["decoder_models"][scale_name](encoding)
            for scale_name, encoding in magno_encodings.items()
        }
        tilt_values = self.models["magno"]["critic_models"]["tilt"](magno_encodings)
        pan_values = self.models["magno"]["critic_models"]["pan"](magno_encodings)

        batch_size = list(fake_frame_by_scale_pavro.values())[0].shape[0]
        # PAVRO
        # encoder
        with tf.GradientTape() as tape:
            total_loss = tf.reduce_sum(self.get_encoder_loss(fake_frame_by_scale_pavro, "pavro"))
            variables = sum([model.variables for model in self.models["pavro"]["encoder_models"].values()] + \
                   [model.variables for model in self.models["pavro"]["decoder_models"].values()], [])
            grads = tape.gradient(total_loss, variables)
            self.encoder_optimizer.apply_gradients(zip(grads, variables))

        # critic
        # vergence
        actions_indices = tf.zeros(shape=(batch_size,), dtype=tf.int32)
        targets = np.zeros(shape=(batch_size,), dtype=np.float32)
        with tf.GradientTape() as tape:
            encodings_by_scale = self.get_encodings(fake_frame_by_scale_pavro, "pavro")
            return_estimates = self.get_return_estimates(encodings_by_scale, "pavro", "vergence")
            # return_estimates # [BS, n_actions]
            # targets          # [BS, ]
            # actions_indices  # [BS, ]
            loss_filter = tf.math.exp(-(tf.cast(
                tf.reshape(actions_indices, (-1, 1)) -
                tf.range(self.n_actions["vergence"])[tf.newaxis],
                tf.float32)) ** 2 / self.actions_neighbourhood_size
            )
            loss_critic = tf.reduce_sum((return_estimates - targets[:, tf.newaxis]) ** 2 * loss_filter)
            variables = self.models["pavro"]["critic_models"]["vergence"].variables
            grads = tape.gradient(loss_critic, variables)
            self.critic_optimizer.apply_gradients(zip(grads, variables))

        # cyclo
        actions_indices = tf.zeros(shape=(batch_size,), dtype=tf.int32)
        targets = np.zeros(shape=(batch_size,), dtype=np.float32)
        with tf.GradientTape() as tape:
            encodings_by_scale = self.get_encodings(fake_frame_by_scale_pavro, "pavro")
            return_estimates = self.get_return_estimates(encodings_by_scale, "pavro", "cyclo")
            # return_estimates # [BS, n_actions]
            # targets          # [BS, ]
            # actions_indices  # [BS, ]
            loss_filter = tf.math.exp(-(tf.cast(
                tf.reshape(actions_indices, (-1, 1)) -
                tf.range(self.n_actions["cyclo"])[tf.newaxis],
                tf.float32)) ** 2 / self.actions_neighbourhood_size
            )
            loss_critic = tf.reduce_sum((return_estimates - targets[:, tf.newaxis]) ** 2 * loss_filter)
            variables = self.models["pavro"]["critic_models"]["cyclo"].variables
            grads = tape.gradient(loss_critic, variables)
            self.critic_optimizer.apply_gradients(zip(grads, variables))

        # MAGNO
        # encoder
        with tf.GradientTape() as tape:
            total_loss = tf.reduce_sum(self.get_encoder_loss(fake_frame_by_scale_magno, "magno"))
            variables = sum([model.variables for model in self.models["magno"]["encoder_models"].values()] + \
                   [model.variables for model in self.models["magno"]["decoder_models"].values()], [])
            grads = tape.gradient(total_loss, variables)
            self.encoder_optimizer.apply_gradients(zip(grads, variables))

        # critic
        # tilt
        actions_indices = tf.zeros(shape=(batch_size,), dtype=tf.int32)
        targets = np.zeros(shape=(batch_size,), dtype=np.float32)
        with tf.GradientTape() as tape:
            encodings_by_scale = self.get_encodings(fake_frame_by_scale_magno, "magno")
            return_estimates = self.get_return_estimates(encodings_by_scale, "magno", "tilt")
            # return_estimates # [BS, n_actions]
            # targets          # [BS, ]
            # actions_indices  # [BS, ]
            loss_filter = tf.math.exp(-(tf.cast(
                tf.reshape(actions_indices, (-1, 1)) -
                tf.range(self.n_actions["tilt"])[tf.newaxis],
                tf.float32)) ** 2 / self.actions_neighbourhood_size
            )
            loss_critic = tf.reduce_sum((return_estimates - targets[:, tf.newaxis]) ** 2 * loss_filter)
            variables = self.models["magno"]["critic_models"]["tilt"].variables
            grads = tape.gradient(loss_critic, variables)
            self.critic_optimizer.apply_gradients(zip(grads, variables))

        # pan
        actions_indices = tf.zeros(shape=(batch_size,), dtype=tf.int32)
        targets = np.zeros(shape=(batch_size,), dtype=np.float32)
        with tf.GradientTape() as tape:
            encodings_by_scale = self.get_encodings(fake_frame_by_scale_magno, "magno")
            return_estimates = self.get_return_estimates(encodings_by_scale, "magno", "pan")
            # return_estimates # [BS, n_actions]
            # targets          # [BS, ]
            # actions_indices  # [BS, ]
            loss_filter = tf.math.exp(-(tf.cast(
                tf.reshape(actions_indices, (-1, 1)) -
                tf.range(self.n_actions["pan"])[tf.newaxis],
                tf.float32)) ** 2 / self.actions_neighbourhood_size
            )
            loss_critic = tf.reduce_sum((return_estimates - targets[:, tf.newaxis]) ** 2 * loss_filter)
            variables = self.models["magno"]["critic_models"]["pan"].variables
            grads = tape.gradient(loss_critic, variables)
            self.critic_optimizer.apply_gradients(zip(grads, variables))

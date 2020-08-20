import numpy as np
from buffer import Buffer
from agent import Agent
from simulation import SimulationPool
from tensorflow.keras.metrics import Mean
import tensorflow as tf
import time
import os
from collections import OrderedDict
from tensorboard.plugins.hparams import api as hp
from imageio import get_writer


def get_snr_db(signal, noise, axis=1):
    mean_signal = np.mean(signal, axis=axis, keepdims=True)
    mean_noise = np.mean(noise, axis=axis, keepdims=True)
    std_signal = np.std(signal - mean_signal, axis=axis)
    std_noise = np.std(noise - mean_noise, axis=axis)
    where = std_signal != 0
    if not where.any():
        print("WARNING: signal to noise can't be computed (constant signal), returning NaN")
        return np.nan
    std_signal = std_signal[where]
    std_noise = std_noise[where]
    rms_signal_db = np.log10(std_signal)
    rms_noise_db = np.log10(std_noise)
    return 20 * (rms_signal_db - rms_noise_db)


class Procedure(object):
    def __init__(self, agent_conf, buffer_conf, simulation_conf, procedure_conf):
        #   PROCEDURE CONF
        self.episode_length = procedure_conf.episode_length
        self.updates_per_sample = procedure_conf.updates_per_sample
        self.batch_size = procedure_conf.batch_size
        self.n_simulations = simulation_conf.n
        self.log_freq = procedure_conf.log_freq
        #    HPARAMS
        self._hparams = OrderedDict([
            ("policy_LR", agent_conf.policy_learning_rate),
            ("critic_LR", agent_conf.critic_learning_rate),
            ("encoder_LR", agent_conf.encoder_learning_rate),
            ("buffer", buffer_conf.size),
            ("update_rate", procedure_conf.updates_per_sample),
            ("ep_length", procedure_conf.episode_length),
            ("batch_size", procedure_conf.batch_size),
            ("noise_std", agent_conf.exploration.stddev),
        ])
        #   OBJECTS
        self.agent = Agent(**agent_conf)
        self.buffer = Buffer(**buffer_conf)
        #   SIMULATION POOL
        guis = list(simulation_conf.guis)
        self.simulation_pool = SimulationPool(
            simulation_conf.n,
            scene="",
            guis=guis
        )
        self.simulation_pool.add_background("ny_times_square")
        self.simulation_pool.add_head()
        for scale in procedure_conf.scales:
            self.simulation_pool.add_scale(scale.name, (scale.resolution, scale.resolution), scale.view_angle)
        self.simulation_pool.add_uniform_motion_screen("/home/aecgroup/aecdata/Textures/mcgillManMade_600x600_png_selection/", size=1.5)
        self.simulation_pool.start_sim()
        self.simulation_pool.step_sim()
        print("[procedure] all simulation started")

        #   DEFINING DATA BUFFERS
        # training
        pavro_dtype = np.dtype([
            (scale.name, (scale.resolution, scale.resolution, 6), np.float32)
            for scale in procedure_conf.scales
        ])
        pavro_dtype = np.dtype([
            (scale.name, (scale.resolution, scale.resolution, 12), np.float32)
            for scale in procedure_conf.scales
        ])
        n_pavro_joints = len(agent_conf.pathways[0].joints)
        n_magno_joints = len(agent_conf.pathways[1].joints)
        self._train_data_type = np.dtype([
            ("pavro_vision", pavro_dtype),
            ("magno_vision", magno_dtype),
            ("pavro_noisy_actions", (n_pavro_joints,), np.float32),
            ("magno_noisy_actions", (n_magno_joints,), np.float32),
            ("pavro_critic_targets", np.float32),
            ("magno_critic_targets", np.float32),
            ("pavro_recerr", np.float32),
            ("magno_recerr", np.float32),
            ("pavro_return_estimates", np.float32)
            ("magno_return_estimates", np.float32)
        ])
        self._train_data_buffer = np.zeros(
            shape=(self.n_simulations, self.episode_length),
            dtype=self._train_data_type
        )
        # evaluation
        self._evaluation_data_type = np.dtype([
            ("pavro_vision", pavro_dtype),
            ("magno_vision", magno_dtype),
            ("pavro_pure_actions", (n_pavro_joints,), np.float32),
            ("magno_pure_actions", (n_magno_joints,), np.float32),
            ("pavro_recerr", np.float32),
            ("magno_recerr", np.float32),
            ("pavro_critic_targets", np.float32),
            ("pavro_return_estimates", np.float32),
            ("magno_critic_targets", np.float32),
            ("magno_return_estimates", np.float32),
        ])
        self._evaluation_data_buffer = np.zeros(
            shape=(self.n_simulations, self.episode_length),
            dtype=self._evaluation_data_type
        )

        # COUNTERS
        self.n_exploration_episodes = 0
        self.n_evaluation_episodes = 0
        self.n_transition_gathered = 0
        self.n_policy_training = 0
        self.n_critic_training = 0
        self.n_encoder_training = 0
        self.n_global_training = 0

        # TENSORBOARD LOGGING
        self.tb = {}
        self.tb["training"] = {}
        self.tb["training"]["policy"] = {}
        self.tb["training"]["policy"]["pavro_loss"] = Mean(
            "training/policy_pavro_loss", dtype=tf.float32)
        self.tb["training"]["policy"]["magno_loss"] = Mean(
            "training/policy_magno_loss", dtype=tf.float32)
        self.tb["training"]["critic"] = {}
        self.tb["training"]["critic"]["pavro_loss"] = Mean(
            "training/critic_pavro_loss", dtype=tf.float32)
        self.tb["training"]["critic"]["magno_loss"] = Mean(
            "training/critic_magno_loss", dtype=tf.float32)
        self.tb["training"]["encoder"] = {}
        self.tb["training"]["encoder"]["pavro_loss"] = Mean(
            "training/encoder_pavro_loss", dtype=tf.float32)
        self.tb["training"]["encoder"]["magno_loss"] = Mean(
            "training/encoder_magno_loss", dtype=tf.float32)
        self.tb["collection"] = {}
        self.tb["collection"]["exploration"] = {}
        self.tb["collection"]["evaluation"] = {}
        self.tb["collection"]["exploration"]["it_per_sec"] = Mean(
            "collection/exploration_it_per_sec", dtype=tf.float32)
        self.tb["collection"]["evaluation"]["it_per_sec"] = Mean(
            "collection/evaluation_it_per_sec", dtype=tf.float32)
        self.tb["collection"]["exploration"]["total_episode_reward_pavro"] = Mean(
            "collection/exploration_total_episode_reward_pavro", dtype=tf.float32)
        self.tb["collection"]["evaluation"]["total_episode_reward_pavro"] = Mean(
            "collection/exploration_total_episode_reward_pavro", dtype=tf.float32)
        self.tb["collection"]["exploration"]["total_episode_reward_magno"] = Mean(
            "collection/exploration_total_episode_reward_magno", dtype=tf.float32)
        self.tb["collection"]["evaluation"]["total_episode_reward_magno"] = Mean(
            "collection/exploration_total_episode_reward_magno", dtype=tf.float32)
        self.tb["collection"]["exploration"]["recerr_pavro"] = Mean(
            "collection/exploration_recerr_pavro", dtype=tf.float32)
        self.tb["collection"]["evaluation"]["recerr_pavro"] = Mean(
            "collection/exploration_recerr_pavro", dtype=tf.float32)
        self.tb["collection"]["exploration"]["recerr_magno"] = Mean(
            "collection/exploration_recerr_magno", dtype=tf.float32)
        self.tb["collection"]["evaluation"]["recerr_magno"] = Mean(
            "collection/exploration_recerr_magno", dtype=tf.float32)
        self.tb["collection"]["exploration"]["final_vergence_error"] = Mean(
            "collection/exploration_final_vergence_error", dtype=tf.float32)
        self.tb["collection"]["evaluation"]["final_vergence_error"] = Mean(
            "collection/exploration_final_vergence_error", dtype=tf.float32)
        self.tb["collection"]["exploration"]["final_tilt_error"] = Mean(
            "collection/exploration_final_tilt_error", dtype=tf.float32)
        self.tb["collection"]["evaluation"]["final_tilt_error"] = Mean(
            "collection/exploration_final_tilt_error", dtype=tf.float32)
        self.tb["collection"]["exploration"]["final_pan_error"] = Mean(
            "collection/exploration_final_pan_error", dtype=tf.float32)
        self.tb["collection"]["evaluation"]["final_pan_error"] = Mean(
            "collection/exploration_final_pan_error", dtype=tf.float32)
        self.tb["collection"]["exploration"]["critic_snr_pavro"] = Mean(
            "collection/exploration_critic_snr_pavro_db", dtype=tf.float32)
        self.tb["collection"]["evaluation"]["critic_snr_pavro"] = Mean(
            "collection/evaluation_critic_snr_pavro_db", dtype=tf.float32)
        self.tb["collection"]["exploration"]["critic_snr_magno"] = Mean(
            "collection/exploration_critic_snr_magno_db", dtype=tf.float32)
        self.tb["collection"]["evaluation"]["critic_snr_magno"] = Mean(
            "collection/evaluation_critic_snr_magno_db", dtype=tf.float32)
        #
        self.summary_writer = tf.summary.create_file_writer("logs")
        with self.summary_writer.as_default():
            hp.hparams(self._hparams)
        # TREE STRUCTURE
        os.makedirs('./replays', exist_ok=True)
        os.makedirs('./visualization_data', exist_ok=True)

    def log_metrics(self, key1, key2, step):
        with self.summary_writer.as_default():
            for name, metric in self.tb[key1][key2].items():
                tf.summary.scalar(metric.name, metric.result(), step=step)
                metric.reset_states()

    def log_summaries(self, exploration=True, evaluation=True, critic=True,
            policy=True, encoders=True):
        if exploration:
            self.log_metrics(
                "collection",
                "exploration",
                self.n_exploration_episodes
            )
        if evaluation:
            self.log_metrics(
                "collection",
                "evaluation",
                self.n_evaluation_episodes
            )
        if critic:
            self.log_metrics(
                "training",
                "critic",
                self.n_critic_training
            )
            self.log_metrics(
                "training",
                "next_critic",
                self.n_critic_training
            )
        if policy:
            self.log_metrics(
                "training",
                "policy",
                self.n_policy_training
            )
        if encoders:
            self.log_metrics(
                "training",
                "encoder",
                self.n_encoder_training
            )

    def _get_current_training_ratio(self):
        if self.n_transition_gathered != 0:
            return self.n_global_training * \
                    self.batch_size / \
                    self.n_transition_gathered
        else:
            return np.inf
    current_training_ratio = property(_get_current_training_ratio)

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()

    def close(self):
        self.simulation_pool.close()

    def save(self):
        """Saves the model in the appropriate directory"""
        path = "./checkpoints/{:08d}".format(self.n_global_training)
        self.agent.save_weights(path)

    def restore(self, path):
        """Restores the weights from a checkpoint"""
        self.agent.load_weights(path)

    def episode_reset_uniform_motion_screen(self, start_distances=None,
            depth_speeds=None, angular_speeds=None, directions=None,
            texture_ids=None, preinit=False):
        with self.simulation_pool.distribute_args():
            self.simulation_pool.episode_reset_uniform_motion_screen(
                start_distances,
                depth_speeds,
                angular_speeds,
                directions,
                texture_ids,
                preinit=[preinit] * self.simulation_pool.n
            )

    def get_vision(self):
        vision_list = self.simulation_pool.get_vision()
        return {
            scale_name: np.vstack([v[scale_name] for v in vision_list], axis=0)
            for scale_name in vision_list[0]
        }

    def apply_action(self, actions):
        with self.simulation_pool.distribute_args():
            self.simulation_pool.apply_action(actions)

    def get_joint_errors(self):
        return tuple(zip(self.simulation_pool.get_joint_errors()))

    def merge_before_after(self, before, after):
        return {
            scale_name: np.concatenate(
                [before[scale_name], after[scale_name]],
                axis=-1
            ) for scale_name in before
        }

    def collect_data(self):
        """Performs one episode of exploration, places data in the buffer"""
        time_start = time.time()
        self.episode_reset_uniform_motion_screen(preinit=True)
        vision_after = self.get_vision()
        self.simulation_pool.step_sim()
        for iteration in range(self.episode_length):
            vision_before = vision_after
            vision_after = self.get_vision()
            pavro_vision = vision_after
            magno_vision = self.merge_before_after(vision_before, vision_after)
            pavro_pure_actions, pavro_noisy_actions = self.agent.get_actions(
                pavro_vision, "pavro", exploration=True)
            magno_pure_actions, magno_noisy_actions = self.agent.get_actions(
                magno_vision, "magno", exploration=True)
            pure_actions = np.concatenate([magno_pure_actions, pavro_pure_actions], axis=-1)
            noisy_actions = np.concatenate([magno_noisy_actions, pavro_noisy_actions], axis=-1)
            pavro_recerr = self.agent.get_encoder_loss(pavro_vision, "pavro") # can be done in batch processing mode after the loop
            magno_recerr = self.agent.get_encoder_loss(magno_vision, "magno") # can be done in batch processing mode after the loop
            for scale_name in pavro_vision:
                self._train_data_buffer[:, iteration]["pavro_vision"][scale_name] = pavro_vision[scale_name]
                self._train_data_buffer[:, iteration]["magno_vision"][scale_name] = magno_vision[scale_name]
            self._train_data_buffer[:, iteration]["pavro_noisy_actions"] = pavro_noisy_actions
            self._train_data_buffer[:, iteration]["magno_noisy_actions"] = magno_noisy_actions
            # not necessary for training but useful for logging:
            self._train_data_buffer[:, iteration]["pavro_recerr"] = pavro_recerr
            self._train_data_buffer[:, iteration]["magno_recerr"] = magno_recerr
            self.apply_action(noisy_actions)
        # COMPUTE TARGET
        self._train_data_buffer[:, :-1]["pavro_critic_targets"] = \
            self._train_data_buffer[:, :-1]["pavro_recerr"] - \
            self._train_data_buffer[:,  1:]["pavro_recerr"]
        self._train_data_buffer[:, :-1]["magno_critic_targets"] = \
            self._train_data_buffer[:, :-1]["magno_recerr"] - \
            self._train_data_buffer[:,  1:]["magno_recerr"]
        # ADD TO BUFFER
        buffer_data = self._train_data_buffer[:, :-1].flatten()
        self.buffer.integrate(buffer_data)
        self.n_transition_gathered += len(buffer_data)
        self.n_exploration_episodes += self.n_simulations
        time_stop = time.time()
        # LOG METRICS
        final_tilt_error, final_pan_error, final_vergence_error = self.get_joint_errors()
        self.accumulate_log_data(
            pavro_return_estimates=self._train_data_buffer[:, 1:-1]["pavro_return_estimates"],
            pavro_critic_targets=self._train_data_buffer[:, 1:-1]["pavro_critic_targets"],
            pavro_recerr=self._train_data_buffer["pavro_recerr"],
            magno_return_estimates=self._train_data_buffer[:, 1:-1]["magno_return_estimates"],
            magno_critic_targets=self._train_data_buffer[:, 1:-1]["magno_critic_targets"],
            magno_recerr=self._train_data_buffer["magno_recerr"],
            final_vergence_error=final_vergence_error,
            final_tilt_error=final_tilt_error,
            final_pan_error=final_pan_error,
            time=time_stop - time_start,
            exploration=True,
        )

    def evaluate(self):
        """Performs one episode of exploration, places data in the buffer"""
        time_start = time.time()
        self.episode_reset_uniform_motion_screen(preinit=True)
        vision_after = self.get_vision()
        self.simulation_pool.step_sim()
        for iteration in range(self.episode_length):
            vision_before = vision_after
            vision_after = self.get_vision()
            pavro_vision = vision_after
            magno_vision = self.merge_before_after(vision_before, vision_after)
            pavro_pure_actions = self.agent.get_actions(pavro_vision, "pavro")
            magno_pure_actions = self.agent.get_actions(magno_vision, "magno")
            pure_actions = np.concatenate([magno_pure_actions, pavro_pure_actions], axis=-1)
            pavro_recerr = self.agent.get_encoder_loss(pavro_vision, "pavro") # can be done in batch processing mode after the loop
            magno_recerr = self.agent.get_encoder_loss(magno_vision, "magno") # can be done in batch processing mode after the loop
            for scale_name in pavro_vision:
                self._evaluation_data_buffer[:, iteration]["pavro_vision"][scale_name] = pavro_vision[scale_name]
                self._evaluation_data_buffer[:, iteration]["magno_vision"][scale_name] = magno_vision[scale_name]
            self._evaluation_data_buffer[:, iteration]["pure_actions"] = pure_actions
            # not necessary for training but useful for logging:
            self._evaluation_data_buffer[:, iteration]["pavro_recerr"] = pavro_recerr
            self._evaluation_data_buffer[:, iteration]["magno_recerr"] = magno_recerr
            self.apply_action(pure_actions)
        # COMPUTE TARGET
        self._evaluation_data_buffer[:, :-1]["pavro_critic_targets"] = \
            self._evaluation_data_buffer[:, :-1]["pavro_recerr"] - \
            self._evaluation_data_buffer[:,  1:]["pavro_recerr"]
        self._evaluation_data_buffer[:, :-1]["magno_critic_targets"] = \
            self._evaluation_data_buffer[:, :-1]["magno_recerr"] - \
            self._evaluation_data_buffer[:,  1:]["magno_recerr"]
        # COUNTER
        self.n_evaluation_episodes += self.n_simulations
        time_stop = time.time()
        # LOG METRICS
        final_tilt_error, final_pan_error, final_vergence_error = self.get_joint_errors()
        self.accumulate_log_data(
            pavro_return_estimates=self._evaluation_data_buffer[:, 1:-1]["pavro_return_estimates"],
            pavro_critic_targets=self._evaluation_data_buffer[:, 1:-1]["pavro_critic_targets"],
            pavro_recerr=self._evaluation_data_buffer["pavro_recerr"],
            magno_return_estimates=self._evaluation_data_buffer[:, 1:-1]["magno_return_estimates"],
            magno_critic_targets=self._evaluation_data_buffer[:, 1:-1]["magno_critic_targets"],
            magno_recerr=self._evaluation_data_buffer["magno_recerr"],
            final_vergence_error=final_vergence_error,
            final_tilt_error=final_tilt_error,
            final_pan_error=final_pan_error,
            time=time_stop - time_start,
            exploration=True,
        )

    def accumulate_log_data(self,
            pavro_return_estimates, pavro_critic_targets, pavro_recerr,
            magno_return_estimates, magno_critic_targets, magno_recerr,
            final_vergence_error, final_tilt_error, final_pan_error,
            time, exploration):
        if exploration:
            tb = self.tb["collection"]["exploration"]
        else:
            tb = self.tb["collection"]["evaluation"]
        #
        n_iterations = self.episode_length * self.n_simulations
        it_per_sec = n_iterations / time
        tb["it_per_sec"](it_per_sec)
        #
        tb["total_episode_reward_pavro"](np.mean(np.sum(pavro_critic_targets, axis=-1)))
        tb["total_episode_reward_magno"](np.mean(np.sum(magno_critic_targets, axis=-1)))
        tb["recerr_pavro"](np.mean(pavro_recerr))
        tb["recerr_magno"](np.mean(magno_recerr))
        tb["final_vergence_error"](np.mean(final_vergence_error))
        tb["final_tilt_error"](np.mean(final_tilt_error))
        tb["final_pan_error"](np.mean(final_pan_error))
        #
        signal = pavro_critic_targets
        noise = pavro_critic_targets - pavro_return_estimates
        critic_snr_pavro = get_snr_db(signal, noise)
        tb["critic_snr"](np.mean(critic_snr_pavro))
        #
        signal = magno_critic_targets
        noise = magno_critic_targets - magno_return_estimates
        critic_snr_magno = get_snr_db(signal, noise)
        tb["critic_snr"](np.mean(critic_snr_magno))
        #

    def train(self, policy=True, critic=True, encoders=True):
        data = self.buffer.sample(self.batch_size)
        tb = self.tb["training"]
        for pathway_name in ["pavro", "magno"]:
            frame_by_scale = {
                scale_name: data["{}_vision".format(pathway_name)][scale_name]
                for scale_name in self.scale_names # todo
            }
            losses = self.agent.train(
                frame_by_scale,
                data["{}_noisy_actions".format(pathway_name)],
                data["{}_critic_targets".format(pathway_name)],
                pathway_name,
                policy=policy,
                critic=critic,
                encoders=encoders,
            )
            if policy:
                tb["policy"]["{}_loss".format(pathway_name)](losses["policy"])
            if critic:
                tb["critic"]["{}_loss".format(pathway_name)](losses["critic"])
            if encoders:
                tb["encoders"]["{}_loss".format(pathway_name)](losses["encoders"])
        if policy:
            self.n_policy_training += 1
        if critic:
            self.n_critic_training += 1
        if encoders:
            self.n_encoder_training += 1
        self.n_global_training += 1
        return losses

    def collect_and_train(self, policy=True, critic=True, encoders=True):
        self.collect_data()
        while self.current_training_ratio < self.updates_per_sample:
            self.train(policy=policy, critic=critic, encoders=encoders)

    def collect_train_and_log(self, policy=True, critic=True, encoders=True,
            evaluation=False):
        self.collect_and_train(policy=policy, critic=critic, encoders=encoders)
        if evaluation:
            self.evaluate()
        if self.n_global_training % self.log_freq == 0:
            self.log_summaries(exploration=True, evaluation=evaluation,
                policy=policy, critic=critic, encoders=encoders)

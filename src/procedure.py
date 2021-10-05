import numpy as np
from buffer import Buffer
from agent import Agent
from simulation import SimulationPool, distance_to_vergence
from tensorflow.keras.metrics import Mean
import tensorflow as tf
import time
import os
from collections import OrderedDict
from tensorboard.plugins.hparams import api as hp
from imageio import get_writer
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from test_data import TestDataContainer
from evaluation import get_evaluation_conf
from hydra.utils import get_original_cwd, to_absolute_path


critic_test_dtype = np.dtype([
    ("return_estimates", np.float32),
    ("gradient", np.float32, (2,)),
])


def angle(pan_in_deg, tilt_in_deg):
    pan = np.deg2rad(pan_in_deg)
    tilt = np.deg2rad(tilt_in_deg) + 1e-8
    length = np.sqrt(pan ** 2 + tilt ** 2)
    return np.arccos(pan / length) * np.sign(tilt)


def add_text(frame, **lines):
    font_size = 10
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    for line_name, string in lines.items():
        h = int(line_name[5:]) + 1
        draw.text((font_size // 4, h * (font_size + font_size // 2)), string, (255, 255, 255), font=font)
    return np.array(image)


def text_frame(height, width, **lines):
    font_size = 10
    SHIFT = 120
    image = Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    for line_name, (string1, string2) in lines.items():
        h = int(line_name[5:]) + 1
        draw.text((font_size // 4, h * (font_size + font_size // 2)), string1, (255, 255, 255), font=font)
        draw.text((font_size // 4 + SHIFT, h * (font_size + font_size // 2)), string2, (255, 255, 255), font=font)
    return np.array(image)


def get_snr_db(signal, noise, axis=1):
    epsilon = 1e-6
    std_signal = np.std(signal, axis=axis)
    std_noise = np.std(noise, axis=axis)
    rms_signal_db = np.log10(std_signal + epsilon)
    rms_noise_db = np.log10(std_noise + epsilon)
    return 20 * (rms_signal_db - rms_noise_db)


anaglyph_matrix = np.array([
    [0.299, 0    , 0    ],
    [0.587, 0    , 0    ],
    [0.114, 0    , 0    ],
    [0    , 0.299, 0.299],
    [0    , 0.587, 0.587],
    [0    , 0.114, 0.114],
    ])


def anaglyph(left_right):
    return np.clip(np.matmul((left_right + 1) * 127.5, anaglyph_matrix), 0, 255).astype(np.uint8)


class Procedure(object):
    def __init__(self, agent_conf, buffer_conf, simulation_conf, procedure_conf):
        #   PROCEDURE CONF
        self.procedure_conf = procedure_conf
        self.episode_length = procedure_conf.episode_length
        self.n_evaluation_batch = procedure_conf.n_evaluation_batch
        self.updates_per_sample = procedure_conf.updates_per_sample
        self.batch_size = procedure_conf.batch_size
        self.n_simulations = simulation_conf.n
        self.reward_scaling = procedure_conf.reward_scaling
        self.max_abs_cyclo_error = procedure_conf.max_abs_cyclo_error
        self.vergence_min_distance_init = procedure_conf.vergence_min_distance_init
        self.vergence_max_distance_init = procedure_conf.vergence_max_distance_init
        self.test_dump_path = "./tests/"
        self.test_plot_path = "./plots/"
        self.dataset_path = "./datasets"
        self.replay_path = './replays'
        # TREE STRUCTURE
        os.makedirs(self.replay_path, exist_ok=True)
        os.makedirs(self.dataset_path, exist_ok=True)
        os.makedirs(self.test_dump_path, exist_ok=True)
        os.makedirs(self.test_plot_path, exist_ok=True)
        #
        self.test_conf = TestDataContainer.load(procedure_conf.test_conf_path)
        #    HPARAMS
        self._hparams = OrderedDict([
            ("buffer", buffer_conf.size),
            ("update_rate", procedure_conf.updates_per_sample),
            ("ep_length", procedure_conf.episode_length),
            ("batch_size", procedure_conf.batch_size),
            ("expl_prob", agent_conf.exploration.prob),
            ("expl_temp", agent_conf.exploration.temperature),
        ])
        #   OBJECTS
        self.buffer = Buffer(**buffer_conf)
        self.scale_names = list(agent_conf.scales.description)
        self.agent = Agent(**agent_conf)
        self.agent.set_critic_learning_rate(procedure_conf.critic_learning_rate[0].lr)
        self.agent.set_encoder_learning_rate(procedure_conf.critic_learning_rate[0].lr)
        #   SIMULATION POOL
        guis = list(simulation_conf.guis)
        self.simulation_pool = SimulationPool(
            simulation_conf.n,
            guis=guis
        )
        self.simulation_pool.add_background("ny_times_square")
        self.simulation_pool.add_head()
        for scale, scale_conf in agent_conf.scales.description.items():
            self.simulation_pool.add_scale(scale, (scale_conf.resolution, scale_conf.resolution), scale_conf.view_angle, scale_conf.downsampling)
        self.simulation_pool.add_uniform_motion_screen(
            textures_path=procedure_conf.screen.textures_path,
            size=procedure_conf.screen.size,
            min_distance=procedure_conf.screen.min_distance,
            max_distance=procedure_conf.screen.max_distance,
            max_depth_speed=procedure_conf.screen.max_depth_speed,
            max_speed_in_deg=procedure_conf.screen.max_speed_in_deg,
        )
        self.simulation_pool.start_sim()
        self.simulation_pool.step_sim()
        print("[procedure] all simulation started")

        fake_frame_by_scale_pavro = self.get_vision()
        fake_frame_by_scale_magno = self.merge_before_after(fake_frame_by_scale_pavro, fake_frame_by_scale_pavro)
        self.agent.create_all_variables(fake_frame_by_scale_pavro, fake_frame_by_scale_magno)

        self.evaluation_conf, n_sample_points = get_evaluation_conf(
            self.n_simulations * self.n_evaluation_batch,
            vergence_error_min=-90 / 320 * 8,
            vergence_error_max= 90 / 320 * 8,
            depth_min=procedure_conf.screen.min_distance,
            depth_max=procedure_conf.screen.max_distance,
            speed_max=procedure_conf.screen.max_speed_in_deg,
            tilt_on=agent_conf.pathways.magno.n_actions.tilt > 1,
            pan_on=agent_conf.pathways.magno.n_actions.pan > 1,
            vergence_on=agent_conf.pathways.pavro.n_actions.vergence > 1,
            cyclo_on=agent_conf.pathways.pavro.n_actions.cyclo > 1,
        )
        print("[procedure] evaluation will be performed on these points:")
        print(n_sample_points)

        #   DEFINING DATA BUFFERS
        # training
        pavro_dtype = np.dtype([
            (scale, np.float32, (scale_conf.resolution, scale_conf.resolution, 6))
            for scale, scale_conf in agent_conf.scales.description.items()
        ])
        magno_dtype = np.dtype([
            (scale, np.float32, (scale_conf.resolution, scale_conf.resolution, 12))
            for scale, scale_conf in agent_conf.scales.description.items()
        ])

        self._train_data_type = np.dtype([
            ("pavro_vision", pavro_dtype),
            ("magno_vision", magno_dtype),
            ("pavro_vision_after", pavro_dtype),
            ("magno_vision_after", magno_dtype),
            ("tilt_actions_indices", np.int32),
            ("pan_actions_indices", np.int32),
            ("vergence_actions_indices", np.int32),
            ("cyclo_actions_indices", np.int32),
            ("tilt_noisy_actions_indices", np.int32),
            ("pan_noisy_actions_indices", np.int32),
            ("vergence_noisy_actions_indices", np.int32),
            ("cyclo_noisy_actions_indices", np.int32),
            ("pavro_critic_targets", np.float32),
            ("magno_critic_targets", np.float32),
            ("pavro_recerr", np.float32),
            ("magno_recerr", np.float32),
            ("tilt_return_estimates", np.float32),
            ("pan_return_estimates", np.float32),
            ("vergence_return_estimates", np.float32),
            ("cyclo_return_estimates", np.float32),
        ])
        self._train_data_buffer = np.zeros(
            shape=(self.n_simulations, self.episode_length),
            dtype=self._train_data_type
        )
        # evaluation
        self._evaluation_data_type = np.dtype([
            ("pavro_vision", pavro_dtype),
            ("magno_vision", magno_dtype),
            ("tilt_actions_indices", np.int32),
            ("pan_actions_indices", np.int32),
            ("vergence_actions_indices", np.int32),
            ("cyclo_actions_indices", np.int32),
            ("pavro_recerr", np.float32),
            ("magno_recerr", np.float32),
            ("pavro_critic_targets", np.float32),
            ("magno_critic_targets", np.float32),
            ("tilt_return_estimates", np.float32),
            ("pan_return_estimates", np.float32),
            ("vergence_return_estimates", np.float32),
            ("cyclo_return_estimates", np.float32),
        ])
        self._evaluation_data_buffer = np.zeros(
            shape=(self.n_simulations * self.n_evaluation_batch, self.episode_length),
            dtype=self._evaluation_data_type
        )

        # COUNTERS
        self.n_exploration_episodes = 0
        self.n_evaluation_episodes = 0
        self.n_transition_gathered = 0
        self.n_critic_training = 0
        self.n_encoder_training = 0
        self.n_global_training = 0

        ### TENSORBOARD LOGGING ###
        self.tb = {}
        # TRAINING
        self.tb["training"] = {}
        # critics
        self.tb["training"]["critic"] = {}
        self.tb["training"]["critic"]["critic_tilt_loss"] = Mean(
            "training/critic_critic_tilt_loss", dtype=tf.float32)
        self.tb["training"]["critic"]["critic_pan_loss"] = Mean(
            "training/critic_critic_pan_loss", dtype=tf.float32)
        self.tb["training"]["critic"]["critic_vergence_loss"] = Mean(
            "training/critic_critic_vergence_loss", dtype=tf.float32)
        self.tb["training"]["critic"]["critic_cyclo_loss"] = Mean(
            "training/critic_critic_cyclo_loss", dtype=tf.float32)
        # encoders
        self.tb["training"]["encoders"] = {}
        self.tb["training"]["encoders"]["pavro_loss"] = Mean(
            "training/encoder_pavro_loss", dtype=tf.float32)
        self.tb["training"]["encoders"]["magno_loss"] = Mean(
            "training/encoder_magno_loss", dtype=tf.float32)

        # COLLECTION
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
            "collection/evaluation_total_episode_reward_pavro", dtype=tf.float32)
        self.tb["collection"]["exploration"]["total_episode_reward_magno"] = Mean(
            "collection/exploration_total_episode_reward_magno", dtype=tf.float32)
        self.tb["collection"]["evaluation"]["total_episode_reward_magno"] = Mean(
            "collection/evaluation_total_episode_reward_magno", dtype=tf.float32)
        self.tb["collection"]["exploration"]["recerr_pavro"] = Mean(
            "collection/exploration_recerr_pavro", dtype=tf.float32)
        self.tb["collection"]["evaluation"]["recerr_pavro"] = Mean(
            "collection/evaluation_recerr_pavro", dtype=tf.float32)
        self.tb["collection"]["exploration"]["recerr_magno"] = Mean(
            "collection/exploration_recerr_magno", dtype=tf.float32)
        self.tb["collection"]["evaluation"]["recerr_magno"] = Mean(
            "collection/evaluation_recerr_magno", dtype=tf.float32)
        self.tb["collection"]["exploration"]["final_vergence_error"] = Mean(
            "collection/exploration_final_vergence_error", dtype=tf.float32)
        self.tb["collection"]["evaluation"]["final_vergence_error"] = Mean(
            "collection/evaluation_final_vergence_error", dtype=tf.float32)
        self.tb["collection"]["exploration"]["final_tilt_error"] = Mean(
            "collection/exploration_final_tilt_error", dtype=tf.float32)
        self.tb["collection"]["evaluation"]["final_tilt_error"] = Mean(
            "collection/evaluation_final_tilt_error", dtype=tf.float32)
        self.tb["collection"]["exploration"]["final_pan_error"] = Mean(
            "collection/exploration_final_pan_error", dtype=tf.float32)
        self.tb["collection"]["evaluation"]["final_pan_error"] = Mean(
            "collection/evaluation_final_pan_error", dtype=tf.float32)
        for i in [1, 2, 3, 4, 6, 8, 10, 16]:
            self.tb["collection"]["exploration"]["final_vergence_{}px".format(i)] = Mean(
                "collection/exploration_final_vergence_{}px".format(i), dtype=tf.float32)
            self.tb["collection"]["evaluation"]["final_vergence_{}px".format(i)] = Mean(
                "collection/evaluation_final_vergence_{}px".format(i), dtype=tf.float32)
            self.tb["collection"]["exploration"]["final_tilt_{}px".format(i)] = Mean(
                "collection/exploration_final_tilt_{}px".format(i), dtype=tf.float32)
            self.tb["collection"]["evaluation"]["final_tilt_{}px".format(i)] = Mean(
                "collection/evaluation_final_tilt_{}px".format(i), dtype=tf.float32)
            self.tb["collection"]["exploration"]["final_pan_{}px".format(i)] = Mean(
                "collection/exploration_final_pan_{}px".format(i), dtype=tf.float32)
            self.tb["collection"]["evaluation"]["final_pan_{}px".format(i)] = Mean(
                "collection/evaluation_final_pan_{}px".format(i), dtype=tf.float32)
        self.tb["collection"]["exploration"]["critic_snr_tilt"] = Mean(
            "collection/exploration_critic_snr_tilt_db", dtype=tf.float32)
        self.tb["collection"]["evaluation"]["critic_snr_tilt"] = Mean(
            "collection/evaluation_critic_snr_tilt_db", dtype=tf.float32)
        self.tb["collection"]["exploration"]["critic_snr_pan"] = Mean(
            "collection/exploration_critic_snr_pan_db", dtype=tf.float32)
        self.tb["collection"]["evaluation"]["critic_snr_pan"] = Mean(
            "collection/evaluation_critic_snr_pan_db", dtype=tf.float32)
        self.tb["collection"]["exploration"]["critic_snr_vergence"] = Mean(
            "collection/exploration_critic_snr_vergence_db", dtype=tf.float32)
        self.tb["collection"]["evaluation"]["critic_snr_vergence"] = Mean(
            "collection/evaluation_critic_snr_vergence_db", dtype=tf.float32)
        self.tb["collection"]["exploration"]["critic_snr_cyclo"] = Mean(
            "collection/exploration_critic_snr_cyclo_db", dtype=tf.float32)
        self.tb["collection"]["evaluation"]["critic_snr_cyclo"] = Mean(
            "collection/evaluation_critic_snr_cyclo_db", dtype=tf.float32)
        #
        self.summary_writer = tf.summary.create_file_writer("logs")
        with self.summary_writer.as_default():
            hp.hparams(self._hparams)

    def log_metrics(self, key1, key2, step):
        with self.summary_writer.as_default():
            for name, metric in self.tb[key1][key2].items():
                tf.summary.scalar(metric.name, metric.result(), step=step)
                metric.reset_states()

    def log_summaries(self, exploration=True, evaluation=True, critic=True,
            encoders=True, wrt="exploration_episodes"):
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
                self.n_exploration_episodes
                # self.n_evaluation_episodes
            )
        if critic:
            self.log_metrics(
                "training",
                "critic",
                self.n_exploration_episodes if wrt == "exploration_episodes" else self.n_critic_training
            )
        if encoders:
            self.log_metrics(
                "training",
                "encoders",
                self.n_exploration_episodes if wrt == "exploration_episodes" else self.n_encoder_training
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

    def save(self, name=None):
        """Saves the model in the appropriate directory"""
        path = "./checkpoints/{:08d}".format(self.n_global_training) if name is None else "./checkpoints/" + name
        self.agent.save_weights(path)
        return path

    def restore(self, path, encoder=True, critic=True):
        """Restores the weights from a checkpoint"""
        self.agent.load_weights(path, encoder=encoder, critic=critic)

    def episode_reset_uniform_motion_screen(self, start_distances=None,
            depth_speeds=None, angular_speeds=None, directions=None,
            texture_ids=None, preinit=False):
        with self.simulation_pool.distribute_args():
            self.simulation_pool.episode_reset_uniform_motion_screen(
                [None] * self.simulation_pool.n if start_distances is None else start_distances,
                [None] * self.simulation_pool.n if depth_speeds is None else depth_speeds,
                [None] * self.simulation_pool.n if angular_speeds is None else angular_speeds,
                [None] * self.simulation_pool.n if directions is None else directions,
                [None] * self.simulation_pool.n if texture_ids is None else texture_ids,
                preinit=[preinit] * self.simulation_pool.n
            )

    def episode_reset_head(self, vergence=None, cyclo=None):
        if vergence is None:
            distance = np.random.uniform(
                low=self.vergence_min_distance_init,
                high=self.vergence_max_distance_init,
                size=self.simulation_pool.n
            )
            vergence = distance_to_vergence(distance)
        if cyclo is None:
            cyclo = np.random.uniform(
                low=-self.max_abs_cyclo_error,
                high=self.max_abs_cyclo_error,
                size=self.simulation_pool.n
            )
        with self.simulation_pool.distribute_args():
            self.simulation_pool.episode_reset_head(vergence, cyclo)

    def get_vision(self):
        with self.simulation_pool.distribute_args():
            vision_list = self.simulation_pool.get_vision()
        return {
            scale_name: np.stack([v[scale_name] for v in vision_list], axis=0)
            for scale_name in vision_list[0]
        }

    def merge_before_after(self, before, after):
        return {
            scale_name: np.concatenate(
                [before[scale_name], after[scale_name]],
                axis=-1
            ) for scale_name in before
        }

    def get_color_means(self):
        self.simulation_pool.episode_reset_head(vergence=0, cyclo=0)
        data = []
        for texture_id in range(20):
            self.simulation_pool.episode_reset_uniform_motion_screen(
                start_distance=2,
                depth_speed=0,
                angular_speed=0,
                direction=0,
                texture_id=texture_id,
                preinit=False,
            )
            self.simulation_pool.step_sim()
            vision_list = self.simulation_pool.get_vision()
            data.append([{
                scale_id: np.mean(0.5 + 0.5 * vision[scale_id], axis=(0, 1))
                for scale_id in vision
            } for vision in vision_list])
        color_means = [{
            scale_id: np.mean([
                stimulus_data[sim_id][scale_id]
                for stimulus_data in data
            ], axis=0)
            for scale_id in vision_list[0]
        } for sim_id in range(self.n_simulations)]
        return color_means

    def apply_action(self, actions):
        with self.simulation_pool.distribute_args():
            self.simulation_pool.apply_action(actions)

    def get_joints_errors(self):
        return tuple(zip(*self.simulation_pool.get_joints_errors()))

    def get_joints_positions(self):
        return tuple(zip(*self.simulation_pool.get_joints_positions()))

    def get_joints_velocities(self):
        return tuple(zip(*self.simulation_pool.get_joints_velocities()))

    def actions_indices_to_actions(self, tilt, pan, vergence, cyclo):
        actions = np.zeros(shape=(tilt.shape[0], 4))
        actions[:, 0] = self.agent.tilt_action_set[tilt]
        actions[:, 1] = self.agent.pan_action_set[pan]
        actions[:, 2] = self.agent.vergence_action_set[vergence]
        actions[:, 3] = self.agent.cyclo_action_set[cyclo]
        return actions

    def record(self, exploration=False, n_episodes=20, video_name='replay', resolution=(320, 240)):
        half_size = (resolution[1] // 2, resolution[0] // 2)
        black_frame = np.zeros(shape=(resolution[1], resolution[0] // 2, 3), dtype=np.uint8)
        video_names = [video_name + "_{:02d}.mp4".format(i) for i in range(self.n_simulations)]
        writers = [get_writer(name, fps=25) for name in video_names]
        self.simulation_pool.add_scale("record", resolution, 90.0, 1)
        n_episodes_done = 0
        while n_episodes_done < n_episodes:
            n_episodes_done += self.n_simulations
            self.episode_reset_uniform_motion_screen(preinit=True)
            self.episode_reset_head()
            vision_after = self.get_vision() # preinit frames
            left_rights = vision_after.pop("record")
            self.apply_action(np.zeros((self.n_simulations, 4)))
            prev_pavro_recerr = np.zeros(self.n_simulations)
            prev_magno_recerr = np.zeros(self.n_simulations)
            for iteration in range(30):
                vision_before = vision_after
                vision_after = self.get_vision()
                left_rights = vision_after.pop("record")
                pavro_vision = vision_after
                magno_vision = self.merge_before_after(vision_before, vision_after)
                tilt_error, pan_error, vergence_error = self.get_joints_errors()
                tilt_speed, pan_speed, vergence_speed, cyclo_speed = self.get_joints_velocities()
                tilt_pos, pan_pos, vergence_pos, cyclo_pos = self.get_joints_positions()
                data = self.agent(pavro_vision, magno_vision)
                if exploration:
                    actions = self.actions_indices_to_actions(
                        data["tilt_noisy_actions_indices"].numpy(),
                        data["pan_noisy_actions_indices"].numpy(),
                        data["vergence_noisy_actions_indices"].numpy(),
                        data["cyclo_noisy_actions_indices"].numpy(),
                    )
                else:
                    actions = self.actions_indices_to_actions(
                        data["tilt_actions_indices"].numpy(),
                        data["pan_actions_indices"].numpy(),
                        data["vergence_actions_indices"].numpy(),
                        data["cyclo_actions_indices"].numpy(),
                    )
                self.apply_action(actions)
                pavro_recerr = data["pavro_recerr"]
                magno_recerr = data["magno_recerr"]
                for i, (writer, left_right) in enumerate(zip(writers, left_rights)):
                    ana = anaglyph(left_right)
                    text_0 = text_frame(height=resolution[1], width=resolution[1],
                        line_0 =("episode", "{: 2d}".format(i * ((n_episodes + self.n_simulations - 1) // self.n_simulations) + n_episodes_done // self.n_simulations)),
                        line_1 =("iteration", "{: 2d}/{: 2d}".format(iteration + 1, 30)),
                        line_2 =("tilt error", "{:.2f}".format(tilt_error[i])),
                        line_3 =("pan error", "{:.2f}".format(pan_error[i])),
                        line_4 =("vergence error", "{:.2f}".format(vergence_error[i])),
                        line_5 =("tilt position", "{:.2f}".format(tilt_pos[i])),
                        line_6 =("pan position", "{:.2f}".format(pan_pos[i])),
                        line_7 =("vergence position", "{:.2f}".format(vergence_pos[i])),
                        line_8 =("cyclo position", "{:.2f}".format(cyclo_pos[i])),
                        line_9 =("tilt speed", "{:.2f}".format(tilt_speed[i])),
                        line_10=("pan speed", "{:.2f}".format(pan_speed[i])),
                        line_11=("vergence speed", "{:.2f}".format(vergence_speed[i])),
                        line_12=("cyclo speed", "{:.2f}".format(cyclo_speed[i])),
                        line_13=("pavro recerr", "{:.2f} - {:.2f} = {:.2f}".format(
                            prev_pavro_recerr[i] * self.reward_scaling,
                            pavro_recerr[i] * self.reward_scaling,
                            (prev_pavro_recerr[i] - pavro_recerr[i]) * self.reward_scaling)),
                        line_14=("magno recerr", "{:.2f} - {:.2f} = {:.2f}".format(
                            prev_magno_recerr[i] * self.reward_scaling,
                            magno_recerr[i] * self.reward_scaling,
                            (prev_magno_recerr[i] - magno_recerr[i]) * self.reward_scaling)),
                    )
                    text_1 = text_frame(height=resolution[1], width=((resolution[1] * 3 // 32) + 1) * 16,
                        line_0=("action tilt", "{:.2f}".format(actions[i, 0])),
                        line_1=("action pan", "{:.2f}".format(actions[i, 1])),
                        line_2=("action vergence", "{:.2f}".format(actions[i, 2])),
                        line_3=("action cyclo", "{:.2f}".format(actions[i, 3])),
                        line_4=("vergence values", ("{: .2f} " * len(data["vergence_return_estimates"][i])).format(*data["vergence_return_estimates"][i])),
                        line_5=("cyclo values", ("{: .2f} " * len(data["cyclo_return_estimates"][i])).format(*data["cyclo_return_estimates"][i])),
                        line_6=("tilt values", ("{: .2f} " * len(data["tilt_return_estimates"][i])).format(*data["tilt_return_estimates"][i])),
                        line_7=("pan values", ("{: .2f} " * len(data["pan_return_estimates"][i])).format(*data["pan_return_estimates"][i])),
                    )
                    # frame = np.concatenate([text_0, text_1, ana, top_bottom], axis=1)
                    frame = np.concatenate([text_0, text_1, ana], axis=1)
                    writer.append_data(frame)
                prev_pavro_recerr = pavro_recerr
                prev_magno_recerr = magno_recerr
        self.simulation_pool.delete_scale("record")
        for writer in writers:
            writer.close()
        with open("file_list.txt", "w") as f:
            for name in video_names:
                f.write("file '{}'\n".format(name))
        os.system("ffmpeg -hide_banner -loglevel panic -f concat -safe 0 -i file_list.txt -c copy {}.mp4".format(video_name))
        # os.remove("file_list.txt")
        for name in video_names:
            os.remove(name)

    def create_dataset(self, name, n_episodes):
        n_episodes_done = 0
        with open(get_original_cwd() + "/../datasets/{}.dat".format(name), "ab") as f:
            while n_episodes_done < n_episodes:
                n_episodes_done += self.n_simulations
                print("{:06d} / {:06d}".format(n_episodes_done, n_episodes))
                self.episode_reset_uniform_motion_screen(preinit=True)
                self.episode_reset_head()
                vision_after = self.get_vision()
                self.simulation_pool.step_sim()
                for iteration in range(self.episode_length):
                    vision_before = vision_after
                    vision_after = self.get_vision()
                    pavro_vision = vision_after
                    magno_vision = self.merge_before_after(vision_before, vision_after)
                    data = self.agent(pavro_vision, magno_vision)
                    noisy_actions = self.actions_indices_to_actions(
                        data["tilt_noisy_actions_indices"].numpy(),
                        data["pan_noisy_actions_indices"].numpy(),
                        data["vergence_noisy_actions_indices"].numpy(),
                        data["cyclo_noisy_actions_indices"].numpy(),
                    )
                    for scale_name in pavro_vision:
                        self._train_data_buffer[:, iteration]["pavro_vision"][scale_name] = pavro_vision[scale_name]
                        self._train_data_buffer[:, iteration]["magno_vision"][scale_name] = magno_vision[scale_name]
                    self._train_data_buffer[:, iteration]["tilt_actions_indices"] = data["tilt_actions_indices"]
                    self._train_data_buffer[:, iteration]["pan_actions_indices"] = data["pan_actions_indices"]
                    self._train_data_buffer[:, iteration]["vergence_actions_indices"] = data["vergence_actions_indices"]
                    self._train_data_buffer[:, iteration]["cyclo_actions_indices"] = data["cyclo_actions_indices"]
                    self._train_data_buffer[:, iteration]["tilt_noisy_actions_indices"] = data["tilt_noisy_actions_indices"]
                    self._train_data_buffer[:, iteration]["pan_noisy_actions_indices"] = data["pan_noisy_actions_indices"]
                    self._train_data_buffer[:, iteration]["vergence_noisy_actions_indices"] = data["vergence_noisy_actions_indices"]
                    self._train_data_buffer[:, iteration]["cyclo_noisy_actions_indices"] = data["cyclo_noisy_actions_indices"]
                    # not necessary for training but useful for logging:
                    self._train_data_buffer[:, iteration]["pavro_recerr"] = data["pavro_recerr"]
                    self._train_data_buffer[:, iteration]["magno_recerr"] = data["magno_recerr"]
                    self._train_data_buffer[:, iteration]["tilt_return_estimates"] = data["tilt_return_estimates"].numpy()[
                        np.arange(self.n_simulations),
                        data["tilt_noisy_actions_indices"].numpy()
                    ]
                    self._train_data_buffer[:, iteration]["pan_return_estimates"] = data["pan_return_estimates"].numpy()[
                        np.arange(self.n_simulations),
                        data["pan_noisy_actions_indices"].numpy()
                    ]
                    self._train_data_buffer[:, iteration]["vergence_return_estimates"] = data["vergence_return_estimates"].numpy()[
                        np.arange(self.n_simulations),
                        data["vergence_noisy_actions_indices"].numpy()
                    ]
                    self._train_data_buffer[:, iteration]["cyclo_return_estimates"] = data["cyclo_return_estimates"].numpy()[
                        np.arange(self.n_simulations),
                        data["cyclo_noisy_actions_indices"].numpy()
                    ]
                    self.apply_action(noisy_actions)
                # COMPUTE TARGET
                self._train_data_buffer[:, :-1]["pavro_critic_targets"] = self.reward_scaling * (
                    self._train_data_buffer[:, :-1]["pavro_recerr"] -
                    self._train_data_buffer[:,  1:]["pavro_recerr"]
                )
                self._train_data_buffer[:, :-1]["magno_critic_targets"] = self.reward_scaling * (
                    self._train_data_buffer[:, :-1]["magno_recerr"] -
                    self._train_data_buffer[:,  1:]["magno_recerr"]
                )
                f.write(self._train_data_buffer.tobytes())

    def create_controlled_dataset(self, name, n_samples, vergence_disparity_std, speed_disparity_std, cyclo_disparity_std):
        n_samples_done = 0
        with open(get_original_cwd() + "/../datasets/{}_vergence_{}_speed_{}_cyclo_{}.dat".format(name, vergence_disparity_std, speed_disparity_std, cyclo_disparity_std), "ab") as f:
            while n_samples_done < n_samples:
                n_samples_done += self.n_simulations
                print("{:06d} / {:06d}".format(n_samples_done, n_samples))
                if speed_disparity_std == 0:
                    angular_speeds = np.zeros(self.n_simulations)
                else:
                    angular_speeds = np.random.normal(
                        size=self.n_simulations,
                        scale=speed_disparity_std * 90 / 320
                    )
                start_distances = np.random.uniform(
                    size=self.n_simulations,
                    low=self.vergence_min_distance_init,
                    high=self.vergence_max_distance_init
                )
                if vergence_disparity_std == 0:
                    vergence_errors = np.zeros(self.n_simulations)
                else:
                    vergence_errors = np.random.normal(
                        size=self.n_simulations,
                        scale=vergence_disparity_std * 90 / 320
                    )
                if cyclo_disparity_std == 0:
                    cyclo = np.zeros(self.n_simulations)
                else:
                    cyclo = np.random.normal(
                        size=self.n_simulations,
                        scale=cyclo_disparity_std,
                    )
                vergence = distance_to_vergence(start_distances) - vergence_errors
                self.episode_reset_uniform_motion_screen(angular_speeds=angular_speeds, start_distances=start_distances)
                self.episode_reset_head(vergence=vergence, cyclo=cyclo)
                vision_after = self.get_vision()
                self.simulation_pool.step_sim()
                for iteration in range(2):
                    vision_before = vision_after
                    vision_after = self.get_vision()
                    pavro_vision = vision_after
                    magno_vision = self.merge_before_after(vision_before, vision_after)
                    data = self.agent(pavro_vision, magno_vision)
                    noisy_actions = self.actions_indices_to_actions(
                        data["tilt_noisy_actions_indices"].numpy(),
                        data["pan_noisy_actions_indices"].numpy(),
                        data["vergence_noisy_actions_indices"].numpy(),
                        data["cyclo_noisy_actions_indices"].numpy(),
                    )
                    for scale_name in pavro_vision:
                        self._train_data_buffer[:, iteration]["pavro_vision"][scale_name] = pavro_vision[scale_name]
                        self._train_data_buffer[:, iteration]["magno_vision"][scale_name] = magno_vision[scale_name]
                    self._train_data_buffer[:, iteration]["tilt_actions_indices"] = data["tilt_actions_indices"]
                    self._train_data_buffer[:, iteration]["pan_actions_indices"] = data["pan_actions_indices"]
                    self._train_data_buffer[:, iteration]["vergence_actions_indices"] = data["vergence_actions_indices"]
                    self._train_data_buffer[:, iteration]["cyclo_actions_indices"] = data["cyclo_actions_indices"]
                    self._train_data_buffer[:, iteration]["tilt_noisy_actions_indices"] = data["tilt_noisy_actions_indices"]
                    self._train_data_buffer[:, iteration]["pan_noisy_actions_indices"] = data["pan_noisy_actions_indices"]
                    self._train_data_buffer[:, iteration]["vergence_noisy_actions_indices"] = data["vergence_noisy_actions_indices"]
                    self._train_data_buffer[:, iteration]["cyclo_noisy_actions_indices"] = data["cyclo_noisy_actions_indices"]
                    # not necessary for training but useful for logging:
                    self._train_data_buffer[:, iteration]["pavro_recerr"] = data["pavro_recerr"]
                    self._train_data_buffer[:, iteration]["magno_recerr"] = data["magno_recerr"]
                    self._train_data_buffer[:, iteration]["tilt_return_estimates"] = data["tilt_return_estimates"].numpy()[
                        np.arange(self.n_simulations),
                        data["tilt_noisy_actions_indices"].numpy()
                    ]
                    self._train_data_buffer[:, iteration]["pan_return_estimates"] = data["pan_return_estimates"].numpy()[
                        np.arange(self.n_simulations),
                        data["pan_noisy_actions_indices"].numpy()
                    ]
                    self._train_data_buffer[:, iteration]["vergence_return_estimates"] = data["vergence_return_estimates"].numpy()[
                        np.arange(self.n_simulations),
                        data["vergence_noisy_actions_indices"].numpy()
                    ]
                    self._train_data_buffer[:, iteration]["cyclo_return_estimates"] = data["cyclo_return_estimates"].numpy()[
                        np.arange(self.n_simulations),
                        data["cyclo_noisy_actions_indices"].numpy()
                    ]
                    self.apply_action(np.zeros_like(noisy_actions))
                # COMPUTE TARGET
                self._train_data_buffer[:, 0]["pavro_critic_targets"] = self.reward_scaling * (
                    self._train_data_buffer[:, 0]["pavro_recerr"] -
                    self._train_data_buffer[:, 1]["pavro_recerr"]
                )
                self._train_data_buffer[:, 0]["magno_critic_targets"] = self.reward_scaling * (
                    self._train_data_buffer[:, 0]["magno_recerr"] -
                    self._train_data_buffer[:, 1]["magno_recerr"]
                )
                f.write(self._train_data_buffer[:, 0].tobytes())

    def collect_data(self):
        """Performs one episode of exploration, places data in the buffer"""
        time_start = time.time()
        self.episode_reset_uniform_motion_screen(preinit=True)
        self.episode_reset_head()
        vision_after = self.get_vision()
        self.simulation_pool.step_sim()
        for iteration in range(self.episode_length):
            vision_before = vision_after
            vision_after = self.get_vision()
            pavro_vision = vision_after
            magno_vision = self.merge_before_after(vision_before, vision_after)
            data = self.agent(pavro_vision, magno_vision)
            noisy_actions = self.actions_indices_to_actions(
                data["tilt_noisy_actions_indices"].numpy(),
                data["pan_noisy_actions_indices"].numpy(),
                data["vergence_noisy_actions_indices"].numpy(),
                data["cyclo_noisy_actions_indices"].numpy(),
            )
            for scale_name in pavro_vision:
                self._train_data_buffer[:, iteration]["pavro_vision"][scale_name] = pavro_vision[scale_name]
                self._train_data_buffer[:, iteration]["magno_vision"][scale_name] = magno_vision[scale_name]
            self._train_data_buffer[:, iteration]["tilt_actions_indices"] = data["tilt_actions_indices"]
            self._train_data_buffer[:, iteration]["pan_actions_indices"] = data["pan_actions_indices"]
            self._train_data_buffer[:, iteration]["vergence_actions_indices"] = data["vergence_actions_indices"]
            self._train_data_buffer[:, iteration]["cyclo_actions_indices"] = data["cyclo_actions_indices"]
            self._train_data_buffer[:, iteration]["tilt_noisy_actions_indices"] = data["tilt_noisy_actions_indices"]
            self._train_data_buffer[:, iteration]["pan_noisy_actions_indices"] = data["pan_noisy_actions_indices"]
            self._train_data_buffer[:, iteration]["vergence_noisy_actions_indices"] = data["vergence_noisy_actions_indices"]
            self._train_data_buffer[:, iteration]["cyclo_noisy_actions_indices"] = data["cyclo_noisy_actions_indices"]
            # not necessary for training but useful for logging:
            self._train_data_buffer[:, iteration]["pavro_recerr"] = data["pavro_recerr"]
            self._train_data_buffer[:, iteration]["magno_recerr"] = data["magno_recerr"]
            self._train_data_buffer[:, iteration]["tilt_return_estimates"] = data["tilt_return_estimates"].numpy()[
                np.arange(self.n_simulations),
                data["tilt_noisy_actions_indices"].numpy()
            ]
            self._train_data_buffer[:, iteration]["pan_return_estimates"] = data["pan_return_estimates"].numpy()[
                np.arange(self.n_simulations),
                data["pan_noisy_actions_indices"].numpy()
            ]
            self._train_data_buffer[:, iteration]["vergence_return_estimates"] = data["vergence_return_estimates"].numpy()[
                np.arange(self.n_simulations),
                data["vergence_noisy_actions_indices"].numpy()
            ]
            self._train_data_buffer[:, iteration]["cyclo_return_estimates"] = data["cyclo_return_estimates"].numpy()[
                np.arange(self.n_simulations),
                data["cyclo_noisy_actions_indices"].numpy()
            ]
            self.apply_action(noisy_actions)
        for scale_name in pavro_vision:
            self._train_data_buffer[:, :-1]["pavro_vision_after"][scale_name] = self._train_data_buffer[:, 1:]["pavro_vision"][scale_name]
            self._train_data_buffer[:, :-1]["magno_vision_after"][scale_name] = self._train_data_buffer[:, 1:]["magno_vision"][scale_name]
        # COMPUTE TARGET
        self._train_data_buffer[:, :-1]["pavro_critic_targets"] = self.reward_scaling * (
            self._train_data_buffer[:, :-1]["pavro_recerr"] -
            self._train_data_buffer[:,  1:]["pavro_recerr"]
        )
        self._train_data_buffer[:, :-1]["magno_critic_targets"] = self.reward_scaling * (
            self._train_data_buffer[:, :-1]["magno_recerr"] -
            self._train_data_buffer[:,  1:]["magno_recerr"]
        )
        # ADD TO BUFFER
        buffer_data = self._train_data_buffer[:, :-1].flatten()
        self.buffer.integrate(buffer_data)
        self.n_transition_gathered += len(buffer_data)
        self.n_exploration_episodes += self.n_simulations
        time_stop = time.time()
        # LOG METRICS
        final_tilt_error, final_pan_error, final_vergence_error = self.get_joints_errors()
        self.accumulate_log_data(
            tilt_return_estimates=self._train_data_buffer[:, :-1]["tilt_return_estimates"],
            pan_return_estimates=self._train_data_buffer[:, :-1]["pan_return_estimates"],
            vergence_return_estimates=self._train_data_buffer[:, :-1]["vergence_return_estimates"],
            cyclo_return_estimates=self._train_data_buffer[:, :-1]["cyclo_return_estimates"],
            pavro_critic_targets=self._train_data_buffer[:, :-1]["pavro_critic_targets"],
            magno_critic_targets=self._train_data_buffer[:, :-1]["magno_critic_targets"],
            pavro_recerr=self._train_data_buffer["pavro_recerr"],
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
        final_tilt_error = np.zeros(self.n_evaluation_batch * self.n_simulations)
        final_pan_error = np.zeros(self.n_evaluation_batch * self.n_simulations)
        final_vergence_error = np.zeros(self.n_evaluation_batch * self.n_simulations)
        for evaluation_batch in range(self.n_evaluation_batch):
            buffer_slice = slice(evaluation_batch * self.n_simulations, (evaluation_batch + 1) * self.n_simulations)
            self.episode_reset_uniform_motion_screen(
                start_distances=self.evaluation_conf["depths"][buffer_slice],
                depth_speeds=[0] * self.n_simulations,
                angular_speeds=self.evaluation_conf["speeds"][buffer_slice],
                directions=self.evaluation_conf["directions"][buffer_slice],
                texture_ids=self.evaluation_conf["stimulus"][buffer_slice],
                preinit=True,
            )
            self.episode_reset_head(
                vergence=distance_to_vergence(self.evaluation_conf["depths"][buffer_slice]) - self.evaluation_conf["vergence_errors"][buffer_slice],
                cyclo=[0] * self.n_simulations,
            )
            vision_after = self.get_vision()
            self.simulation_pool.step_sim()
            for iteration in range(self.episode_length):
                vision_before = vision_after
                vision_after = self.get_vision()
                pavro_vision = vision_after
                magno_vision = self.merge_before_after(vision_before, vision_after)
                data = self.agent(pavro_vision, magno_vision)
                actions = self.actions_indices_to_actions(
                    data["tilt_actions_indices"].numpy(),
                    data["pan_actions_indices"].numpy(),
                    data["vergence_actions_indices"].numpy(),
                    data["cyclo_actions_indices"].numpy(),
                )
                for scale_name in pavro_vision:
                    self._evaluation_data_buffer[buffer_slice, iteration]["pavro_vision"][scale_name] = pavro_vision[scale_name]
                    self._evaluation_data_buffer[buffer_slice, iteration]["magno_vision"][scale_name] = magno_vision[scale_name]
                self._evaluation_data_buffer[buffer_slice, iteration]["tilt_actions_indices"] = data["tilt_actions_indices"]
                self._evaluation_data_buffer[buffer_slice, iteration]["pan_actions_indices"] = data["pan_actions_indices"]
                self._evaluation_data_buffer[buffer_slice, iteration]["vergence_actions_indices"] = data["vergence_actions_indices"]
                self._evaluation_data_buffer[buffer_slice, iteration]["cyclo_actions_indices"] = data["cyclo_actions_indices"]
                self._evaluation_data_buffer[buffer_slice, iteration]["pavro_recerr"] = data["pavro_recerr"]
                self._evaluation_data_buffer[buffer_slice, iteration]["magno_recerr"] = data["magno_recerr"]
                self._evaluation_data_buffer[buffer_slice, iteration]["tilt_return_estimates"] = data["tilt_return_estimates"].numpy()[
                    np.arange(self.n_simulations),
                    data["tilt_actions_indices"].numpy()
                ]
                self._evaluation_data_buffer[buffer_slice, iteration]["pan_return_estimates"] = data["pan_return_estimates"].numpy()[
                    np.arange(self.n_simulations),
                    data["pan_actions_indices"].numpy()
                ]
                self._evaluation_data_buffer[buffer_slice, iteration]["vergence_return_estimates"] = data["vergence_return_estimates"].numpy()[
                    np.arange(self.n_simulations),
                    data["vergence_actions_indices"].numpy()
                ]
                self._evaluation_data_buffer[buffer_slice, iteration]["cyclo_return_estimates"] = data["cyclo_return_estimates"].numpy()[
                    np.arange(self.n_simulations),
                    data["cyclo_actions_indices"].numpy()
                ]
                self.apply_action(actions)
            tilt_error, pan_error, vergence_error = self.get_joints_errors()
            final_tilt_error[buffer_slice] = tilt_error
            final_pan_error[buffer_slice] = pan_error
            final_vergence_error[buffer_slice] = vergence_error
        for conf, tilt, pan, vergence in zip(self.evaluation_conf, final_tilt_error, final_pan_error, final_vergence_error):
            print(conf, vergence, tilt, pan)
        # COMPUTE TARGET
        self._evaluation_data_buffer[:, :-1]["pavro_critic_targets"] = self.reward_scaling * (
            self._evaluation_data_buffer[:, :-1]["pavro_recerr"] -
            self._evaluation_data_buffer[:,  1:]["pavro_recerr"]
        )
        self._evaluation_data_buffer[:, :-1]["magno_critic_targets"] = self.reward_scaling * (
            self._evaluation_data_buffer[:, :-1]["magno_recerr"] -
            self._evaluation_data_buffer[:,  1:]["magno_recerr"]
        )
        # COUNTER
        self.n_evaluation_episodes += self.n_simulations
        time_stop = time.time()
        # LOG METRICS
        with self.summary_writer.as_default():
            tf.summary.histogram("tilt", final_tilt_error, step=self.n_exploration_episodes)
            tf.summary.histogram("pan", final_pan_error, step=self.n_exploration_episodes)
            tf.summary.histogram("vergence", final_vergence_error, step=self.n_exploration_episodes)

        self.accumulate_log_data(
            tilt_return_estimates=self._evaluation_data_buffer[:, :-1]["tilt_return_estimates"],
            pan_return_estimates=self._evaluation_data_buffer[:, :-1]["pan_return_estimates"],
            vergence_return_estimates=self._evaluation_data_buffer[:, :-1]["vergence_return_estimates"],
            cyclo_return_estimates=self._evaluation_data_buffer[:, :-1]["cyclo_return_estimates"],
            pavro_critic_targets=self._evaluation_data_buffer[:, :-1]["pavro_critic_targets"],
            magno_critic_targets=self._evaluation_data_buffer[:, :-1]["magno_critic_targets"],
            pavro_recerr=self._evaluation_data_buffer["pavro_recerr"],
            magno_recerr=self._evaluation_data_buffer["magno_recerr"],
            final_vergence_error=final_vergence_error,
            final_tilt_error=final_tilt_error,
            final_pan_error=final_pan_error,
            time=(time_stop - time_start) / self.n_evaluation_batch,
            exploration=False,
        )

    def test(self, test_conf_path=None, dump_path=None, plot_path=None, name=None):
        if test_conf_path is None:
            test_conf = self.test_conf
        else:
            test_conf = TestDataContainer.load(test_conf_path)
        ##### CRITIC TESTING #####
        # n_stimulus = 20
        # test_conf.test_critic_data = self.get_critic_test_data(list(range(n_stimulus)), 0, 0, 0, 0, 0, "pavro")
        ##### GENERAL TESTING #####
        remaining = len(test_conf)
        for length in test_conf.get_tests_lengths():
            print("Testing length {}".format(length))
            for chunk in test_conf.tests_by_chunks(length, self.n_simulations):
                chunk_size = len(chunk)
                remaining -= chunk_size
                print("new chunk, remaining:", remaining)
                with self.simulation_pool.specific(list(range(chunk_size))):
                    conf = chunk["conf"]
                    angular_speeds_deg = np.sqrt(conf["tilt_error"] ** 2 + conf["pan_error"] ** 2)
                    directions = angle(conf["pan_error"], conf["tilt_error"])
                    self.episode_reset_uniform_motion_screen(
                        start_distances=conf["object_distance"],
                        depth_speeds=[0] * chunk_size,
                        angular_speeds=angular_speeds_deg,
                        directions=directions,
                        texture_ids=conf["stimulus"],
                        preinit=True,
                    )
                    self.episode_reset_head(
                        vergence=distance_to_vergence(conf["object_distance"]) - conf["vergence_error"],
                        cyclo=conf["cyclo_pos"],
                    )
                    vision_after = self.get_vision()
                    self.simulation_pool.step_sim()
                    for iteration in range(length):
                        vision_before = vision_after
                        vision_after = self.get_vision()
                        pavro_vision = vision_after
                        magno_vision = self.merge_before_after(vision_before, vision_after)
                        data = self.agent(pavro_vision, magno_vision)
                        actions = self.actions_indices_to_actions(
                            data["tilt_actions_indices"].numpy(),
                            data["pan_actions_indices"].numpy(),
                            data["vergence_actions_indices"].numpy(),
                            data["cyclo_actions_indices"].numpy(),
                        )
                        tilt_error, pan_error, vergence_error = self.get_joints_errors()
                        tilt_pos, pan_pos, vergence_pos, cyclo_pos = self.get_joints_positions()
                        tilt_speed, pan_speed, vergence_speed, cyclo_speed = self.get_joints_velocities()
                        chunk["result"]["vergence_error"][:, iteration] = vergence_error
                        chunk["result"]["pan_error"][:, iteration] = pan_error
                        chunk["result"]["tilt_error"][:, iteration] = tilt_error
                        chunk["result"]["recerr_magno"][:, iteration] = data["magno_recerr"]
                        chunk["result"]["recerr_pavro"][:, iteration] = data["pavro_recerr"]
                        chunk["result"]["tilt_return_estimates"][:, iteration] = data["tilt_return_estimates"].numpy()[
                            np.arange(chunk_size),
                            data["tilt_actions_indices"].numpy()
                        ]
                        chunk["result"]["pan_return_estimates"][:, iteration] = data["pan_return_estimates"].numpy()[
                            np.arange(chunk_size),
                            data["pan_actions_indices"].numpy()
                        ]
                        chunk["result"]["vergence_return_estimates"][:, iteration] = data["vergence_return_estimates"].numpy()[
                            np.arange(chunk_size),
                            data["vergence_actions_indices"].numpy()
                        ]
                        chunk["result"]["cyclo_return_estimates"][:, iteration] = data["cyclo_return_estimates"].numpy()[
                            np.arange(chunk_size),
                            data["cyclo_actions_indices"].numpy()
                        ]
                        chunk["result"]["pan_pos"][:, iteration] = pan_pos
                        chunk["result"]["tilt_pos"][:, iteration] = tilt_pos
                        chunk["result"]["vergence_pos"][:, iteration] = vergence_pos
                        chunk["result"]["cyclo_pos"][:, iteration] = cyclo_pos
                        chunk["result"]["pan_speed"][:, iteration] = pan_speed
                        chunk["result"]["tilt_speed"][:, iteration] = tilt_speed
                        chunk["result"]["vergence_speed"][:, iteration] = vergence_speed
                        chunk["result"]["cyclo_speed"][:, iteration] = cyclo_speed
                        chunk["result"]["tilt_action"][:, iteration] = actions[:, 0]
                        chunk["result"]["pan_action"][:, iteration] = actions[:, 1]
                        chunk["result"]["vergence_action"][:, iteration] = actions[:, 2]
                        chunk["result"]["cyclo_action"][:, iteration] = actions[:, 3]
                        self.apply_action(actions)
        filepath = self.test_dump_path if dump_path is None else dump_path
        name = "/{}_{:06d}".format(test_conf.name, self.n_exploration_episodes) if name is None else name
        test_conf.dump(filepath, name=name)
        path = self.test_plot_path + "/" + name if plot_path is None else plot_path
        test_conf.plot(path)

    # def get_critic_test_data(self, stimulus_list, action_axis,
    #         tilt_error, pan_error, vergence_error, cyclo_error,
    #         pathway_name, n_test_points=1000, distance=2.0):
    #     angular_speed = np.sqrt(pan_error ** 2 + tilt_error ** 2)
    #     direction = angle(pan_error, tilt_error)
    #     n_stimulus = len(stimulus_list)
    #     results = np.zeros(shape=(n_test_points, n_stimulus), dtype=critic_test_dtype)
    #     if action_axis == 0:
    #         actions = np.stack([
    #             np.linspace(-1, 1, n_test_points),
    #             np.zeros(n_test_points),
    #         ], axis=-1)
    #     else:
    #         actions = np.stack([
    #             np.zeros(n_test_points),
    #             np.linspace(-1, 1, n_test_points),
    #         ], axis=-1)
    #     while len(stimulus_list):
    #         stimulus_processed_now = stimulus_list[:self.n_simulations]
    #         stimulus_list = stimulus_list[self.n_simulations:]
    #         n_processed_now = len(stimulus_processed_now)
    #         with self.simulation_pool.specific(list(range(n_processed_now))):
    #             self.episode_reset_uniform_motion_screen(
    #                 start_distances=[distance] * n_processed_now,
    #                 depth_speeds=[0] * n_processed_now,
    #                 angular_speeds=[angular_speed] * n_processed_now,
    #                 directions=[direction] * n_processed_now,
    #                 texture_ids=stimulus_processed_now,
    #                 preinit=True,
    #             )
    #             self.episode_reset_head(
    #                 vergence=[distance_to_vergence(distance) - vergence_error] * n_processed_now,
    #                 cyclo=[cyclo_error] * n_processed_now,
    #             )
    #             vision_before = self.get_vision()
    #             self.simulation_pool.step_sim()
    #             vision_after = self.get_vision()
    #             pavro_vision = vision_after
    #             magno_vision = self.merge_before_after(vision_before, vision_after)
    #
    #         for i in range(n_processed_now):
    #             repeated_magno_vision = {scale: np.repeat(v[i:i + 1], n_test_points, axis=0) for scale, v in magno_vision.items()}
    #             repeated_pavro_vision = {scale: np.repeat(v[i:i + 1], n_test_points, axis=0) for scale, v in pavro_vision.items()}
    #             if pathway_name == "magno":
    #                 vision = repeated_magno_vision
    #             elif pathway_name == "pavro":
    #                 vision = repeated_pavro_vision
    #             else:
    #                 raise ValueError("Unrecognized pathway name {}".format(pathway_name))
    #             return_estimates = self.agent.get_return_estimates(vision, actions, pathway_name)
    #             gradient = self.agent.get_gradient(vision, actions, pathway_name)
    #             results["return_estimates"][:, stimulus_processed_now[i]] = return_estimates[..., 0]
    #             results["gradient"][:, stimulus_processed_now[i]] = gradient
    #     return results

    def accumulate_log_data(self,
            tilt_return_estimates, pan_return_estimates,
            vergence_return_estimates, cyclo_return_estimates,
            pavro_critic_targets, magno_critic_targets, pavro_recerr,
            magno_recerr, final_vergence_error, final_tilt_error,
            final_pan_error, time, exploration):
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
        tb["final_vergence_error"](np.mean(np.abs(final_vergence_error)))
        tb["final_tilt_error"](np.mean(np.abs(final_tilt_error)))
        tb["final_pan_error"](np.mean(np.abs(final_pan_error)))
        for i in [1, 2, 3, 4, 6, 8, 10, 16]:
            tb["final_tilt_{}px".format(i)](np.sum(np.abs(final_tilt_error) < i * 90 / 320) * 100 / len(final_tilt_error))
            tb["final_pan_{}px".format(i)](np.sum(np.abs(final_pan_error) < i * 90 / 320) * 100 / len(final_pan_error))
            tb["final_vergence_{}px".format(i)](np.sum(np.abs(final_vergence_error) < i * 90 / 320) * 100 / len(final_vergence_error))
        #
        signal = magno_critic_targets
        noise = magno_critic_targets - tilt_return_estimates
        critic_snr_tilt = get_snr_db(signal, noise)
        tb["critic_snr_tilt"](np.mean(critic_snr_tilt))
        #
        signal = magno_critic_targets
        noise = magno_critic_targets - pan_return_estimates
        critic_snr_pan = get_snr_db(signal, noise)
        tb["critic_snr_pan"](np.mean(critic_snr_pan))
        #
        signal = pavro_critic_targets
        noise = pavro_critic_targets - vergence_return_estimates
        critic_snr_vergence = get_snr_db(signal, noise)
        tb["critic_snr_vergence"](np.mean(critic_snr_vergence))
        #
        signal = pavro_critic_targets
        noise = pavro_critic_targets - cyclo_return_estimates
        critic_snr_cyclo = get_snr_db(signal, noise)
        tb["critic_snr_cyclo"](np.mean(critic_snr_cyclo))
        #

    def train(self, critic=True, encoders=True):
        self.n_global_training += 1

        critic_schedule = [(x.iteration, x.lr) for x in self.procedure_conf.critic_learning_rate]
        for it, lr in reversed(critic_schedule):
            if self.n_exploration_episodes > it:
                self.agent.set_critic_learning_rate(lr)
                break

        encoder_schedule = [(x.iteration, x.lr) for x in self.procedure_conf.encoder_learning_rate]
        for it, lr in reversed(encoder_schedule):
            if self.n_exploration_episodes > it:
                self.agent.set_encoder_learning_rate(lr)
                break

        if self.buffer.enough(self.batch_size):
            indices, data = self.buffer.sample(self.batch_size)
            tb = self.tb["training"]
            pavro_frame_by_scale = {
                scale_name: data["pavro_vision"][scale_name]
                for scale_name in self.scale_names
            }
            magno_frame_by_scale = {
                scale_name: data["magno_vision"][scale_name]
                for scale_name in self.scale_names
            }
            pavro_frame_by_scale_after = {
                scale_name: data["pavro_vision_after"][scale_name]
                for scale_name in self.scale_names
            }
            magno_frame_by_scale_after = {
                scale_name: data["magno_vision_after"][scale_name]
                for scale_name in self.scale_names
            }
            losses = self.agent.train(
                pavro_frame_by_scale,
                magno_frame_by_scale,
                data["tilt_noisy_actions_indices"],
                data["pan_noisy_actions_indices"],
                data["vergence_noisy_actions_indices"],
                data["cyclo_noisy_actions_indices"],
                data["magno_critic_targets"],
                data["pavro_critic_targets"],
                encoders=encoders,
                critic=critic)
            if critic:
                tb["critic"]["critic_tilt_loss"](losses["critic_tilt"])
                tb["critic"]["critic_pan_loss"](losses["critic_pan"])
                tb["critic"]["critic_vergence_loss"](losses["critic_vergence"])
                tb["critic"]["critic_cyclo_loss"](losses["critic_cyclo"])
                self.n_critic_training += 1
            if encoders:
                tb["encoders"]["pavro_loss"](losses["pavro_encoders"])
                tb["encoders"]["magno_loss"](losses["magno_encoders"])
                self.n_encoder_training += 1
            pavro_recerr = self.agent.get_encoder_loss(pavro_frame_by_scale, 'pavro')
            magno_recerr = self.agent.get_encoder_loss(magno_frame_by_scale, 'magno')
            pavro_recerr_after = self.agent.get_encoder_loss(pavro_frame_by_scale_after, 'pavro')
            magno_recerr_after = self.agent.get_encoder_loss(magno_frame_by_scale_after, 'magno')
            pavro_critic_targets = self.reward_scaling * (pavro_recerr - pavro_recerr_after)
            magno_critic_targets = self.reward_scaling * (magno_recerr - magno_recerr_after)
            self.buffer.buffer["pavro_critic_targets"][indices] = pavro_critic_targets
            self.buffer.buffer["magno_critic_targets"][indices] = magno_critic_targets
            return losses

    def train_from_dataset(self, cfg):
        dataset_path = to_absolute_path(cfg.dataset_path)
        dtype_path = None if cfg.dtype_path is None else to_absolute_path(cfg.dtype_path)
        n_training_steps = cfg.n_training_steps
        critic = cfg.critic
        encoders = cfg.encoders

        if dtype_path is not None:
            import pickle
            with open(dtype_path, 'rb') as f:
                dtype = pickle.load(f)
            print("[procedure] dataset type loaded from path ({})".format(cfg.dtype_path))
        else:
            dtype = self._train_data_type
            print("[procedure] default dataset type")

        with open(dataset_path, "rb") as f:
            dataset = np.frombuffer(f.read(), dtype=dtype).flatten()

        self.n_global_training += 1

        self.agent.set_critic_learning_rate(cfg.critic_learning_rate)
        self.agent.set_encoder_learning_rate(cfg.encoder_learning_rate)
        batch_size = cfg.batch_size

        for i in range(n_training_steps):
            print("Training step {: 7d} / {: 7d}".format(i + 1, n_training_steps))
            data = dataset[np.random.choice(dataset.shape[0], size=batch_size, replace=False)]
            tb = self.tb["training"]
            pavro_frame_by_scale = {
                scale_name: data["pavro_vision"][scale_name]
                for scale_name in self.scale_names
            }
            magno_frame_by_scale = {
                scale_name: data["magno_vision"][scale_name]
                for scale_name in self.scale_names
            }
            losses = self.agent.train(
                pavro_frame_by_scale,
                magno_frame_by_scale,
                data["tilt_noisy_actions_indices"],
                data["pan_noisy_actions_indices"],
                data["vergence_noisy_actions_indices"],
                data["cyclo_noisy_actions_indices"],
                data["magno_critic_targets"],
                data["pavro_critic_targets"],
                encoders=encoders,
                critic=critic)
            if critic:
                tb["critic"]["critic_tilt_loss"](losses["critic_tilt"])
                tb["critic"]["critic_pan_loss"](losses["critic_pan"])
                tb["critic"]["critic_vergence_loss"](losses["critic_vergence"])
                tb["critic"]["critic_cyclo_loss"](losses["critic_cyclo"])
                self.n_critic_training += 1
            if encoders:
                tb["encoders"]["pavro_loss"](losses["pavro_encoders"])
                tb["encoders"]["magno_loss"](losses["magno_encoders"])
                self.n_encoder_training += 1
            self.log_summaries(exploration=False, evaluation=False, critic=critic, encoders=encoders, wrt='training')

    def collect_and_train(self, critic=True, encoders=True):
        self.collect_data()
        while self.current_training_ratio < self.updates_per_sample:
            self.train(critic=critic, encoders=encoders)

    def collect_train_and_log(self, critic=True, encoders=True, evaluation=False):
        self.collect_and_train(critic=critic, encoders=encoders)
        if evaluation:
            self.evaluate()
            self.log_summaries(exploration=True, evaluation=True, critic=critic, encoders=encoders)

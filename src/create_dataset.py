import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from procedure import Procedure
import os
import tensorflow as tf
import custom_interpolations
from test_data import TestDataContainer


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_conf(path):
    cfg = OmegaConf.load(path + '/../../.hydra/config.yaml')
    return cfg


@hydra.main(config_path='../config/scripts/', config_name='create_dataset.yaml')
def main(cfg):
    create_dataset(cfg)


def create_dataset(cfg):
    checkpoint_path = to_absolute_path(cfg.checkpoint_path)
    experiment_cfg = get_conf(checkpoint_path)
    agent_conf = experiment_cfg.agent
    buffer_conf = experiment_cfg.buffer
    simulation_conf = experiment_cfg.simulation
    procedure_conf = experiment_cfg.procedure
    procedure_conf.screen.min_distance = 2
    experiment_conf = experiment_cfg.experiment
    simulation_conf.n = cfg.override_n_sim
    tf.config.threading.set_intra_op_parallelism_threads(8)
    with Procedure(agent_conf, buffer_conf, simulation_conf, procedure_conf) as procedure:
        procedure.restore(checkpoint_path, encoder=True, critic=True)
        if cfg.controlled:
            procedure.create_controlled_dataset(
                cfg.dataset_name,
                cfg.n_samples,
                cfg.vergence_disparity_std,
                cfg.speed_disparity_std,
                cfg.cyclo_disparity_std
            )
        else:
            procedure.create_dataset(cfg.dataset_name, cfg.n_episodes)


if __name__ == "__main__":
    main()

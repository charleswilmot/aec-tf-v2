import hydra
from omegaconf import OmegaConf
from procedure import Procedure
import os
import tensorflow as tf
import custom_interpolations
from test_data import TestDataContainer


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_conf():
    cfg = OmegaConf.load('../.hydra/config.yaml')
    return cfg


@hydra.main(config_path='../config/script/', config_name='create_dataset.yaml')
def main(cfg):
    create_dataset(cfg)


def create_dataset(cfg):
    experiment_cfg = get_conf()
    agent_conf = experiment_cfg.agent
    buffer_conf = experiment_cfg.buffer
    simulation_conf = experiment_cfg.simulation
    procedure_conf = experiment_cfg.procedure
    experiment_conf = experiment_cfg.experiment
    tf.config.threading.set_intra_op_parallelism_threads(8)
    checkpoint_path = os.getcwd() + "/../checkpoints/" + cfg.path.rstrip('/').split('/')[-1]
    with Procedure(agent_conf, buffer_conf, simulation_conf, procedure_conf) as procedure:
        procedure.restore(checkpoint_path, encoder=True, critic=True)
        procedure.create_dataset(cfg.n_episodes)


if __name__ == "__main__":
    main()

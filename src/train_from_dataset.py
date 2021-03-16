import hydra
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf
from procedure import Procedure
import os
import tensorflow as tf
import custom_interpolations
from test_data import TestDataContainer


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


@hydra.main(config_path='../config/script/', config_name='train_from_dataset.yaml')
def main(cfg):
    train_from_dataset(cfg)


def train_from_dataset(cfg):
    experiment_cfg = OmegaConf.load(get_original_cwd() + '/' + cfg.other_conf_path)
    agent_conf = experiment_cfg.agent
    buffer_conf = experiment_cfg.buffer
    simulation_conf = experiment_cfg.simulation
    procedure_conf = experiment_cfg.procedure
    experiment_conf = experiment_cfg.experiment
    tf.config.threading.set_intra_op_parallelism_threads(8)
    with Procedure(agent_conf, buffer_conf, simulation_conf, procedure_conf) as procedure:
        procedure.train_from_dataset(cfg)
        procedure.test()


if __name__ == "__main__":
    main()

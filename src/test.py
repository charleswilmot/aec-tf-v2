from procedure import Procedure
from omegaconf import OmegaConf
from pathlib import Path
import hydra
import sys
from test_data import TestDataContainer
import custom_interpolations


def get_conf():
    cfg = OmegaConf.load('../.hydra/config.yaml')
    return cfg


@hydra.main(config_path="../config/scripts/", config_name="test.yaml")
def main(cfg):
    test(cfg)


def test(cfg):
    experiment_cfg = get_conf()
    agent_conf = experiment_cfg.agent
    buffer_conf = experiment_cfg.buffer
    simulation_conf = experiment_cfg.simulation
    procedure_conf = experiment_cfg.procedure
    simulation_conf.n = cfg.n_simulations
    if cfg.gui:
        simulation_conf.guis = [0]

    relative_checkpoint_path = "../checkpoints/" + Path(cfg.path).stem
    with Procedure(agent_conf, buffer_conf, simulation_conf,
            procedure_conf) as procedure:
        print("[TEST] restoring from checkpoint")
        procedure.restore(relative_checkpoint_path)
        print("[TEST] start tests")
        procedure.test(test_conf_path=cfg.test_conf_path + "/" + cfg.test_conf_name)
        print("[TEST] finished, exiting")

if __name__ == '__main__':
    main()

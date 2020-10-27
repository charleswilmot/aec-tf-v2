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


@hydra.main(config_path="../config/scripts/", config_name="replay.yaml", strict=True)
def main(cfg):
    replay(cfg)


def replay(cfg):
    experiment_cfg = get_conf()
    agent_conf = experiment_cfg.agent
    buffer_conf = experiment_cfg.buffer
    simulation_conf = experiment_cfg.simulation
    procedure_conf = experiment_cfg.procedure
    simulation_conf.n = 4
    if cfg.gui:
        simulation_conf.guis = [0]

    video_name = cfg.name + '_exploration' if cfg.exploration else cfg.name
    relative_checkpoint_path = "../checkpoints/" + Path(cfg.path).stem
    with Procedure(agent_conf, buffer_conf, simulation_conf,
            procedure_conf) as procedure:
        procedure.restore(relative_checkpoint_path)
        procedure.record(
            exploration=cfg.exploration,
            n_episodes=cfg.n_episodes,
            video_name=video_name,
            resolution=cfg.resolution,
        )


if __name__ == '__main__':
    main()

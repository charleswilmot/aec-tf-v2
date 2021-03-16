from create_dataset import create_dataset
import hydra
import json
import omegaconf
import os
from test_data import TestDataContainer


@hydra.main(config_path="../config/scripts/", config_name="cluster.yaml")
def remote_create_dataset(cfg):
    with open(cfg.rundir + '/cfg.json', 'r') as f:
        other_cfg_json = json.load(f)
    os.chdir(cfg.rundir)
    cfg = omegaconf.OmegaConf.create(other_cfg_json)
    create_dataset(cfg)


if __name__ == '__main__':
    remote_create_dataset()

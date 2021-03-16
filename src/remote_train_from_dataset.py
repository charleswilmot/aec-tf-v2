from train_from_dataset import train_from_dataset
import hydra
import json
import omegaconf
import os
from test_data import TestDataContainer


@hydra.main(config_path="../config/scripts/", config_name="cluster.yaml")
def remote_train_from_dataset(cfg):
    with open(cfg.rundir + '/cfg.json', 'r') as f:
        other_cfg_json = json.load(f)
    os.chdir(cfg.rundir)
    cfg = omegaconf.OmegaConf.create(other_cfg_json)
    train_from_dataset(cfg)


if __name__ == '__main__':
    remote_train_from_dataset()

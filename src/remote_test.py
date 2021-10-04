from test import test
import hydra
import json
import omegaconf
import os
from test_data import TestDataContainer


@hydra.main(config_path="../config/scripts/", config_name="cluster.yaml")
def remote_test(cfg):
    with open(cfg.rundir + '/cfg.json', 'r') as f:
        other_cfg_json = json.load(f)
    os.chdir(cfg.rundir)
    cfg = omegaconf.OmegaConf.create(other_cfg_json)
    test(cfg)


if __name__ == '__main__':
    remote_test()

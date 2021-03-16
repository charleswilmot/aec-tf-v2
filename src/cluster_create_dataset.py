from omegaconf import OmegaConf
from cluster_utils import get_job_name, get_partition_reservation, ssh_command
import hydra
import os
from hydra.utils import get_original_cwd
from socket import gethostname
import sys
import json
import time
import custom_interpolations


REMOTE_HOST_NAME = 'otto'


@hydra.main(config_path='../config/scripts/', config_name='create_dataset.yaml')
def start_job(cfg):
        experiment_path = os.getcwd()
        pickle_conf_path = experiment_path + '/cfg.json'
        with open(pickle_conf_path, "w") as f:
            json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=4)
        command_line_args  = " rundir=" + experiment_path
        job_name = get_job_name()
        output_flag = "--output {outdir}/%N_%j.joblog".format(outdir=experiment_path)
        job_name_flag = "--job-name {job_name}".format(job_name=job_name)
        partition, reservation = get_partition_reservation()
        partition_flag = "--partition {partition}".format(partition=partition)
        reservation_flag = "--reservation {reservation}".format(reservation=reservation) if reservation is not None else ""
        os.chdir(get_original_cwd())
        command_line = "sbatch {output_flag} {job_name_flag} {partition_flag} {reservation_flag} cluster_create_dataset.sh ".format(
            output_flag=output_flag,
            job_name_flag=job_name_flag,
            partition_flag=partition_flag,
            reservation_flag=reservation_flag,
        ) + command_line_args
        print(command_line, flush=True)
        os.system(command_line)
        time.sleep(20)


if __name__ == "__main__" and gethostname() == REMOTE_HOST_NAME:
    start_job()
if __name__ == "__main__" and gethostname() != REMOTE_HOST_NAME:
    ssh_command("python " + " ".join(sys.argv))

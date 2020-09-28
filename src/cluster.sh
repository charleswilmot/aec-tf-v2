#!/usr/bin/env bash
#SBATCH --mincpus 12
#SBATCH --mem 40000
#SBATCH --exclude rockford,steele,hammer,conan,blomquist,wolfe,knatterton,holmes,lenssen,scuderi,matula,marlowe,poirot,monk
#SBATCH -LXserver
#SBATCH --gres gpu:1

##SBATCH --exclude turbine,vane


#srun -u xvfb-run -a python remote_experiment.py "$@"
srun -u python remote_experiment.py "$@"

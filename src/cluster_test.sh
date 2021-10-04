#!/usr/bin/env bash
#SBATCH --mincpus 8
#SBATCH --mem 31200
#SBATCH --exclude rockford,steele,hammer,conan,blomquist,wolfe,knatterton,holmes,lenssen,scuderi,matula,marlowe,poirot,monk
#SBATCH -LXserver
#SBATCH --gres gpu:1


srun -u python remote_test.py "$@"

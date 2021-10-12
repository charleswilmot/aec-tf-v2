# Active efficient coding TF v2

## Installing

The script `install_fias.sh` does the following:
- Clone this repository under `~/Code/aec-tf-v2/`
- Install all requirements using pip (see `requirements.txt`)
- Downloads `CoppeliaSim_Edu_V4_2_0` and installs it under `~/.Softwares`
- Configures the `~/.bashrc` according to [PyRep](https://github.com/stepjam/PyRep)'s documentation


If you are not confident with doing all that yourself, just run the script.


In order to run the code on FIAS' cluster, you will have to ask the permission to use the `sleuths` partition. For this, Jochen will have to send an email for you to the it@fias to let them know to grant you access to `sleuths`.




## Usage

### Running locally

First thing you should do after installing is running the `simulation.py` script to check that the installation process succeeded.

```
python simulation.py
```

Next you can run an experiment (ie. training and agent) locally

```
python experiment.py simulation.n=4 experiment.n_episodes=1000
```

This command will run an experiment of 1000 episodes with 4 simulated environments in parallel.
You will find the results of the experiment under the `/experiments` directory.

- `/experiments/path_to_the_experiments_results/tests/` contains pickle files which themselves contain measurments about the performance of the agent
- `/experiments/path_to_the_experiments_results/plots/` contains plots showing the performances of the agent. These plot can be regenerated from the previously mentioned pickle files
- `/experiments/path_to_the_experiments_results/checkpoints/` contains TensorFlow checkpoint files that you can use to restore the model's weights

You can run three repetitions of the experiment with the command

```
python experiment.py --multirun simulation.n=4 experiment.n_episodes=1000 experiment.repetition=0,1,2
```

the `--multirun` flag means that many experiments will get started by pooling over all the values `0,1,2` for the experiment option `repetition`. (This mechanism also works for all other options)


Once the experiment has completed, you can generate a video showing the agent's performance with the command

```
python replay.py path=../experiments/path_to_the_experiments_results/checkpoints/0001234567/
```

If needed, you can generate new test files (same as those in `/experiments/path_to_the_experiments_results/tests/`) when testing under different conditions (ie. different screen distances / different screen speeds etc) with the command

```
python test.py path=../experiments/path_to_the_experiments_results/checkpoints/0001234567/ test_conf_name=name_of_a_conf_file
```

where `name_of_a_conf_file` is the filename of one of the files that you'll find in `/config/test_conf/`. These test_conf files describe how the agent's perf should be tested (ie. at what screen distance, screen speed etc).

You can create new test_conf files yourself by updating the script at the end of the file `/src/test_data.py` and running it with the command

```
python test_data.py
```



### Running on the cluster

in order to run the experiment on the cluster, which I highly advise, you can use the script files starting by `cluster_*.py`:

```
python cluster.py experiment.n_episodes=100000
```

will queue up a job on the cluster which will run a full experiment (100000 episodes).

For this to work however, you will have to edit the content of the file `cluster_utils.py`, lines 69 to 74. Checkout the content of your `~/.bashrc`. We can also call or text if you need help with this.

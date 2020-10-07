import hydra
from omegaconf import OmegaConf
from procedure import Procedure
import os
import tensorflow as tf
import custom_interpolations
from test_data import TestDataContainer


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


@hydra.main(config_path='../config/algo/', config_name='config.yaml')
def main(cfg):
    experiment(cfg)


def experiment(cfg):
    print(OmegaConf.to_yaml(cfg, resolve=True), end="\n\n\n")
    agent_conf = cfg.agent
    buffer_conf = cfg.buffer
    simulation_conf = cfg.simulation
    procedure_conf = cfg.procedure
    experiment_conf = cfg.experiment
    with Procedure(agent_conf, buffer_conf, simulation_conf, procedure_conf) as procedure:
        n_episode_batch = experiment_conf.n_episodes // simulation_conf.n
        if experiment_conf.test_at_start:
            procedure.test()
        for episode_batch in range(n_episode_batch):
            critic = (episode_batch + 1) % experiment_conf.critic_every == 0
            encoders = (episode_batch + 1) % experiment_conf.encoders_every == 0
            evaluation = (episode_batch + 1) % experiment_conf.evaluate_every == 0
            test = (episode_batch + 1) % experiment_conf.test_every == 0
            save = (episode_batch + 1) % experiment_conf.save_every == 0
            record = (episode_batch + 1) % experiment_conf.record_episode_every == 0
            print_info = (episode_batch + 1) % 10 == 0
            print("batch {: 5d}\tevaluation:{}\tcritic:{}\tencoders:{}\tsave:{}\trecord:{}".format(
                episode_batch + 1,
                evaluation,
                critic,
                encoders,
                save,
                record,
            ))
            if record:
                procedure.record(
                    video_name='./replays/replay_{:05d}'.format(episode_batch + 1),
                    n_episodes=1,
                    exploration=False
                )
                procedure.record(
                    video_name='./replays/replay_exploration_{:05d}'.format(episode_batch + 1),
                    n_episodes=1,
                    exploration=True
                )
            if test:
                procedure.test()
            if save:
                procedure.save()
            procedure.collect_train_and_log(critic=critic, encoders=encoders, evaluation=evaluation)
            if print_info:
                print('n_exploration_episodes  ...  ', procedure.n_exploration_episodes)
                print('n_evaluation_episodes  ....  ', procedure.n_evaluation_episodes)
                print('n_transition_gathered  ....  ', procedure.n_transition_gathered)
                print('n_critic_training  ........  ', procedure.n_critic_training)
                print('n_encoder_training  .......  ', procedure.n_encoder_training)
                print('n_global_training  ........  ', procedure.n_global_training)
        if not save:
            procedure.save()
        if not test:
            procedure.test()
        if experiment_conf.final_recording:
            print("Generating final recording (without exploration)")
            procedure.record(
                video_name='./replays/replay_final',
                n_episodes=4,
                exploration=False
            )
        print("Experiment finished, hope it worked. Good bye!")


if __name__ == "__main__":
    main()

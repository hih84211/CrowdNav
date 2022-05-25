import sys
import logging
import argparse
import configparser
import os
import shutil
import torch
import gym
import git
from crowd_sim.envs.utils.robot import Robot
from crowd_nav.utils.trainer import Trainer
from crowd_nav.utils.memory import ReplayMemory
from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory


def main(argv=None):
    if argv is None:
        argv = sys.argv
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--config', type=str, default='configs/params.config',
                        help='Directory of the environment configuration file')
    parser.add_argument('--output_dir', type=str, default='data/output',
                        help='Directory to store the trained model weights')
    parser.add_argument('--weights', type=str)
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    args = parser.parse_args(argv)

    # configure paths
    make_new_dir = True
    if os.path.exists(args.output_dir):
        key = input('Output directory already exists! Overwrite the folder? (y/n)')
        if key == 'y' and not args.resume:
            shutil.rmtree(args.output_dir)
        else:
            make_new_dir = False
            args.config = os.path.join(args.output_dir, os.path.basename(args.config))
    if make_new_dir:
        os.makedirs(args.output_dir)
    log_file = os.path.join(args.output_dir, 'output.log')
    rl_weight_file = os.path.join(args.output_dir, 'rl_model.pth')

    # configure logging
    mode = 'a' if args.resume else 'w'
    file_handler = logging.FileHandler(log_file, mode=mode)
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO if not args.debug else logging.DEBUG
    logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    repo = git.Repo(search_parent_directories=True)
    logging.info('Current git head hash code: %s'.format(repo.head.object.hexsha))
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)

    config = configparser.RawConfigParser()
    config.read(args.config)

    # configure policy
    policy = policy_factory['carl']()
    if not policy.trainable:
        parser.error('Policy has to be trainable')

    policy.set_device(device)

    # configure environment
    env = gym.make('CrowdSim-v0')
    env.configure(config)
    robot = Robot(config, 'robot')
    env.set_robot(robot)

    # read training parameters
    learning_rate = config.getfloat('train', 'learning_rate')
    train_batches = config.getint('train', 'train_batches')
    train_episodes = config.getint('train', 'train_episodes')
    sample_episodes = config.getint('train', 'sample_episodes')
    target_update_interval = config.getint('train', 'target_update_interval')
    evaluation_interval = config.getint('train', 'evaluation_interval')
    capacity = config.getint('train', 'capacity')
    epsilon_start = config.getfloat('train', 'epsilon_start')
    epsilon_end = config.getfloat('train', 'epsilon_end')
    epsilon_decay = config.getfloat('train', 'epsilon_decay')
    checkpoint_interval = config.getint('train', 'checkpoint_interval')

    # configure trainer and explorer
    memory = ReplayMemory(capacity)
    model = policy.get_model()
    batch_size = config.getint('train', 'batch_size')
    trainer = Trainer(model, memory, device, batch_size)
    explorer = Explorer(env, robot, device, memory, policy.gamma, target_policy=policy)

    if args.resume:
        if not os.path.exists(rl_weight_file):
            logging.error('RL weights does not exist')
        model.load_state_dict(torch.load(rl_weight_file))
        rl_weight_file = os.path.join(args.output_dir, 'resumed_rl_model.pth')
        logging.info('Load reinforcement learning trained weights. Resume training')

        if robot.visible:
            safety_space = 0

    explorer.update_target_model(model)

    # reinforcement learning
    policy.set_env(env)
    robot.set_policy(policy)
    robot.print_info()
    trainer.set_learning_rate(learning_rate)
    # fill the memory pool with some RL experience
    if args.resume:
        robot.policy.set_epsilon(epsilon_end)
        explorer.run_k_episodes(100, 'train', update_memory=True, episode=0)
        logging.info('Experience set size: %d/%d', len(memory), memory.capacity)
    episode = 0
    while episode < train_episodes:
        if args.resume:
            epsilon = epsilon_end
        else:
            if episode < epsilon_decay:
                epsilon = epsilon_start + (epsilon_end - epsilon_start) / epsilon_decay * episode
            else:
                epsilon = epsilon_end
        robot.policy.set_epsilon(epsilon)

        # evaluate the model
        if episode % evaluation_interval == 0:
            explorer.run_k_episodes(env.case_size['val'], 'val', episode=episode)

        # sample k episodes into memory and optimize over the generated memory
        explorer.run_k_episodes(sample_episodes, 'train', update_memory=True, episode=episode)
        trainer.optimize_batch(train_batches)
        episode += 1

        if episode % target_update_interval == 0:
            explorer.update_target_model(model)

        if episode != 0 and episode % checkpoint_interval == 0:
            torch.save(model.state_dict(), rl_weight_file)

    # final test
    explorer.run_k_episodes(env.case_size['test'], 'test', episode=episode)


if __name__ == '__main__':
    """
    Arguments
    --config: type=str, default='configs/env_modified.config'
    --output_dir: type=str, default='data/output2'
    --weights: type=str
    --resume: default=False, action='store_true'
    --gpu: default=False, action='store_true'
    --debug: default=False, action='store_true'
    """


    carl = ['--output_dir', 'data/model',
            '--config', 'configs/params.config']

    # main(['--policy', 'sarl'])
    main(carl)


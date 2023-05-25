import argparse

import gymnasium as gym
from gymnasium.wrappers import atari_preprocessing
import torch

import config
from utils import preprocess
from evaluate import evaluate_policy
from dqn import DQN, ReplayMemory, optimize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--env', choices=['CartPole-v1', 'ALE/Pong-v5'], default='ALE/Pong-v5')
parser.add_argument('--evaluate_freq', type=int, default=25, help='How often to run evaluation.', nargs='?')
parser.add_argument('--evaluation_episodes', type=int, default=5, help='Number of evaluation episodes.', nargs='?')

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
    'CartPole-v1': config.CartPole,
    'ALE/Pong-v5': config.Pong
}

if __name__ == '__main__':
    args = parser.parse_args()

    # Initialize environment and config.
    env = gym.make(args.env)
    env = gym.wrappers.AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=1, noop_max=30)
    env_config = ENV_CONFIGS[args.env]

    # Initialize deep Q-networks.
    dqn = DQN(env_config=env_config).to(device)
    # TODO: Create and initialize target Q-network.
    # Initialize target Q-network
    target_dqn = DQN(env_config=env_config).to(device)

    # Create replay memory.
    memory = ReplayMemory(env_config['memory_size'])

    # Initialize optimizer used for training the DQN. We use Adam rather than RMSProp.
    optimizer = torch.optim.Adam(dqn.parameters(), lr=env_config['lr'])

    # Keep track of best evaluation mean return achieved so far.
    best_mean_return = -float("Inf")

    for episode in range(env_config['n_episodes']):
        terminated = False
        obs, info = env.reset()

        obs = preprocess(obs, env=args.env).unsqueeze(0)
        #print("1")
        obs_stack = torch.cat(env_config['obs_stack_size'] * [obs]).unsqueeze(0).to(device)
        #print("obs_stack shapebefore")
        #print(obs_stack.shape)
        k = 0
        while not terminated:
            print(k)
            k += 1
            # TODO: Get action from DQN.
            #print("obs_stack shape")
            #print("WHYYYYYYYYY")
            #print(obs_stack.shape)

            #print("2")
            action = dqn.act(obs_stack)
            # Act in the true environment.
            next_obs, reward, terminated, truncated, info = env.step(action.item())

            # Preprocess incoming observation.
            if not terminated:
                obs = preprocess(obs, env=args.env).unsqueeze(0)
            
            # TODO: Add the transition to the replay memory. Remember to convert
            #       everything to PyTorch tensors!

            obs = torch.as_tensor(obs)
            action = torch.as_tensor(action)
            next_obs = torch.as_tensor(next_obs)
            reward = torch.as_tensor(reward)

            # resize the tensors
            obs = obs.view(1,84,84)
            next_obs = next_obs.view(1,84,84)
            reward = reward.view(1)
            action = action.view(1)

            next_obs_stack = torch.cat((obs_stack[:, 1:, ...], obs.unsqueeze(0)), dim=1).to(device)

            memory.push(obs_stack, action, next_obs_stack, reward)

            obs = next_obs
            obs_stack = next_obs_stack
            # TODO: Run DQN.optimize() every env_config["train_frequency"] steps.

            if episode % env_config["train_frequency"] == 0:
                #print("3")
                optimize(dqn, target_dqn, memory, optimizer)

            # TODO: Update the target network every env_config["target_update_frequency"] steps.
            if episode % env_config["target_update_frequency"] == 0:
                #print("4")
                target_dqn = dqn

        # Evaluate the current agent.
        if episode % args.evaluate_freq == 0:
            #print("5")
            mean_return = evaluate_policy(dqn, env, env_config, args, n_episodes=args.evaluation_episodes)
            print(f'Episode {episode+1}/{env_config["n_episodes"]}: {mean_return}')

            # Save current agent if it has the best performance so far.
            if mean_return >= best_mean_return:
                best_mean_return = mean_return

                print('Best performance so far! Saving model.')
                torch.save(dqn, f'models/{args.env}_best.pt')
        
    # Close environment after training is completed.
    env.close()

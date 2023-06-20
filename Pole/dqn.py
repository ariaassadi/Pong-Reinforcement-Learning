import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def push(self, obs, action, next_obs, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = (obs, action, next_obs, reward)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Samples batch_size transitions from the replay memory and returns a tuple
            (obs, action, next_obs, reward)
        """
        sample = random.sample(self.memory, batch_size)
        return tuple(zip(*sample))


class DQN(nn.Module):
    def __init__(self, env_config):
        super(DQN, self).__init__()

        # Save hyperparameters needed in the DQN class.
        self.batch_size = env_config["batch_size"]
        self.gamma = env_config["gamma"]
        self.eps_start = env_config["eps_start"]
        self.eps_end = env_config["eps_end"]
        self.anneal_length = env_config["anneal_length"]
        self.n_actions = env_config["n_actions"]

        self.fc1 = nn.Linear(4, 256)
        self.fc2 = nn.Linear(256, self.n_actions)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        self.n_steps = 0

    def forward(self, x):
        """Runs the forward pass of the NN depending on architecture."""
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def act(self, observation, exploit=False):
        """Selects an action with an epsilon-greedy exploration strategy."""
        
        self.n_steps = self.n_steps + 1
        greedy_rand = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.n_steps / self.anneal_length) # Calculates the epsilon threshold

        if greedy_rand > eps_threshold or exploit: # If the random number is greater than the threshold, exploit and choose the best action using the DQN
            with torch.no_grad():
                q_values = self.forward(observation.to(device))   # Calculate Q-values
                # q_values = q_values.view(1, 2) # Ugly hack to make it work, for some reason shape would change randomly
                _, action = q_values.max(dim=1)  # Get the indices of the maximum Q-values
                action = action.view(1, 1)  # Reshape the action tensor
                
                #action = self.forward(observation).max(1)[1].view(1, 1)
        else: # Otherwise, explore and choose a random action
            action = torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long)

        return action

def optimize(dqn, target_dqn, memory, optimizer):
    """This function samples a batch from the replay buffer and optimizes the Q-network."""
    # If we don't have enough transitions stored yet, we don't train.
    if len(memory) < dqn.batch_size:
        return
    
    batch = memory.sample(dqn.batch_size)
    obs = torch.cat(batch[0]).to(device)
    action = torch.cat(batch[1]).to(device)
    next_obs = torch.cat(batch[2]).to(device)
    reward = torch.cat([s for s in batch[3] if s is not None]).to(device) # Handle the case where next_state is None
      
    q_values = dqn.forward(obs).gather(1, action)
    
    # Compute the Q-value targets for non-terminal transitions
    q_value_targets = torch.zeros(dqn.batch_size, device=device)

    for i in range(dqn.batch_size):
        # If non-terminal state
        if next_obs[i].sum() != 0:
            q_value_targets[i] = reward[i] + dqn.gamma * target_dqn.forward(next_obs[i]).max(0)[0]
        # If terminal state
        else:
            q_value_targets[i] = reward[i]
    
    # Compute loss
    loss = F.mse_loss(q_values.squeeze(), q_value_targets)

    # Perform gradient descent.
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    return loss.item()

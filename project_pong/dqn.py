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

        # check that obs and next_obs has shape (1,4,84,84)
        if obs.shape != (1,4,84,84):
            raise ValueError('obs_stack has wrong shape')
        if next_obs.shape != (1,4,84,84):
            raise ValueError('next_obs_stack has wrong shape')



        obs = obs.to(device)
        action = action.to(device)
        next_obs = next_obs.to(device)
        reward = reward.to(device)

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

        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, self.n_actions)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        """Runs the forward pass of the NN depending on architecture."""

        # check if tensor is 4D
        # if 5D remove dimension
        #print(x.shape)
        if x.shape == (32, 1, 4, 84, 84):
            x = x.squeeze(1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

    def act(self, observation, exploit=False):
        """Selects an action with an epsilon-greedy exploration strategy."""
        # TODO: Implement action selection using the Deep Q-network. This function
        #       takes an observation tensor and should return a tensor of actions.
        #       For example, if the state dimension is 4 and the batch size is 32,
        #       the input would be a [32, 4] tensor and the output a [32, 1] tensor.
        # TODO: Implement epsilon-greedy exploration.
        
        greedy_rand = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.n_actions / self.anneal_length) # Calculates the epsilon threshold
        #eps_threshold = 0
        #print("2.1")
        if greedy_rand > eps_threshold or exploit: # If the random number is greater than the threshold, exploit and choose the best action using the DQN
            with torch.no_grad():
                #print("2.2")
                #print("act")
                #print(observation.shape)
                q_values = self.forward(observation)  # Calculate Q-values
                q_values = q_values.view(1, 2) # Ugly hack to make it work, for some reason shape would change randomly
                _, action = q_values.max(dim=1)  # Get the indices of the maximum Q-values
                action = action.view(1, 1)  # Reshape the action tensor
                
                #action = self.forward(observation).max(1)[1].view(1, 1)
        else: # Otherwise, explore and choose a random action
            #print("act random")
            action = torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long)

        return action

def optimize(dqn, target_dqn, memory, optimizer):
    """This function samples a batch from the replay buffer and optimizes the Q-network."""
    # If we don't have enough transitions stored yet, we don't train.
    if len(memory) < dqn.batch_size:
        #print("3.1")
        return

    # TODO: Sample a batch from the replay memory and concatenate so that there are
    #       four tensors in total: observations, actions, next observations and rewards.
    #       Remember to move them to GPU if it is available, e.g., by using Tensor.to(device).
    #       Note that special care is needed for terminal transitions!
    
    batch = memory.sample(dqn.batch_size)
    obs = torch.stack(batch[0]).to(device)
    action = torch.stack(batch[1]).to(device)
    next_obs = torch.stack(batch[2]).to(device)
    reward = torch.stack(batch[3]).to(device)
    
    # TODO: Compute the current estimates of the Q-values for each state-action
    #       pair (s,a). Here, torch.gather() is useful for selecting the Q-values
    #       corresponding to the chosen actions.   
    #print("optimize") 
    #print("3.2")
    q_values = dqn.forward(obs).gather(1, action)
    
    # TODO: Compute the Q-value targets. Only do this for non-terminal transitions!

    # Compute the Q-value targets for non-terminal transitions
    q_value_targets = torch.zeros(dqn.batch_size, device=device).unsqueeze(1)

    for i in range(dqn.batch_size):
        # If non-terminal state
        if next_obs[i].sum() != 0:
            #print("3.3")
            #print("optimize batch")
            q_value_targets[i] = reward[i] + dqn.gamma * target_dqn.forward(next_obs[i]).squeeze(0).max(0)[0]
        # If terminal state
        else:
            q_value_targets[i] = reward[i]
    
    # Compute loss
    loss = F.mse_loss(q_values.squeeze(), q_value_targets)

    # Perform gradient descent.
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()
    #print("3.4")

    return loss.item()

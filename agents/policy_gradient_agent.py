import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.distributions import Categorical

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.distributions import Categorical

class PolicyGradientNetwork(nn.Module):
    """
    A neural network for policy gradient methods.

    Args:
        input_size (int): The number of input features.
        hidden_sizes (list of int): The list of sizes for hidden layers.
        output_size (int): The number of output features.

    Attributes:
        input_size (int): The number of input features.
        output_size (int): The number of output features.
        hidden_layers (ModuleList): List containing hidden layer modules.
        output_layer (Linear): Output layer module.

    Example usage:
        input_size, hidden_sizes, output_size = 8, [10, 10], 4
        network = PolicyGradientNetwork(input_size, hidden_sizes, output_size)
    """
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = nn.ModuleList()
        
        # Add input layer
        self.hidden_layers.append(nn.Linear(input_size, hidden_sizes[0]))
        
        # Add hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        
        # Add output layer
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, state):
        """
        Forward pass through the network.

        Args:
            state (tensor): Input state tensor.

        Returns:
            tensor: Output action probabilities.
        """
        # Forward pass through each hidden layer
        for layer in self.hidden_layers:
            state = F.relu(layer(state))
        
        # Output layer
        return F.softmax(self.output_layer(state), dim=-1)
    

class PolicyGradientAgent():
    """
    An agent implementing policy gradient methods.

    Args:
        network (PolicyGradientNetwork): The policy gradient network.
        lr (float): Learning rate for optimizer (default: 1e-3).
        epochs (int): Number of epochs for training (default: 500).

    Attributes:
        lr (float): Learning rate for optimizer.
        epochs (int): Number of epochs for training.
        network (PolicyGradientNetwork): The policy gradient network.
        optimizer (Optimizer): Optimizer for updating network parameters.
        lr_scheduler (CosineAnnealingLR): Learning rate scheduler.

    Example usage:
        input_size, hidden_sizes, output_size = 8, [10, 10], 4
        network = PolicyGradientNetwork(input_size, hidden_sizes, output_size)
        agent = PolicyGradientAgent(network)
    """
    def __init__(self, network, lr=1e-3, epochs=500):
        self.lr = lr
        self.epochs = epochs
        self.network = network
        self.optimizer = optim.SGD(self.network.parameters(), lr=1e-3)
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, self.epochs, eta_min=1e-6)

    def forward(self, state):
        """
        Forward pass through the network.

        Args:
            state (tensor): Input state tensor.

        Returns:
            tensor: Output action probabilities.
        """
        return self.network(state)

    def learn(self, log_probs, rewards):
        """
        Perform a learning step.

        Args:
            log_probs (tensor): Log probabilities of selected actions.
            rewards (tensor): Rewards associated with selected actions.
        """
        loss = (-log_probs * rewards).sum() # You don't need to revise this to pass simple baseline (but you can)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        
    def sample(self, state, greedy=False):
        """
        Sample an action from the network's output distribution.

        Args:
            state (array_like): Input state.
            greedy (bool): Whether to sample greedily (default: False).

        Returns:
            tuple: Action and corresponding log probability.
        """
        action_prob = self.network(torch.FloatTensor(np.array(state)))
        action_dist = Categorical(action_prob)
        if not greedy:
            action = action_dist.sample()
        else:
            action = action_prob.argmax()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob

    def save(self, path:str): # You should not revise this
        """
        Save the agent's network and optimizer parameters to a file.

        Args:
            path (str): Path to save the parameters.
        """
        agent_dict = {
            "network" : self.network.state_dict(),
            "optimizer" : self.optimizer.state_dict()
        }
        torch.save(agent_dict, path)

    def load(self, path:str): # You should not revise this
        """
        Load the agent's network and optimizer parameters from a file.

        Args:
            path (str): Path to load the parameters.
        """
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint["network"])
        #如果要儲存過程或是中斷訓練後想繼續可以用喔
        self.optimizer.load_state_dict(checkpoint["optimizer"])



import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam

class PolicyNet(nn.Module):
    def __init__(self, input, output, hidden_sizes):
        super().__init__()
        layers = []
        sizes = [input] + hidden_sizes

        # hidden layers
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))

        self.hidden_layers = nn.ModuleList(layers)

        # policy head
        self.output_layer = nn.Linear(hidden_sizes[-1], output)

    def forward(self, obs):
        x = obs
        for layer in self.hidden_layers:
            x = F.relu(layer(x))

        logits = self.output_layer(x)
        probs = F.softmax(logits, dim=-1)
        return probs
    

class CentralizedIrrigationModel:
    def __init__(self, centralized_dim, lr, gamma):
        # Centralized Dim will be the input from the surrogate model 
        self.model = PolicyNet(input = centralized_dim, 
                               output = 2, 
                               hidden_sizes=[64,64]) # Output is 2 because we want water or not
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma

    def select_action(self, state):

        """
        Returns:
            The selected action (as an int)
            The log probability of the selected action (as a tensor)
        
        """
        state = torch.as_tensor(state)
        probs = self.model(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob

    def compute_returns(self, rewards):
        
        G = 0.0
        returns = []
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.append(G)

        returns.reverse()

        return returns

    def update_model(self, log_probs, rewards):
        returns = torch.tensor(self.compute_returns(rewards))

        loss = 0
        for log_p, G in zip(log_probs, returns):
            loss += -log_p * G

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class LocalIrrigationModel:
    def __init__(self, centralized_dim, lr, gamma):
        # Local Dim will be the input from the surrogate model 
        self.model = PolicyNet(input = centralized_dim, 
                               output = 5, 
                               hidden_sizes=[64,64]) # Output is 2 because we want water or not
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma

    def select_action(self, state):

        """
        Returns:
            The selected action (as an int)
            The log probability of the selected action (as a tensor)
        
        """
        state = torch.as_tensor(state)
        probs = self.model(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob

    def compute_returns(self, rewards):
        
        G = 0.0
        returns = []
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.append(G)

        returns.reverse()

        return returns

    def update_model(self, log_probs, rewards):
        returns = torch.tensor(self.compute_returns(rewards))

        loss = 0
        for log_p, G in zip(log_probs, returns):
            loss += -log_p * G

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



        

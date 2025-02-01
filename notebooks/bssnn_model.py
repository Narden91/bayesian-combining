import torch
import torch.nn as nn


class BSSNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BSSNN, self).__init__()

        # Joint pathway: P(y, X)
        self.fc1_joint = nn.Linear(input_size, hidden_size)
        self.relu_joint = nn.ReLU()
        self.fc2_joint = nn.Linear(hidden_size, 1)

        # Marginal pathway: P(X)
        self.fc1_marginal = nn.Linear(input_size, hidden_size)
        self.relu_marginal = nn.ReLU()
        self.fc2_marginal = nn.Linear(hidden_size, 1)

        # Sigmoid activation for probability
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Joint probability computation
        joint = self.relu_joint(self.fc1_joint(x))
        joint = self.fc2_joint(joint)  # Output: logit for P(y, X)

        # Marginal probability computation
        marginal = self.relu_marginal(self.fc1_marginal(x))
        marginal = self.fc2_marginal(marginal)  # Output: logit for P(X)

        # Bayesian division: P(y|X) = P(y, X) / P(X)
        conditional = joint - marginal  # Log-space division
        return self.sigmoid(conditional)  # Probability score
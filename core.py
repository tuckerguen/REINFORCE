import typing as tp
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np


def create_torch_nn(obs_dim: int,
                    act_dim: int,
                    hidden_shapes: tp.Tuple[int] = (64, 64),
                    activation=nn.Tanh,
                    output_activation=nn.Identity):
    """
    Creates a pytorch NN w/ given parameters
    :param obs_dim: Input shape (in RL, dimension of observation spcae)
    :param act_dim: Output shape (in RL, dimension of action space
    :param hidden_shapes: tuple of hidden layer sizes (ex: (64, 64))
    :param activation: Activation function
    :return: torch nn.Sequential NN
    """
    layers = []
    # Concat shapes to single list
    nn_shape = [obs_dim] + list(hidden_shapes) + [act_dim]
    # Loop over layers up to output layer
    for i, in_size in enumerate(nn_shape[:-1]):
        # Select correct activation function
        act = activation if i < len(nn_shape) - 2 else output_activation
        # Create linear layer of correct size and activation
        layers.append(nn.Linear(in_size, nn_shape[i + 1]))
    # Create sequential NN
    return nn.Sequential(*layers)


def discount_return(ret, gamma):
    """
    Computes total discounted return over a trajectory
    :param ret: list of returns for each time step t
    :param gamma: discount factor
    :return: list of discounted return at each time step t
    """
    n = len(ret)
    # Create discount factors ([y**0, y**1, ..., y**n])
    gammas = gamma ** np.arange(n)
    # Total discounted return at each time step t',
    # all returns from t'->T * discount factor from y**0->y**n-t'
    return [sum(ret[tp:] * gammas[:n - tp]) for tp in range(n)]


class MLPActor(nn.Module):
    """
    Actor with policy pi given by an MLP
    Strictly discrete action spaces
    """

    def __init__(self, obs_dim, act_dim, hidden_shapes, activation=nn.Tanh):
        super(MLPActor, self).__init__()
        self.pi = create_torch_nn(obs_dim, act_dim, hidden_shapes, activation, activation)

    def sample_action(self, obs):
        """
        Samples an action from the current policy given
        an observation
        :param obs:
        :return:
        """
        # Get categorical distribution of actions over observations
        pi = self.distribution(obs)
        # Sample an action from the distribution
        a = pi.sample()
        return a.numpy()

    def distribution(self, obs):
        return Categorical(logits=self.pi(obs))

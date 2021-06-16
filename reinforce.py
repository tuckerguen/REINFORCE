import gym
import time
import torch
import numpy as np
from torch.optim import Adam
from core import MLPActor, discount_return
from tqdm import tqdm
from algos.util.logger import Logger
import matplotlib.pyplot as plt
import queue


def reinforce(env_fn, hidden_shape=(64,64), seed=0, steps_per_epoch=4000, epochs=50, pi_lr=3e-4,
              gamma=0.99, max_ep_len=1000, record_frac=0.8, exp_name='reinforce',
              video_dir='videos', data_dir='data'):
    """
    REINFORCE deep RL algorithm
    :param env_fn: function that creates copy of an OpenAI gym environment
    :param hidden_shape: tuple specifying sizes of neural net hidden layers
    :param seed: seed value for RNG
    :param steps_per_epoch: number of environment interactions per epoch
    :param epochs: number of policy updates to perform
    :param pi_lr: learning rate of pi
    :param gamma: discount factor for discounted reward
    :param max_ep_len: maximum length of a trajectory / episode
    :param record_frac: fraction of final epoch to record video of. If recording not
                        desired, use record_frac = 0.
                        Be aware that recording requires rendering the trajectories
                        during the final epoch and will significantly slow the
                        runtime of that final fraction of the epoch.
    :param exp_name: Video file prefix
    :param video_dir: directory to save video to
    :param data_dir: directory to data to
    """
    # Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create env
    env = env_fn()

    # Init logger
    logger = Logger(env, exp_name, record_frac, video_dir=video_dir, data_dir=data_dir)
    params = [['env', env.unwrapped.spec.id], ['hidden', hidden_shape], ['seed', seed],
              ['steps per epoch', steps_per_epoch], ['epochs', epochs], ['policy learning rate', pi_lr],
              ['gamma', gamma], ['max episode length', max_ep_len]]
    logger.print_table(params, ['Param', 'Value'], label='EXPERIMENT PARAMETERS')

    time.sleep(3)

    # Observation space dimension
    obs_dim = env.observation_space.shape[0]
    # Action space dimension
    act_dim = env.action_space.n

    # Create reinforce actor
    actor = MLPActor(obs_dim, act_dim, hidden_shape)

    # Count variables
    n_vars = sum([np.prod(p.shape) for p in actor.parameters()])
    print(f'Number of parameters: pi: {n_vars}')

    # Set up policy optimizer
    pi_optimizer = Adam(actor.pi.parameters(), lr=pi_lr)

    # Initialize environment and tracking variables
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    avg_ep_rets = []
    avg_ep_len = []

    # Loop over epochs (training batches)
    for epoch in range(epochs):
        # Lists to store epoch data in
        epoch_disc_rets, epoch_rets = [], []
        # Store data per episode
        ep_obs, ep_acts = np.zeros((steps_per_epoch, obs_dim)), np.zeros(steps_per_epoch)
        baselines, rewards = [], []

        act_q = queue.Queue()
        act_q.put(0)
        act_q.put(0)
        # Generate N trajectories with a maximum of steps_per_epoch total steps
        # across all generated trajectories
        for t in tqdm(range(steps_per_epoch)):
            # Take an environment step: sample action from pi given observation
            tensor_o = torch.as_tensor(o, dtype=torch.float32)
            a = actor.sample_action(tensor_o)
            # Track observations, actions, episode length
            ep_obs[t] = tensor_o
            ep_acts[t] = a
            ep_len += 1

            if epoch == epochs-1 and t > record_frac * steps_per_epoch:
                logger.record_frame()

            # Update environment
            act_q.put(a)
            next_o, r, d, _ = env.step(a)

            # Save reward
            rewards.append(r)
            # Update observation
            o = next_o

            # Check for termination condition
            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t == steps_per_epoch - 1
            if terminal or epoch_ended:
                if epoch_ended and not terminal:
                    print('Warning: trajectory cut off by epoch at {ep_len} steps.')

                # End of i_th trajectory, save trajectory data
                epoch_rets.extend(rewards)
                avg_ep_rets.append(np.sum(rewards))
                avg_ep_len.append(len(rewards))
                epoch_disc_rets.extend(discount_return(rewards, gamma=gamma))

                # Reset env and episode tracking vars
                o, ep_ret, ep_len, rewards = env.reset(), 0, 0, []

        # Metrics computed over all trajectories
        # Compute baseline as the average of all trajectory rewards
        epoch_bls = torch.ones(steps_per_epoch) * (np.sum(epoch_rets) / len(epoch_rets))
        # Compute log probability of action given observation at each timestep
        pi = actor.distribution(torch.as_tensor(ep_obs, dtype=torch.float32))
        epoch_logps = pi.log_prob(torch.as_tensor(ep_acts, dtype=torch.float32))

        # Update model
        # Compute loss
        loss = -(epoch_logps * np.subtract(epoch_disc_rets, epoch_bls)).mean()

        # Do gradient step
        pi_optimizer.zero_grad()
        loss.backward()
        pi_optimizer.step()

        # Log info about epoch
        epoch_log = [['epoch', epoch], ['avg return', np.mean(avg_ep_rets)],
                     ['avg episode length', np.mean(avg_ep_len)],
                     ['total env interacts', (epoch + 1) * steps_per_epoch],
                     ['time', time.time() - start_time]]
        logger.print_table(epoch_log, ['Metric', 'Value'])

    env.close()
    logger.close_rec()
    metrics = [['avg return', np.mean(avg_ep_rets)],
               ['max return', np.max(avg_ep_rets)],
               ['min return', np.min(avg_ep_rets)],
               ['avg episode length', np.mean(avg_ep_len)],
               ['time', time.time() - start_time]]
    logger.save_run(params, metrics)
    plt.plot(avg_ep_len)
    plt.title('Average reward over epochs')
    plt.show()

    # TODO: Save the learned policy


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--pi_lr', type=float, default=3e-4)
    parser.add_argument('--record_frac', type=float, default=0.8)
    parser.add_argument('--video_dir', type=str, default='videos')
    parser.add_argument('--data_dir', type=str, default='data')
    args = parser.parse_args()

    reinforce((lambda: gym.make(args.env)),
              hidden_shape=(args.hid,) * args.l,
              seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
              pi_lr=args.pi_lr, gamma=args.gamma, record_frac=args.record_frac,
              video_dir=args.video_dir, data_dir=args.data_dir)

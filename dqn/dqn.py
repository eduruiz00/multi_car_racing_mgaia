import argparse
import gym
import torch
from torch import nn
import torch.optim as optim
import numpy as np
import copy
from simhash import Simhash

from dqn_model import DQN
from selection_action import SelectionMethod
from replay_buffer import ReplayBuffer, BufferElement
from helper import smooth, LearningCurvePlot

parser = argparse.ArgumentParser()
parser.add_argument("--experience_replay", default=False, action="store_true", help="Whether to use experience replay")
parser.add_argument("--target_network", default=False, action="store_true", help="Whether to use a target network")
parser.add_argument("--save_model", default=False, action="store_true", help="Whether to save model after training")
args = parser.parse_args()

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dqn_algorithm(n_episodes: int, steps: int, n_sampling: int, env, target_update: int, experience_replay: bool,
                  target_network: bool, selection_action: SelectionMethod, optim_name: str = "Adam", lr: float = 1e-4,
                  model: nn.Module = DQN):
    """
    Pseudocode: https://arxiv.org/pdf/1312.5602v1.pdf
    :return:
    """
    if model == DQN:
        dqn = DQN(num_channels=env.observation_space.shape[0], n_actions=env.action_space.n).to(device)
        target_dqn = DQN(num_channels=env.observation_space.shape[0], n_actions=env.action_space.n).to(device)
    else:
        dqn = model.to(device)
        target_dqn = copy.deepcopy(dqn)  # Different reference

    target_dqn.load_state_dict(dqn.state_dict())
    optimizer = getattr(optim, optim_name)(dqn.parameters(), lr=lr)

    gamma = 0.99

    replay_buffer = ReplayBuffer(capacity=1000)
    episode_rewards = []
    for episode in range(n_episodes):
        s, _ = env.reset()
        episode_reward = 0
        states_episode = []
        for step in range(steps):
            # Select action
            a = selection_action.run(step, s, dqn, env)

            # Perform the action and observe the reward
            s_next, r, done, _, _ = env.step(a)
            episode_reward += r

            if experience_replay:
                buffer_element = BufferElement(s, a, r, s_next, done)
                replay_buffer.add(buffer_element)
                if len(replay_buffer.buffer) > n_sampling:
                    batch_states, batch_actions, batch_rewards, batch_next_state, batch_done = replay_buffer.sample(
                        n_sampling)

                    q_values = dqn(batch_states)
                    output = q_values.gather(1, batch_actions)

                    if selection_action.novelty:
                        # hashing the states
                        batch_states_cpu = copy.deepcopy(batch_states).to('cpu').numpy()
                        h = [Simhash(" ".join(str(round(x, 1)) for x in np.array(s))).value for s in batch_states_cpu]
                        for i in range(len(h)):
                            if h[i] in selection_action.counts.keys():
                                batch_rewards[i] += selection_action.rewards[h[i]]

                    if target_network:
                        next_q_values = target_dqn(batch_next_state).max(1)[0].detach()
                    else:
                        next_q_values = dqn(batch_next_state).max(1)[0].detach()

                    # If done then target = batch_rewards
                    target = batch_rewards + gamma * next_q_values * (1 - batch_done)

                    loss = nn.functional.mse_loss(output, target.unsqueeze(1))

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            else:  # No experience replay, page 71 book RL
                output = dqn(torch.tensor(s).to(device))[a]
                if target_network:
                    next_q_values = target_dqn(torch.tensor(s_next).to(device)).max().detach()
                else:
                    next_q_values = dqn(torch.tensor(s_next).to(device)).max().detach()

                if selection_action.novelty:
                    # hashing the states
                    state_str = " ".join(str(round(x, 1)) for x in s)
                    hash_state = Simhash(state_str).value
                    if hash_state in selection_action.counts.keys():
                        r += selection_action.rewards[hash_state]
                # If done then target = batch_rewards
                target = r + gamma * next_q_values * (1 - done)

                loss = torch.nn.functional.mse_loss(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            states_episode.append(s)
            s = s_next

            if step % target_update == 0:
                target_dqn.load_state_dict(dqn.state_dict())

            if done:
                break

        episode_rewards.append(episode_reward)
        selection_action.explore(states_episode)  # Update novelty

        if episode % 10 == 0:
            print(f"Episode {episode} got a mean reward last 10 episodes of {np.mean(episode_rewards[-10:])}")

    env.close()
    # return episode_rewards
    return episode_rewards, dqn


def run_experiment(selection_method: SelectionMethod, experience_replay=args.experience_replay,
                   target_network=args.target_network, plot=True, n_episodes=500, render=False):
    if render:
        env = gym.make('CartPole-v1', render_mode="human")
    else:
        env = gym.make('CartPole-v1')

    selection_method.num_episodes = n_episodes

    rewards, dqn_model = dqn_algorithm(n_episodes=n_episodes, steps=500, n_sampling=32, env=env, target_update=75,
                                 experience_replay=experience_replay, target_network=target_network,
                                 selection_action=selection_method)

    if plot:
        graph = LearningCurvePlot(title=selection_method.method)
        graph.add_curve(smooth([rewards], 101), label=r'Smooth line')
        graph.save(f'{selection_method.method}.png')

    return rewards, dqn_model


if __name__ == '__main__':
    selection_method = SelectionMethod(method="boltzmann", temp=0.1)
    _, dqn = run_experiment(selection_method, n_episodes=1000)
    if args.save_model:
        torch.save(dqn, 'dqn.pt')
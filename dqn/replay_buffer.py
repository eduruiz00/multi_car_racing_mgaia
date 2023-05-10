import numpy as np
import torch

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BufferElement:
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done


class ReplayBuffer:
    def __init__(self, capacity: int = int(10e6)):
        self.buffer: list[BufferElement] = []
        self.capacity: int = capacity

    def add(self, element: BufferElement):
        if len(self.buffer) >= self.capacity:
            self.buffer = self.buffer[1:]  # We remove the first element
        self.buffer.append(element)
        # print(self.buffer[-1].state)

    def sample(self, n_sampling: int):
        if n_sampling > len(self.buffer):  # Not enough samples inside the buffer
            return None

        indices = np.random.choice(len(self.buffer), size=n_sampling, replace=False)
        sampled_elements = [self.buffer[idx] for idx in indices]

        batch_states = torch.tensor(np.stack([x.state for x in sampled_elements]), dtype=torch.float32).to(device)
        batch_actions = torch.tensor([x.action for x in sampled_elements], dtype=torch.int64).to(device).unsqueeze(1)
        batch_rewards = torch.tensor([x.reward for x in sampled_elements], dtype=torch.float32).to(device)
        batch_next_state = torch.tensor(np.stack([x.next_state for x in sampled_elements]), dtype=torch.float32).to(device)
        batch_done = torch.tensor([x.done for x in sampled_elements], dtype=torch.float32).to(device)
        return batch_states, batch_actions, batch_rewards, batch_next_state, batch_done

"""
https://github.com/vermouth1992/deep-learning-playground/blob/master/pytorch/replay_buffer.py
"""

from collections import deque
import random
import numpy as np

from td3.utils.logs.logger import WithLogger


@WithLogger()
class ReplayBuffer(object):
    """
    A simple FIFO experience replay buffer for TD3 agents.
    """

    def __init__(self, buffer_size: int, random_seed: int = 123):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, s, a, r, t, s2) -> None:
        """
        Add a new experience to the buffer. If the buffer is full, the oldest experience will be removed.

        Args:
            s: State at time t
            a: Action taken at time t
            r: Reward received after taking action a in state s
            t: Done flag indicating whether the episode has ended after taking action a in state s
            s2: State at time t+1 after taking action a in state s

        """
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        """Return the current size of internal memory."""
        return self.count

    def sample_batch(self, batch_size: int):
        """Return a random sample of experiences from the buffer."""
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

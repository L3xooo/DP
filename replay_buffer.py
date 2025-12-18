"""
Source: https://github.com/vermouth1992/deep-learning-playground/blob/master/tensorflow/ddpg/replay_buffer.py
"""
from collections import deque
import random
import numpy as np

from utils.logger import WithLogger, log_values_with_color


@WithLogger()
class ReplayBuffer(object):
    """
    A ReplayBuffer is a data structure used in reinforcement learning to store
    the agent's experiences during training. The experiences are stored as
    tuples (state, action, reward, done, next_state) and can be sampled later
    to train the agent.

    The buffer allows the agent to learn from a wide variety of experiences by
    sampling random batches from the buffer (which helps break temporal correlations
    in the training data and improves learning stability).
    """

    def __init__(self, buffer_size, random_seed=123):
        """
        Initializes the replay buffer with a maximum size, and an empty deque to store experiences.
        The buffer will store the most recent experiences, with the oldest ones being removed
        once the buffer is full.

        Args:
            buffer_size (int): The maximum size of the buffer. When the buffer reaches this size,
                                older experiences are removed to make space for new ones.
            random_seed (int): The seed for random operations, ensuring reproducibility of results.
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, s, a, r, t, s2):

        """
        Adds a new experience (state, action, reward, done, next_state) to the buffer.
        If the buffer is full, the oldest experience is removed to make space for the new one.

        Args:
            s (np.array): The current state of the environment (observation).
            a (np.array): The action taken by the agent in state `s`.
            r (float): The reward received after taking action `a` in state `s`.
            t (bool): Whether the episode has ended (True if done, False otherwise).
            s2 (np.array): The next state observed after taking action `a`.
        """
        formatted_state = [round(value, 2) for value in s]
        formatted_new_state = [round(value, 2) for value in s2]

        # Uloženie do dočasných premenných
        formatted_state_str = ', '.join(map(str, formatted_state))
        formatted_new_state_str = ', '.join(map(str, formatted_new_state))

        # Logovanie sformátovaných hodnôt
        # log_values_with_color(self.logger, {"State": formatted_state_str})
        # log_values_with_color(self.logger, {"New_state": formatted_new_state_str})
        # log_values_with_color(self.logger, {"Action": a})
        #
        # self.logger.info("Adding new reward to replay buffer: %.3f", r)
        experience = (s, a, r, t, s2)
        # self.logger.info(f"Adding experience to buffer: {experience}")
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        """
        Returns the current size of the replay buffer (number of experiences stored).

        Returns:
            int: The number of experiences currently stored in the buffer.
        """
        return self.count

    def sample_batch(self, batch_size):
        """
        Samples a random batch of experiences from the buffer for training.
        If the buffer contains fewer experiences than the requested batch size,
        it will return all the experiences in the buffer.

        Args:
            batch_size (int): The number of experiences to sample from the buffer.

        Returns:
            tuple: A tuple containing numpy arrays of states, actions, rewards, done flags,
                   and next states for the sampled batch:
                   (s_batch, a_batch, r_batch, t_batch, s2_batch)
        """
        # If the buffer contains fewer experiences than the batch size, sample all of them
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])  # States
        a_batch = np.array([_[1] for _ in batch])  # Actions
        r_batch = np.array([_[2] for _ in batch])  # Rewards
        t_batch = np.array([_[3] for _ in batch])  # Done flags (True if episode ended)
        s2_batch = np.array([_[4] for _ in batch])  # Next states

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        """
        Clears the buffer by removing all stored experiences and resetting the count.

        This can be useful when the agent needs to start fresh, or when you want to clear the
        buffer at the end of training or after a certain condition is met.
        """
        self.buffer.clear()
        self.count = 0

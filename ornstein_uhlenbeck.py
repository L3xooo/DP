"""
Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
"""

import numpy as np

class OrnsteinUhlenbeckActionNoise:
    """
    Implementation of Ornstein-Uhlenbeck noise for reinforcement learning (RL) algorithms,
    such as DDPG (Deep Deterministic Policy Gradient) or TD3 (Twin Delayed Deep Deterministic
    Policy Gradient). This noise is used to ensure **exploration** during training of an agent
    learning optimal actions for portfolio management or other tasks.

    The Ornstein-Uhlenbeck process generates correlated noise, which means the noise is
    not independent like regular Gaussian noise. This helps to **smooth** the actions
    and prevents the agent from getting stuck in **local minima** during training.

    The process is **mean-reverting**, and the noise gradually returns to a target mean value (`mu`).
    This is controlled by the `theta` parameter, which determines how fast the noise reverts to the mean.
    The noise is also scaled by a random component.

    Parameters:
        mu (numpy.array): The target mean value to which the noise will revert.
                          For agents, this is often the **mean value** around which the noise stabilizes.
        sigma (float): The spread (standard deviation) of the noise. Larger values result in more randomness
                       in the action taken by the agent.
        theta (float): The rate at which the noise reverts to the target mean (`mu`). Larger values
                       mean the noise returns to the mean more quickly.
        dt (float): Time step (time interval) between updates to the noise. Controls how often the noise changes.
        x0 (numpy.array, optional): Initial value of the noise. If not provided, defaults to zero.

    Methods:
        __call__: Generates new noise based on the previous noise value and the Ornstein-Uhlenbeck
                  process parameters (`theta`, `mu`, `sigma`, and `dt`).
        reset: Resets the noise state to its initial value (`x0`).
        __repr__: Returns a string representation of the object, displaying the current noise parameters
                  (`mu`, `sigma`).
    """

    def __init__(self, mu, sigma=0.2, theta=.15, dt=1e-2, x0=None):
        """
        Initializes the Ornstein-Uhlenbeck noise generator with the given parameters.

        Args:
            mu (numpy.array): The target mean value the noise will revert to.
            sigma (float): The spread (standard deviation) of the noise. Default is 0.2.
            theta (float): The rate at which the noise reverts to `mu`. Default is 0.15.
            dt (float): The time step (interval) for updating the noise. Default is 1e-2.
            x0 (numpy.array, optional): The initial noise value. If not provided, defaults to zero.
        """
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        """
        Generates a new noise value using the Ornstein-Uhlenbeck process, based on the previous noise value.
        The noise is returned to the mean value `mu` over time according to the `theta` parameter,
        with a random component controlled by `sigma`.

        Returns:
            numpy.array: The generated noise value, with the same shape as `mu`.
        """
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x  # Update the previous noise value for the next call
        return x

    def reset(self):
        """
        Resets the state of the noise generator to its initial value (`x0`).
        This can be useful when you want to restart the noise generation process,
        for example, at the start of each episode or training cycle.

        If no initial value `x0` is provided, the noise is reset to zero.
        """
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        """
        Returns a string representation of the Ornstein-Uhlenbeck noise object,
        showing the current values of `mu` and `sigma`.

        Returns:
            str: A string representing the noise generator with its parameters.
        """
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

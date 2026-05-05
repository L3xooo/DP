"""
A portfolio management environment compatible with the OpenAI Gymnasium interface.

Author: Peter Likavec
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

from td3.config.app_config import AppConfig
from td3.replay.replay_buffer import ReplayBuffer
from td3.utils.logs.logger import WithLogger
from td3.utils.prev_curr import PrevCurr

DEFAULT_PORTFOLIO_VALUE = 10000.0


@WithLogger()
class PortfolioEnv(gym.Env):
    """
    A portfolio management environment compatible with the OpenAI Gymnasium interface.

    The agent controls a portfolio of ``num_assets`` assets plus a cash position.
    At each time step it receives a feature observation and outputs a weight vector
    that determines how the total portfolio value is redistributed across assets and
    cash.

    Observation space:
        Box of shape ``(num_assets * feature_dim,)`` — the flattened feature matrix
        for the current time step.

    Action space:
        Box of shape ``(num_assets + 1,)`` in ``[0, 1]`` — target portfolio weights
        where index 0 is cash and indices 1…N are the individual assets.  Weights
        are expected to sum to 1 but this is not enforced internally.
    """

    def __init__(
        self,
        features: np.ndarray,
        prices: np.ndarray = None,
        tickers: list = None,
        app_config: AppConfig = None,
    ):
        super(PortfolioEnv, self).__init__()

        assert features.ndim == 3, "features must be a 3D numpy array (T, N, F)"
        self.features = features.astype(np.float32)
        self.current_step = None
        # From the shape get number of steps, assets and feature dimension
        self.num_steps, self.num_assets, self.feature_dim = self.features.shape

        self.tickers = tickers
        self.seed_value = None
        self.replay_buffer = ReplayBuffer(app_config.replay_buffer_size)
        self.app_config = app_config

        # Coefficients
        self.transaction_coefficient = 0.001
        self.risk_aversion_coefficient = 0.005

        self.features = features.astype(np.float32)
        self.prices = prices.astype(np.float32)

        # Changeable state variables
        self.weights = None
        self.assets_prices = None
        self.shares = None
        self.portfolio_cash = None
        self.portfolio_value = None

        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.num_assets + 1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_assets * self.feature_dim,),
            dtype=np.float32,
        )
        # self.logger.info("State shape: %s", self.features.shape)

    def _get_seed(self, seed=None):
        """Sets the random seed for reproducibility."""
        self.seed_value = seed
        np.random.seed(seed)
        random.seed(seed)

    def _get_features_current(self) -> np.ndarray:
        """Returns the features for the current step as a flattened array."""
        return self.features[self.current_step].flatten()

    def _get_features_next(self) -> np.ndarray:
        """Returns the features for the next step as a flattened array."""
        next_step = self.current_step + 1
        if next_step >= self.num_steps:
            raise IndexError

        # self.logger.info("Current step: %s | next_step: %s", self.current_step, next_step)
        return self.features[next_step].flatten()

    def _get_state(self):
        """Returns the current state as a flattened array of features."""
        return np.concatenate([self._get_features_current().flatten()], axis=0).astype(np.float32)

    def _get_state_next(self):
        # self.logger.info("getting next state")
        """Returns the next state as a flattened array of features."""
        return np.concatenate([self._get_features_next().flatten()], axis=0).astype(np.float32)

    def _calculate_portfolio_value(self, prices: np.ndarray = None) -> float:
        """Calculates the current portfolio value based on cash and shares held."""
        p = self._get_prices() if prices is None else prices

        if p.shape != self.shares.curr.shape:
            raise ValueError(f"prices shape {p.shape} != shares shape {self.shares.curr.shape}")

        return float(self.portfolio_cash.curr + p.dot(self.shares.curr))

    def _get_prices(self, extra_step: int = 0) -> np.ndarray:
        """Returns the asset prices for the current step plus an optional extra step."""
        idx = self.current_step + extra_step
        if idx < 0 or idx >= len(self.prices):
            raise IndexError()
        return self.prices[idx].flatten()

    def _calculate_reward(self):
        """
        Computes the reward for the current step.

        Returns:
            Float reward value calculated as the log return of the portfolio value from the current step to the next step.
        """

        portfolio_value = self._calculate_portfolio_value()
        next_portfolio_value = self._calculate_portfolio_value(self._get_prices(1))
        self.logger.info(f"Portfolio value Current t: {portfolio_value} | Next t+1: {next_portfolio_value}" )
        log_return = np.log((next_portfolio_value + 1e-12) / (portfolio_value + 1e-12))
        #
        # turnover = 0.0
        # if self.weights is not None:
        #     turnover = float(np.sum(np.abs(self.weights.curr - self.weights.prev)))
        # turnover_penalty = 0.0035
        # return log_return - turnover_penalty * turnover
        return log_return

    def reset(self, seed=None, options=None):
        """
        Reset the environment to an initial state and return the initial observation.

        Args:
            seed: Optional random seed for reproducibility.
            options: Optional dictionary of additional reset options (not used in this implementation).
        """

        self._get_seed(seed)
        self.current_step = 0
        w0 = np.zeros(self.num_assets + 1, dtype=np.float32)
        w0[0] = 1.0
        zeros_assets = np.zeros(self.num_assets, dtype=np.float32)

        self.portfolio_value = PrevCurr(prev=float(0), curr=float(DEFAULT_PORTFOLIO_VALUE))
        self.portfolio_cash = PrevCurr(prev=float(0), curr=float(DEFAULT_PORTFOLIO_VALUE))

        # Shares & assets prices everything zero (1 step all in cash)
        self.shares = PrevCurr(prev=zeros_assets.copy(), curr=zeros_assets.copy())
        self.assets_prices = PrevCurr(prev=zeros_assets.copy(), curr=zeros_assets.copy())

        # Weights in each asset & cash
        self.weights = PrevCurr(prev=zeros_assets.copy(), curr=w0.copy())

        return self._get_state()

    def _step_v2(self, action):
        """Executes one time step within the environment based on the given action."""

        # Save the previous values, check if even needed
        self.weights.set_prev_from_curr()
        self.portfolio_value.set_prev_from_curr()
        self.portfolio_cash.set_prev_from_curr()
        self.shares.set_prev_from_curr()
        self.assets_prices.set_prev_from_curr()

        self.logger.info("Prices: %s", self._get_prices())
        try:
            self.logger.info("Next prices: %s", self._get_prices(1))
        except IndexError:
            self.logger.warn("Cannot retrieve the next prices index error")
        # Calculate the current portfolio value with previous cash and shares held
        self.portfolio_value.set_curr(
            self.portfolio_cash.prev + self._get_prices().dot(self.shares.prev)
        )
        self.logger.debug("PortfolioValue Prev: %s | Curr: %s ", self.portfolio_value.prev, self.portfolio_value.curr)

        # Do the changes in portfolio based on the new weights
        self.weights.set_curr(action)
        self.logger.debug("Weights Prev: %s | Curr: %s", self.weights.prev, self.weights.curr)

        self.portfolio_cash.set_curr(self.portfolio_value.curr * self.weights.curr[0])
        self.logger.debug("Cash Prev: %s | Curr: %s ", self.portfolio_cash.prev, self.portfolio_cash.curr)

        self.assets_prices.set_curr(self.weights.curr[1:] * self.portfolio_value.curr)
        self.logger.debug("Asset Prices Prev: %s | Curr: %s ", self.assets_prices.prev, self.assets_prices.curr)

        # Calculate how many shares per asset with new prices
        self.shares.set_curr(self.assets_prices.curr / self._get_prices())

        reward = 0
        try:
            reward = self._calculate_reward()
        except IndexError:
            self.logger.warn("Error on reward calculation")

        next_state = None
        try:
            next_state = self._get_state_next()
        except IndexError:
            self.logger.warn("Error on getting next state")

        episode_end = self.current_step >= self.num_steps - 1
        self.current_step += 1

        return (
            next_state,
            float(reward),
            episode_end,
            False,
            {},
        )

    def step(self, action):
        return self._step_v2(action)
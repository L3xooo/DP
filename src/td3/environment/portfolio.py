import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

from td3.config.app_config import AppConfig
from td3.replay.replay_buffer import ReplayBuffer
from td3.utils.logger import WithLogger, log_values_with_color, log_stock_value
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

        # Historical values
        self.portfolio_value_history = []
        self.portfolio_weights_history = []

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
        return self.features[next_step].flatten()

    def _get_state(self):
        """Returns the current state as a flattened array of features."""
        return np.concatenate([self._get_features_current().flatten()], axis=0).astype(np.float32)

    def _get_state_next(self):
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
        """Calculates the reward as the log return of the portfolio value from the current step to the next step."""
        portfolio_value = self._calculate_portfolio_value()
        next_portfolio_value = self._calculate_portfolio_value(self._get_prices(1))
        return np.log((next_portfolio_value + 1e-12) / (portfolio_value + 1e-12))

    # def _calculate_risk_cost(self):
    #     """
    #     Computes risk cost.
    #     Formula: β × σ²_t
    #     """
    #     return self.risk_aversion_coefficient * self._calculate_portfolio_variance()

    # def _calculate_portfolio_variance(self, window: int = 20) -> float:
    #     """
    #     Computes portfolio variance σ²_t over last `window` returns.
    #     Formula: σ²_t = mean( (r_i - μ)² ), where r_i = (P_i - P_{i-1}) / P_{i-1}
    #     """
    #
    #     if len(self.portfolio_value_history) < window + 1:
    #         return 0.0
    #     recent = np.array(self.portfolio_value_history[-(window + 1) :])
    #     returns = (recent[1:] - recent[:-1]) / recent[:-1]
    #     return float(np.mean((returns - returns.mean()) ** 2))

    # def _calculate_transaction_cost(self, shares_changes: np.ndarray):
    #     """
    #     Calculate the transaction cost for the given shares changes.
    #     Formula: cᵗʳᵃⁿₜ = ξ × (pₜᵀ · |kₜ|)
    #     """
    #     return self.transaction_coefficient * np.dot(self._get_prices(), np.abs(shares_changes))

    def reset(self, seed=None, options=None):
        """Resets the environment to the initial state."""
        self._get_seed(seed)
        self.current_step = 0
        w0 = np.zeros(self.num_assets + 1, dtype=np.float32)
        w0[0] = 1.0
        zeros_assets = np.zeros(self.num_assets, dtype=np.float32)

        self.portfolio_value_history = [DEFAULT_PORTFOLIO_VALUE]
        self.portfolio_weights_history = [w0]

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
        log_stock_value(self.logger, self.app_config.ticker_config.tickers_with_cash, self.weights.curr, "Action Weights", decimals=4,
                        use_color=False)
        log_stock_value(self.logger, self.app_config.ticker_config.tickers, self._get_prices(), "Stock Price", decimals=4,
                        use_color=True)
        log_stock_value(self.logger, self.app_config.ticker_config.tickers, self.assets_prices.curr, "Asset Price",
                        decimals=4,
                        use_color=True)
        # Save the previous values, check if even needed
        self.weights.set_prev_from_curr()
        self.portfolio_value.set_prev_from_curr()
        self.portfolio_cash.set_prev_from_curr()
        self.shares.set_prev_from_curr()
        self.assets_prices.set_prev_from_curr()

        # Calculate the current portfolio value with previous cash and shares held
        self.portfolio_value.set_curr(
            self.portfolio_cash.prev + self._get_prices().dot(self.shares.prev)
        )
        log_values_with_color(self.logger, {"Portfolio Value": self.portfolio_value.curr})

        # Do the changes in portfolio based on the new weights
        self.weights.set_curr(action)
        self.portfolio_cash.set_curr(self.portfolio_value.curr * self.weights.curr[0])
        self.assets_prices.set_curr(self.weights.curr[1:] * self.portfolio_value.curr)

        # Calculate how many shares per asset with new prices
        self.shares.set_curr(self.assets_prices.curr / self._get_prices())
        reward = self._calculate_reward()

        self.current_step += 1

        return (
            self._get_state_next(),
            float(reward),
            self.current_step + 1 >= self.num_steps - 1,
            False,
            {},
        )

    def step(self, action):
        return self._step_v2(action)

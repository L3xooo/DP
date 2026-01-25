import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

from replay_buffer import ReplayBuffer
from utils.logger import WithLogger, log_stock_value, log_values_with_color
from utils.prev_curr import PrevCurr

DEFAULT_PORTFOLIO_VALUE = 10000.0

@WithLogger()
class PortfolioEnv(gym.Env):
    def __init__(self,
                 features: np.ndarray,
                 prices: np.ndarray = None,
                 tickers: list = None):
        super(PortfolioEnv, self).__init__()

        assert features.ndim == 3, "features must be a 3D numpy array (T, N, F)"
        self.features = features.astype(np.float32)
        self.current_step = None
        # From the shape get number of steps, assets and feature dimension
        self.num_steps, self.num_assets, self.feature_dim = self.features.shape

        self.tickers = tickers
        self.seed_value = None
        # Initialize the Replay Buffer
        self.replay_buffer = ReplayBuffer(100000)

        ## New values
        self.transaction_coefficient = 0.001
        self.risk_aversion_coefficient = 0.005
        self.new_portfolio_value_history = []
        self.new_portfolio_weights_history = []

        self.new_features = features.astype(np.float32)
        self.new_prices = prices.astype(np.float32)
        self.prices = prices.astype(np.float32)


        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.num_assets + 1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_assets * self.feature_dim,), dtype=np.float32)

    def _get_seed(self, seed=None):
        self.seed_value = seed
        np.random.seed(seed)
        random.seed(seed)

    def _get_state(self):
        """
        Returns the current state: flattened features.
        """
        return np.concatenate([self._get_features_current().flatten()], axis=0).astype(np.float32)

    def _get_state_next(self):
        """
        Returns the next state: flattened features.
        """
        return np.concatenate([self._get_features_next().flatten()], axis=0).astype(np.float32)

    def _calculate_reward(self, shares_changes: np.ndarray):
        """
        Computes reward for step.
        Formula: rₜ = pₜᵀmₜ − βσₜ² − ξ (pₜᵀ |kₜ|) − α * H(W)
        """
        transaction_cost = self._calculate_transaction_cost(shares_changes)
        risk_cost = self._calculate_risk_cost()
        portfolio_value = self._calculate_portfolio_value()

        next_portfolio_value = self._calculate_portfolio_value(self._get_prices(1))

        net_new_value = next_portfolio_value - risk_cost - transaction_cost

        rel_return = ((net_new_value - portfolio_value) / (portfolio_value + 1e-12))
        # self.logger.info(f"Reward = {rel_return:.4f}")
        # log_values_with_color(self.logger, {"Transaction Cost": transaction_cost, "Risk Cost": risk_cost,
        #                                     "Portfolio Value": portfolio_value, "Next Portfolio Value": next_portfolio_value,
        #                                     "Cash Return": float(next_portfolio_value - portfolio_value),
        #                                                        }, True)
        return float(rel_return)

    def _calculate_risk_cost(self):
        """
        Computes risk cost.
        Formula: β × σ²_t
        """
        return self.risk_aversion_coefficient * self._calculate_portfolio_variance()

    def _calculate_portfolio_variance(self, window: int = 20) -> float:
        """
        Computes portfolio variance σ²_t over last `window` returns.
        Formula: σ²_t = mean( (r_i - μ)² ), where r_i = (P_i - P_{i-1}) / P_{i-1}
        """

        if len(self.new_portfolio_value_history) < window + 1:
            return 0.0
        recent = np.array(self.new_portfolio_value_history[-(window + 1):])
        returns = (recent[1:] - recent[:-1]) / recent[:-1]
        return float(np.mean((returns - returns.mean()) ** 2))

    def _calculate_portfolio_value(self, prices: np.ndarray = None) -> float:
        if prices is None:
            return self.portfolio_cash.curr + self._get_prices().dot(self.shares.curr)
        else:
            return self.portfolio_cash.curr + prices.dot(self.shares.curr)

    def _calculate_transaction_cost(self, shares_changes: np.ndarray):
        """
        Calculate the transaction cost for the given shares changes.
        Formula: cᵗʳᵃⁿₜ = ξ × (pₜᵀ · |kₜ|)
        """
        return self.transaction_coefficient * np.dot(self._get_prices(), np.abs(shares_changes))

    def _get_features_current(self) -> np.ndarray:
        return self.new_features[self.current_step].flatten()

    def _get_features_next(self) -> np.ndarray:
        return self.new_features[self.current_step + 1].flatten()

    def _get_prices(self, extra_step: int = 0) -> np.ndarray:
        idx = self.current_step + extra_step
        if idx < 0 or idx >= len(self.prices):
            raise IndexError()
        return self.prices[idx].flatten()

    def reset(self, seed=None, options=None):
        self._get_seed(seed)
        self.current_step = 0

        self.new_portfolio_value_history = [DEFAULT_PORTFOLIO_VALUE]
        self.new_portfolio_weights_history = []

        self.portfolio_value = PrevCurr(prev=float(0), curr=float(DEFAULT_PORTFOLIO_VALUE))
        self.portfolio_cash = PrevCurr(prev=float(0), curr=float(DEFAULT_PORTFOLIO_VALUE))

        zeros_assets = np.zeros(self.num_assets, dtype=np.float32)

        # Shares & assets prices everything zero (1 step all in cash)
        self.shares = PrevCurr(prev=zeros_assets.copy(), curr=zeros_assets.copy())
        self.assets_prices = PrevCurr(prev=zeros_assets.copy(), curr=zeros_assets.copy())

        # Weights in each asset & cash
        w0 = np.zeros(self.num_assets + 1, dtype=np.float32)
        w0[0] = 1.0
        self.weights = PrevCurr(prev=zeros_assets.copy(), curr=w0.copy())

        return self._get_state()

    def _step_v2(self, action):
        # Save the previous values, check if even needed
        self.weights.set_prev_from_curr()
        self.portfolio_value.set_prev_from_curr()
        self.portfolio_cash.set_prev_from_curr()
        self.shares.set_prev_from_curr()
        self.assets_prices.set_prev_from_curr()

        log_stock_value(self.logger, self.tickers, self._get_prices(), "Asset Prices - Step Start")
        # Calculate the current portfolio value with previous cash and shares held
        self.portfolio_value.set_curr(self.portfolio_cash.prev + self._get_prices().dot(self.shares.prev))
        # log_values_with_color(self.logger, {"Portfolio Value": self.portfolio_value.curr})

        # Do the changes in portfolio based on the new weights
        self.weights.set_curr(action)
        log_values_with_color(self.logger, {"Weights": self.weights.curr})
        self.tickers.insert(0, "Cash")
        # log_stock_value(self.logger, self.tickers, self.weights.curr, "Weights")

        self.portfolio_cash.set_curr(self.portfolio_value.curr * self.weights.curr[0])
        # log_values_with_color(self.logger, {"Portfolio Cash": self.portfolio_cash.curr, "Invested in Asset": self.portfolio_value.curr - self.portfolio_cash.curr})

        self.assets_prices.set_curr(self.weights.curr[1:] * self.portfolio_value.curr)

        self.tickers.pop(0)
        # log_stock_value(self.logger, self.tickers, self.assets_prices.curr, "Assets Prices")
        # Calculate how many shares per asset with new prices
        self.shares.set_curr(self.assets_prices.curr / self._get_prices())
        # log_stock_value(self.logger, self.tickers, self.shares.curr, "Shares")

        # log_values_with_color(self.logger, {"Portfolio Value": self._calculate_portfolio_value()})

        shares_changes = self.shares.curr - self.shares.prev
        reward = self._calculate_reward(shares_changes)

        self.current_step +=1

        return self._get_state_next(), float(reward), self.current_step + 1 >= self.num_steps - 1, False, {}

    def step(self, action):
        # self.logger.info(f"------------------------ Step {self.current_step} ------------------------")
        return self._step_v2(action)
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

from replay_buffer import ReplayBuffer
from utils.logger import WithLogger, log_stock_value, log_values_with_color

DEFAULT_PORTFOLIO_VALUE = 10000.0

@WithLogger()
class PortfolioV2Env(gym.Env):
    """
    Environment adapted to 3D features tensor: features shape = (T, N, F)
    State returned = flattened features for current step (N*F) concatenated with portfolio_shares (N,)
    Assumes feature index 0 == close price for each asset.
    """
    def __init__(self,
                 features: np.ndarray,
                 prices: np.ndarray = None,
                 tickers: list = None):
        super(PortfolioV2Env, self).__init__()

        assert features.ndim == 3, "features must be a 3D numpy array (T, N, F)"
        self.features = features.astype(np.float32)
        self.current_step = None
        self.num_steps, self.num_assets, self.feature_dim = self.features.shape

        self.tickers = tickers
        self.seed_value = None
        self.replay_buffer = ReplayBuffer(100000)

        ## New values
        self.transaction_coefficient = 0.001
        self.risk_aversion_coefficient = 0.005

        self.features = features.astype(np.float32)
        self.prices = prices.astype(np.float32)

        self.portfolio_value_prev = None
        self.portfolio_value = None
        self.portfolio_value_history = []

        self.shares_change = None
        self.shares_hold = None
        self.shares_weights = None

        self.shares_hold_prev = None
        self.shares_weights_prev = None
        self.shares_changes_prev = None


        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.num_assets,), dtype=np.float32)
        obs_dim = self.num_assets * self.feature_dim + self.num_assets
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def _get_seed(self, seed=None):
        self.seed_value = seed
        np.random.seed(seed)
        random.seed(seed)

    def _get_state(self):
        """
        Returns the current state: flattened features + portfolio shares held.
        """
        features_flat = self._get_features_current().flatten() # Format like [price, indicator1, indicator2, ..., price, indicator1, indicator2, ...]
        return np.concatenate([features_flat, self.shares_hold_prev], axis=0).astype(np.float32)

    def _calculate_reward(self, shares_changes: np.ndarray):
        """
        Computes reward for step.
        Formula: rₜ = pₜᵀmₜ − βσₜ² − ξ (pₜᵀ |kₜ|)
        """
        transaction_cost = self._calculate_transaction_cost(shares_changes)
        risk_cost = self._calculate_risk_cost()
        total_reward = self._calculate_portfolio_value(shares_changes) - risk_cost - transaction_cost
        # log_values_with_color(self.logger, {"Transaction Cost" : transaction_cost, "Risk Cost": risk_cost, "Total Reward": total_reward }, use_color=False, level="info",)
        return total_reward

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

        if len(self.portfolio_value_history) < window + 1:
            return 0.0
        recent = np.array(self.portfolio_value_history[-(window + 1):])
        returns = (recent[1:] - recent[:-1]) / recent[:-1]
        return float(np.mean((returns - returns.mean()) ** 2))

    def _calculate_transaction_cost(self, shares_changes: np.ndarray):
        """
        Calculate the transaction cost for the given shares changes.
        Formula: cᵗʳᵃⁿₜ = ξ × (pₜᵀ · |kₜ|)
        """
        return self.transaction_coefficient * np.dot(self._get_prices_current(), np.abs(shares_changes))

    def _calculate_portfolio_value(self, shares_changes: np.ndarray, ):
        """
        Calculate the portfolio value from previous value and shares changes.
        Formula: Pₜ = Pₜ₋₁ + pₜᵀ kₜ
        """
        if self.current_step == 0:
            return self._get_prices_current().dot(shares_changes)
        else:
            return self.portfolio_value_prev + self._get_prices_current().dot(shares_changes)

    def _get_features_current(self) -> np.ndarray:
        return self.features[self.current_step].flatten()

    def get_prices_previous(self) -> np.ndarray:
        return self._get_prices_previous()

    def get_prices_current(self) -> np.ndarray:
        return self._get_prices_current()

    def _get_prices_current(self) -> np.ndarray:
        return self.prices[self.current_step].flatten()

    def _get_prices_previous(self) -> np.ndarray:
        if self.current_step > 0:
            return self.prices[self.current_step - 1].flatten()
        else:
            return self._get_prices_current()


    def reset(self, seed=None, options=None):
        self._get_seed(seed)
        self.current_step = 0

        # New Values
        self.portfolio_value_history = [DEFAULT_PORTFOLIO_VALUE]

        self.portfolio_value_prev = DEFAULT_PORTFOLIO_VALUE
        self.portfolio_value = DEFAULT_PORTFOLIO_VALUE

        self.shares_change = np.zeros(self.num_assets, dtype=np.float32)
        self.shares_hold = np.zeros(self.num_assets, dtype=np.float32)
        self.shares_weights = np.zeros(self.num_assets, dtype=np.float32)

        self.shares_hold_prev = np.zeros(self.num_assets, dtype=np.float32)
        self.shares_changes_prev = np.zeros(self.num_assets, dtype=np.float32)
        self.shares_weights_prev = np.zeros(self.num_assets, dtype=np.float32)

        return self._get_state()

    def _step_v2(self, action):
        self.logger.info(f"Starting step {self.current_step} with portfolio init 10000")
        init_portfolio_value = DEFAULT_PORTFOLIO_VALUE
        action = np.array(action).flatten()

        portfolio_shares_values = action * init_portfolio_value # Get the dollar value per stock where I invested the cash
        portfolio_shares_count =  portfolio_shares_values / (self._get_prices_current() + 1e-12) # Get how many shares I can buy per stock
        # control_portfolio_value = self.

    def _step(self, action):
        # log_stock_value(self.logger, self.tickers, self.shares_weights, "Prev Weights")
        # log_stock_value(self.logger, self.tickers, action, "Current Weights")
        # log_stock_value(self.logger, self.tickers, action - self.shares_weights, "Diff Weights", use_color=True)

        # log_stock_value(self.logger, self.tickers, self.shares_weights, "Prev Shares Hold")
        # log_stock_value(self.logger, self.tickers, self.shares_hold_prev, "Current Shares Hold")
        # log_stock_value(self.logger, self.tickers, self.shares_hold - self.shares_hold_prev, "Diff Shares Hold", use_color=True)

        self.portfolio_value_prev = self.portfolio_value
        self.shares_changes_prev = self.shares_change
        self.shares_hold_prev = self.shares_hold
        self.shares_weights_prev = self.shares_weights

        # Get the action and flatten it
        action = np.array(action).flatten()
        # Update the shares here
        self.shares_weights = action

        # Calculate the prices of each asset based on the weight
        target_asset_values = self.shares_weights * self.portfolio_value_prev
        # log_stock_value(self.logger, self.tickers, target_asset_values, "Calculated Asset Price")
        # Count how many shares I have per asset
        shares_hold = target_asset_values / (self._get_prices_current() + 1e-12)

        # Get how shares changes from previous step (see how much you buy/sell per each)
        shares_changes = shares_hold - self.shares_hold_prev
        # log_stock_value(self.logger, self.tickers, self.new_shares_hold_prev, "Asset Count Before Change", use_color=True)
        # log_stock_value(self.logger, self.tickers, shares_changes, "Asset Count Change", use_color=True)
        portfolio_value = self._calculate_portfolio_value(shares_changes)
        log_values_with_color(self.logger,{"Prev Portfolio" : self.portfolio_value_prev, "Portfolio Value": portfolio_value , "Portfolio Change": portfolio_value - self.portfolio_value_prev}, use_color=True)
        # self.logger.info(f"Prev Portfolio Value: {self.new_portfolio_value:.2f} | New Portfolio Value: {portfolio_value:.2f} | Portfolio Change: {(portfolio_value-self.new_portfolio_value_prev):.2f}" )

        self.shares_hold = shares_hold.astype(np.float32)
        self.shares_changes = shares_changes.astype(np.float32)
        # self.new_portfolio_value = float(portfolio_value)
        self.portfolio_value_history.append(self.portfolio_value)

        # print("Portfolio value: ", portfolio_value)

        reward = self._calculate_reward(shares_changes)
        self.logger.info("Reward: {}".format(reward))
        self.current_step += 1
        done = self.current_step >= self.num_steps - 1
        return self._get_state(), float(reward), done, False, {}

    def step(self, action):
        return self._step(action)
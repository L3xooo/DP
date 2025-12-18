import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

from replay_buffer import ReplayBuffer
from utils.logger import WithLogger, log_stock_value, log_values_with_color

DEFAULT_PORTFOLIO_VALUE = 10000.0

@WithLogger()
class PortfolioEnv(gym.Env):
    """
    Environment adapted to 3D features tensor: features shape = (T, N, F)
    State returned = flattened features for current step (N*F) concatenated with portfolio_shares (N,)
    Assumes feature index 0 == close price for each asset.
    """
    def __init__(self,
                 features: np.ndarray,
                 num_assets: int = None,
                 max_shares=100,
                 prices: np.ndarray = None,
                 tickers: list = None,
                 ):
        super(PortfolioEnv, self).__init__()

        self.portfolio_shares_prices = None
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
        self.new_portfolio_value_history = []
        self.new_portfolio_weights_history = []

        self.new_features = features.astype(np.float32)
        self.new_prices = prices.astype(np.float32)

        self.new_portfolio_value_prev = None
        self.new_portfolio_value = None

        self.new_shares_changes = None
        self.new_shares_hold = None
        self.new_shares_weights = None

        self.new_shares_hold_prev = None
        self.new_shares_weights_prev = None
        self.new_shares_changes_prev = None

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
        return np.concatenate([features_flat, self.new_shares_hold_prev], axis=0).astype(np.float32)

    def _calculate_reward(self, shares_changes: np.ndarray):
        """
        Computes reward for step.
        Formula: rₜ = pₜᵀmₜ − βσₜ² − ξ (pₜᵀ |kₜ|) − α * H(W)
        """
        transaction_cost = self._calculate_transaction_cost(shares_changes)
        risk_cost = self._calculate_risk_cost()
        new_portfolio_value = self._calculate_portfolio_value_v2()

        # Dynamická penalizácia za diverzifikáciu na základe váh menších ako 0.01
        log_values_with_color(self.logger, {"New Shares Weights": self.new_shares_weights}, use_color=False, level="info",)
        small_weights_count = sum(1 for w in self.new_shares_weights if w < 0.001)
        total_weights = len(self.new_shares_weights)

        # Dynamická penalizácia
        diversification_penalty = small_weights_count / total_weights * 100  # Väčší násobok pre silnú penalizáciu

        # self.logger.info("Diversification Penalty (Entropy): %.6f", diversification_penalty)

        # Výpočet novej hodnoty portfólia po odpočítaní nákladov
        net_new_value = new_portfolio_value - risk_cost - transaction_cost

        if self.new_portfolio_value_prev is None or self.new_portfolio_value_prev == 0:
            return float(net_new_value), diversification_penalty

        # Relatívna návratnosť so zohľadnením penalizácie za diverzifikáciu
        rel_return = ((net_new_value - self.new_portfolio_value_prev) / (
                    self.new_portfolio_value_prev + 1e-12) * 100) - diversification_penalty

        return float(rel_return), diversification_penalty
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

    def _calculate_portfolio_value_v2(self):
        """

        Returns:

        """
        if self.current_step == 0:
            return DEFAULT_PORTFOLIO_VALUE
        return self._get_prices_current().dot(self.new_shares_hold_prev)

    def _calculate_transaction_cost(self, shares_changes: np.ndarray):
        """
        Calculate the transaction cost for the given shares changes.
        Formula: cᵗʳᵃⁿₜ = ξ × (pₜᵀ · |kₜ|)
        """
        return self.transaction_coefficient * np.dot(self._get_prices_current(), np.abs(shares_changes))

    def _calculate_portfolio_value(self, shares_changes: np.ndarray):
        """
        Calculate the portfolio value from previous value and shares changes.
        Formula: Pₜ = Pₜ₋₁ + pₜᵀ kₜ
        """
        if self.current_step == 0:
            return self._get_prices_current().dot(shares_changes)
        else:
            # self.logger.info(f"_calculate_portfolio_value {self.current_step} prev: {self.new_portfolio_value_prev} changes dot: {self._get_prices_current().dot(shares_changes)}")
            return self.new_portfolio_value_prev + self._get_prices_current().dot(shares_changes)

    def _get_features_current(self) -> np.ndarray:
        return self.new_features[self.current_step].flatten()

    def get_prices_previous(self) -> np.ndarray:
        return self._get_prices_previous()

    def get_prices_current(self) -> np.ndarray:
        return self._get_prices_current()

    def _get_prices_current(self) -> np.ndarray:
        return self.new_prices[self.current_step].flatten()

    def _get_prices_previous(self) -> np.ndarray:
        if self.current_step > 0:
            return self.new_prices[self.current_step - 1].flatten()
        else:
            return self._get_prices_current()

    def reset(self, seed=None, options=None):

        self._get_seed(seed)
        self.current_step = 0

        # New Values
        self.new_portfolio_value_history = [DEFAULT_PORTFOLIO_VALUE]

        self.new_portfolio_value_prev = DEFAULT_PORTFOLIO_VALUE
        self.new_portfolio_value = DEFAULT_PORTFOLIO_VALUE

        self.portfolio_shares_prices = np.zeros(self.num_assets, dtype=np.float32)
        self.new_shares_changes = np.zeros(self.num_assets, dtype=np.float32)
        self.new_shares_hold = np.zeros(self.num_assets, dtype=np.float32)
        self.new_shares_weights = np.zeros(self.num_assets, dtype=np.float32)

        self.portfolio_shares_prices_prev = np.zeros(self.num_assets, dtype=np.float32)
        self.new_shares_hold_prev = np.zeros(self.num_assets, dtype=np.float32)
        self.new_shares_changes_prev = np.zeros(self.num_assets, dtype=np.float32)
        self.new_shares_weights_prev = np.zeros(self.num_assets, dtype=np.float32)

        return self._get_state()

    def _step(self, action):
        log_stock_value(self.logger, self.tickers, self.new_shares_weights, "P Weights")
        log_stock_value(self.logger, self.tickers, action, "C Weights")
        # log_stock_value(self.logger, self.tickers, action - self.new_shares_weights, "D Weights", use_color=True)

        self.new_portfolio_value_prev = self.new_portfolio_value
        self.new_shares_changes_prev = self.new_shares_changes
        self.new_shares_hold_prev = self.new_shares_hold
        self.new_shares_weights_prev = self.new_shares_weights
        # self.logger.info("[Shares Portfolio Value]: %.2f", self._calculate_portfolio_value_v2())

        # Get the action and flatten it
        action = np.array(action).flatten()
        # Update the shares here

        self.new_shares_weights = action
        self.new_portfolio_weights_history.append(self.new_shares_weights)


        # Calculate the prices of each asset based on the weight
        target_asset_values = self.new_shares_weights * self._calculate_portfolio_value_v2()
        # log_stock_value(self.logger, self.tickers, self.new_shares, "Calculated Asset Price")
        # log_stock_value(self.logger, self.tickers, target_asset_values, "Calculated Asset Price")
        # Count how many shares I have per asset
        shares_hold = target_asset_values / (self._get_prices_current() + 1e-12)

        # Get how shares changes from previous step (see how much you buy/sell per each)
        shares_changes = shares_hold - self.new_shares_hold_prev

        # log_stock_value(self.logger, self.tickers, self.new_shares_hold_prev, "P Shares Count")
        # log_stock_value(self.logger, self.tickers, shares_changes, "D Shares Count", use_color=True)
        # log_stock_value(self.logger, self.tickers, shares_hold, "T Shares Count", use_color=False)

        portfolio_value = self._calculate_portfolio_value_v2()
        # log_portfolio_value_change(self.logger, self.new_portfolio_value, portfolio_value, self.new_portfolio_value_prev, log_name="Portfolio", use_color=True)
        # self.logger.info(f"Prev Portfolio Value: {self.new_portfolio_value:.2f} | New Portfolio Value: {portfolio_value:.2f} | Portfolio Change: {(portfolio_value-self.new_portfolio_value_prev):.2f}" )
        self.new_shares_hold = shares_hold.astype(np.float32)
        self.new_shares_changes = shares_changes.astype(np.float32)
        self.new_portfolio_value = float(portfolio_value)
        self.new_portfolio_value_history.append(self.new_portfolio_value)

        # print("Portfolio value: ", portfolio_value)

        reward, diversification_penalty = self._calculate_reward(shares_changes)
        log_values_with_color(self.logger, {"Prev Portfolio Value": self.new_portfolio_value_prev, "New Portfolio Value": portfolio_value, "Portfolio Change": (portfolio_value - self.new_portfolio_value_prev), "Reward": reward, "Diversification Penalty": diversification_penalty}, use_color=True, level="info",)

        # log_values_with_color(self.logger, {"Reward": reward}, use_color=True, level="info",)
        self.current_step += 1
        done = self.current_step >= self.num_steps - 1

        return self._get_state(), float(reward), done, False, {}

    def step(self, action):
        # self.logger.info(f"------------------------ Step {self.current_step} ------------------------")
        return self._step(action)
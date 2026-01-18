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


        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.num_assets + 1,), dtype=np.float32)
        obs_dim = self.num_assets * self.feature_dim
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def _get_seed(self, seed=None):
        self.seed_value = seed
        np.random.seed(seed)
        random.seed(seed)

    def _get_state(self):
        """
        Returns the current state: flattened features.
        """
        return np.concatenate([self._get_features_current().flatten()], axis=0).astype(np.float32)

    def _calculate_reward(self, shares_changes: np.ndarray):
        """
        Computes reward for step.
        Formula: rₜ = pₜᵀmₜ − βσₜ² − ξ (pₜᵀ |kₜ|) − α * H(W)
        """
        transaction_cost = self._calculate_transaction_cost(shares_changes)
        risk_cost = self._calculate_risk_cost()
        new_portfolio_value = self._calculate_portfolio_value_v2()


        # Výpočet novej hodnoty portfólia po odpočítaní nákladov
        net_new_value = new_portfolio_value - risk_cost - transaction_cost

        if self.portfolio_value.prev is None or self.portfolio_value.prev == 0:
            return float(net_new_value)

        rel_return = ((net_new_value - self.portfolio_value.prev) / (
                    self.portfolio_value.prev + 1e-12) * 100)
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

    def _calculate_portfolio_value_v2(self):
        # if self.current_step == 0:
        #     return float(DEFAULT_PORTFOLIO_VALUE)
        # return float(self._get_prices_current().dot(self.portfolio_shares_hold.prev))
        # return self.portfolio_ca
        return self.portfolio_cash.curr + self._get_prices_current().dot(self.portfolio_shares_hold.prev)

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

        self.new_portfolio_value_history = [DEFAULT_PORTFOLIO_VALUE]
        self.new_portfolio_weights_history = []

        self.portfolio_value = PrevCurr(prev=float(DEFAULT_PORTFOLIO_VALUE), curr=float(DEFAULT_PORTFOLIO_VALUE))
        self.portfolio_cash = PrevCurr(prev=float(DEFAULT_PORTFOLIO_VALUE), curr=float(DEFAULT_PORTFOLIO_VALUE))


        zeros_assets = np.zeros(self.num_assets, dtype=np.float32)
        self.portfolio_shares_hold = PrevCurr(prev=zeros_assets.copy(), curr=zeros_assets.copy())
        self.portfolio_shares_changes = PrevCurr(prev=zeros_assets.copy(), curr=zeros_assets.copy())
        self.portfolio_weights = PrevCurr(prev=zeros_assets.copy(), curr=zeros_assets.copy())

        prices0 = self._get_prices_current().astype(np.float32)
        self.portfolio_shares_prices = PrevCurr(prev=prices0.copy(), curr=prices0.copy())


        return self._get_state()

    def _step(self, action):
        log_values_with_color(self.logger, {"Cash" : self.portfolio_cash.curr, "Portfolio" : self.portfolio_value.curr})
        log_values_with_color(self.logger, {"Action": action})
        self.portfolio_cash.prev = self.portfolio_cash.curr
        self.portfolio_value.prev = self.portfolio_value.curr

        self.portfolio_shares_hold.update(self.portfolio_shares_hold.curr)
        self.portfolio_shares_changes.update(self.portfolio_shares_changes.curr)
        self.portfolio_weights.update(self.portfolio_weights.curr)
        self.portfolio_shares_prices.update(self.portfolio_shares_prices.curr)

        self.portfolio_weights.set_curr(np.array(action, dtype=np.float32).flatten())
        self.new_portfolio_weights_history.append(self.portfolio_weights.curr.copy())

        pv_for_alloc = self.portfolio_cash.prev + self._get_prices_current().dot(self.portfolio_shares_hold.prev)
        # self.logger.info("Portfolio Value for Allocation: {:.2f}".format(pv_for_alloc))

        new_cash = pv_for_alloc * self.portfolio_weights.curr[0]
        self.portfolio_cash.curr = new_cash
        # self.logger.info("New cash: {:.2f}".format(new_cash))

        target_asset_values = self.portfolio_weights.curr[1:]  * pv_for_alloc

        # self.logger.info("Target asset values: {:.2f}".format(np.sum(target_asset_values)))

        self.portfolio_shares_prices.set_curr(target_asset_values.copy())

        shares_hold = target_asset_values / (self.get_prices_current())

        # self.logger.info(f"Target asset values: {shares_hold}")

        shares_changes = shares_hold - self.portfolio_shares_hold.prev

        # log_values_with_color(self.logger, {"Changes": shares_changes}, True)

        # 6) Update curr
        portfolio_value = self.portfolio_cash.prev + self._get_prices_current().dot(self.portfolio_shares_hold.prev)


        self.portfolio_shares_hold.set_curr(shares_hold.astype(np.float32))
        self.portfolio_shares_changes.set_curr(shares_changes.astype(np.float32))
        self.portfolio_value.set_curr(float(portfolio_value))
        self.new_portfolio_value_history.append(self.portfolio_value.curr)

        reward = self._calculate_reward(self.portfolio_shares_changes.curr)
        log_values_with_color(self.logger, {"Reward": reward}, True)
        self.current_step += 1
        return self._get_state(), float(reward), self.current_step >= self.num_steps - 1, False, {}

    def step(self, action):
        # self.logger.info(f"------------------------ Step {self.current_step} ------------------------")
        return self._step(action)
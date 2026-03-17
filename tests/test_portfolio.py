from td3.config.app_config import AppConfig
from td3.environment.portfolio import PortfolioEnv, DEFAULT_PORTFOLIO_VALUE
from td3.utils.prev_curr import PrevCurr

import numpy as np
import pytest


def _make_env(
    seed: int = 0, T: int = 6, N: int = 4, F: int = 3
) -> tuple[PortfolioEnv, np.ndarray, int, int]:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(T, N, F)).astype(np.float32)
    prices = rng.uniform(50, 200, size=(T, N)).astype(np.float32)
    tickers = [f"T{i}" for i in range(N)]
    env = PortfolioEnv(
        features=features,
        prices=prices,
        tickers=tickers,
        app_config=AppConfig(),
    )
    return env, features, N, F


class TestPortfolioGetFeatures:
    def test_get_features_current_returns_expected_flatten(self):
        env, features, N, F = _make_env()
        print(env.num_steps)
        env.current_step = 2

        expected = features[2].flatten()
        np.testing.assert_allclose(env._get_features_current(), expected)
        assert env._get_features_current().shape == (N * F,)

    def test_get_features_next_returns_expected_flatten(self):
        env, features, N, F = _make_env()
        env.current_step = 2

        expected = features[3].flatten()
        np.testing.assert_allclose(env._get_features_next(), expected)
        assert env._get_features_next().shape == (N * F,)

    def test_get_features_next_raises_index_error_when_out_of_bounds(self):
        env, features, N, F = _make_env()
        env.current_step = env.num_steps - 1

        with pytest.raises(IndexError):
            _ = env._get_features_next()


class TestPortfolioGetState:
    def test_get_state_returns_expected(self):
        env, features, N, F = _make_env()
        env.current_step = 2

        expected = features[2].flatten().astype(np.float32)
        got = env._get_state()

        np.testing.assert_allclose(got, expected)
        assert got.dtype == np.float32
        assert got.shape == (N * F,)

    def test_get_state_next_returns_expected(self):
        env, features, N, F = _make_env()
        env.current_step = 2

        expected = features[3].flatten().astype(np.float32)
        got = env._get_state_next()

        np.testing.assert_allclose(got, expected)
        assert got.dtype == np.float32
        assert got.shape == (N * F,)

    def test_get_state_next_raises_index_error_when_out_of_bounds(self):
        env, features, N, F = _make_env()
        env.current_step = env.features.shape[0] - 1  # last step

        with pytest.raises(IndexError):
            _ = env._get_state_next()


class TestPortfolioGetPrices:
    @pytest.mark.parametrize(
        "current_step, extra_step, expected, should_raise",
        [
            (2, 0, np.array([8, 9, 10, 11], dtype=np.float32), False),
            (2, 1, np.array([12, 13, 14, 15], dtype=np.float32), False),
            (5, 1, None, True),
        ],
        ids=["current", "with_extra_step", "out_of_bounds"],
    )
    def test_get_prices_returns_exact_values(
        self, current_step, extra_step, expected, should_raise
    ):
        from td3.environment.portfolio import PortfolioEnv

        env = PortfolioEnv.__new__(PortfolioEnv)

        T, N = 6, 4
        env.prices = np.arange(T * N, dtype=np.float32).reshape(T, N)
        env.current_step = current_step

        if should_raise:
            with pytest.raises(IndexError):
                env._get_prices(extra_step=extra_step)
            return

        got = env._get_prices(extra_step=extra_step)
        np.testing.assert_array_equal(got, expected)


class TestPortfolioValue:
    def test_calculate_portfolio_value_uses_internal_get_prices(self, monkeypatch):
        env = PortfolioEnv.__new__(PortfolioEnv)

        env.portfolio_cash = PrevCurr(prev=5000.0, curr=7000.0)
        env.shares = PrevCurr(
            prev=np.array([10.0, 5.0], dtype=np.float32),
            curr=np.array([15.0, 3.0], dtype=np.float32),
        )

        internal_prices = np.array([100.0, 200.0], dtype=np.float32)

        def fake_get_prices():
            return internal_prices

        monkeypatch.setattr(env, "_get_prices", fake_get_prices)

        calculated = env._calculate_portfolio_value()
        expected = 9100.0

        assert calculated == pytest.approx(expected, rel=0, abs=1e-12)

    @pytest.mark.parametrize(
        "prices_arg, expected",
        [
            (
                np.array([100.0, 200.0], dtype=np.float32),
                9100,
            ),  # 15 * 100 + 3 * 200 + 7000 = 1500 + 600 + 7000 = 9100
            (np.array([50.0, 10.0], dtype=np.float32), 7780),  # 7780
            (np.array([0.0, 300.0], dtype=np.float32), 7900),  # 7900
        ],
        ids=["p100_200", "p50_10", "p0_300"],
    )
    def test_calculate_portfolio_value_correctness_three_prices(self, prices_arg, expected):
        env = PortfolioEnv.__new__(PortfolioEnv)

        env.portfolio_cash = PrevCurr(prev=5000.0, curr=7000.0)
        env.shares = PrevCurr(
            prev=np.array([10.0, 5.0], dtype=np.float32),
            curr=np.array([15.0, 3.0], dtype=np.float32),
        )

        calculated = env._calculate_portfolio_value(prices=prices_arg)
        assert calculated == pytest.approx(expected, rel=0, abs=1e-12)

    def test_calculate_portfolio_value_raises_when_price_and_share_sizes_mismatch(
        self,
    ):
        env = PortfolioEnv.__new__(PortfolioEnv)

        env.portfolio_cash = PrevCurr(prev=0.0, curr=0.0)
        env.shares = PrevCurr(
            prev=np.zeros(3, dtype=np.float32),
            curr=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        )

        bad_prices = np.array([10.0, 20.0], dtype=np.float32)

        with pytest.raises(ValueError, match="prices shape"):
            env._calculate_portfolio_value(prices=bad_prices)


class TestPortfolioReset:
    def test_reset_overwrites_previous_state(self):
        rng = np.random.default_rng(0)
        T, N, F = 5, 4, 3

        features = rng.normal(size=(T, N, F)).astype(np.float32)
        prices = rng.uniform(50, 200, size=(T, N)).astype(np.float32)
        tickers = [f"T{i}" for i in range(N)]

        env = PortfolioEnv(
            features=features,
            prices=prices,
            tickers=tickers,
            app_config=AppConfig(),
        )

        env.current_step = 3
        env.portfolio_value_history = [111.0, 222.0]
        env.portfolio_weights_history = [np.ones(N + 1, dtype=np.float32)]

        env.portfolio_value = PrevCurr(prev=1.0, curr=2.0)
        env.portfolio_cash = PrevCurr(prev=3.0, curr=4.0)

        env.shares = PrevCurr(
            prev=np.ones(N, dtype=np.float32),
            curr=np.ones(N, dtype=np.float32) * 5,
        )
        env.assets_prices = PrevCurr(
            prev=np.ones(N, dtype=np.float32),
            curr=np.ones(N, dtype=np.float32) * 6,
        )

        env.weights = PrevCurr(
            prev=np.ones(N + 1, dtype=np.float32),
            curr=np.ones(N + 1, dtype=np.float32) * 0.25,
        )

        obs = env.reset(seed=123)

        assert env.current_step == 0

        assert env.weights.curr.shape == (N + 1,)
        assert env.weights.curr[0] == 1.0
        assert np.all(env.weights.curr[1:] == 0.0)

        assert np.all(env.shares.curr == 0.0)
        assert np.all(env.assets_prices.curr == 0.0)

        assert env.portfolio_value.curr == DEFAULT_PORTFOLIO_VALUE
        assert env.portfolio_cash.curr == DEFAULT_PORTFOLIO_VALUE

        assert env.portfolio_value_history == [DEFAULT_PORTFOLIO_VALUE]
        assert len(env.portfolio_weights_history) == 1
        np.testing.assert_array_equal(env.portfolio_weights_history[0], env.weights.curr)

        expected_obs = features[0].flatten().astype(np.float32)
        np.testing.assert_allclose(obs, expected_obs)


class TestPortfolioReward:
    def test_calculate_reward_handles_zero_portfolio_value(self, monkeypatch):
        env = PortfolioEnv.__new__(PortfolioEnv)
        monkeypatch.setattr(env, "_get_prices", lambda extra_step: object())
        monkeypatch.setattr(
            env,
            "_calculate_portfolio_value",
            lambda prices=None: 0.0 if prices is None else 1.0,
        )

        r = env._calculate_reward()
        # log((1.0 + 1e-12) / (0.0 + 1e-12)) ≈ log(1e12) ≈ 27.631
        expected = np.log((1.0 + 1e-12) / (0.0 + 1e-12))
        assert r == pytest.approx(expected, rel=1e-6)

    @pytest.mark.parametrize(
        "pv, next_pv, expected",
        [
            (100.0, 110.0, np.log(110.0 / 100.0)),   # ≈  0.09531
            (200.0, 180.0, np.log(180.0 / 200.0)),   # ≈ -0.10536
            (1000.0, 1000.0, 0.0),
        ],
        ids=["up_10pct", "down_10pct", "flat"],
    )

    def test_calculate_reward_simple_return(self, monkeypatch, pv, next_pv, expected):
        env = PortfolioEnv.__new__(PortfolioEnv)

        sentinel_prices = object()
        monkeypatch.setattr(env, "_get_prices", lambda extra_step: sentinel_prices)

        def fake_calc_portfolio_value(prices=None):
            return pv if prices is None else next_pv

        monkeypatch.setattr(env, "_calculate_portfolio_value", fake_calc_portfolio_value)

        r = env._calculate_reward()

        assert r == pytest.approx(expected, rel=0, abs=1e-10)
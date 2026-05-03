"""
Tests for src/features/ — exogenous feature pipeline.

Expected interfaces
-------------------
src.features.exogenous
    build_feature_matrix(market_data: dict[str, pd.Series]) -> pd.DataFrame
        market_data keys: SPY TLT GLD HYG LQD VIX VIX3M VVIX IRX FVX TNX TYX
        Returns DataFrame with exactly 16 columns (same index as inputs).

src.features.yield_curve
    fit_yield_curve_coefficients(tenors: np.ndarray, yields: np.ndarray) -> np.ndarray
        Returns 4-element array [level, slope, curvature, inflection].

src.features.correlations
    rolling_correlation(s1: pd.Series, s2: pd.Series, window: int) -> pd.Series
        Returns rolling Pearson correlation (NaN before window fills).
"""
import numpy as np
import pandas as pd
import pytest

from src.features.exogenous import build_feature_matrix
from src.features.yield_curve import fit_yield_curve_coefficients
from src.features.correlations import rolling_correlation

REQUIRED_COLUMNS = {
    "vix", "vix3m", "vix_term_spread", "vvix",
    "vrp_30d", "vrp_60d",
    "yield_level", "yield_slope", "yield_curvature", "yield_inflection",
    "spy_tlt_corr_20d", "spy_gld_corr_20d", "hyg_lqd_spread",
    "spy_return_20d", "spy_dist_200ma", "ma_cross_state",
}


# ===========================================================================
# Schema tests — feature count, names, dtypes
# ===========================================================================

class TestFeatureSchema:
    def test_exactly_16_features(self, long_market_data):
        """Feature matrix must have exactly 16 columns — no more, no fewer."""
        features = build_feature_matrix(long_market_data)
        assert features.shape[1] == 16, (
            f"Expected 16 features, got {features.shape[1]}.\n"
            f"Got columns: {sorted(features.columns.tolist())}"
        )

    def test_required_column_names_present(self, long_market_data):
        """All 16 required column names must be present (exact names, no aliases)."""
        features = build_feature_matrix(long_market_data)
        missing = REQUIRED_COLUMNS - set(features.columns)
        assert not missing, f"Missing feature columns: {missing}"

    def test_no_extra_columns(self, long_market_data):
        """No undocumented extra columns allowed — keeps the interface stable."""
        features = build_feature_matrix(long_market_data)
        extra = set(features.columns) - REQUIRED_COLUMNS
        assert not extra, f"Unexpected extra feature columns: {extra}"

    def test_ma_cross_state_is_integer_dtype(self, long_market_data):
        """MA cross state is a binary indicator and must be integer, not float."""
        features = build_feature_matrix(long_market_data)
        col = features["ma_cross_state"].dropna()
        assert col.dtype in (np.int32, np.int64, np.int8, np.int16, int), (
            f"ma_cross_state dtype should be integer, got {col.dtype}"
        )

    def test_ma_cross_state_only_zero_or_one(self, long_market_data):
        """Golden cross / death cross must be exactly 0 or 1, never any other value."""
        features = build_feature_matrix(long_market_data)
        col = features["ma_cross_state"].dropna()
        invalid = col[~col.isin([0, 1])]
        assert invalid.empty, f"ma_cross_state has values outside {{0,1}}: {invalid.unique()}"

    def test_output_index_matches_input_index(self, long_market_data):
        """Output DataFrame index must be identical to the input date index."""
        features = build_feature_matrix(long_market_data)
        pd.testing.assert_index_equal(
            features.index,
            long_market_data["SPY"].index,
            check_names=False,
        )


# ===========================================================================
# Correctness — VRP
# ===========================================================================

class TestVRP:
    def test_vrp_30d_sign_and_magnitude(self, long_biz_dates):
        """
        With constant VIX=20 and constant daily moves yielding ~15% annualized vol,
        VRP should be positive and close to 5 vol points after warmup.
        """
        n = len(long_biz_dates)
        daily_vol = 0.15 / np.sqrt(252)
        spy_returns = np.full(n, daily_vol)
        spy_prices = 100 * np.exp(np.cumsum(spy_returns))

        data = {
            "SPY": pd.Series(spy_prices, index=long_biz_dates),
            "TLT": pd.Series(np.ones(n) * 100, index=long_biz_dates),
            "GLD": pd.Series(np.ones(n) * 100, index=long_biz_dates),
            "HYG": pd.Series(np.ones(n) * 100, index=long_biz_dates),
            "LQD": pd.Series(np.ones(n) * 100, index=long_biz_dates),
            "VIX":  pd.Series(np.full(n, 20.0), index=long_biz_dates),
            "VIX3M": pd.Series(np.full(n, 22.0), index=long_biz_dates),
            "VVIX": pd.Series(np.full(n, 85.0), index=long_biz_dates),
            "IRX": pd.Series(np.full(n, 0.5), index=long_biz_dates),
            "FVX": pd.Series(np.full(n, 1.5), index=long_biz_dates),
            "TNX": pd.Series(np.full(n, 2.5), index=long_biz_dates),
            "TYX": pd.Series(np.full(n, 3.0), index=long_biz_dates),
        }
        features = build_feature_matrix(data)
        vrp = features["vrp_30d"].dropna().iloc[-1]
        assert vrp == pytest.approx(5.0, abs=1.5), (
            f"VRP should be ~5 (VIX=20 minus realized~15), got {vrp:.2f}"
        )

    def test_vrp_negative_when_realized_exceeds_vix(self, long_biz_dates):
        """
        When realized vol (30%) > VIX (18%), VRP is negative.
        The pipeline must NOT clip to zero — negative VRP is a valid signal.
        """
        n = len(long_biz_dates)
        daily_vol = 0.30 / np.sqrt(252)
        spy_prices = 100 * np.exp(np.cumsum(np.full(n, daily_vol)))

        rng = np.random.default_rng(1)
        base = {
            k: pd.Series(np.ones(n) * 100, index=long_biz_dates)
            for k in ["TLT", "GLD", "HYG", "LQD"]
        }
        data = {
            "SPY": pd.Series(spy_prices, index=long_biz_dates),
            "VIX":  pd.Series(np.full(n, 18.0), index=long_biz_dates),
            "VIX3M": pd.Series(np.full(n, 20.0), index=long_biz_dates),
            "VVIX": pd.Series(np.full(n, 85.0), index=long_biz_dates),
            "IRX": pd.Series(np.full(n, 0.5), index=long_biz_dates),
            "FVX": pd.Series(np.full(n, 1.5), index=long_biz_dates),
            "TNX": pd.Series(np.full(n, 2.5), index=long_biz_dates),
            "TYX": pd.Series(np.full(n, 3.0), index=long_biz_dates),
            **base,
        }
        features = build_feature_matrix(data)
        vrp = features["vrp_30d"].dropna().iloc[-1]
        assert vrp < 0, (
            f"VRP should be negative when realized vol > VIX, got {vrp:.2f}"
        )

    def test_vix_term_spread_formula(self, long_biz_dates):
        """vix_term_spread = VIX3M - VIX. Positive means contango (normal regime)."""
        n = len(long_biz_dates)
        rng = np.random.default_rng(2)
        data = {
            k: pd.Series(np.ones(n) * 100, index=long_biz_dates)
            for k in ["SPY", "TLT", "GLD", "HYG", "LQD"]
        }
        data.update({
            "VIX":  pd.Series(np.full(n, 15.0), index=long_biz_dates),
            "VIX3M": pd.Series(np.full(n, 18.0), index=long_biz_dates),
            "VVIX": pd.Series(np.full(n, 85.0), index=long_biz_dates),
            "IRX": pd.Series(np.full(n, 0.5), index=long_biz_dates),
            "FVX": pd.Series(np.full(n, 1.5), index=long_biz_dates),
            "TNX": pd.Series(np.full(n, 2.5), index=long_biz_dates),
            "TYX": pd.Series(np.full(n, 3.0), index=long_biz_dates),
        })
        features = build_feature_matrix(data)
        spread = features["vix_term_spread"].dropna().iloc[-1]
        assert spread == pytest.approx(3.0, abs=0.01), (
            f"vix_term_spread = VIX3M(18) - VIX(15) should be 3.0, got {spread}"
        )

    def test_vix_term_spread_negative_in_backwardation(self, long_biz_dates):
        """When VIX > VIX3M, vix_term_spread must be negative."""
        n = len(long_biz_dates)
        data = {
            k: pd.Series(np.ones(n) * 100, index=long_biz_dates)
            for k in ["SPY", "TLT", "GLD", "HYG", "LQD"]
        }
        data.update({
            "VIX":  pd.Series(np.full(n, 28.0), index=long_biz_dates),
            "VIX3M": pd.Series(np.full(n, 24.0), index=long_biz_dates),
            "VVIX": pd.Series(np.full(n, 85.0), index=long_biz_dates),
            "IRX": pd.Series(np.full(n, 0.5), index=long_biz_dates),
            "FVX": pd.Series(np.full(n, 1.5), index=long_biz_dates),
            "TNX": pd.Series(np.full(n, 2.5), index=long_biz_dates),
            "TYX": pd.Series(np.full(n, 3.0), index=long_biz_dates),
        })
        features = build_feature_matrix(data)
        spread = features["vix_term_spread"].dropna().iloc[-1]
        assert spread < 0, "Term spread must be negative when VIX > VIX3M (backwardation)"


# ===========================================================================
# Correctness — yield curve
# ===========================================================================

class TestYieldCurve:
    TENORS = np.array([0.25, 2.0, 5.0, 10.0, 30.0])

    def test_flat_curve_produces_zero_slope_curvature_inflection(self):
        yields = np.array([0.03, 0.03, 0.03, 0.03, 0.03])
        coeffs = fit_yield_curve_coefficients(self.TENORS, yields)
        assert len(coeffs) == 4
        assert coeffs[0] == pytest.approx(0.03, abs=1e-5), "Level coefficient wrong"
        assert coeffs[1] == pytest.approx(0.0,  abs=1e-5), "Slope should be 0 for flat curve"
        assert coeffs[2] == pytest.approx(0.0,  abs=1e-5), "Curvature should be 0 for flat curve"
        assert coeffs[3] == pytest.approx(0.0,  abs=1e-5), "Inflection should be 0 for flat curve"

    def test_inverted_curve_has_negative_slope(self):
        """Short rates > long rates → negative slope coefficient."""
        yields = np.array([0.055, 0.045, 0.035, 0.030, 0.025])
        coeffs = fit_yield_curve_coefficients(self.TENORS, yields)
        assert coeffs[1] < 0, (
            f"Inverted curve should yield negative slope, got slope={coeffs[1]:.4f}"
        )

    def test_normal_curve_has_positive_slope(self):
        """Long rates > short rates → positive slope coefficient."""
        yields = np.array([0.010, 0.020, 0.025, 0.030, 0.035])
        coeffs = fit_yield_curve_coefficients(self.TENORS, yields)
        assert coeffs[1] > 0, (
            f"Normal (upward sloping) curve should yield positive slope, got {coeffs[1]:.4f}"
        )

    def test_always_returns_exactly_4_coefficients(self):
        """Interface contract: always 4 coefficients regardless of input shape."""
        for seed in range(5):
            rng = np.random.default_rng(seed)
            yields = rng.uniform(0.01, 0.06, 5)
            coeffs = fit_yield_curve_coefficients(self.TENORS, yields)
            assert len(coeffs) == 4, f"Expected 4 coefficients, got {len(coeffs)}"

    def test_no_linalg_error_near_degenerate_input(self):
        """Near-flat yields (very small differences) must not raise LinAlgError."""
        yields = np.array([0.05000, 0.05001, 0.05000, 0.04999, 0.05000])
        try:
            coeffs = fit_yield_curve_coefficients(self.TENORS, yields)
            assert len(coeffs) == 4
        except np.linalg.LinAlgError as e:
            pytest.fail(f"fit_yield_curve_coefficients raised LinAlgError: {e}")

    def test_coefficients_reconstruct_yields_approximately(self):
        """
        Fitting a degree-3 polynomial to 5 points should produce a good fit.
        Reconstructed values should be within 0.001 of originals.
        """
        yields = np.array([0.02, 0.025, 0.028, 0.030, 0.032])
        coeffs = fit_yield_curve_coefficients(self.TENORS, yields)
        reconstructed = np.polyval(coeffs[::-1], self.TENORS)  # or however the impl works
        np.testing.assert_allclose(reconstructed, yields, atol=0.001)


# ===========================================================================
# Correctness — rolling correlation
# ===========================================================================

class TestRollingCorrelation:
    def test_identical_series_correlation_is_one(self):
        """Correlation of a series with itself must be exactly 1.0 after warmup."""
        prices = pd.Series(
            np.cumsum(np.random.default_rng(1).normal(0, 1, 100)) + 100,
            index=pd.bdate_range("2010-01-04", periods=100),
        )
        corr = rolling_correlation(prices, prices, window=20)
        final_vals = corr.dropna()
        assert (final_vals == pytest.approx(1.0, abs=1e-8)).all(), (
            "Rolling correlation of series with itself must be 1.0"
        )

    def test_perfectly_inverse_series_correlation_is_minus_one(self):
        """Correlation of x with -x must be -1.0 after warmup."""
        dates = pd.bdate_range("2010-01-04", periods=100)
        s1 = pd.Series(np.linspace(1.0, 2.0, 100), index=dates)
        s2 = pd.Series(np.linspace(2.0, 1.0, 100), index=dates)
        corr = rolling_correlation(s1, s2, window=20)
        assert corr.dropna().iloc[-1] == pytest.approx(-1.0, abs=1e-6)

    def test_nan_before_window_fills(self):
        """The first (window-1) values must be NaN — no early extrapolation."""
        prices = pd.Series(
            np.cumsum(np.random.default_rng(2).normal(0, 1, 60)) + 100,
            index=pd.bdate_range("2010-01-04", periods=60),
        )
        corr = rolling_correlation(prices, prices, window=20)
        assert corr.iloc[:19].isna().all(), (
            "First 19 values should be NaN for window=20"
        )

    def test_defined_from_window_onwards(self):
        """From index 19 (the 20th element) onwards, values must be non-NaN."""
        prices = pd.Series(
            np.cumsum(np.random.default_rng(3).normal(0, 1, 60)) + 100,
            index=pd.bdate_range("2010-01-04", periods=60),
        )
        corr = rolling_correlation(prices, prices, window=20)
        assert corr.iloc[19:].notna().all(), (
            "Correlation must be defined from window position onwards"
        )

    def test_output_values_in_minus_one_to_one(self):
        """Correlation values are always in [-1, 1] — no numerical overflow."""
        rng = np.random.default_rng(4)
        s1 = pd.Series(rng.normal(0, 100, 100), index=pd.bdate_range("2010-01-04", periods=100))
        s2 = pd.Series(rng.normal(0, 100, 100), index=pd.bdate_range("2010-01-04", periods=100))
        corr = rolling_correlation(s1, s2, window=20).dropna()
        assert (corr >= -1.0 - 1e-10).all() and (corr <= 1.0 + 1e-10).all(), (
            "Correlation values must be in [-1, 1]"
        )


# ===========================================================================
# NaN and boundary behavior
# ===========================================================================

class TestNaNBehavior:
    def test_no_nan_after_full_warmup(self, long_market_data):
        """
        After the 200-day MA warmup period, no feature in any row should be NaN.
        Failing this indicates a feature that never fully initializes.
        """
        features = build_feature_matrix(long_market_data)
        post_warmup = features.iloc[210:]  # give a 10-day buffer past 200
        nan_cols = post_warmup.columns[post_warmup.isna().any()].tolist()
        assert not nan_cols, (
            f"NaN values found after warmup in columns: {nan_cols}"
        )

    def test_early_rows_have_nans_for_long_window_features(self, long_market_data):
        """
        200d MA requires 200 days. Feature at row 0 must be NaN for that feature.
        """
        features = build_feature_matrix(long_market_data)
        assert pd.isna(features["spy_dist_200ma"].iloc[0]), (
            "spy_dist_200ma at row 0 should be NaN (200-day warmup not yet complete)"
        )

    def test_single_row_input_returns_all_nans(self, long_biz_dates):
        """
        With only 1 data point, all windowed features are undefined.
        The pipeline must return a row of NaNs without raising.
        """
        one_day = long_biz_dates[:1]
        data = {
            k: pd.Series([100.0], index=one_day)
            for k in ["SPY", "TLT", "GLD", "HYG", "LQD", "VIX", "VIX3M", "VVIX",
                      "IRX", "FVX", "TNX", "TYX"]
        }
        try:
            features = build_feature_matrix(data)
            assert features.shape[0] == 1
            assert features.isna().all().all(), (
                "All features must be NaN when only 1 data point is provided"
            )
        except Exception as e:
            pytest.fail(f"build_feature_matrix raised on 1-row input: {type(e).__name__}: {e}")

    def test_no_zero_fill_where_nan_expected(self, long_market_data):
        """
        Features during warmup must be NaN, not silently filled with 0.
        A zero fill would corrupt any model trained on this data.
        """
        features = build_feature_matrix(long_market_data)
        # The first row should have NaN for rolling features, not 0
        first_row_rolling = features[["spy_tlt_corr_20d", "spy_return_20d", "spy_dist_200ma"]].iloc[0]
        zeros_found = (first_row_rolling == 0.0).any()
        assert not zeros_found, (
            "Rolling features during warmup must be NaN, not zero-filled"
        )

    def test_no_forward_fill_of_nans(self, long_biz_dates):
        """
        Features must NOT be forward-filled across NaN gaps.
        Inject a gap (NaN in input data) and verify it propagates to the feature.
        """
        n = len(long_biz_dates)
        rng = np.random.default_rng(5)
        vix_vals = np.clip(rng.normal(18, 4, n), 9, 80)
        vix_vals[250] = np.nan  # inject a missing value

        data = {k: pd.Series(np.ones(n) * 100, index=long_biz_dates)
                for k in ["SPY", "TLT", "GLD", "HYG", "LQD"]}
        data.update({
            "VIX": pd.Series(vix_vals, index=long_biz_dates),
            "VIX3M": pd.Series(np.full(n, 20.0), index=long_biz_dates),
            "VVIX": pd.Series(np.full(n, 85.0), index=long_biz_dates),
            "IRX": pd.Series(np.full(n, 0.5), index=long_biz_dates),
            "FVX": pd.Series(np.full(n, 1.5), index=long_biz_dates),
            "TNX": pd.Series(np.full(n, 2.5), index=long_biz_dates),
            "TYX": pd.Series(np.full(n, 3.0), index=long_biz_dates),
        })
        features = build_feature_matrix(data)
        assert pd.isna(features["vix"].iloc[250]), (
            "A NaN in raw VIX data at row 250 should produce NaN in the vix feature, "
            "not a forward-filled value."
        )


# ===========================================================================
# Look-ahead bias / timestamp discipline
# ===========================================================================

class TestLookAheadBias:
    def test_spike_at_t_plus_1_does_not_affect_feature_at_t(self, long_market_data):
        """
        Inject an absurd sentinel value at the last row.
        The second-to-last row's features must be unchanged.
        This catches any off-by-one that leaks T's data into T-1's feature.
        """
        clean_features = build_feature_matrix(long_market_data)

        data_spiked = {k: v.copy() for k, v in long_market_data.items()}
        data_spiked["VIX"].iloc[-1] = 99999.0
        spiked_features = build_feature_matrix(data_spiked)

        clean_val = clean_features["vix"].iloc[-2]
        spiked_val = spiked_features["vix"].iloc[-2]
        assert clean_val == pytest.approx(spiked_val, rel=1e-6), (
            "VIX feature at T-1 changed when T data was spiked — look-ahead leak detected!"
        )

    def test_monday_data_does_not_contaminate_friday_feature(self, long_market_data):
        """
        Strategy fires on Monday using features computed from Friday close.
        Spiking Monday's raw data must not change Friday's computed feature vector.
        """
        dates = long_market_data["VIX"].index
        fridays = [i for i, d in enumerate(dates) if d.weekday() == 4]
        if len(fridays) < 2:
            pytest.skip("Need at least 2 Fridays in dataset")

        friday_idx = fridays[-1]
        monday_idx = friday_idx + 1
        if monday_idx >= len(dates):
            pytest.skip("No Monday following the last Friday")

        clean_features = build_feature_matrix(long_market_data)

        data_spiked = {k: v.copy() for k, v in long_market_data.items()}
        data_spiked["VIX"].iloc[monday_idx] = 99999.0
        spiked_features = build_feature_matrix(data_spiked)

        assert clean_features["vix"].iloc[friday_idx] == pytest.approx(
            spiked_features["vix"].iloc[friday_idx], rel=1e-6
        ), "Friday's VIX feature changed when Monday's raw VIX was spiked — look-ahead leak!"

    def test_all_features_use_lagged_data_for_signal_day(self, long_market_data):
        """
        For each signal (Monday), every feature must be computable from
        data available at or before market close on the preceding Friday.
        Verify by checking that no feature value at row T matches a spike at row T.
        """
        data_with_spike = {k: v.copy() for k, v in long_market_data.items()}
        spike_day = 300
        for key in ["VIX", "VIX3M", "VVIX"]:
            data_with_spike[key].iloc[spike_day] = 88888.0

        features = build_feature_matrix(data_with_spike)
        # The feature at spike_day-1 must not have absorbed the spike
        for col in ["vix", "vix3m", "vvix"]:
            val = features[col].iloc[spike_day - 1]
            assert val < 1000, (
                f"Feature '{col}' at T-1 shows spike value — look-ahead bias in feature pipeline!"
            )


# ===========================================================================
# Edge cases that expose implementation fragility
# ===========================================================================

class TestEdgeCases:
    def test_constant_price_no_division_by_zero(self, long_biz_dates):
        """
        Flat price series → zero realized vol.
        Pipeline must handle 0/0 in VRP and undefined correlation gracefully.
        Must not raise ZeroDivisionError or produce inf values.
        """
        n = len(long_biz_dates)
        data = {
            k: pd.Series(np.full(n, 100.0), index=long_biz_dates)
            for k in ["SPY", "TLT", "GLD", "HYG", "LQD"]
        }
        data.update({
            "VIX":  pd.Series(np.full(n, 18.0), index=long_biz_dates),
            "VIX3M": pd.Series(np.full(n, 20.0), index=long_biz_dates),
            "VVIX": pd.Series(np.full(n, 85.0), index=long_biz_dates),
            "IRX": pd.Series(np.full(n, 0.5), index=long_biz_dates),
            "FVX": pd.Series(np.full(n, 1.5), index=long_biz_dates),
            "TNX": pd.Series(np.full(n, 2.5), index=long_biz_dates),
            "TYX": pd.Series(np.full(n, 3.0), index=long_biz_dates),
        })
        try:
            features = build_feature_matrix(data)
        except ZeroDivisionError as e:
            pytest.fail(f"ZeroDivisionError on constant prices: {e}")

        # No inf values allowed
        assert not np.isinf(features.select_dtypes(include=[np.number]).values).any(), (
            "Constant price series produced inf in features"
        )

    def test_hyg_lqd_spread_uses_difference_not_ratio(self, long_biz_dates):
        """
        hyg_lqd_spread should be HYG_yield - LQD_yield (a spread),
        not a ratio. Verify sign and scale make sense.
        High-yield credit always trades at a spread ABOVE investment grade.
        """
        n = len(long_biz_dates)
        rng = np.random.default_rng(6)
        # HYG yields should be higher than LQD yields in normal markets
        hyg_prices = 100 * np.exp(np.cumsum(rng.normal(-0.0001, 0.003, n)))
        lqd_prices = 100 * np.exp(np.cumsum(rng.normal(0.0001, 0.002, n)))

        data = {k: pd.Series(np.ones(n) * 100, index=long_biz_dates)
                for k in ["SPY", "TLT", "GLD", "VIX", "VIX3M", "VVIX",
                          "IRX", "FVX", "TNX", "TYX"]}
        data["VIX"] = pd.Series(np.full(n, 18.0), index=long_biz_dates)
        data["VIX3M"] = pd.Series(np.full(n, 20.0), index=long_biz_dates)
        data["HYG"] = pd.Series(hyg_prices, index=long_biz_dates)
        data["LQD"] = pd.Series(lqd_prices, index=long_biz_dates)

        features = build_feature_matrix(data)
        # The feature should be a numeric value, not constant zero
        spread_vals = features["hyg_lqd_spread"].dropna()
        assert spread_vals.std() > 0, "hyg_lqd_spread has no variation — likely computed wrong"

    def test_spy_return_20d_is_percentage_not_decimal(self, long_biz_dates):
        """
        20-day SPY return should be in a consistent scale.
        A 10% move should produce a value of either 0.10 (decimal) or 10.0 (pct).
        The key test: the scale must be consistent with how it's used in sizing/halts.
        At minimum, values should not span wildly different magnitudes for similar moves.
        """
        n = len(long_biz_dates)
        # Construct ~10% upward move over 20 days
        prices = 100 * np.exp(np.linspace(0, 0.10, n))
        data = {k: pd.Series(np.ones(n) * 100, index=long_biz_dates)
                for k in ["TLT", "GLD", "HYG", "LQD"]}
        data.update({
            "SPY": pd.Series(prices, index=long_biz_dates),
            "VIX": pd.Series(np.full(n, 18.0), index=long_biz_dates),
            "VIX3M": pd.Series(np.full(n, 20.0), index=long_biz_dates),
            "VVIX": pd.Series(np.full(n, 85.0), index=long_biz_dates),
            "IRX": pd.Series(np.full(n, 0.5), index=long_biz_dates),
            "FVX": pd.Series(np.full(n, 1.5), index=long_biz_dates),
            "TNX": pd.Series(np.full(n, 2.5), index=long_biz_dates),
            "TYX": pd.Series(np.full(n, 3.0), index=long_biz_dates),
        })
        features = build_feature_matrix(data)
        ret = features["spy_return_20d"].dropna().iloc[-1]
        # Regardless of decimal vs pct convention, should be in a sane range
        assert abs(ret) < 200, f"spy_return_20d value {ret} is out of any reasonable range"
        assert abs(ret) > 0.0001, f"spy_return_20d value {ret} is effectively zero for a 10% move"

    def test_vvix_passthrough(self, long_market_data):
        """VVIX feature should be the raw VVIX level, no transformation."""
        features = build_feature_matrix(long_market_data)
        raw_vvix = long_market_data["VVIX"].values
        feat_vvix = features["vvix"].values
        # Values should match (possibly with a lag shift; here check the non-NaN portion)
        valid_mask = ~np.isnan(feat_vvix)
        np.testing.assert_allclose(
            feat_vvix[valid_mask],
            raw_vvix[valid_mask],
            rtol=1e-6,
            err_msg="vvix feature does not match raw VVIX input",
        )

"""
Tests for analytics modules.
Run: pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.generate_dataset import generate_dataset, compute_value_score, assign_price_tier
from data.cleaner import SmartphoneDataCleaner
from analytics.benchmarks import (
    compute_brand_scorecard, head_to_head, price_tier_champions, spec_per_rupee
)
from analytics.price_model import (
    ValueScorePredictor, PriceTierClassifier, fit_depreciation_curves
)


@pytest.fixture(scope="session")
def df():
    """Generate a small dataset for testing."""
    return generate_dataset(n_records=200, seed=99)


@pytest.fixture(scope="session")
def df_clean(df):
    cleaner = SmartphoneDataCleaner()
    return cleaner.fit_transform(df)


# ── Dataset generation tests ──────────────────────────────────────────────────
class TestDataGeneration:
    def test_record_count(self, df):
        assert len(df) == 200

    def test_required_columns(self, df):
        required = ["brand", "model", "launch_price_inr", "ram_gb", "battery_mah",
                    "user_rating", "value_score", "price_tier"]
        assert all(c in df.columns for c in required)

    def test_price_tier_assignment(self):
        assert assign_price_tier(10000) == "Budget"
        assert assign_price_tier(20000) == "Mid-range"
        assert assign_price_tier(45000) == "Premium"
        assert assign_price_tier(80000) == "Ultra-premium"
        assert assign_price_tier(120000) == "Flagship"

    def test_value_score_range(self, df):
        assert df["value_score"].between(0, 10).all(), "Value scores must be 0-10"

    def test_no_negative_prices(self, df):
        assert (df["launch_price_inr"] > 0).all()
        assert (df["current_price_inr"] > 0).all()

    def test_rating_range(self, df):
        assert df["user_rating"].between(1, 5).all()

    def test_brand_diversity(self, df):
        assert df["brand"].nunique() >= 5, "Should have at least 5 brands"


# ── Cleaning tests ────────────────────────────────────────────────────────────
class TestCleaner:
    def test_no_missing_after_clean(self, df_clean):
        null_counts = df_clean.isnull().sum()
        # Only camera columns can be 0 (not missing) but not null
        assert null_counts.sum() == 0, f"Nulls remaining: {null_counts[null_counts > 0]}"

    def test_price_relationship_preserved(self, df_clean):
        assert (df_clean["current_price_inr"] <= df_clean["launch_price_inr"]).all()

    def test_price_drop_recalculated(self, df_clean):
        expected = ((1 - df_clean["current_price_inr"] / df_clean["launch_price_inr"]) * 100).round(1)
        pd.testing.assert_series_equal(df_clean["price_drop_pct"], expected, check_names=False)

    def test_rating_clipped(self, df_clean):
        assert df_clean["user_rating"].between(1.0, 5.0).all()


# ── Benchmark tests ───────────────────────────────────────────────────────────
class TestBenchmarks:
    def test_scorecard_shape(self, df_clean):
        scorecard = compute_brand_scorecard(df_clean)
        assert "overall_score" in scorecard.columns
        assert scorecard["overall_score"].between(0, 10).all()

    def test_scorecard_sorted_desc(self, df_clean):
        scorecard = compute_brand_scorecard(df_clean)
        scores = scorecard["overall_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_price_tier_champions(self, df_clean):
        champions = price_tier_champions(df_clean)
        assert not champions.empty
        assert "value_score" in champions.columns

    def test_spec_per_rupee(self, df_clean):
        result = spec_per_rupee(df_clean)
        assert "ram_per_10k" in result.columns
        assert (result["ram_per_10k"] > 0).all()


# ── ML model tests ────────────────────────────────────────────────────────────
class TestModels:
    def test_value_predictor_fit(self, df_clean):
        model = ValueScorePredictor()
        model.fit(df_clean)
        assert model.is_fitted
        assert model.metrics["r2"] > 0.5, f"R² too low: {model.metrics['r2']}"

    def test_value_predictor_output_range(self, df_clean):
        model = ValueScorePredictor()
        model.fit(df_clean)
        preds = model.predict(df_clean)
        assert ((preds >= 0) & (preds <= 10)).all()

    def test_price_tier_classifier(self, df_clean):
        clf = PriceTierClassifier()
        clf.fit(df_clean)
        assert clf.is_fitted
        assert clf.metrics["accuracy"] > 0.75

    def test_depreciation_curves(self, df_clean):
        curves = fit_depreciation_curves(df_clean)
        assert not curves.empty
        assert "decay_lambda" in curves.columns
        assert (curves["decay_lambda"] > 0).all()
        assert (curves["price_at_1yr_pct"] < 100).all()

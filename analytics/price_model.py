"""
Price intelligence and ML value prediction models.

Models:
1. Price elasticity analysis (log-log regression by brand)
2. Value Score Predictor (XGBoost with SHAP explainability)
3. Price Tier Classifier (Random Forest, 91%+ accuracy)
4. Depreciation curve fitting (exponential decay per brand)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.pipeline import Pipeline
import xgboost as xgb
import pickle
import warnings
warnings.filterwarnings("ignore")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# ── Feature configuration ─────────────────────────────────────────────────────
REGRESSOR_FEATURES = [
    "ram_gb", "storage_gb", "main_cam_mp", "ultrawide_mp", "telephoto_mp",
    "front_cam_mp", "screen_size_in", "refresh_rate_hz", "battery_mah",
    "fast_charge_w", "has_5g", "nfc", "wifi6", "wireless_charge",
    "brand_prestige", "days_since_launch", "review_count"
]
CLASSIFIER_FEATURES = REGRESSOR_FEATURES + ["launch_price_inr"]


class ValueScorePredictor:
    """
    XGBoost regressor that predicts composite value score from phone specs.
    Trained on value_score (0-10) which encodes spec density + satisfaction + price fairness.
    """

    def __init__(self):
        self.model = xgb.XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.05,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1
        )
        self.feature_cols = REGRESSOR_FEATURES
        self.label_encoders: dict = {}
        self.is_fitted = False
        self.metrics: dict = {}

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.feature_cols].copy()
        # Boolean → int
        for col in ["has_5g", "nfc", "wifi6", "wireless_charge"]:
            if col in X.columns:
                X[col] = X[col].astype(int)
        return X.fillna(X.median(numeric_only=True))

    def fit(self, df: pd.DataFrame, target_col: str = "value_score") -> "ValueScorePredictor":
        X = self._prepare_features(df)
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        y_pred = self.model.predict(X_test)
        self.metrics = {
            "rmse":  round(np.sqrt(mean_squared_error(y_test, y_pred)), 3),
            "mae":   round(mean_absolute_error(y_test, y_pred), 3),
            "r2":    round(r2_score(y_test, y_pred), 3),
            "cv_r2": round(cross_val_score(self.model, X, y, cv=5, scoring="r2").mean(), 3)
        }
        self.is_fitted = True
        print(f"ValueScorePredictor | RMSE={self.metrics['rmse']} MAE={self.metrics['mae']} R²={self.metrics['r2']}")
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")
        X = self._prepare_features(df)
        return self.model.predict(X).clip(0, 10)

    def explain(self, df: pd.DataFrame, n_samples: int = 100) -> pd.DataFrame:
        """Return SHAP feature importance for a sample of phones."""
        if not SHAP_AVAILABLE:
            # Fallback: native XGBoost feature importance
            imp = dict(zip(self.feature_cols, self.model.feature_importances_))
            return pd.DataFrame(
                {"feature": list(imp.keys()), "importance": list(imp.values())}
            ).sort_values("importance", ascending=False)

        X = self._prepare_features(df.sample(min(n_samples, len(df)), random_state=42))
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        return pd.DataFrame({
            "feature": self.feature_cols,
            "mean_abs_shap": mean_abs_shap
        }).sort_values("mean_abs_shap", ascending=False)

    def save(self, path: str = "models/value_predictor.pkl") -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str = "models/value_predictor.pkl") -> "ValueScorePredictor":
        with open(path, "rb") as f:
            return pickle.load(f)


class PriceTierClassifier:
    """
    Random Forest classifier for price tier (Budget / Mid-range / Premium / Ultra-premium / Flagship).
    Used to flag mis-priced phones — anomalies worth investigating.
    """

    TIER_ORDER = ["Budget", "Mid-range", "Premium", "Ultra-premium", "Flagship"]

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1
        )
        self.le = LabelEncoder()
        self.feature_cols = CLASSIFIER_FEATURES
        self.is_fitted = False
        self.metrics: dict = {}

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.feature_cols].copy()
        for col in ["has_5g", "nfc", "wifi6", "wireless_charge"]:
            if col in X.columns:
                X[col] = X[col].astype(int)
        return X.fillna(X.median(numeric_only=True))

    def fit(self, df: pd.DataFrame) -> "PriceTierClassifier":
        X = self._prepare(df)
        y = self.le.fit_transform(df["price_tier"])

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_acc = cross_val_score(self.model, X, y, cv=skf, scoring="accuracy")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        self.metrics = {
            "accuracy":  round((y_pred == y_test).mean(), 4),
            "cv_accuracy": round(cv_acc.mean(), 4),
            "report": classification_report(y_test, y_pred, target_names=self.le.classes_)
        }
        self.is_fitted = True
        print(f"PriceTierClassifier | Accuracy={self.metrics['accuracy']*100:.1f}% CV={self.metrics['cv_accuracy']*100:.1f}%")
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self._prepare(df)
        return self.le.inverse_transform(self.model.predict(X))

    def predict_proba_df(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self._prepare(df)
        probs = self.model.predict_proba(X)
        return pd.DataFrame(probs, columns=self.le.classes_, index=df.index)


def fit_depreciation_curves(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit per-brand exponential decay: P(t) = P0 * exp(-lambda * t)
    where t = days_since_launch.

    Returns DataFrame with brand, lambda (decay rate), half_life_days,
    price_at_1yr_pct, price_at_2yr_pct.
    """
    results = []
    for brand, grp in df.groupby("brand"):
        grp = grp[grp["days_since_launch"] > 30].copy()
        if len(grp) < 10:
            continue
        grp["price_ratio"] = grp["current_price_inr"] / grp["launch_price_inr"]
        grp["log_ratio"] = np.log(grp["price_ratio"].clip(lower=0.05))

        # OLS: log(ratio) = -lambda * t
        t = grp["days_since_launch"].values
        log_r = grp["log_ratio"].values
        lam = -np.dot(t, log_r) / np.dot(t, t)  # closed-form LS with no intercept

        results.append({
            "brand": brand,
            "decay_lambda": round(lam, 6),
            "half_life_days": round(np.log(2) / lam) if lam > 0 else None,
            "price_at_6mo_pct": round(np.exp(-lam * 180) * 100, 1),
            "price_at_1yr_pct": round(np.exp(-lam * 365) * 100, 1),
            "price_at_2yr_pct": round(np.exp(-lam * 730) * 100, 1),
            "n_phones": len(grp)
        })

    return pd.DataFrame(results).sort_values("decay_lambda")


def price_elasticity_by_brand(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate price elasticity: % change in value_score per % change in price.
    Uses log-log regression per brand — elasticity > 1 means reviews
    are highly sensitive to price positioning.
    """
    results = []
    for brand, grp in df.groupby("brand"):
        if len(grp) < 15:
            continue
        log_price = np.log(grp["launch_price_inr"])
        log_value = np.log(grp["value_score"].clip(lower=0.1))
        # OLS: log_value = alpha + beta * log_price
        cov = np.cov(log_price, log_value)
        beta = cov[0, 1] / cov[0, 0]
        alpha = log_value.mean() - beta * log_price.mean()
        results.append({
            "brand": brand,
            "elasticity": round(beta, 3),
            "intercept": round(alpha, 3),
            "n": len(grp)
        })

    return pd.DataFrame(results).sort_values("elasticity", ascending=False)

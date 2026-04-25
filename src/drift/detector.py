import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class DriftResult:
    drift_detected: bool
    warning_detected: bool
    severity: str
    metric_name: str
    metric_value: float
    threshold_value: float
    description: str


class DDMDetector:
    """
    Drift Detection Method (DDM) — Gama et al. 2004.
    Monitors the error rate of a stream of binary predictions.
    Fires a warning when error rate increases significantly,
    fires a drift alert when it increases further.

    How it works:
    - Tracks running mean and std of prediction errors
    - Warning level: mean + 2*std exceeds minimum seen so far
    - Drift level: mean + 3*std exceeds minimum seen so far
    """
    def __init__(self,
                 warning_threshold: float = 2.0,
                 drift_threshold: float = 3.0,
                 min_samples: int = 30):
        self.warning_threshold = warning_threshold
        self.drift_threshold = drift_threshold
        self.min_samples = min_samples
        self.reset()

    def reset(self):
        self.n = 0
        self.mean = 0.0
        self.variance = 0.0
        self.min_mean_plus_std = float("inf")
        self.in_warning = False

    def update(self, error: int) -> DriftResult:
        """
        Update with a single binary error (1 = wrong, 0 = correct).
        Returns a DriftResult indicating current state.
        """
        self.n += 1

        # Welford online mean and variance
        old_mean = self.mean
        self.mean += (error - self.mean) / self.n
        self.variance += (error - old_mean) * (error - self.mean)

        if self.n < self.min_samples:
            return DriftResult(
                drift_detected=False,
                warning_detected=False,
                severity="none",
                metric_name="error_rate",
                metric_value=self.mean,
                threshold_value=self.drift_threshold,
                description="Insufficient samples"
            )

        std = np.sqrt(self.variance / self.n)
        mean_plus_std = self.mean + std

        # Update minimum
        if mean_plus_std < self.min_mean_plus_std:
            self.min_mean_plus_std = mean_plus_std
            self.in_warning = False

        # Check drift level
        if self.mean + self.drift_threshold * std > \
                self.min_mean_plus_std + self.drift_threshold * np.sqrt(
                    self.min_mean_plus_std * (1 - self.min_mean_plus_std) / self.n
                ):
            self.reset()
            return DriftResult(
                drift_detected=True,
                warning_detected=False,
                severity="critical",
                metric_name="error_rate",
                metric_value=self.mean,
                threshold_value=self.drift_threshold,
                description=f"DDM drift detected — error rate {self.mean:.3f} "
                            f"exceeded drift threshold"
            )

        # Check warning level
        if self.mean + self.warning_threshold * std > \
                self.min_mean_plus_std + self.warning_threshold * np.sqrt(
                    self.min_mean_plus_std * (1 - self.min_mean_plus_std) / self.n
                ):
            self.in_warning = True
            return DriftResult(
                drift_detected=False,
                warning_detected=True,
                severity="warning",
                metric_name="error_rate",
                metric_value=self.mean,
                threshold_value=self.warning_threshold,
                description=f"DDM warning — error rate {self.mean:.3f} "
                            f"rising above baseline"
            )

        return DriftResult(
            drift_detected=False,
            warning_detected=False,
            severity="none",
            metric_name="error_rate",
            metric_value=self.mean,
            threshold_value=self.drift_threshold,
            description="No drift detected"
        )


class SlidingWindowDetector:
    """
    Sliding window drift detector — compares two consecutive windows
    of predictions and flags drift when metrics drop significantly.

    This is simpler than DDM and more interpretable —
    it directly compares AUC/fraud rate between windows.
    Used alongside DDM for complementary signals.
    """
    def __init__(self,
                 window_size: int = 500,
                 auc_drop_threshold: float = 0.05,
                 fraud_rate_shift_threshold: float = 0.02):
        self.window_size = window_size
        self.auc_drop_threshold = auc_drop_threshold
        self.fraud_rate_shift_threshold = fraud_rate_shift_threshold
        self.reference_window = None
        self.reference_auc = None
        self.reference_fraud_rate = None

    def set_reference(self, auc: float, fraud_rate: float):
        """Set the baseline window to compare against."""
        self.reference_auc = auc
        self.reference_fraud_rate = fraud_rate
        print(f"Drift reference set — AUC: {auc:.4f}, "
              f"fraud rate: {fraud_rate:.4f}")

    def check(self, current_auc: float,
              current_fraud_rate: float) -> DriftResult:
        """
        Compare current window metrics against reference.
        Returns DriftResult.
        """
        if self.reference_auc is None:
            self.set_reference(current_auc, current_fraud_rate)
            return DriftResult(
                drift_detected=False,
                warning_detected=False,
                severity="none",
                metric_name="auc",
                metric_value=current_auc,
                threshold_value=self.auc_drop_threshold,
                description="Reference window initialized"
            )

        auc_drop = self.reference_auc - current_auc
        fraud_rate_shift = abs(
            current_fraud_rate - self.reference_fraud_rate
        )

        # AUC dropped significantly
        if auc_drop >= self.auc_drop_threshold:
            return DriftResult(
                drift_detected=True,
                warning_detected=False,
                severity="critical",
                metric_name="auc",
                metric_value=current_auc,
                threshold_value=self.reference_auc - self.auc_drop_threshold,
                description=f"AUC dropped from {self.reference_auc:.4f} "
                            f"to {current_auc:.4f} "
                            f"(drop: {auc_drop:.4f})"
            )

        # AUC warning
        if auc_drop >= self.auc_drop_threshold * 0.6:
            return DriftResult(
                drift_detected=False,
                warning_detected=True,
                severity="warning",
                metric_name="auc",
                metric_value=current_auc,
                threshold_value=self.reference_auc - self.auc_drop_threshold,
                description=f"AUC declining — {self.reference_auc:.4f} "
                            f"to {current_auc:.4f}"
            )

        # Fraud rate shifted significantly
        if fraud_rate_shift >= self.fraud_rate_shift_threshold:
            return DriftResult(
                drift_detected=True,
                warning_detected=False,
                severity="high",
                metric_name="fraud_rate",
                metric_value=current_fraud_rate,
                threshold_value=self.reference_fraud_rate,
                description=f"Fraud rate shifted from "
                            f"{self.reference_fraud_rate:.4f} "
                            f"to {current_fraud_rate:.4f}"
            )

        return DriftResult(
            drift_detected=False,
            warning_detected=False,
            severity="none",
            metric_name="auc",
            metric_value=current_auc,
            threshold_value=self.auc_drop_threshold,
            description="No drift detected"
        )


class SHAPDriftDetector:
    """
    Explanation drift detector — tracks whether the SHAP feature
    importance distribution is shifting over time.

    This catches cases where fraud patterns change but accuracy
    hasn't dropped yet — an early warning signal.

    Uses Jensen-Shannon divergence between consecutive windows
    of mean absolute SHAP values.
    """
    def __init__(self,
                 window_size: int = 200,
                 js_threshold: float = 0.15):
        self.window_size = window_size
        self.js_threshold = js_threshold
        self.reference_shap = None
        self.feature_names = None

    def _js_divergence(self, p: np.ndarray,
                       q: np.ndarray) -> float:
        """Jensen-Shannon divergence between two distributions."""
        p = np.array(p, dtype=float)
        q = np.array(q, dtype=float)

        # Normalize to probability distributions
        p_sum = p.sum()
        q_sum = q.sum()

        if p_sum == 0 or q_sum == 0:
            return 0.0

        p = p / p_sum
        q = q / q_sum

        m = 0.5 * (p + q)

        # Avoid log(0)
        p = np.where(p > 0, p, 1e-10)
        q = np.where(q > 0, q, 1e-10)
        m = np.where(m > 0, m, 1e-10)

        js = 0.5 * np.sum(p * np.log(p / m)) + \
             0.5 * np.sum(q * np.log(q / m))

        return float(np.clip(js, 0, 1))

    def update(self, shap_window: list) -> Optional[DriftResult]:
        """
        shap_window: list of SHAP value dicts from recent predictions.
        Each item is a list of {"feature": str, "shap_value": float}.
        """
        if not shap_window or len(shap_window) < 10:
            return None

        # Build feature importance vector for this window
        feature_totals = {}
        for prediction_shap in shap_window:
            if not prediction_shap:
                continue
            for item in prediction_shap:
                fname = item["feature"]
                val = abs(item["shap_value"])
                feature_totals[fname] = feature_totals.get(fname, 0) + val

        if not feature_totals:
            return None

        current_vector = np.array(list(feature_totals.values()))

        if self.reference_shap is None:
            self.reference_shap = feature_totals
            return DriftResult(
                drift_detected=False,
                warning_detected=False,
                severity="none",
                metric_name="shap_js_divergence",
                metric_value=0.0,
                threshold_value=self.js_threshold,
                description="SHAP reference window initialized"
            )

        # Align features between reference and current
        all_features = set(self.reference_shap.keys()) | \
                       set(feature_totals.keys())
        ref_vec = np.array([
            self.reference_shap.get(f, 0) for f in all_features
        ])
        cur_vec = np.array([
            feature_totals.get(f, 0) for f in all_features
        ])

        js_div = self._js_divergence(ref_vec, cur_vec)

        # Find top shifting features
        shifts = {
            f: abs(feature_totals.get(f, 0) - self.reference_shap.get(f, 0))
            for f in all_features
        }
        top_shifted = sorted(
            shifts.items(), key=lambda x: x[1], reverse=True
        )[:3]
        top_names = [f for f, _ in top_shifted]

        if js_div >= self.js_threshold:
            return DriftResult(
                drift_detected=True,
                warning_detected=False,
                severity="high",
                metric_name="shap_js_divergence",
                metric_value=js_div,
                threshold_value=self.js_threshold,
                description=f"Explanation drift detected — JS divergence: "
                            f"{js_div:.4f}. "
                            f"Top shifting features: {', '.join(top_names)}"
            )

        if js_div >= self.js_threshold * 0.6:
            return DriftResult(
                drift_detected=False,
                warning_detected=True,
                severity="warning",
                metric_name="shap_js_divergence",
                metric_value=js_div,
                threshold_value=self.js_threshold,
                description=f"Explanation shift warning — JS: {js_div:.4f}. "
                            f"Shifting features: {', '.join(top_names)}"
            )

        return DriftResult(
            drift_detected=False,
            warning_detected=False,
            severity="none",
            metric_name="shap_js_divergence",
            metric_value=js_div,
            threshold_value=self.js_threshold,
            description=f"SHAP stable — JS divergence: {js_div:.4f}"
        )
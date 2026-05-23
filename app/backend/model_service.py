from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import DMatrix

from .fft_service import (
    calculate_kinematic_params,
    compute_amplitude_spectrum,
    compute_crest_factor,
    compute_kurtosis,
    compute_peak_value,
    compute_rms,
    extract_band_metrics,
    extract_window_signal,
)

HARMONIC_ORDERS = (1, 2, 3, 4, 5)
HARMONIC_BAND_HALF_WIDTH_HZ = 10.0


class SimplifiedTabularModelService:
    def __init__(self, artifacts_dir: Path, class_map: dict[int, str]):
        self.artifacts_dir = artifacts_dir
        self.class_map = class_map
        self.model = None
        self.feature_columns: list[str] = []
        self.metadata: dict = {}
        self.load_error: str | None = None
        self.model_type: str | None = None
        self._load()

    def _load(self) -> None:
        generic_model_path = self.artifacts_dir / "modelo_rockpi_simplificado.joblib"
        fallback_lightgbm_path = self.artifacts_dir / "modelo_lightgbm_rockpi_simplificado.joblib"
        columns_path = self.artifacts_dir / "feature_columns_rockpi_simplificado.json"
        metadata_path = self.artifacts_dir / "model_metadata_rockpi_simplificado.json"

        model_path = generic_model_path if generic_model_path.exists() else fallback_lightgbm_path
        required_paths = [columns_path, metadata_path]
        missing = [str(path.name) for path in required_paths if not path.exists()]
        if not model_path.exists():
            missing.insert(0, generic_model_path.name)

        if missing:
            self.load_error = f"Model artifacts not found: {', '.join(missing)}"
            self.model = None
            self.feature_columns = []
            self.metadata = {}
            self.model_type = None
            return

        try:
            self.model = joblib.load(model_path)
            self.feature_columns = json.loads(columns_path.read_text(encoding="utf-8"))
            self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.model_type = str(self.metadata.get("model_type") or type(self.model).__name__)
            self.load_error = None
        except Exception as exc:
            self.model = None
            self.feature_columns = []
            self.metadata = {}
            self.model_type = None
            self.load_error = f"Failed to load model artifacts: {exc}"

    @property
    def available(self) -> bool:
        return self.model is not None and not self.load_error

    def _extract_time_features(self, signal: np.ndarray, axis: str) -> dict[str, float]:
        return {
            f"rms_{axis}": compute_rms(signal),
            f"kurtosis_{axis}": compute_kurtosis(signal),
            f"peak_value_{axis}": compute_peak_value(signal),
            f"crest_factor_{axis}": compute_crest_factor(signal),
        }

    def _extract_v2_features_from_signal(self, signal: np.ndarray, axis: str, fm2_hz: float) -> dict[str, float]:
        features = self._extract_time_features(signal, axis)
        freq, amplitude = compute_amplitude_spectrum(signal, apply_hann=True)
        total_spectral_energy = float(np.sum(np.square(amplitude)))

        harmonic_energies: list[float] = []
        harmonic_peaks: list[float] = []

        for harmonic_order in (1, 2, 3):
            center_hz = harmonic_order * fm2_hz
            band_energy, band_peak = extract_band_metrics(freq, amplitude, center_hz, half_width_hz=5.0)
            features[f"energy_rel_fm2_h{harmonic_order}_{axis}"] = float(
                band_energy / (total_spectral_energy + 1e-12)
            )
            features[f"amp_max_fm2_h{harmonic_order}_{axis}"] = band_peak
            harmonic_energies.append(band_energy)
            harmonic_peaks.append(band_peak)

        features[f"ratio_energy_fm2_h2_h1_{axis}"] = float(harmonic_energies[1] / (harmonic_energies[0] + 1e-12))
        features[f"ratio_energy_fm2_h3_h1_{axis}"] = float(harmonic_energies[2] / (harmonic_energies[0] + 1e-12))
        features[f"ratio_amp_fm2_h2_h1_{axis}"] = float(harmonic_peaks[1] / (harmonic_peaks[0] + 1e-12))
        features[f"ratio_amp_fm2_h3_h1_{axis}"] = float(harmonic_peaks[2] / (harmonic_peaks[0] + 1e-12))
        return features

    def _extract_harmonic_features_from_signal(
        self,
        signal: np.ndarray,
        axis: str,
        fm1_hz: float,
        fm2_hz: float,
    ) -> dict[str, float]:
        features = self._extract_time_features(signal, axis)
        freq, amplitude = compute_amplitude_spectrum(signal, apply_hann=True)

        for stage_name, base_frequency_hz in (("fm1", fm1_hz), ("fm2", fm2_hz)):
            for harmonic_order in HARMONIC_ORDERS:
                center_hz = harmonic_order * base_frequency_hz
                band_energy, band_peak = extract_band_metrics(
                    freq,
                    amplitude,
                    center_hz,
                    half_width_hz=HARMONIC_BAND_HALF_WIDTH_HZ,
                )
                features[f"energy_{stage_name}_h{harmonic_order}_{axis}"] = band_energy
                features[f"amp_max_{stage_name}_h{harmonic_order}_{axis}"] = band_peak

        return features

    def _extract_signal_features(
        self,
        signal: np.ndarray,
        axis: str,
        kinematic_params: dict[str, float],
    ) -> dict[str, float]:
        feature_set = str(self.metadata.get("feature_set", "")).strip()
        if feature_set == "tempo_fm2_v2":
            return self._extract_v2_features_from_signal(signal, axis, kinematic_params["fm2"])

        return self._extract_harmonic_features_from_signal(
            signal,
            axis,
            fm1_hz=kinematic_params["fm1"],
            fm2_hz=kinematic_params["fm2"],
        )

    def extract_features_from_sample(self, sample_npz, window_index: int) -> tuple[dict[str, float], float, float]:
        features: dict[str, float] = {}
        window_start_s = 0.0
        window_end_s = 0.0
        rpm_value = float(sample_npz["rpm"].item() if hasattr(sample_npz["rpm"], "item") else sample_npz["rpm"])
        kinematic_params = calculate_kinematic_params(rpm_value)

        for axis in ["x", "y", "z"]:
            signal, window_start_s, window_end_s = extract_window_signal(sample_npz, axis, window_index)
            features.update(self._extract_signal_features(signal, axis, kinematic_params))

        return features, window_start_s, window_end_s

    def _build_input_frame(self, features: dict[str, float]) -> pd.DataFrame:
        missing_features = [name for name in self.feature_columns if name not in features]
        if missing_features:
            raise RuntimeError(f"Missing extracted features for model input: {', '.join(missing_features)}")
        return pd.DataFrame([[features[col] for col in self.feature_columns]], columns=self.feature_columns)

    def predict_window(self, sample_npz, window_index: int) -> dict:
        if not self.available:
            raise RuntimeError(self.load_error or "Model not available.")

        features, window_start_s, window_end_s = self.extract_features_from_sample(sample_npz, window_index)
        X_input = self._build_input_frame(features)

        probabilities = self.model.predict_proba(X_input)[0]
        classes = [int(cls) for cls in self.model.classes_]
        best_position = int(np.argmax(probabilities))
        predicted_class = classes[best_position]

        probability_map = {str(cls): float(prob) for cls, prob in zip(classes, probabilities)}

        return {
            "window_start_s": float(window_start_s),
            "window_end_s": float(window_end_s),
            "predicted_class": int(predicted_class),
            "predicted_class_name": self.class_map[int(predicted_class)],
            "predicted_probability": float(probabilities[best_position]),
            "class_probabilities": probability_map,
            "feature_vector": {name: float(features[name]) for name in self.feature_columns},
        }

    def explain_window(self, sample_npz, window_index: int, top_k: int = 5) -> dict:
        if not self.available:
            raise RuntimeError(self.load_error or "Model not available.")

        features, window_start_s, window_end_s = self.extract_features_from_sample(sample_npz, window_index)
        X_input = self._build_input_frame(features)

        probabilities = self.model.predict_proba(X_input)[0]
        classes = [int(cls) for cls in self.model.classes_]
        best_position = int(np.argmax(probabilities))
        predicted_class = classes[best_position]

        contrib_matrix = self._compute_contributions(X_input, len(classes))
        class_index = classes.index(predicted_class)
        class_contrib = contrib_matrix[class_index]

        shap_values = class_contrib[:-1]
        expected_value = float(class_contrib[-1])
        total_abs_impact = float(np.sum(np.abs(shap_values))) + 1e-12

        order = np.argsort(np.abs(shap_values))[::-1][:top_k]
        top_contributions = []
        for rank, feature_idx in enumerate(order, start=1):
            feature_name = self.feature_columns[int(feature_idx)]
            impact_abs = float(abs(shap_values[int(feature_idx)]))
            top_contributions.append(
                {
                    "rank": rank,
                    "feature": feature_name,
                    "feature_value": float(features[feature_name]),
                    "shap_value": float(shap_values[int(feature_idx)]),
                    "impact_abs": impact_abs,
                    "impact_pct": float((impact_abs / total_abs_impact) * 100.0),
                }
            )

        return {
            "window_start_s": float(window_start_s),
            "window_end_s": float(window_end_s),
            "predicted_class": int(predicted_class),
            "predicted_class_name": self.class_map[int(predicted_class)],
            "predicted_probability": float(probabilities[best_position]),
            "expected_value": expected_value,
            "top_contributions": top_contributions,
        }

    def _compute_contributions(self, X_input: pd.DataFrame, n_classes: int) -> np.ndarray:
        model_type = (self.model_type or "").lower()

        if "xgb" in model_type:
            booster = self.model.get_booster()
            dmatrix = DMatrix(X_input, feature_names=self.feature_columns)
            contrib_raw = booster.predict(dmatrix, pred_contribs=True)
            contrib_array = np.asarray(contrib_raw)

            if contrib_array.ndim == 3 and contrib_array.shape[0] == 1:
                return contrib_array[0]

            if contrib_array.ndim == 2 and contrib_array.shape[0] == 1:
                total_columns = contrib_array.shape[1]
                expected_total = n_classes * (len(self.feature_columns) + 1)
                if total_columns == expected_total:
                    return contrib_array[0].reshape(n_classes, len(self.feature_columns) + 1)
                if total_columns == len(self.feature_columns) + 1:
                    return np.expand_dims(contrib_array[0], axis=0)

            raise RuntimeError(f"Unsupported XGBoost contribution shape: {contrib_array.shape}")

        contrib_raw = self.model.predict(X_input, pred_contrib=True)
        return self._normalize_lightgbm_contrib_output(contrib_raw, n_classes, len(self.feature_columns))

    def _normalize_lightgbm_contrib_output(self, contrib_raw, n_classes: int, n_features: int) -> np.ndarray:
        if isinstance(contrib_raw, list):
            contrib_array = np.stack([np.asarray(item)[0] for item in contrib_raw], axis=0)
            return contrib_array

        contrib_array = np.asarray(contrib_raw)

        if contrib_array.ndim == 3:
            if contrib_array.shape[0] == 1 and contrib_array.shape[1] == n_classes:
                return contrib_array[0]
            if contrib_array.shape[0] == n_classes and contrib_array.shape[1] == 1:
                return contrib_array[:, 0, :]

        if contrib_array.ndim == 2 and contrib_array.shape[0] == 1:
            total_columns = contrib_array.shape[1]
            if total_columns == n_features + 1:
                return np.expand_dims(contrib_array[0], axis=0)
            expected_total = n_classes * (n_features + 1)
            if total_columns == expected_total:
                return contrib_array[0].reshape(n_classes, n_features + 1)

        raise RuntimeError(f"Unsupported LightGBM contribution shape: {contrib_array.shape}")


# Backward-compatible alias for existing imports if needed elsewhere.
SimplifiedLightGBMService = SimplifiedTabularModelService

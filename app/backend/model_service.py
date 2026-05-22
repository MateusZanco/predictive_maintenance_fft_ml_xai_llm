from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .fft_service import compute_kurtosis, compute_peak_value, compute_rms, extract_window_signal


class SimplifiedLightGBMService:
    def __init__(self, artifacts_dir: Path, class_map: dict[int, str]):
        self.artifacts_dir = artifacts_dir
        self.class_map = class_map
        self.model = None
        self.feature_columns: list[str] = []
        self.metadata: dict = {}
        self.load_error: str | None = None
        self._load()

    def _load(self) -> None:
        model_path = self.artifacts_dir / "modelo_lightgbm_rockpi_simplificado.joblib"
        columns_path = self.artifacts_dir / "feature_columns_rockpi_simplificado.json"
        metadata_path = self.artifacts_dir / "model_metadata_rockpi_simplificado.json"

        missing = [str(path.name) for path in [model_path, columns_path, metadata_path] if not path.exists()]
        if missing:
            self.load_error = f"Model artifacts not found: {', '.join(missing)}"
            self.model = None
            self.feature_columns = []
            self.metadata = {}
            return

        try:
            self.model = joblib.load(model_path)
            self.feature_columns = json.loads(columns_path.read_text(encoding="utf-8"))
            self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.load_error = None
        except Exception as exc:
            self.model = None
            self.feature_columns = []
            self.metadata = {}
            self.load_error = f"Failed to load model artifacts: {exc}"

    @property
    def available(self) -> bool:
        return self.model is not None and not self.load_error

    def extract_features_from_sample(self, sample_npz, window_index: int) -> tuple[dict[str, float], float, float]:
        features: dict[str, float] = {}
        window_start_s = 0.0
        window_end_s = 0.0

        for axis in ["x", "y", "z"]:
            signal, window_start_s, window_end_s = extract_window_signal(sample_npz, axis, window_index)
            features[f"rms_{axis}"] = compute_rms(signal)
            features[f"kurtosis_{axis}"] = compute_kurtosis(signal)
            features[f"peak_value_{axis}"] = compute_peak_value(signal)

        return features, window_start_s, window_end_s

    def predict_window(self, sample_npz, window_index: int) -> dict:
        if not self.available:
            raise RuntimeError(self.load_error or "Model not available.")

        features, window_start_s, window_end_s = self.extract_features_from_sample(sample_npz, window_index)
        X_input = pd.DataFrame([[features[col] for col in self.feature_columns]], columns=self.feature_columns)

        probabilities = self.model.predict_proba(X_input)[0]
        classes = [int(cls) for cls in self.model.classes_]
        best_position = int(np.argmax(probabilities))
        predicted_class = classes[best_position]

        probability_map = {
            str(cls): float(prob)
            for cls, prob in zip(classes, probabilities)
        }

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
        X_input = pd.DataFrame([[features[col] for col in self.feature_columns]], columns=self.feature_columns)

        probabilities = self.model.predict_proba(X_input)[0]
        classes = [int(cls) for cls in self.model.classes_]
        best_position = int(np.argmax(probabilities))
        predicted_class = classes[best_position]

        contrib_raw = self.model.predict(X_input, pred_contrib=True)
        contrib_matrix = self._normalize_contrib_output(contrib_raw, len(classes), len(self.feature_columns))
        class_index = classes.index(predicted_class)
        class_contrib = contrib_matrix[class_index]

        shap_values = class_contrib[:-1]
        expected_value = float(class_contrib[-1])

        order = np.argsort(np.abs(shap_values))[::-1][:top_k]
        top_contributions = []
        for rank, feature_idx in enumerate(order, start=1):
            feature_name = self.feature_columns[int(feature_idx)]
            top_contributions.append(
                {
                    "rank": rank,
                    "feature": feature_name,
                    "feature_value": float(features[feature_name]),
                    "shap_value": float(shap_values[int(feature_idx)]),
                    "impact_abs": float(abs(shap_values[int(feature_idx)])),
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

    def _normalize_contrib_output(self, contrib_raw, n_classes: int, n_features: int) -> np.ndarray:
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

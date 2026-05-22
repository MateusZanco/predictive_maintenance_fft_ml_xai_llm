from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SampleSummary(BaseModel):
    sample_id: str
    dataset_operacao: str
    condicao_operacao: str
    rpm: float
    torque_nm: float
    duration_s: float
    available_windows: int
    classe: int | None = None
    classe_nome: str | None = None


class SampleMetadata(SampleSummary):
    fs: int
    points_per_row: int
    row_duration_s: float
    total_rows: int
    total_flat_samples_per_axis: int
    available_axes: list[str]
    fm1: float
    fm2: float


class FftRequest(BaseModel):
    sample_id: str
    axis: Literal["x", "y", "z"] = "x"
    window_index: int = Field(ge=0)
    fmin: float = Field(default=0.0, ge=0.0)
    fmax: float = Field(default=5000.0, gt=0.0)
    apply_hann: bool = True


class SignalRequest(BaseModel):
    sample_id: str
    axis: Literal["x", "y", "z"] = "x"
    window_index: int = Field(ge=0)


class SignalResponse(BaseModel):
    sample_id: str
    dataset_operacao: str
    condicao_operacao: str
    axis: str
    window_index: int
    window_start_s: float
    window_end_s: float
    time: list[float]
    signal: list[float]
    points_per_window: int


class FeatureRequest(BaseModel):
    sample_id: str
    axis: Literal["x", "y", "z"] = "x"
    window_index: int = Field(ge=0)


class FeatureResponse(BaseModel):
    sample_id: str
    dataset_operacao: str
    condicao_operacao: str
    classe: int | None = None
    classe_nome: str | None = None
    axis: str
    window_index: int
    window_start_s: float
    window_end_s: float
    rms: float
    kurtosis: float
    peak_value: float


class PredictRequest(BaseModel):
    sample_id: str
    window_index: int = Field(ge=0)


class PredictResponse(BaseModel):
    sample_id: str
    dataset_operacao: str
    condicao_operacao: str
    classe_real: int | None = None
    classe_real_nome: str | None = None
    window_index: int
    window_start_s: float
    window_end_s: float
    predicted_class: int
    predicted_class_name: str
    predicted_probability: float
    class_probabilities: dict[str, float]
    feature_vector: dict[str, float]


class ShapRequest(BaseModel):
    sample_id: str
    window_index: int = Field(ge=0)
    top_k: int = Field(default=5, ge=1, le=20)


class ShapContribution(BaseModel):
    rank: int
    feature: str
    feature_value: float
    shap_value: float
    impact_abs: float


class ShapResponse(BaseModel):
    sample_id: str
    dataset_operacao: str
    condicao_operacao: str
    classe_real: int | None = None
    classe_real_nome: str | None = None
    window_index: int
    window_start_s: float
    window_end_s: float
    predicted_class: int
    predicted_class_name: str
    predicted_probability: float
    expected_value: float
    top_contributions: list[ShapContribution]


class ExplainRequest(BaseModel):
    sample_id: str
    window_index: int = Field(ge=0)
    top_k: int = Field(default=5, ge=1, le=20)


class ExplainResponse(BaseModel):
    sample_id: str
    dataset_operacao: str
    condicao_operacao: str
    classe_real: int | None = None
    classe_real_nome: str | None = None
    window_index: int
    window_start_s: float
    window_end_s: float
    predicted_class: int
    predicted_class_name: str
    predicted_probability: float
    interpretacao_vibracional: str
    interpretacao_mecanica: str
    system_prompt: str
    user_prompt: str
    raw_response: str


class FftResponse(BaseModel):
    sample_id: str
    dataset_operacao: str
    condicao_operacao: str
    axis: str
    window_index: int
    window_start_s: float
    window_end_s: float
    freq: list[float]
    amp: list[float]
    fm1: float
    fm2: float
    points_per_window: int
    apply_hann: bool

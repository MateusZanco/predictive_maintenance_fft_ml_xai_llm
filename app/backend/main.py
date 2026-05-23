from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .fft_service import (
    AMOSTRAS_POR_JANELA,
    FS,
    calculate_kinematic_params,
    compute_crest_factor,
    compute_fft,
    compute_kurtosis,
    compute_peak_value,
    compute_rms,
    extract_window_signal,
)
from .llm_service import LocalLlamaExplanationService
from .model_service import SimplifiedTabularModelService
from .sample_service import SampleRepository
from .schemas import (
    ExplainRequest,
    ExplainResponse,
    FeatureRequest,
    FeatureResponse,
    FftRequest,
    FftResponse,
    PredictRequest,
    PredictResponse,
    SampleMetadata,
    SampleSummary,
    ShapRequest,
    ShapResponse,
    SignalRequest,
    SignalResponse,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(os.getenv("ROCKPI_STATIC_DIR", PROJECT_ROOT / "app" / "backend" / "static"))
SAMPLE_DIR = Path(os.getenv("ROCKPI_SAMPLE_DIR", PROJECT_ROOT / "outputs" / "rockpi_test_samples"))
MODEL_ARTIFACTS_DIR = Path(
    os.getenv("ROCKPI_MODEL_ARTIFACTS_DIR", PROJECT_ROOT / "outputs" / "model_artifacts_rockpi_simplificado")
)

MAPA_CLASSES = {
    0: "Normal",
    1: "Desgaste Superficial",
    2: "Dente Trincado",
    3: "Dente Lascado",
    4: "Dente Ausente",
}

app = FastAPI(title="Rock Pi FFT Viewer", version="0.1.0")
repo = SampleRepository(SAMPLE_DIR)
model_service = SimplifiedTabularModelService(MODEL_ARTIFACTS_DIR, MAPA_CLASSES)
llm_service = LocalLlamaExplanationService()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "model_available": "true" if model_service.available else "false",
        "model_type": model_service.model_type or "",
    }


@app.get("/api/samples", response_model=list[SampleSummary])
def list_samples() -> list[SampleSummary]:
    return repo.list_samples()


@app.get("/api/samples/{sample_id}/meta", response_model=SampleMetadata)
def sample_meta(sample_id: str) -> SampleMetadata:
    try:
        return repo.get_metadata(sample_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Sample not found: {sample_id}") from exc


@app.post("/api/fft", response_model=FftResponse)
def calculate_fft(request: FftRequest) -> FftResponse:
    try:
        sample_npz = repo.open_sample(request.sample_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Sample not found: {request.sample_id}") from exc

    with sample_npz:
        try:
            signal, window_start_s, window_end_s = extract_window_signal(sample_npz, request.axis, request.window_index)
        except (KeyError, IndexError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        freq, amp = compute_fft(signal, request.fmin, request.fmax, request.apply_hann)
        rpm = float(sample_npz["rpm"].item() if hasattr(sample_npz["rpm"], "item") else sample_npz["rpm"])
        params = calculate_kinematic_params(rpm)
        dataset_operacao = str(sample_npz["dataset_operacao"].item() if hasattr(sample_npz["dataset_operacao"], "item") else sample_npz["dataset_operacao"])
        condicao_operacao = str(sample_npz["condicao_operacao"].item() if hasattr(sample_npz["condicao_operacao"], "item") else sample_npz["condicao_operacao"])

    return FftResponse(
        sample_id=request.sample_id,
        dataset_operacao=dataset_operacao,
        condicao_operacao=condicao_operacao,
        axis=request.axis,
        window_index=request.window_index,
        window_start_s=window_start_s,
        window_end_s=window_end_s,
        freq=freq.tolist(),
        amp=amp.tolist(),
        fm1=params["fm1"],
        fm2=params["fm2"],
        points_per_window=AMOSTRAS_POR_JANELA,
        apply_hann=request.apply_hann,
    )


@app.post("/api/signal", response_model=SignalResponse)
def get_signal(request: SignalRequest) -> SignalResponse:
    try:
        sample_npz = repo.open_sample(request.sample_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Sample not found: {request.sample_id}") from exc

    with sample_npz:
        try:
            signal, window_start_s, window_end_s = extract_window_signal(sample_npz, request.axis, request.window_index)
        except (KeyError, IndexError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        dataset_operacao = str(sample_npz["dataset_operacao"].item() if hasattr(sample_npz["dataset_operacao"], "item") else sample_npz["dataset_operacao"])
        condicao_operacao = str(sample_npz["condicao_operacao"].item() if hasattr(sample_npz["condicao_operacao"], "item") else sample_npz["condicao_operacao"])
        time = np.arange(signal.shape[0], dtype=float) / FS + window_start_s

    return SignalResponse(
        sample_id=request.sample_id,
        dataset_operacao=dataset_operacao,
        condicao_operacao=condicao_operacao,
        axis=request.axis,
        window_index=request.window_index,
        window_start_s=window_start_s,
        window_end_s=window_end_s,
        time=time.tolist(),
        signal=signal.tolist(),
        points_per_window=AMOSTRAS_POR_JANELA,
    )


@app.post("/api/features", response_model=FeatureResponse)
def get_features(request: FeatureRequest) -> FeatureResponse:
    try:
        sample_npz = repo.open_sample(request.sample_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Sample not found: {request.sample_id}") from exc

    with sample_npz:
        try:
            signal, window_start_s, window_end_s = extract_window_signal(sample_npz, request.axis, request.window_index)
        except (KeyError, IndexError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        dataset_operacao = str(sample_npz["dataset_operacao"].item() if hasattr(sample_npz["dataset_operacao"], "item") else sample_npz["dataset_operacao"])
        condicao_operacao = str(sample_npz["condicao_operacao"].item() if hasattr(sample_npz["condicao_operacao"], "item") else sample_npz["condicao_operacao"])
        classe = None
        classe_nome = None
        if "classe" in sample_npz.files:
            classe = int(sample_npz["classe"].item() if hasattr(sample_npz["classe"], "item") else sample_npz["classe"])
        if "classe_nome" in sample_npz.files:
            classe_nome = str(sample_npz["classe_nome"].item() if hasattr(sample_npz["classe_nome"], "item") else sample_npz["classe_nome"])

    return FeatureResponse(
        sample_id=request.sample_id,
        dataset_operacao=dataset_operacao,
        condicao_operacao=condicao_operacao,
        classe=classe,
        classe_nome=classe_nome,
        axis=request.axis,
        window_index=request.window_index,
        window_start_s=window_start_s,
        window_end_s=window_end_s,
        rms=compute_rms(signal),
        kurtosis=compute_kurtosis(signal),
        peak_value=compute_peak_value(signal),
        crest_factor=compute_crest_factor(signal),
    )


@app.post("/api/predict", response_model=PredictResponse)
def predict_window(request: PredictRequest) -> PredictResponse:
    if not model_service.available:
        raise HTTPException(
            status_code=503,
            detail=(
                model_service.load_error
                or "Model artifacts unavailable. Run the simplified training notebook and copy the exported artifacts."
            ),
        )

    try:
        sample_npz = repo.open_sample(request.sample_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Sample not found: {request.sample_id}") from exc

    with sample_npz:
        try:
            prediction = model_service.predict_window(sample_npz, request.window_index)
        except (KeyError, IndexError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        dataset_operacao = str(sample_npz["dataset_operacao"].item() if hasattr(sample_npz["dataset_operacao"], "item") else sample_npz["dataset_operacao"])
        condicao_operacao = str(sample_npz["condicao_operacao"].item() if hasattr(sample_npz["condicao_operacao"], "item") else sample_npz["condicao_operacao"])
        classe_real = None
        classe_real_nome = None
        if "classe" in sample_npz.files:
            classe_real = int(sample_npz["classe"].item() if hasattr(sample_npz["classe"], "item") else sample_npz["classe"])
            classe_real_nome = MAPA_CLASSES.get(classe_real)

    return PredictResponse(
        sample_id=request.sample_id,
        dataset_operacao=dataset_operacao,
        condicao_operacao=condicao_operacao,
        classe_real=classe_real,
        classe_real_nome=classe_real_nome,
        window_index=request.window_index,
        window_start_s=prediction["window_start_s"],
        window_end_s=prediction["window_end_s"],
        predicted_class=prediction["predicted_class"],
        predicted_class_name=prediction["predicted_class_name"],
        predicted_probability=prediction["predicted_probability"],
        class_probabilities=prediction["class_probabilities"],
        feature_vector=prediction["feature_vector"],
    )


@app.post("/api/shap", response_model=ShapResponse)
def explain_window_shap(request: ShapRequest) -> ShapResponse:
    if not model_service.available:
        raise HTTPException(
            status_code=503,
            detail=(
                model_service.load_error
                or "Model artifacts unavailable. Run the simplified training notebook and copy the exported artifacts."
            ),
        )

    try:
        sample_npz = repo.open_sample(request.sample_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Sample not found: {request.sample_id}") from exc

    with sample_npz:
        try:
            explanation = model_service.explain_window(sample_npz, request.window_index, top_k=request.top_k)
        except (KeyError, IndexError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        dataset_operacao = str(sample_npz["dataset_operacao"].item() if hasattr(sample_npz["dataset_operacao"], "item") else sample_npz["dataset_operacao"])
        condicao_operacao = str(sample_npz["condicao_operacao"].item() if hasattr(sample_npz["condicao_operacao"], "item") else sample_npz["condicao_operacao"])
        classe_real = None
        classe_real_nome = None
        if "classe" in sample_npz.files:
            classe_real = int(sample_npz["classe"].item() if hasattr(sample_npz["classe"], "item") else sample_npz["classe"])
            classe_real_nome = MAPA_CLASSES.get(classe_real)

    return ShapResponse(
        sample_id=request.sample_id,
        dataset_operacao=dataset_operacao,
        condicao_operacao=condicao_operacao,
        classe_real=classe_real,
        classe_real_nome=classe_real_nome,
        window_index=request.window_index,
        window_start_s=explanation["window_start_s"],
        window_end_s=explanation["window_end_s"],
        predicted_class=explanation["predicted_class"],
        predicted_class_name=explanation["predicted_class_name"],
        predicted_probability=explanation["predicted_probability"],
        expected_value=explanation["expected_value"],
        top_contributions=explanation["top_contributions"],
    )


@app.post("/api/explain", response_model=ExplainResponse)
def explain_window_llm(request: ExplainRequest) -> ExplainResponse:
    if not model_service.available:
        raise HTTPException(
            status_code=503,
            detail=(
                model_service.load_error
                or "Model artifacts unavailable. Run the simplified training notebook and copy the exported artifacts."
            ),
        )

    try:
        sample_npz = repo.open_sample(request.sample_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Sample not found: {request.sample_id}") from exc

    with sample_npz:
        try:
            shap_explanation = model_service.explain_window(sample_npz, request.window_index, top_k=request.top_k)
            llm_explanation = llm_service.generate_explanation(
                condicao_operacao=str(
                    sample_npz["condicao_operacao"].item()
                    if hasattr(sample_npz["condicao_operacao"], "item")
                    else sample_npz["condicao_operacao"]
                ),
                predicted_class_name=shap_explanation["predicted_class_name"],
                predicted_probability=shap_explanation["predicted_probability"],
                top_contributions=shap_explanation["top_contributions"],
            )
        except (KeyError, IndexError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            mensagem = str(exc)
            if "Timeout" in mensagem:
                raise HTTPException(status_code=504, detail=mensagem) from exc
            raise HTTPException(status_code=502, detail=mensagem) from exc

        dataset_operacao = str(sample_npz["dataset_operacao"].item() if hasattr(sample_npz["dataset_operacao"], "item") else sample_npz["dataset_operacao"])
        condicao_operacao = str(sample_npz["condicao_operacao"].item() if hasattr(sample_npz["condicao_operacao"], "item") else sample_npz["condicao_operacao"])
        classe_real = None
        classe_real_nome = None
        if "classe" in sample_npz.files:
            classe_real = int(sample_npz["classe"].item() if hasattr(sample_npz["classe"], "item") else sample_npz["classe"])
            classe_real_nome = MAPA_CLASSES.get(classe_real)

    return ExplainResponse(
        sample_id=request.sample_id,
        dataset_operacao=dataset_operacao,
        condicao_operacao=condicao_operacao,
        classe_real=classe_real,
        classe_real_nome=classe_real_nome,
        window_index=request.window_index,
        window_start_s=shap_explanation["window_start_s"],
        window_end_s=shap_explanation["window_end_s"],
        predicted_class=shap_explanation["predicted_class"],
        predicted_class_name=shap_explanation["predicted_class_name"],
        predicted_probability=shap_explanation["predicted_probability"],
        interpretacao_vibracional=llm_explanation["interpretacao_vibracional"],
        interpretacao_mecanica=llm_explanation["interpretacao_mecanica"],
        system_prompt=llm_explanation["system_prompt"],
        user_prompt=llm_explanation["user_prompt"],
        raw_response=llm_explanation["raw_response"],
    )


if STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")


@app.get("/{full_path:path}")
def serve_frontend(full_path: str):
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {
        "message": "Frontend build not found.",
        "hint": "Build app/frontend and copy dist to app/backend/static or use Docker.",
    }

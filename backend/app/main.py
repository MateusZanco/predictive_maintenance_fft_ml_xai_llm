from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .llm import gerar_explicacao_local, montar_auditoria_features
from .pipeline import AnalysisArtifacts, AnalysisParams, explain_sample, list_datasets, run_analysis, sample_fft_payload


REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_CACHE: dict[str, tuple[AnalysisArtifacts, dict[str, Any]]] = {}

app = FastAPI(title="Predictive Maintenance Web API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalysisRequest(BaseModel):
    dataset_key: str
    fs: float = Field(default=10_000.0)
    pontos_por_linha: int = Field(default=200)
    duracao_intervalo_s: float = Field(default=1.0)
    largura_busca_fm_real_hz: float = Field(default=10.0)
    largura_banda_harmonica_hz: float = Field(default=10.0)
    ordens_harmonicas_fm: int = Field(default=5)
    ordens_harmonicas_fm1: int = Field(default=5)
    ressonancia_min_hz: float = Field(default=3000.0)
    ressonancia_max_hz: float = Field(default=5000.0)
    test_size: float = Field(default=0.3)
    explanation_samples_per_class: int = Field(default=3)
    shap_top_k: int = Field(default=15)
    random_state: int = Field(default=42)


class ExplainRequest(BaseModel):
    analysis_id: str
    sample_index: int
    shap_source_model: str = Field(default="XGBoost")
    shap_top_k: int = Field(default=15)
    generate_llm: bool = Field(default=True)
    ollama_model: str = Field(default="llama3.1")
    temperature: float = Field(default=0.2)
    num_predict: int = Field(default=500)


class SampleFFTRequest(BaseModel):
    analysis_id: str
    sample_index: int
    frequencia_min_hz: float = Field(default=0.0)
    frequencia_max_hz: float | None = Field(default=5000.0)


def _analysis_id(params: AnalysisRequest) -> str:
    values = params.model_dump()
    return "|".join(f"{key}={values[key]}" for key in sorted(values))


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in value)


def _persist_feature_audit(
    repo_root: Path,
    analysis_id: str,
    explanation_payload: dict[str, Any],
    feature_audit_payload: dict[str, Any],
) -> Path:
    audit_dir = repo_root / "outputs" / "llm_feature_audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    sample_index = int(explanation_payload["sample_index"])
    source_model = _safe_name(str(explanation_payload["source_model"]))
    analysis_slug = _safe_name(analysis_id)[:80]
    file_path = audit_dir / f"{analysis_slug}__sample_{sample_index}__{source_model}.json"
    file_path.write_text(
        json.dumps(feature_audit_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return file_path


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/datasets")
def datasets() -> dict[str, Any]:
    return {"datasets": list_datasets(REPO_ROOT)}


@app.post("/api/analysis")
def analysis(request: AnalysisRequest) -> dict[str, Any]:
    analysis_id = _analysis_id(request)
    if analysis_id not in ANALYSIS_CACHE:
        params = AnalysisParams(**request.model_dump())
        artifacts, summary = run_analysis(REPO_ROOT, params, analysis_id)
        ANALYSIS_CACHE[analysis_id] = (artifacts, summary)
    _, summary = ANALYSIS_CACHE[analysis_id]
    return {"analysis_id": analysis_id, **summary}


@app.post("/api/sample-fft")
def sample_fft(request: SampleFFTRequest) -> dict[str, Any]:
    cached = ANALYSIS_CACHE.get(request.analysis_id)
    if cached is None:
        raise HTTPException(status_code=404, detail="Analise nao encontrada. Execute /api/analysis primeiro.")
    artifacts, _ = cached
    try:
        return sample_fft_payload(
            artifacts,
            sample_index=request.sample_index,
            frequencia_min_hz=request.frequencia_min_hz,
            frequencia_max_hz=request.frequencia_max_hz,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/explain")
def explain(request: ExplainRequest) -> dict[str, Any]:
    cached = ANALYSIS_CACHE.get(request.analysis_id)
    if cached is None:
        raise HTTPException(status_code=404, detail="Analise nao encontrada. Execute /api/analysis primeiro.")

    artifacts, summary = cached
    try:
        explanation_payload = explain_sample(
            artifacts,
            sample_index=request.sample_index,
            shap_source_model=request.shap_source_model,
            top_k=request.shap_top_k,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    feature_audit_payload = montar_auditoria_features(explanation_payload)
    feature_audit_path = _persist_feature_audit(
        REPO_ROOT,
        request.analysis_id,
        explanation_payload,
        feature_audit_payload,
    )

    llm_payload: dict[str, Any] | None = None
    if request.generate_llm:
        try:
            llm_payload = gerar_explicacao_local(
                explanation_payload,
                ollama_model=request.ollama_model,
                temperature=request.temperature,
                num_predict=request.num_predict,
            )
        except Exception as exc:  # noqa: BLE001
            llm_payload = {
                "audit": {
                    "ollama_model": request.ollama_model,
                    "temperature": request.temperature,
                    "num_predict": request.num_predict,
                    "source_model": request.shap_source_model,
                },
                "response_text": "",
                "error": str(exc),
            }

    return {
        "analysis_id": request.analysis_id,
        "analysis_parameters": summary["analysis_parameters"],
        "explanation": explanation_payload,
        "feature_audit": {
            "file_path": str(feature_audit_path),
            "payload": feature_audit_payload,
        },
        "llm": llm_payload,
    }

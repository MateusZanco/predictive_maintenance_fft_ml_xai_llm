from __future__ import annotations

import re
import time
from typing import Any

import ollama


def descrever_feature_tecnica(feature: str, valor_feature: float, shap_value: float) -> str:
    eixo_map = {"x": "X", "y": "Y", "z": "Z"}
    direcao = "increased" if shap_value >= 0 else "reduced"
    tendencia = "stronger evidence" if shap_value >= 0 else "weaker evidence"

    def ordinal_label(valor: str) -> str:
        mapa = {
            "1": "1st",
            "2": "2nd",
            "3": "3rd",
            "4": "4th",
            "5": "5th",
            "6": "6th",
            "7": "7th",
            "8": "8th",
            "9": "9th",
            "10": "10th",
            "11": "11th",
            "12": "12th",
        }
        return mapa.get(str(valor), f"{valor}th")

    match = re.match(r"rms_([xyz])$", feature)
    if match:
        eixo = match.group(1)
        return f"The RMS value on axis {eixo_map[eixo]} was {valor_feature:.6f} and {direcao} the evidence for the predicted class, representing the effective vibration level of the segment."

    match = re.match(r"peak_value_([xyz])$", feature)
    if match:
        eixo = match.group(1)
        return f"The absolute peak value on axis {eixo_map[eixo]} was {valor_feature:.6f} and {direcao} the evidence for the predicted class, indicating the intensity of the highest vibration excursion in the segment."

    match = re.match(r"kurtosis_([xyz])$", feature)
    if match:
        eixo = match.group(1)
        return f"The kurtosis on axis {eixo_map[eixo]} was {valor_feature:.6f} and {direcao} the evidence for the predicted class, reflecting a more or less impulsive vibration pattern."

    match = re.match(r"crest_factor_([xyz])$", feature)
    if match:
        eixo = match.group(1)
        return f"The crest factor on axis {eixo_map[eixo]} was {valor_feature:.6f} and {direcao} the evidence for the predicted class, indicating the relationship between transient peaks and the effective vibration level."

    match = re.match(r"fm_real_([xyz])$", feature)
    if match:
        eixo = match.group(1)
        return f"The detected real meshing frequency of the second stage on axis {eixo_map[eixo]} was {valor_feature:.2f} Hz and {direcao} the evidence for the predicted class, indicating a shift or concentration around the expected mesh component."

    match = re.match(r"amp_fm_h(\d+)_([xyz])$", feature)
    if match:
        harmonico, eixo = match.groups()
        ordinal = ordinal_label(harmonico)
        return f"The summed squared spectral amplitude within the band around the {ordinal} harmonic of the second-stage meshing frequency on axis {eixo_map[eixo]} was {valor_feature:.6f} and {direcao} the evidence for the predicted class, reflecting the vibrational intensity concentrated near this harmonic."

    match = re.match(r"amp_fcsd_(esq|dir)_([xyz])$", feature)
    if match:
        lado, eixo = match.groups()
        side = "lower" if lado == "esq" else "upper"
        return f"The amplitude at the {side} sideband associated with the second-stage distributed fault frequency on axis {eixo_map[eixo]} was {valor_feature:.6f} and {direcao} the evidence for the predicted class, indicating modulation around the normal meshing component."

    match = re.match(r"amp_fcsl_(esq|dir)_([xyz])$", feature)
    if match:
        lado, eixo = match.groups()
        side = "lower" if lado == "esq" else "upper"
        return f"The amplitude at the {side} sideband associated with the second-stage local fault frequency on axis {eixo_map[eixo]} was {valor_feature:.6f} and {direcao} the evidence for the predicted class, suggesting local modulation effects near the mesh frequency."

    match = re.match(r"fm1_real_([xyz])$", feature)
    if match:
        eixo = match.group(1)
        return f"The detected real meshing frequency of the first stage on axis {eixo_map[eixo]} was {valor_feature:.2f} Hz and {direcao} the evidence for the predicted class, acting as a control indicator for first-stage spectral activity."

    match = re.match(r"amp_fm1_h(\d+)_([xyz])$", feature)
    if match:
        harmonico, eixo = match.groups()
        ordinal = ordinal_label(harmonico)
        return f"The summed squared spectral amplitude within the band around the {ordinal} harmonic of the first-stage meshing frequency on axis {eixo_map[eixo]} was {valor_feature:.6f} and {direcao} the evidence for the predicted class, representing control information from the first-stage mesh response."

    match = re.match(r"amp_fcsd1_dir_([xyz])$", feature)
    if match:
        eixo = match.group(1)
        return f"The amplitude at the upper sideband associated with the first-stage distributed fault frequency on axis {eixo_map[eixo]} was {valor_feature:.6f} and {direcao} the evidence for the predicted class, indicating modulation linked to the first-stage mesh."

    match = re.match(r"amp_fcsl1_dir_([xyz])$", feature)
    if match:
        eixo = match.group(1)
        return f"The amplitude at the upper sideband associated with the first-stage local fault frequency on axis {eixo_map[eixo]} was {valor_feature:.6f} and {direcao} the evidence for the predicted class, indicating localized modulation linked to the first-stage mesh."

    match = re.match(r"energia_ressonancia_([xyz])$", feature)
    if match:
        eixo = match.group(1)
        return f"The summed squared spectral amplitude in the 2000 to 5000 Hz resonance band on axis {eixo_map[eixo]} was {valor_feature:.6f} and {direcao} the evidence for the predicted class, reflecting high-frequency vibrational excitation and possible structural impact response."

    return f"Feature {feature} had a value of {valor_feature:.6f} and {direcao} the evidence for the predicted class, contributing to {tendencia} in the diagnostic decision."


def traduzir_feature_para_llm(item: dict[str, Any]) -> str:
    leitura = descrever_feature_tecnica(item["feature"], item["valor_feature"], item["shap_value"])
    efeito = "supports the predicted class" if item["shap_value"] >= 0 else "weakens the predicted class"
    return (
        f"{int(item['rank'])}. {leitura} "
        f"Diagnostic effect: {efeito}. "
        f"Absolute impact: {float(item['impacto_absoluto']):.6f}."
    )


def montar_prompts(explanation_payload: dict[str, Any]) -> dict[str, str]:
    meta = explanation_payload["sample_metadata"]
    top_features_traduzidas_llm = "\n".join(
        traduzir_feature_para_llm(item) for item in explanation_payload["top_features"]
    )

    legenda_fisica_variaveis_llm = """
Physical legend of variables:
- RMS: effective vibration level of the segment.
- peak_value: highest absolute vibration peak in the segment.
- kurtosis: impulsiveness indicator; higher values suggest more impulsive events.
- crest_factor: ratio between absolute peak and RMS; higher values suggest stronger transients.
- fm_real: detected real meshing frequency of the second stage.
- amp_fm_hN: summed squared spectral amplitude around the Nth harmonic of the second-stage meshing frequency.
- amp_fcsd_esq/dir: lower or upper sideband amplitude associated with the distributed fault frequency of the second stage.
- amp_fcsl_esq/dir: lower or upper sideband amplitude associated with the local fault frequency of the second stage.
- fm1_real: detected real meshing frequency of the first stage, used as a control reference.
- amp_fm1_hN: summed squared spectral amplitude around the Nth harmonic of the first-stage meshing frequency.
- amp_fcsd1_dir and amp_fcsl1_dir: first-stage sideband amplitudes used as control references.
- energia_ressonancia: summed squared spectral amplitude in the 2000 to 5000 Hz resonance band.
"""

    instrucoes_llm = f"""
You are a predictive maintenance specialist for industrial rotating equipment.
Write the explanation in Brazilian Portuguese.
Use concise and technically precise maintenance language.
Do not mention SHAP, feature importance, AI, model internals, or prompt instructions.
Your response must contain exactly the following two section headers, in this exact order:
Interpretação Vibracional:
Interpretação Mecânica:
Do not add any other headers, bullet lists, or recommendation sections.
Base the explanation only on the predicted class, predicted probability, and translated evidence provided in the prompt.
Be direct and objective.
Do not invent causes, components, frequencies, inspection findings, or actions that are not supported by the provided evidence.
If the evidence is insufficient for a stronger conclusion, say that the indication is inconclusive or suggestive rather than certain.
Do not mention parts or subsystems unless they are justified by the predicted class or by the provided evidence.
In Interpretação Vibracional, explain the vibration evidence and indicate the most relevant axes and frequency components or bands.
In Interpretação Mecânica, describe the most likely mechanical interpretation and keep the statement proportional to the evidence.
Keep the full response concise, factual, and grounded in the evidence.
\n{legenda_fisica_variaveis_llm}
"""

    prompt_usuario = f"""
Generate a maintenance-oriented explanation for the following predictive maintenance case.

Model source: {explanation_payload['source_model']}
Predicted class: {int(meta['classe_predita'])} - {meta['classe_predita_nome']}
Predicted probability: {float(meta['probabilidade_predita']):.4f}

Translated evidence from the predictive model:
{top_features_traduzidas_llm}
"""

    return {
        "system_prompt": instrucoes_llm.strip(),
        "user_prompt": prompt_usuario.strip(),
        "evidencias_traduzidas": top_features_traduzidas_llm,
    }


def montar_auditoria_features(explanation_payload: dict[str, Any]) -> dict[str, Any]:
    features_auditadas = []
    for item in explanation_payload["top_features"]:
        features_auditadas.append(
            {
                "rank": int(item["rank"]),
                "feature": item["feature"],
                "valor_feature": float(item["valor_feature"]),
                "shap_value": float(item["shap_value"]),
                "impacto_absoluto": float(item["impacto_absoluto"]),
                "leitura_tecnica": descrever_feature_tecnica(
                    item["feature"],
                    item["valor_feature"],
                    item["shap_value"],
                ),
                "evidencia_llm": traduzir_feature_para_llm(item),
            }
        )

    return {
        "source_model": explanation_payload["source_model"],
        "sample_index": int(explanation_payload["sample_index"]),
        "sample_metadata": explanation_payload["sample_metadata"],
        "features_auditadas": features_auditadas,
    }


def gerar_explicacao_local(
    explanation_payload: dict[str, Any],
    ollama_model: str,
    temperature: float,
    num_predict: int,
) -> dict[str, Any]:
    prompts = montar_prompts(explanation_payload)
    started_at = time.perf_counter()
    resposta = ollama.chat(
        model=ollama_model,
        messages=[
            {"role": "system", "content": prompts["system_prompt"]},
            {"role": "user", "content": prompts["user_prompt"]},
        ],
        options={
            "temperature": temperature,
            "num_predict": num_predict,
        },
    )
    elapsed_ms = (time.perf_counter() - started_at) * 1000.0
    content = resposta["message"]["content"].strip()
    prompt_eval_count = resposta.get("prompt_eval_count")
    eval_count = resposta.get("eval_count")
    prompt_eval_duration_ns = resposta.get("prompt_eval_duration")
    eval_duration_ns = resposta.get("eval_duration")
    total_duration_ns = resposta.get("total_duration")
    load_duration_ns = resposta.get("load_duration")

    performance = {
        "latency_ms": round(elapsed_ms, 2),
        "prompt_tokens": int(prompt_eval_count) if prompt_eval_count is not None else None,
        "response_tokens": int(eval_count) if eval_count is not None else None,
        "total_tokens": (
            int(prompt_eval_count) + int(eval_count)
            if prompt_eval_count is not None and eval_count is not None
            else None
        ),
        "prompt_eval_duration_ms": (
            round(float(prompt_eval_duration_ns) / 1_000_000.0, 2)
            if prompt_eval_duration_ns is not None
            else None
        ),
        "response_eval_duration_ms": (
            round(float(eval_duration_ns) / 1_000_000.0, 2)
            if eval_duration_ns is not None
            else None
        ),
        "total_duration_ms": (
            round(float(total_duration_ns) / 1_000_000.0, 2)
            if total_duration_ns is not None
            else None
        ),
        "load_duration_ms": (
            round(float(load_duration_ns) / 1_000_000.0, 2)
            if load_duration_ns is not None
            else None
        ),
        "tokens_per_second": (
            round(float(eval_count) / (float(eval_duration_ns) / 1_000_000_000.0), 2)
            if eval_count is not None and eval_duration_ns not in (None, 0)
            else None
        ),
        "done": resposta.get("done"),
        "done_reason": resposta.get("done_reason"),
        "created_at": resposta.get("created_at"),
    }
    return {
        "audit": {
            "ollama_model": ollama_model,
            "temperature": temperature,
            "num_predict": num_predict,
            "source_model": explanation_payload["source_model"],
            "system_prompt": prompts["system_prompt"],
            "user_prompt": prompts["user_prompt"],
            "evidencias_traduzidas": prompts["evidencias_traduzidas"],
        },
        "performance": performance,
        "raw_usage": {
            "prompt_eval_count": prompt_eval_count,
            "eval_count": eval_count,
            "prompt_eval_duration_ns": prompt_eval_duration_ns,
            "eval_duration_ns": eval_duration_ns,
            "total_duration_ns": total_duration_ns,
            "load_duration_ns": load_duration_ns,
        },
        "response_text": content,
    }

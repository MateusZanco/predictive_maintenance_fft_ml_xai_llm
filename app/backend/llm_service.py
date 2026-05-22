from __future__ import annotations

import json
import os
import socket
from typing import Any
from urllib import error, request


class LocalLlamaExplanationService:
    def __init__(self) -> None:
        self.chat_completions_url = os.getenv("LLM_CHAT_COMPLETIONS_URL", "http://127.0.0.1:8080/v1/chat/completions")
        self.model_name = os.getenv("LLM_MODEL", "").strip()
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
        self.top_p = float(os.getenv("LLM_TOP_P", "0.8"))
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "180"))
        self.timeout_seconds = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))

    def build_messages(
        self,
        condicao_operacao: str,
        predicted_class_name: str,
        predicted_probability: float,
        top_contributions: list[dict[str, Any]],
        concise: bool = False,
    ) -> tuple[str, str]:
        system_prompt = """You are a specialist in vibration analysis for planetary gearboxes.

Rules:
- Use only the provided evidence.
- The operating condition is context only and must not be treated as fault evidence.
- Do not invent frequencies, causes, severity, extreme conditions, stress, tension, or symptoms.
- Do not describe a variable as high, low, normal, abnormal, extreme, critical, within range, or outside range without an explicit reference.
- Do not mention mechanical stress, imbalance, misalignment, wear progression, looseness, or other specific mechanisms unless they are explicitly provided in the evidence.
- If the evidence is limited, use cautious language equivalent to "compatible with" or "suggests".
- Do not mention SHAP, model, AI, or prompt.
- Do not provide recommended actions.
- Write the final answer only in Brazilian Portuguese.
- Do not use English words or English sentences in the final answer.
- Use at most 2 short sentences in each field.
- Return only valid JSON in this format:
{
  "interpretacao_vibracional": "...",
  "interpretacao_mecanica": "..."
}"""

        evidencias = []
        for idx, item in enumerate(top_contributions, start=1):
            evidencias.append(f"{idx}. {self._describe_contribution(item)}")

        user_prompt = f"""/no_think
Equipment: two-stage planetary gearbox.
Monitored component: second-stage sun gear.
Operating condition: {condicao_operacao}.
Predicted class: {predicted_class_name}.
Predicted probability: {predicted_probability:.4f}.

Variable legend:
- RMS: global vibration level of the time segment on one axis.
- kurtosis: impulsiveness / tail-heaviness of the time signal on one axis.
- peak value: highest absolute amplitude observed in the time segment on one axis.
- "contributed positively to the predicted class": this variable pushed the model toward the predicted class.
- "contributed negatively to the predicted class": this variable pushed the model away from the predicted class.

Observed evidence for this window:
{chr(10).join(evidencias)}

Explain what was observed in the signal and what this suggests mechanically.
Do not treat the operating condition as evidence.
Do not extrapolate beyond the provided variables.
In interpretacao_vibracional, only describe the observed variables and axes.
In interpretacao_mecanica, state only whether the evidence is compatible with the predicted class, with cautious wording if needed.
Do not compare with normal ranges or reference limits.
{"Be even more brief and direct." if concise else "Generate the final explanation."}"""

        return system_prompt, user_prompt

    def generate_explanation(
        self,
        condicao_operacao: str,
        predicted_class_name: str,
        predicted_probability: float,
        top_contributions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        system_prompt, user_prompt = self.build_messages(
            condicao_operacao=condicao_operacao,
            predicted_class_name=predicted_class_name,
            predicted_probability=predicted_probability,
            top_contributions=top_contributions,
        )

        content = self._chat_completion(system_prompt, user_prompt, self.max_tokens)
        try:
            explanation_json = self._extract_json_object(content)
            raw_response = content
        except RuntimeError:
            retry_system_prompt, retry_user_prompt = self.build_messages(
                condicao_operacao=condicao_operacao,
                predicted_class_name=predicted_class_name,
                predicted_probability=predicted_probability,
                top_contributions=top_contributions,
                concise=True,
            )
            retry_max_tokens = min(max(self.max_tokens + 80, int(self.max_tokens * 1.5)), 400)
            retry_content = self._chat_completion(retry_system_prompt, retry_user_prompt, retry_max_tokens)
            explanation_json = self._extract_json_object(retry_content)
            system_prompt = retry_system_prompt
            user_prompt = retry_user_prompt
            raw_response = retry_content

        return {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "raw_response": raw_response,
            "interpretacao_vibracional": str(explanation_json.get("interpretacao_vibracional", "")).strip(),
            "interpretacao_mecanica": str(explanation_json.get("interpretacao_mecanica", "")).strip(),
        }

    def _chat_completion(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        payload: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if self.model_name:
            payload["model"] = self.model_name

        req = request.Request(
            self.chat_completions_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"llama-server HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Falha ao conectar ao llama-server em {self.chat_completions_url}: {exc}") from exc
        except TimeoutError as exc:
            raise RuntimeError(
                f"Timeout ao aguardar resposta do llama-server após {self.timeout_seconds:.0f} s."
            ) from exc
        except socket.timeout as exc:
            raise RuntimeError(
                f"Timeout ao aguardar resposta do llama-server após {self.timeout_seconds:.0f} s."
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Falha inesperada ao consultar o llama-server: {exc}") from exc

        try:
            parsed = json.loads(raw)
            return parsed["choices"][0]["message"]["content"]
        except Exception as exc:
            raise RuntimeError(f"Resposta inválida do llama-server: {raw}") from exc

    def _describe_contribution(self, item: dict[str, Any]) -> str:
        feature = str(item["feature"])
        value = float(item["feature_value"])
        shap_value = float(item["shap_value"])
        direction = "positively" if shap_value >= 0 else "negatively"

        if feature.startswith("rms_"):
            axis = feature.split("_")[-1].upper()
            return f"RMS on axis {axis} = {value:.6f}; contributed {direction} to the predicted class."
        if feature.startswith("kurtosis_"):
            axis = feature.split("_")[-1].upper()
            return f"kurtosis on axis {axis} = {value:.6f}; contributed {direction} to the predicted class."
        if feature.startswith("peak_value_"):
            axis = feature.split("_")[-1].upper()
            return f"peak value on axis {axis} = {value:.6f}; contributed {direction} to the predicted class."

        return f"{feature} = {value:.6f}; contributed {direction} to the predicted class."

    def _extract_json_object(self, text: str) -> dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise RuntimeError(f"Não foi possível extrair JSON da resposta do modelo: {text}")

        json_text = cleaned[start : end + 1]
        try:
            return json.loads(json_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"JSON inválido retornado pelo modelo: {json_text}") from exc

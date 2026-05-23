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
    ) -> tuple[str, str, list[dict[str, str]]]:
        system_prompt = """You are a specialist in vibration analysis for planetary gearboxes.

Rules:
- Use only the provided evidence.
- The operating condition is context only and must not be treated as fault evidence.
- Do not invent frequencies, causes, severity, extreme conditions, mechanical stress, tension, or symptoms.
- Do not describe a variable as high, low, normal, abnormal, extreme, critical, within range, or outside range without an explicit reference.
- Do not mention imbalance, misalignment, looseness, wear progression, mechanical stress, or other specific mechanisms unless they are explicitly supported by the evidence.
- If the evidence is limited, use cautious language such as "compatible with" or "suggests".
- Do not mention SHAP, model, AI, or prompt.
- Do not provide recommended actions.
- Write the final answer only in Brazilian Portuguese.
- Return only valid JSON in this format:
{
  "interpretacao_vibracional": "...",
  "interpretacao_mecanica": "..."
}"""

        evidence_lines = []
        for idx, item in enumerate(top_contributions, start=1):
            evidence_lines.append(f"{idx}. {self._describe_contribution(item)}")

        user_prompt = f"""/no_think
Equipment: two-stage planetary gearbox.
Monitored component: second-stage sun gear.
Operating condition: {condicao_operacao}.
Predicted class: {predicted_class_name}.
Predicted class probability: {predicted_probability:.4f}.

Variable legend:
- RMS: global vibration level of the segment on the analyzed axis.
- kurtosis: impulsiveness of the time signal on the analyzed axis.
- peak value: highest absolute amplitude observed in the segment on the analyzed axis.
- crest factor: ratio between peak value and RMS on the analyzed axis.
- energy around an Fm1 harmonic: spectral energy inside a +/-10 Hz band around a first-stage gear mesh harmonic.
- energy around an Fm2 harmonic: spectral energy inside a +/-10 Hz band around a second-stage gear mesh harmonic.
- maximum amplitude around an Fm1 harmonic: highest spectral amplitude inside a +/-10 Hz band around an Fm1 harmonic.
- maximum amplitude around an Fm2 harmonic: highest spectral amplitude inside a +/-10 Hz band around an Fm2 harmonic.
- "contributed positively to the predicted class": this variable pushed the model toward the predicted class.
- "contributed negatively to the predicted class": this variable pushed the model away from the predicted class.

Observed evidence for this window:
{chr(10).join(evidence_lines)}

Explain what was observed in the signal and what this suggests mechanically.
Do not treat the operating condition as evidence.
Do not extrapolate beyond the provided variables.
In interpretacao_vibracional, describe only the observed variables and associated axes.
In interpretacao_mecanica, state only whether the evidence is compatible with the predicted class, using cautious wording when needed.
Do not compare with normal ranges, reference limits, or expected values.
{"Be even more brief and direct." if concise else "Generate the final explanation."}"""

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self._few_shot_examples())
        messages.append({"role": "user", "content": user_prompt})
        return system_prompt, user_prompt, messages

    def generate_explanation(
        self,
        condicao_operacao: str,
        predicted_class_name: str,
        predicted_probability: float,
        top_contributions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        system_prompt, user_prompt, messages = self.build_messages(
            condicao_operacao=condicao_operacao,
            predicted_class_name=predicted_class_name,
            predicted_probability=predicted_probability,
            top_contributions=top_contributions,
        )

        content = self._chat_completion(messages, self.max_tokens)
        try:
            explanation_json = self._extract_json_object(content)
            raw_response = content
        except RuntimeError:
            retry_system_prompt, retry_user_prompt, retry_messages = self.build_messages(
                condicao_operacao=condicao_operacao,
                predicted_class_name=predicted_class_name,
                predicted_probability=predicted_probability,
                top_contributions=top_contributions,
                concise=True,
            )
            retry_max_tokens = min(max(self.max_tokens + 80, int(self.max_tokens * 1.5)), 400)
            retry_content = self._chat_completion(retry_messages, retry_max_tokens)
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

    def _chat_completion(self, messages: list[dict[str, str]], max_tokens: int) -> str:
        payload: dict[str, Any] = {
            "messages": messages,
            "cache_prompt": True,
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
                f"Timeout ao aguardar resposta do llama-server apos {self.timeout_seconds:.0f} s."
            ) from exc
        except socket.timeout as exc:
            raise RuntimeError(
                f"Timeout ao aguardar resposta do llama-server apos {self.timeout_seconds:.0f} s."
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Falha inesperada ao consultar o llama-server: {exc}") from exc

        try:
            parsed = json.loads(raw)
            return parsed["choices"][0]["message"]["content"]
        except Exception as exc:
            raise RuntimeError(f"Resposta invalida do llama-server: {raw}") from exc

    def _few_shot_examples(self) -> list[dict[str, str]]:
        example_1_user = """/no_think
Equipment: two-stage planetary gearbox.
Monitored component: second-stage sun gear.
Operating condition: 1500 rpm / 10 Nm.
Predicted class: Desgaste Superficial.
Predicted class probability: 0.9810.

Variable legend:
- RMS: global vibration level of the segment on the analyzed axis.
- kurtosis: impulsiveness of the time signal on the analyzed axis.
- peak value: highest absolute amplitude observed in the segment on the analyzed axis.
- crest factor: ratio between peak value and RMS on the analyzed axis.
- energy around an Fm1 harmonic: spectral energy inside a +/-10 Hz band around a first-stage gear mesh harmonic.
- energy around an Fm2 harmonic: spectral energy inside a +/-10 Hz band around a second-stage gear mesh harmonic.
- maximum amplitude around an Fm1 harmonic: highest spectral amplitude inside a +/-10 Hz band around an Fm1 harmonic.
- maximum amplitude around an Fm2 harmonic: highest spectral amplitude inside a +/-10 Hz band around an Fm2 harmonic.
- "contributed positively to the predicted class": this variable pushed the model toward the predicted class.
- "contributed negatively to the predicted class": this variable pushed the model away from the predicted class.

Observed evidence for this window:
1. RMS on axis Y = 0.207204; contributed positively to the predicted class; approximate local importance = 31.0%.
2. peak value on axis Z = 1.153589; contributed positively to the predicted class; approximate local importance = 27.5%.
3. peak value on axis Y = 0.969533; contributed positively to the predicted class; approximate local importance = 22.4%.
4. kurtosis on axis Y = 3.127697; contributed positively to the predicted class; approximate local importance = 12.8%.

Explain what was observed in the signal and what this suggests mechanically.
Do not treat the operating condition as evidence.
Do not extrapolate beyond the provided variables.
In interpretacao_vibracional, describe only the observed variables and associated axes.
In interpretacao_mecanica, state only whether the evidence is compatible with the predicted class, using cautious wording when needed.
Do not compare with normal ranges, reference limits, or expected values.
Generate the final explanation."""

        example_1_assistant = """{
  "interpretacao_vibracional": "A janela apresenta contribuicoes relevantes de RMS, peak value e kurtosis, principalmente nos eixos Y e Z. Essas variaveis indicam alteracao no nivel global de vibracao, na amplitude de pico e na impulsividade do sinal nessa janela.",
  "interpretacao_mecanica": "O conjunto de evidencias e compativel com a classe predita de Desgaste Superficial. A interpretacao deve ser feita com cautela, pois as evidencias fornecidas descrevem indicadores do sinal e nao um mecanismo mecanico especifico."
}"""

        example_2_user = """/no_think
Equipment: two-stage planetary gearbox.
Monitored component: second-stage sun gear.
Operating condition: 2700 rpm / 25 Nm.
Predicted class: Dente Trincado.
Predicted class probability: 0.9925.

Variable legend:
- RMS: global vibration level of the segment on the analyzed axis.
- kurtosis: impulsiveness of the time signal on the analyzed axis.
- peak value: highest absolute amplitude observed in the segment on the analyzed axis.
- crest factor: ratio between peak value and RMS on the analyzed axis.
- energy around an Fm1 harmonic: spectral energy inside a +/-10 Hz band around a first-stage gear mesh harmonic.
- energy around an Fm2 harmonic: spectral energy inside a +/-10 Hz band around a second-stage gear mesh harmonic.
- maximum amplitude around an Fm1 harmonic: highest spectral amplitude inside a +/-10 Hz band around an Fm1 harmonic.
- maximum amplitude around an Fm2 harmonic: highest spectral amplitude inside a +/-10 Hz band around an Fm2 harmonic.
- "contributed positively to the predicted class": this variable pushed the model toward the predicted class.
- "contributed negatively to the predicted class": this variable pushed the model away from the predicted class.

Observed evidence for this window:
1. energy around harmonic 1 of Fm2 on axis X = 0.004812; contributed positively to the predicted class; approximate local importance = 29.4%.
2. maximum amplitude around harmonic 2 of Fm2 on axis Z = 0.021334; contributed positively to the predicted class; approximate local importance = 24.7%.
3. energy around harmonic 3 of Fm1 on axis Y = 0.002105; contributed positively to the predicted class; approximate local importance = 18.2%.
4. crest factor on axis Z = 5.184220; contributed positively to the predicted class; approximate local importance = 14.6%.

Explain what was observed in the signal and what this suggests mechanically.
Do not treat the operating condition as evidence.
Do not extrapolate beyond the provided variables.
In interpretacao_vibracional, describe only the observed variables and associated axes.
In interpretacao_mecanica, state only whether the evidence is compatible with the predicted class, using cautious wording when needed.
Do not compare with normal ranges, reference limits, or expected values.
Generate the final explanation."""

        example_2_assistant = """{
  "interpretacao_vibracional": "A janela mostra contribuicoes de energia e amplitude maxima em harmonicas de Fm1 e Fm2, alem de crest factor no eixo Z. Essas evidencias apontam para participacao de componentes harmonicas do engrenamento e para alteracoes na resposta temporal e espectral do segmento.",
  "interpretacao_mecanica": "O conjunto de evidencias e compativel com a classe predita de Dente Trincado. Ainda assim, a interpretacao mecanica deve ser vista com cautela, porque as evidencias fornecidas representam contribuicoes de variaveis e nao uma confirmacao fisica direta do mecanismo de falha."
}"""

        return [
            {"role": "user", "content": example_1_user},
            {"role": "assistant", "content": example_1_assistant},
            {"role": "user", "content": example_2_user},
            {"role": "assistant", "content": example_2_assistant},
        ]

    def _describe_contribution(self, item: dict[str, Any]) -> str:
        feature = str(item["feature"])
        value = float(item["feature_value"])
        shap_value = float(item["shap_value"])
        impact_pct = float(item.get("impact_pct", 0.0))
        direction = "positively" if shap_value >= 0 else "negatively"

        if feature.startswith("rms_"):
            axis = feature.split("_")[-1].upper()
            return (
                f"RMS on axis {axis} = {value:.6f}; "
                f"contributed {direction} to the predicted class; "
                f"approximate local importance = {impact_pct:.1f}%."
            )
        if feature.startswith("kurtosis_"):
            axis = feature.split("_")[-1].upper()
            return (
                f"kurtosis on axis {axis} = {value:.6f}; "
                f"contributed {direction} to the predicted class; "
                f"approximate local importance = {impact_pct:.1f}%."
            )
        if feature.startswith("peak_value_"):
            axis = feature.split("_")[-1].upper()
            return (
                f"peak value on axis {axis} = {value:.6f}; "
                f"contributed {direction} to the predicted class; "
                f"approximate local importance = {impact_pct:.1f}%."
            )
        if feature.startswith("crest_factor_"):
            axis = feature.split("_")[-1].upper()
            return (
                f"crest factor on axis {axis} = {value:.6f}; "
                f"contributed {direction} to the predicted class; "
                f"approximate local importance = {impact_pct:.1f}%."
            )
        if feature.startswith("energy_fm1_h"):
            parts = feature.split("_")
            harmonic = parts[2].replace("h", "")
            axis = parts[-1].upper()
            return (
                f"energy around harmonic {harmonic} of Fm1 on axis {axis} = {value:.6f}; "
                f"contributed {direction} to the predicted class; "
                f"approximate local importance = {impact_pct:.1f}%."
            )
        if feature.startswith("amp_max_fm1_h"):
            parts = feature.split("_")
            harmonic = parts[3].replace("h", "")
            axis = parts[-1].upper()
            return (
                f"maximum amplitude around harmonic {harmonic} of Fm1 on axis {axis} = {value:.6f}; "
                f"contributed {direction} to the predicted class; "
                f"approximate local importance = {impact_pct:.1f}%."
            )
        if feature.startswith("energy_fm2_h"):
            parts = feature.split("_")
            harmonic = parts[2].replace("h", "")
            axis = parts[-1].upper()
            return (
                f"energy around harmonic {harmonic} of Fm2 on axis {axis} = {value:.6f}; "
                f"contributed {direction} to the predicted class; "
                f"approximate local importance = {impact_pct:.1f}%."
            )
        if feature.startswith("amp_max_fm2_h"):
            parts = feature.split("_")
            harmonic = parts[3].replace("h", "")
            axis = parts[-1].upper()
            return (
                f"maximum amplitude around harmonic {harmonic} of Fm2 on axis {axis} = {value:.6f}; "
                f"contributed {direction} to the predicted class; "
                f"approximate local importance = {impact_pct:.1f}%."
            )
        if feature.startswith("energy_rel_fm2_h"):
            parts = feature.split("_")
            harmonic = parts[3].replace("h", "")
            axis = parts[-1].upper()
            return (
                f"relative energy around harmonic {harmonic} of Fm2 on axis {axis} = {value:.6f}; "
                f"contributed {direction} to the predicted class; "
                f"approximate local importance = {impact_pct:.1f}%."
            )
        if feature.startswith("ratio_energy_fm2_h"):
            parts = feature.split("_")
            ratio_name = parts[3].upper()
            axis = parts[-1].upper()
            return (
                f"harmonic energy ratio {ratio_name} on axis {axis} = {value:.6f}; "
                f"contributed {direction} to the predicted class; "
                f"approximate local importance = {impact_pct:.1f}%."
            )
        if feature.startswith("ratio_amp_fm2_h"):
            parts = feature.split("_")
            ratio_name = parts[3].upper()
            axis = parts[-1].upper()
            return (
                f"harmonic amplitude ratio {ratio_name} on axis {axis} = {value:.6f}; "
                f"contributed {direction} to the predicted class; "
                f"approximate local importance = {impact_pct:.1f}%."
            )
        return (
            f"{feature} = {value:.6f}; "
            f"contributed {direction} to the predicted class; "
            f"approximate local importance = {impact_pct:.1f}%."
        )

    def _extract_json_object(self, text: str) -> dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise RuntimeError(f"Nao foi possivel extrair JSON da resposta do modelo: {text}")

        json_text = cleaned[start : end + 1]
        try:
            return json.loads(json_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"JSON invalido retornado pelo modelo: {json_text}") from exc

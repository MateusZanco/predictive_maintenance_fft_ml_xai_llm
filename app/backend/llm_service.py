from __future__ import annotations

import json
import os
import socket
from typing import Any
from urllib import error, request
from xml.sax.saxutils import escape


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
        prompt_strategy: str = "few_shot",
        concise: bool = False,
    ) -> tuple[str, str, list[dict[str, str]]]:
        del condicao_operacao

        system_prompt = """<SYSTEM_PROMPT>
  <ROLE>You are a specialist in vibration analysis for planetary gearboxes.</ROLE>
  <STRICT_RULES>
    <RULE>Use only the provided evidence list.</RULE>
    <RULE>The evidence list contains exactly the provided items and no other hidden evidence.</RULE>
    <RULE>Do not invent frequencies, causes, severity, numeric ranges, thresholds, extreme conditions, mechanical stress, tension, or symptoms.</RULE>
    <RULE>If no explicit reference range is provided, never say normal, abnormal, elevated, reduced, within range, or outside range.</RULE>
    <RULE>Do not create numeric limits such as "0.10 to 0.20" or similar.</RULE>
    <RULE>Mention only concepts supported by variable families present in the evidence list.</RULE>
    <RULE>Without RMS, do not mention global vibration level.</RULE>
    <RULE>Without kurtosis, do not mention impulsiveness.</RULE>
    <RULE>Without peak value, do not mention peak amplitude in the time signal.</RULE>
    <RULE>Without crest factor, do not mention the relationship between peak and RMS.</RULE>
    <RULE>Without harmonic energy features, do not mention concentration of energy in harmonic bands.</RULE>
    <RULE>Without maximum harmonic amplitude features, do not mention maximum spectral amplitude in a harmonic band.</RULE>
    <RULE>Do not mention imbalance, misalignment, looseness, wear progression, mechanical stress, or other specific mechanisms unless they are explicitly supported by the evidence.</RULE>
    <RULE>Do not use generic expressions such as "high-frequency components", "system response", or "system behavior" unless they are directly supported by the evidence block.</RULE>
    <RULE>If the evidence is limited, use cautious language such as "compatible with" or "suggests".</RULE>
    <RULE>Prefer citing the highest-ranked evidence items first.</RULE>
    <RULE>When 3 or more evidence items are provided, mention at least 3 of them in interpretacao_vibracional.</RULE>
    <RULE>Treat every percentage in the evidence list as an approximate share of absolute local explanatory impact, not as probability share.</RULE>
    <RULE>Do not produce checklist-style summaries of all possible concept groups.</RULE>
    <RULE>Do not use parenthetical qualifiers such as "if applicable" or "se aplicável".</RULE>
    <RULE>In interpretacao_vibracional, mention only the concepts directly supported by the evidence items actually listed in the current window.</RULE>
    <RULE>If the predicted class is Normal, do not describe the result as a fault confirmation or as compatibility with a specific fault mechanism.</RULE>
    <RULE>Do not mention SHAP, model, AI, or prompt.</RULE>
    <RULE>Do not provide recommended actions.</RULE>
    <RULE>Write the final answer only in Brazilian Portuguese.</RULE>
    <RULE>Do not reveal the internal procedure. Return only the final JSON.</RULE>
  </STRICT_RULES>
  <PROCEDURE>
    <STEP index="1">Read only the evidence items listed in OBSERVED_EVIDENCE.</STEP>
    <STEP index="2">Identify which variable families are present in the evidence list.</STEP>
    <STEP index="3">Map each present variable family to its allowed concept.</STEP>
    <STEP index="4">Ignore any concept whose variable family is absent.</STEP>
    <STEP index="5">Write interpretacao_vibracional citing the highest-ranked evidence items first, with variable value and approximate local importance.</STEP>
    <STEP index="6">If you write a final synthesis sentence in interpretacao_vibracional, include only concept groups supported by the listed evidence.</STEP>
    <STEP index="7">If the predicted class is Normal, write interpretacao_mecanica using the Normal-specific instruction from TASK and include the predicted class probability.</STEP>
    <STEP index="8">If the predicted class is not Normal, write interpretacao_mecanica only as compatibility with the predicted class, using cautious wording, and include the predicted class probability.</STEP>
    <STEP index="9">Return only valid JSON.</STEP>
  </PROCEDURE>
  <OUTPUT_FORMAT>
    <TYPE>Return only valid JSON.</TYPE>
    <JSON_TEMPLATE>{
  "interpretacao_vibracional": "...",
  "interpretacao_mecanica": "..."
}</JSON_TEMPLATE>
  </OUTPUT_FORMAT>
</SYSTEM_PROMPT>"""

        fixed_user_prefix = self._fixed_user_prefix()
        evidence_lines = [
            f'    <EVIDENCE index="{idx}">{escape(self._describe_contribution(item))}</EVIDENCE>'
            for idx, item in enumerate(top_contributions, start=1)
        ]
        final_instruction = "Be even more brief and direct." if concise else "Generate the final explanation."

        user_prompt = f"""<USER_PROMPT>
{fixed_user_prefix}
  <INPUT_DATA>
    <PREDICTED_CLASS>{escape(predicted_class_name)}</PREDICTED_CLASS>
    <PREDICTED_CLASS_PROBABILITY>{predicted_probability:.4f}</PREDICTED_CLASS_PROBABILITY>
    <OBSERVED_EVIDENCE>
{chr(10).join(evidence_lines)}
    </OBSERVED_EVIDENCE>
    <FINAL_INSTRUCTION>{escape(final_instruction)}</FINAL_INSTRUCTION>
  </INPUT_DATA>
</USER_PROMPT>"""

        messages = [{"role": "system", "content": system_prompt}]
        if prompt_strategy == "few_shot":
            messages.extend(self._few_shot_examples())
        messages.append({"role": "user", "content": user_prompt})
        return system_prompt, user_prompt, messages

    def generate_explanation(
        self,
        condicao_operacao: str,
        predicted_class_name: str,
        predicted_probability: float,
        top_contributions: list[dict[str, Any]],
        prompt_strategy: str = "few_shot",
    ) -> dict[str, Any]:
        system_prompt, user_prompt, messages = self.build_messages(
            condicao_operacao=condicao_operacao,
            predicted_class_name=predicted_class_name,
            predicted_probability=predicted_probability,
            top_contributions=top_contributions,
            prompt_strategy=prompt_strategy,
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
                prompt_strategy=prompt_strategy,
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
            raise RuntimeError(f"Timeout ao aguardar resposta do llama-server apos {self.timeout_seconds:.0f} s.") from exc
        except socket.timeout as exc:
            raise RuntimeError(f"Timeout ao aguardar resposta do llama-server apos {self.timeout_seconds:.0f} s.") from exc
        except Exception as exc:
            raise RuntimeError(f"Falha inesperada ao consultar o llama-server: {exc}") from exc

        try:
            parsed = json.loads(raw)
            return parsed["choices"][0]["message"]["content"]
        except Exception as exc:
            raise RuntimeError(f"Resposta invalida do llama-server: {raw}") from exc

    def _few_shot_examples(self) -> list[dict[str, str]]:
        fixed_user_prefix = self._fixed_user_prefix()

        example_user = f"""<USER_PROMPT>
{fixed_user_prefix}
  <INPUT_DATA>
    <PREDICTED_CLASS>Dente Trincado</PREDICTED_CLASS>
    <PREDICTED_CLASS_PROBABILITY>0.9925</PREDICTED_CLASS_PROBABILITY>
    <OBSERVED_EVIDENCE>
      <EVIDENCE index="1">RMS on axis Y = 0.214310; contributed positively to the predicted class; approximate local importance = 28.7%.</EVIDENCE>
      <EVIDENCE index="2">energy around harmonic 1 of Fm2 on axis X = 0.004812; contributed positively to the predicted class; approximate local importance = 24.9%.</EVIDENCE>
      <EVIDENCE index="3">maximum amplitude around harmonic 2 of Fm2 on axis Z = 0.021334; contributed positively to the predicted class; approximate local importance = 19.8%.</EVIDENCE>
      <EVIDENCE index="4">crest factor on axis Z = 5.184220; contributed positively to the predicted class; approximate local importance = 11.5%.</EVIDENCE>
    </OBSERVED_EVIDENCE>
    <FINAL_INSTRUCTION>Generate the final explanation.</FINAL_INSTRUCTION>
  </INPUT_DATA>
</USER_PROMPT>"""

        example_assistant = """{
  "interpretacao_vibracional": "A janela mostra RMS no eixo Y = 0.214310, com importância local aproximada de 28.7%, energia em torno da harmônica 1 de Fm2 no eixo X = 0.004812, com 24.9%, amplitude máxima em torno da harmônica 2 de Fm2 no eixo Z = 0.021334, com 19.8%, e crest factor no eixo Z = 5.184220, com 11.5%. Essas evidências descrevem contribuições associadas ao nível global de vibração, à concentração de energia em bandas harmônicas, à amplitude máxima em banda harmônica e à relação entre pico e RMS nesta janela.",
  "interpretacao_mecanica": "O conjunto de evidências é compatível com a classe predita de Dente Trincado. Essa interpretação deve ser vista com cautela, porque as evidências fornecidas descrevem variáveis do sinal e bandas harmônicas relevantes, mas não constituem confirmação direta de um mecanismo específico de falha."
}"""

        return [
            {"role": "user", "content": example_user},
            {"role": "assistant", "content": example_assistant},
        ]

    def _fixed_user_prefix(self) -> str:
        return """  <FIXED_CONTEXT>
    <EQUIPMENT>two-stage planetary gearbox</EQUIPMENT>
    <MONITORED_COMPONENT>second-stage sun gear</MONITORED_COMPONENT>
  </FIXED_CONTEXT>
  <VARIABLE_LEGEND>
    <TIME_DOMAIN_VARIABLES>
      <VARIABLE name="RMS">global vibration level of the segment on the analyzed axis</VARIABLE>
      <VARIABLE name="kurtosis">impulsiveness of the time signal on the analyzed axis</VARIABLE>
      <VARIABLE name="peak_value">highest absolute amplitude observed in the segment on the analyzed axis</VARIABLE>
      <VARIABLE name="crest_factor">ratio between peak value and RMS on the analyzed axis</VARIABLE>
    </TIME_DOMAIN_VARIABLES>
    <FREQUENCY_DOMAIN_HARMONIC_BAND_VARIABLES>
      <VARIABLE name="energy_around_Fm1_harmonic">spectral energy inside a +/-10 Hz band around a first-stage gear mesh harmonic</VARIABLE>
      <VARIABLE name="energy_around_Fm2_harmonic">spectral energy inside a +/-10 Hz band around a second-stage gear mesh harmonic</VARIABLE>
      <VARIABLE name="maximum_amplitude_around_Fm1_harmonic">highest spectral amplitude inside a +/-10 Hz band around an Fm1 harmonic</VARIABLE>
      <VARIABLE name="maximum_amplitude_around_Fm2_harmonic">highest spectral amplitude inside a +/-10 Hz band around an Fm2 harmonic</VARIABLE>
    </FREQUENCY_DOMAIN_HARMONIC_BAND_VARIABLES>
    <ATTRIBUTION_TERMS>
      <TERM name="contributed_positively">this variable pushed the model toward the predicted class</TERM>
      <TERM name="contributed_negatively">this variable pushed the model away from the predicted class</TERM>
      <TERM name="approximate_local_importance">approximate share of absolute local explanatory impact attributed to that variable in the current window</TERM>
    </ATTRIBUTION_TERMS>
  </VARIABLE_LEGEND>
  <INTERPRETATION_MAPPING_RULES>
    <RULE>RMS allows mentioning global vibration level.</RULE>
    <RULE>kurtosis allows mentioning impulsiveness of the time signal.</RULE>
    <RULE>peak value allows mentioning peak amplitude in the time signal.</RULE>
    <RULE>crest factor allows mentioning the relationship between peak value and RMS.</RULE>
    <RULE>energy around an Fm1 or Fm2 harmonic allows mentioning concentration of energy inside harmonic bands.</RULE>
    <RULE>maximum amplitude around an Fm1 or Fm2 harmonic allows mentioning the highest spectral amplitude inside a harmonic band.</RULE>
    <RULE>Use only concepts associated with variable families present in the evidence block.</RULE>
    <RULE>If a variable family is absent, do not mention the concept associated with it.</RULE>
    <RULE>Do not summarize absent concepts as optional or hypothetical concepts.</RULE>
    <RULE>Without kurtosis, do not mention impulsiveness.</RULE>
    <RULE>Without peak value, do not mention peak amplitude in the time signal.</RULE>
    <RULE>Without harmonic energy, do not mention concentration of energy inside harmonic bands.</RULE>
    <RULE>Without maximum harmonic amplitude, do not mention the highest spectral amplitude inside a harmonic band.</RULE>
    <RULE>Without crest factor, do not mention the relationship between peak and RMS in any synthesis sentence.</RULE>
  </INTERPRETATION_MAPPING_RULES>
  <TASK>
    <ITEM target="interpretacao_vibracional">Describe only the observed variables and their axes or harmonics.</ITEM>
    <ITEM target="interpretacao_vibracional">When useful, explicitly cite the variable value and the approximate local importance percentage.</ITEM>
    <ITEM target="interpretacao_vibracional">Prefer literal descriptions of the reported variables instead of generic summaries.</ITEM>
    <ITEM target="interpretacao_vibracional">Distinguish clearly between time-domain variables and frequency-domain harmonic-band variables when describing the evidence.</ITEM>
    <ITEM target="interpretacao_vibracional">If you write a final synthesis sentence, include only concept groups directly supported by the evidence listed in INPUT_DATA.</ITEM>
    <ITEM target="interpretacao_vibracional">Never mention impulsiveness, peak amplitude in the time signal, relationship between peak and RMS, concentration of energy in harmonic bands, or maximum spectral amplitude in a harmonic band unless the corresponding variable family appears explicitly in OBSERVED_EVIDENCE.</ITEM>
    <ITEM target="interpretacao_mecanica">State only whether the evidence is compatible with the predicted class, using cautious wording.</ITEM>
    <ITEM target="interpretacao_mecanica">Include the predicted class probability as "probabilidade estimada de X% para a classe predita".</ITEM>
    <ITEM target="interpretacao_mecanica">If the predicted class is Normal, start with exactly: "A principal evidência nesta janela não indicou predominância de um padrão compatível com as classes de falha modeladas."</ITEM>
    <ITEM target="interpretacao_mecanica">If the predicted class is not Normal, do not present the fault class as confirmed; use language compatible with or suggestive of the predicted class.</ITEM>
    <ITEM target="general">Do not compare with normal ranges, reference limits, or expected values.</ITEM>
    <ITEM target="general">Do not mention operating condition, rotation, torque, or reduction ratio unless they appear explicitly in the input data.</ITEM>
    <ITEM target="general">If a variable family is absent from the evidence, avoid introducing the associated concept.</ITEM>
  </TASK>"""

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

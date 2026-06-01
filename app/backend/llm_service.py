from __future__ import annotations

import json
import os
import socket
from typing import Any
from urllib import error, request
from xml.sax.saxutils import escape


AudienceProfile = str


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
        audience_profile: AudienceProfile = "tecnico",
        concise: bool = False,
    ) -> tuple[str, str, list[dict[str, str]]]:
        del condicao_operacao
        del concise
        audience_profile = self._normalize_audience_profile(audience_profile)
        audience_rule = escape(self._audience_profile_rule(audience_profile))
        audience_vibrational_instruction = escape(self._audience_vibrational_instruction(audience_profile))
        audience_importance_instruction = escape(self._audience_importance_instruction(audience_profile))
        audience_negative_importance_instruction = escape(self._audience_negative_importance_instruction(audience_profile))
        audience_mechanical_instruction = escape(self._audience_mechanical_instruction(audience_profile))

        system_prompt = f"""<SYSTEM_PROMPT>
  <ROLE>You are a specialist in vibration analysis for planetary gearboxes.</ROLE>
  <FIXED_CONTEXT>
    <EQUIPMENT>two-stage planetary gearbox</EQUIPMENT>
    <MONITORED_COMPONENT>second-stage sun gear</MONITORED_COMPONENT>
  </FIXED_CONTEXT>
  <RULES>
    <RULE>Use only OBSERVED_EVIDENCE.</RULE>
    <RULE>Do not invent or infer ranges, baselines, causes, severity, symptoms, or specific failure mechanisms.</RULE>
    <RULE>Mention only concepts supported by the variable families present in OBSERVED_EVIDENCE.</RULE>
    <RULE>If an evidence item contributed negatively to the predicted class, describe it as negative evidence using the wording required by AUDIENCE_PROFILE, never as positive support.</RULE>
    <RULE>Do not mention SHAP, model, AI, algorithm, or prompt.</RULE>
    <RULE>Write only in Brazilian Portuguese.</RULE>
    <RULE>Return only valid JSON.</RULE>
  </RULES>
  <LEGEND>
    <TIME_DOMAIN_VARIABLES>
      <VARIABLE name="RMS" concept="valor RMS e nível global de vibração">root-mean-square amplitude of the time segment on the analyzed axis</VARIABLE>
      <VARIABLE name="kurtosis" concept="impulsividade do sinal no tempo">fourth standardized moment of the time signal on the analyzed axis</VARIABLE>
      <VARIABLE name="peak_value" concept="valor de pico no sinal no tempo">highest absolute amplitude observed in the time segment on the analyzed axis</VARIABLE>
      <VARIABLE name="crest_factor" concept="fator de crista e relação entre valor de pico e RMS">ratio between peak value and RMS on the analyzed axis</VARIABLE>
    </TIME_DOMAIN_VARIABLES>
    <FREQUENCY_DOMAIN_VARIABLES>
      <VARIABLE name="energy_around_harmonic" concept="energia espectral e concentração de energia espectral em bandas harmônicas">spectral energy inside a ±10 Hz band around an Fm1 or Fm2 harmonic</VARIABLE>
      <VARIABLE name="maximum_amplitude_around_harmonic" concept="amplitude espectral máxima em banda harmônica">highest spectral amplitude inside a ±10 Hz band around an Fm1 or Fm2 harmonic</VARIABLE>
    </FREQUENCY_DOMAIN_VARIABLES>
    <ATTRIBUTION_TERMS>
      <TERM name="contributed_positively">this variable pushed the predicted class upward</TERM>
      <TERM name="contributed_negatively">this variable pushed the predicted class downward</TERM>
      <TERM name="approximate_local_importance">share of absolute local explanatory impact in the current window</TERM>
    </ATTRIBUTION_TERMS>
  </LEGEND>
  <ACTIVE_AUDIENCE_PROFILE name="{audience_profile}">{audience_rule}</ACTIVE_AUDIENCE_PROFILE>
  <TASK>
    <VIBRATIONAL>Write interpretacao_vibracional as a single JSON string with exactly 5 bullet lines, one line for each evidence item, each starting with "- ".</VIBRATIONAL>
    <VIBRATIONAL>In each bullet, cite the variable, axis or harmonic order, value, and relative percentage contribution for the current window.</VIBRATIONAL>
    <VIBRATIONAL>{audience_importance_instruction} {audience_negative_importance_instruction}</VIBRATIONAL>
    <VIBRATIONAL>Do not add an opening paragraph or a final synthesis paragraph.</VIBRATIONAL>
    <VIBRATIONAL>{audience_vibrational_instruction}</VIBRATIONAL>
    <PREDICTED_CLASS_INTERPRETATION>Write interpretacao_classe_predita only as cautious compatibility with the predicted class, including "probabilidade estimada de X% para a classe predita".</PREDICTED_CLASS_INTERPRETATION>
    <PREDICTED_CLASS_INTERPRETATION>{audience_mechanical_instruction}</PREDICTED_CLASS_INTERPRETATION>
  </TASK>
  <PROCEDURE>
    <STEP>Read AUDIENCE_PROFILE and OBSERVED_EVIDENCE.</STEP>
    <STEP>Identify the present variable families and whether each evidence item contributed positively or negatively.</STEP>
    <STEP>Use all 5 evidence items in interpretacao_vibracional with the audience-specific percentage wording.</STEP>
    <STEP>Return only the JSON object defined in OUTPUT_FORMAT.</STEP>
  </PROCEDURE>
  <OUTPUT_FORMAT>{{
  "interpretacao_vibracional": "...",
  "interpretacao_classe_predita": "..."
}}</OUTPUT_FORMAT>
</SYSTEM_PROMPT>"""

        evidence_lines = [
            f'      <EVIDENCE index="{idx}">{escape(self._describe_contribution(item))}</EVIDENCE>'
            for idx, item in enumerate(top_contributions, start=1)
        ]

        user_prompt = f"""<USER_PROMPT>
  <AUDIENCE_PROFILE>{escape(audience_profile)}</AUDIENCE_PROFILE>
  <PREDICTED_CLASS>{escape(predicted_class_name)}</PREDICTED_CLASS>
  <PREDICTED_CLASS_PROBABILITY>{predicted_probability:.4f}</PREDICTED_CLASS_PROBABILITY>
  <OBSERVED_EVIDENCE>
{chr(10).join(evidence_lines)}
  </OBSERVED_EVIDENCE>
</USER_PROMPT>"""

        messages = [{"role": "system", "content": system_prompt}]
        if prompt_strategy == "few_shot":
            messages.extend(self._few_shot_examples(audience_profile))
        messages.append({"role": "user", "content": user_prompt})
        return system_prompt, user_prompt, messages

    def generate_explanation(
        self,
        condicao_operacao: str,
        predicted_class_name: str,
        predicted_probability: float,
        top_contributions: list[dict[str, Any]],
        prompt_strategy: str = "few_shot",
        audience_profile: AudienceProfile = "tecnico",
    ) -> dict[str, Any]:
        audience_profile = self._normalize_audience_profile(audience_profile)
        system_prompt, user_prompt, messages = self.build_messages(
            condicao_operacao=condicao_operacao,
            predicted_class_name=predicted_class_name,
            predicted_probability=predicted_probability,
            top_contributions=top_contributions,
            prompt_strategy=prompt_strategy,
            audience_profile=audience_profile,
        )

        content = self._chat_completion(messages, self.max_tokens)
        try:
            explanation_json = self._extract_json_object(content)
            raw_response = content
        except RuntimeError:
            if prompt_strategy == "zero_shot":
                return {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "raw_response": content,
                    "audience_profile": audience_profile,
                    "response_format": "raw_text",
                    "unstructured_response": content.strip(),
                    "interpretacao_vibracional": "",
                    "interpretacao_classe_predita": "",
                }
            raise

        interpretacao_classe_predita = str(
            explanation_json.get("interpretacao_classe_predita", explanation_json.get("interpretacao_mecanica", ""))
        ).strip()

        return {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "raw_response": raw_response,
            "audience_profile": audience_profile,
            "response_format": "json",
            "unstructured_response": None,
            "interpretacao_vibracional": str(explanation_json.get("interpretacao_vibracional", "")).strip(),
            "interpretacao_classe_predita": interpretacao_classe_predita,
        }

    def _chat_completion(self, messages: list[dict[str, str]], max_tokens: int) -> str:
        payload: dict[str, Any] = {
            "messages": messages,
            "cache_prompt": True,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": max_tokens,
            "response_format": self._response_format_schema(),
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
            raise RuntimeError(f"Timeout ao aguardar resposta do llama-server após {self.timeout_seconds:.0f} s.") from exc
        except socket.timeout as exc:
            raise RuntimeError(f"Timeout ao aguardar resposta do llama-server após {self.timeout_seconds:.0f} s.") from exc
        except Exception as exc:
            raise RuntimeError(f"Falha inesperada ao consultar o llama-server: {exc}") from exc

        try:
            parsed = json.loads(raw)
            return parsed["choices"][0]["message"]["content"]
        except Exception as exc:
            raise RuntimeError(f"Resposta inválida do llama-server: {raw}") from exc

    def _response_format_schema(self) -> dict[str, Any]:
        return {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "interpretacao_vibracional": {"type": "string"},
                    "interpretacao_classe_predita": {"type": "string"},
                },
                "required": [
                    "interpretacao_vibracional",
                    "interpretacao_classe_predita",
                ],
                "additionalProperties": False,
            },
        }

    def _normalize_audience_profile(self, audience_profile: AudienceProfile) -> AudienceProfile:
        normalized = str(audience_profile).strip().lower()
        aliases = {
            "engenharia": "tecnico",
            "manutencao": "contextualizado",
            "operacao": "didatico",
        }
        if normalized in {"tecnico", "contextualizado", "didatico"}:
            return normalized
        return aliases.get(normalized, "tecnico")

    def _audience_profile_rule(self, audience_profile: AudienceProfile) -> str:
        rules = {
            "tecnico": (
                "Use the most rigorous technical vocabulary available in the provided evidence. "
                "Distinguish clearly between time-domain metrics and frequency-domain harmonic-band metrics."
            ),
            "contextualizado": (
                "Keep technical accuracy, but prefer contextualized wording linked to the practical meaning of the signal. "
                "Explain what each relevant quantity represents in practice, such as level of overall vibration, "
                "strongest signal peak, vibration concentrated at a specific harmonic order of the first or second stage, "
                "or maximum intensity observed at that harmonic. Avoid overly academic wording."
            ),
            "didatico": (
                "Use simpler wording, shorter sentences, and minimal jargon. "
                "Do not use the terms RMS, kurtosis, crest factor, harmonic, spectral energy, spectral amplitude, Fm1, or Fm2 in the final text. "
                "Translate them into accessible phrases such as overall vibration level, strongest signal peak, difference between peaks and overall level, "
                "or vibration concentrated in a specific band of the first or second stage. Keep the exact values and the axis or stage reference."
            ),
        }
        return rules[self._normalize_audience_profile(audience_profile)]

    def _audience_vibrational_instruction(self, audience_profile: AudienceProfile) -> str:
        instructions = {
            "tecnico": (
                "Preserve the most rigorous technical wording and explicit distinction between time-domain and harmonic-band evidence."
            ),
            "contextualizado": (
                "Explain in practical vibration-monitoring terms what each variable represents and what in the signal deserves technical attention."
            ),
            "didatico": (
                "Explain the same evidence in plain operational language focused on what the machine signal is showing."
            ),
        }
        return instructions[self._normalize_audience_profile(audience_profile)]

    def _audience_importance_instruction(self, audience_profile: AudienceProfile) -> str:
        instructions = {
            "tecnico": (
                'When citing each percentage, use wording such as "participação relativa de X% no impacto explicativo local absoluto".'
            ),
            "contextualizado": (
                'When citing each percentage, use wording such as "respondeu por cerca de X% do impacto local da classificação".'
            ),
            "didatico": (
                'When citing each percentage, use wording such as "teve participação de cerca de X% na explicação da decisão do modelo nesta janela".'
            ),
        }
        return instructions[self._normalize_audience_profile(audience_profile)]

    def _audience_negative_importance_instruction(self, audience_profile: AudienceProfile) -> str:
        instructions = {
            "tecnico": (
                'When an evidence item contributed negatively, use wording such as "participação relativa de X% no impacto explicativo local absoluto, atuando em sentido oposto à classe predita".'
            ),
            "contextualizado": (
                'When an evidence item contributed negatively, use wording such as "com participação relativa negativa de X% do impacto local da classificação".'
            ),
            "didatico": (
                'When an evidence item contributed negatively, use wording such as "contribuiu negativamente com cerca de X% na explicação da decisão do modelo nesta janela".'
            ),
        }
        return instructions[self._normalize_audience_profile(audience_profile)]

    def _audience_mechanical_instruction(self, audience_profile: AudienceProfile) -> str:
        instructions = {
            "tecnico": "Keep the most technical tone.",
            "contextualizado": "Keep a contextualized diagnostic tone centered on what the signal is indicating in practice.",
            "didatico": "Explicitly state in simple language that the listed variables led the model to the predicted class.",
        }
        return instructions[self._normalize_audience_profile(audience_profile)]

    def _few_shot_examples(self, audience_profile: AudienceProfile) -> list[dict[str, str]]:
        example_user_1 = f"""<USER_PROMPT>
  <AUDIENCE_PROFILE>{escape(audience_profile)}</AUDIENCE_PROFILE>
  <PREDICTED_CLASS>Dente Trincado</PREDICTED_CLASS>
  <PREDICTED_CLASS_PROBABILITY>0.9925</PREDICTED_CLASS_PROBABILITY>
  <OBSERVED_EVIDENCE>
      <EVIDENCE index="1">RMS value on axis Y = 0.214310; contributed positively to the predicted class; approximate local importance = 28.7%.</EVIDENCE>
      <EVIDENCE index="2">spectral energy within the ±10 Hz band around the 1st-order harmonic of Fm2 on axis X = 0.004812; contributed positively to the predicted class; approximate local importance = 24.9%.</EVIDENCE>
      <EVIDENCE index="3">maximum spectral amplitude within the ±10 Hz band around the 2nd-order harmonic of Fm2 on axis Z = 0.021334; contributed positively to the predicted class; approximate local importance = 19.8%.</EVIDENCE>
      <EVIDENCE index="4">crest factor on axis Z = 5.184220; contributed positively to the predicted class; approximate local importance = 11.5%.</EVIDENCE>
      <EVIDENCE index="5">RMS value on axis X = 0.118400; contributed positively to the predicted class; approximate local importance = 8.2%.</EVIDENCE>
  </OBSERVED_EVIDENCE>
</USER_PROMPT>"""

        example_user_2 = f"""<USER_PROMPT>
  <AUDIENCE_PROFILE>{escape(audience_profile)}</AUDIENCE_PROFILE>
  <PREDICTED_CLASS>Desgaste Superficial</PREDICTED_CLASS>
  <PREDICTED_CLASS_PROBABILITY>0.9640</PREDICTED_CLASS_PROBABILITY>
  <OBSERVED_EVIDENCE>
      <EVIDENCE index="1">RMS value on axis X = 0.091390; contributed positively to the predicted class; approximate local importance = 24.6%.</EVIDENCE>
      <EVIDENCE index="2">peak value on axis Y = 0.969533; contributed positively to the predicted class; approximate local importance = 21.4%.</EVIDENCE>
      <EVIDENCE index="3">spectral energy within the ±10 Hz band around the 2nd-order harmonic of Fm1 on axis Y = 0.008880; contributed positively to the predicted class; approximate local importance = 15.6%.</EVIDENCE>
      <EVIDENCE index="4">maximum spectral amplitude within the ±10 Hz band around the 5th-order harmonic of Fm1 on axis X = 0.003614; contributed positively to the predicted class; approximate local importance = 9.9%.</EVIDENCE>
      <EVIDENCE index="5">spectral energy within the ±10 Hz band around the 5th-order harmonic of Fm1 on axis Y = 0.000314; contributed negatively to the predicted class; approximate local importance = 9.4%.</EVIDENCE>
  </OBSERVED_EVIDENCE>
</USER_PROMPT>"""

        example_user_3 = f"""<USER_PROMPT>
  <AUDIENCE_PROFILE>{escape(audience_profile)}</AUDIENCE_PROFILE>
  <PREDICTED_CLASS>Normal</PREDICTED_CLASS>
  <PREDICTED_CLASS_PROBABILITY>0.9510</PREDICTED_CLASS_PROBABILITY>
  <OBSERVED_EVIDENCE>
      <EVIDENCE index="1">RMS value on axis Y = 0.170664; contributed positively to the predicted class; approximate local importance = 31.5%.</EVIDENCE>
      <EVIDENCE index="2">RMS value on axis Z = 0.240076; contributed positively to the predicted class; approximate local importance = 18.6%.</EVIDENCE>
      <EVIDENCE index="3">spectral energy within the ±10 Hz band around the 5th-order harmonic of Fm2 on axis Y = 0.000190; contributed positively to the predicted class; approximate local importance = 10.6%.</EVIDENCE>
      <EVIDENCE index="4">RMS value on axis X = 0.083640; contributed positively to the predicted class; approximate local importance = 7.6%.</EVIDENCE>
      <EVIDENCE index="5">spectral energy within the ±10 Hz band around the 1st-order harmonic of Fm1 on axis Y = 0.001764; contributed negatively to the predicted class; approximate local importance = 6.4%.</EVIDENCE>
  </OBSERVED_EVIDENCE>
</USER_PROMPT>"""

        examples_by_audience = {
            "tecnico": [
                """{
  "interpretacao_vibracional": "- valor RMS no eixo Y = 0.214310, com participação relativa de 28.7% no impacto explicativo local absoluto.\\n- energia espectral na faixa de ±10 Hz em torno da harmônica de 1ª ordem de Fm2 no eixo X = 0.004812, com participação relativa de 24.9% no impacto explicativo local absoluto.\\n- amplitude espectral máxima na faixa de ±10 Hz em torno da harmônica de 2ª ordem de Fm2 no eixo Z = 0.021334, com participação relativa de 19.8% no impacto explicativo local absoluto.\\n- fator de crista no eixo Z = 5.184220, com participação relativa de 11.5% no impacto explicativo local absoluto.\\n- valor RMS no eixo X = 0.118400, com participação relativa de 8.2% no impacto explicativo local absoluto.",
  "interpretacao_classe_predita": "O conjunto de evidências é compatível com a classe predita de Dente Trincado, com probabilidade estimada de 99.3% para a classe predita. Essa interpretação deve ser vista com cautela, porque as evidências fornecidas descrevem variáveis do sinal e bandas harmônicas relevantes, mas não constituem confirmação direta de um mecanismo específico de falha."
}""",
                """{
  "interpretacao_vibracional": "- valor RMS no eixo X = 0.091390, com participação relativa de 24.6% no impacto explicativo local absoluto.\\n- valor de pico no eixo Y = 0.969533, com participação relativa de 21.4% no impacto explicativo local absoluto.\\n- energia espectral na faixa de ±10 Hz em torno da harmônica de 2ª ordem de Fm1 no eixo Y = 0.008880, com participação relativa de 15.6% no impacto explicativo local absoluto.\\n- amplitude espectral máxima na faixa de ±10 Hz em torno da harmônica de 5ª ordem de Fm1 no eixo X = 0.003614, com participação relativa de 9.9% no impacto explicativo local absoluto.\\n- energia espectral na faixa de ±10 Hz em torno da harmônica de 5ª ordem de Fm1 no eixo Y = 0.000314, com participação relativa de 9.4% no impacto explicativo local absoluto, atuando em sentido oposto à classe predita.",
  "interpretacao_classe_predita": "O conjunto de evidências é compatível com a classe predita de Desgaste Superficial, com probabilidade estimada de 96.4% para a classe predita. Essa interpretação deve ser vista com cautela, porque as evidências fornecidas descrevem variáveis do sinal e bandas harmônicas relevantes, mas não constituem confirmação direta de um mecanismo específico de falha."
}""",
                """{
  "interpretacao_vibracional": "- valor RMS no eixo Y = 0.170664, com participação relativa de 31.5% no impacto explicativo local absoluto.\\n- valor RMS no eixo Z = 0.240076, com participação relativa de 18.6% no impacto explicativo local absoluto.\\n- energia espectral na faixa de ±10 Hz em torno da harmônica de 5ª ordem de Fm2 no eixo Y = 0.000190, com participação relativa de 10.6% no impacto explicativo local absoluto.\\n- valor RMS no eixo X = 0.083640, com participação relativa de 7.6% no impacto explicativo local absoluto.\\n- energia espectral na faixa de ±10 Hz em torno da harmônica de 1ª ordem de Fm1 no eixo Y = 0.001764, com participação relativa de 6.4% no impacto explicativo local absoluto, atuando em sentido oposto à classe predita.",
  "interpretacao_classe_predita": "O conjunto de evidências é compatível com a classe predita de Normal, com probabilidade estimada de 95.1% para a classe predita. Essa interpretação deve ser vista com cautela, porque as evidências fornecidas descrevem variáveis do sinal e bandas harmônicas relevantes, sem constituir confirmação direta de um mecanismo específico de falha."
}""",
            ],
            "contextualizado": [
                """{
  "interpretacao_vibracional": "- nível geral de vibração no eixo Y = 0.214310, que respondeu por cerca de 28.7% do impacto local da classificação.\\n- concentração de vibração na harmônica de 1ª ordem do segundo estágio no eixo X = 0.004812, que respondeu por cerca de 24.9% do impacto local da classificação.\\n- maior intensidade observada nessa harmônica do segundo estágio no eixo Z = 0.021334, que respondeu por cerca de 19.8% do impacto local da classificação.\\n- diferença entre picos e nível geral no eixo Z = 5.184220, que respondeu por cerca de 11.5% do impacto local da classificação.\\n- nível geral de vibração no eixo X = 0.118400, que respondeu por cerca de 8.2% do impacto local da classificação.",
  "interpretacao_classe_predita": "De forma contextualizada, o conjunto de evidências é compatível com a classe predita de Dente Trincado, com probabilidade estimada de 99.3% para a classe predita. Essa leitura deve ser tratada com cautela, porque os indicadores apresentados caracterizam o comportamento vibracional observado, mas não confirmam sozinhos um mecanismo específico de falha."
}""",
                """{
  "interpretacao_vibracional": "- nível geral de vibração no eixo X = 0.091390, que respondeu por cerca de 24.6% do impacto local da classificação.\\n- pico mais forte do sinal no eixo Y = 0.969533, que respondeu por cerca de 21.4% do impacto local da classificação.\\n- concentração de vibração na harmônica de 2ª ordem do primeiro estágio no eixo Y = 0.008880, que respondeu por cerca de 15.6% do impacto local da classificação.\\n- maior intensidade observada na harmônica de 5ª ordem do primeiro estágio no eixo X = 0.003614, que respondeu por cerca de 9.9% do impacto local da classificação.\\n- outra concentração de vibração na harmônica de 5ª ordem do primeiro estágio no eixo Y = 0.000314, com participação relativa negativa de 9.4% do impacto local da classificação.",
  "interpretacao_classe_predita": "De forma contextualizada, o conjunto de evidências é compatível com a classe predita de Desgaste Superficial, com probabilidade estimada de 96.4% para a classe predita. Essa interpretação deve ser vista com cautela, porque os indicadores descrevem o padrão vibracional observado, sem confirmar diretamente um mecanismo específico de falha."
}""",
                """{
  "interpretacao_vibracional": "- nível geral de vibração no eixo Y = 0.170664, que respondeu por cerca de 31.5% do impacto local da classificação.\\n- nível geral de vibração no eixo Z = 0.240076, que respondeu por cerca de 18.6% do impacto local da classificação.\\n- concentração de vibração na harmônica de 5ª ordem do segundo estágio no eixo Y = 0.000190, que respondeu por cerca de 10.6% do impacto local da classificação.\\n- nível geral de vibração no eixo X = 0.083640, que respondeu por cerca de 7.6% do impacto local da classificação.\\n- concentração de vibração na harmônica de 1ª ordem do primeiro estágio no eixo Y = 0.001764, com participação relativa negativa de 6.4% do impacto local da classificação.",
  "interpretacao_classe_predita": "De forma contextualizada, o conjunto de evidências é compatível com a classe predita de Normal, com probabilidade estimada de 95.1% para a classe predita. Essa leitura deve ser tratada com cautela, porque os indicadores apresentados descrevem o comportamento vibracional da janela, sem constituir confirmação direta de falha."
}""",
            ],
            "didatico": [
                """{
  "interpretacao_vibracional": "- nível geral de vibração no eixo Y = 0.214310, que teve participação de cerca de 28.7% na explicação da decisão do modelo nesta janela.\\n- concentração de vibração em uma faixa específica do segundo estágio no eixo X = 0.004812, que teve participação de cerca de 24.9% na explicação da decisão do modelo nesta janela.\\n- maior intensidade dessa faixa no eixo Z = 0.021334, que teve participação de cerca de 19.8% na explicação da decisão do modelo nesta janela.\\n- diferença entre picos e nível geral no eixo Z = 5.184220, que teve participação de cerca de 11.5% na explicação da decisão do modelo nesta janela.\\n- nível geral de vibração no eixo X = 0.118400, que teve participação de cerca de 8.2% na explicação da decisão do modelo nesta janela.",
  "interpretacao_classe_predita": "De forma didática, esse conjunto de variáveis levou o modelo a classificar a janela como Dente Trincado, com probabilidade estimada de 99.3% para a classe predita. Essa leitura deve ser interpretada com cautela, porque os indicadores descrevem o comportamento do sinal, mas não confirmam sozinhos uma falha específica."
}""",
                """{
  "interpretacao_vibracional": "- nível geral de vibração no eixo X = 0.091390, que teve participação de cerca de 24.6% na explicação da decisão do modelo nesta janela.\\n- maior pico do sinal no eixo Y = 0.969533, que teve participação de cerca de 21.4% na explicação da decisão do modelo nesta janela.\\n- concentração de vibração em uma faixa específica do primeiro estágio no eixo Y = 0.008880, que teve participação de cerca de 15.6% na explicação da decisão do modelo nesta janela.\\n- maior intensidade dessa faixa no eixo X = 0.003614, que teve participação de cerca de 9.9% na explicação da decisão do modelo nesta janela.\\n- outra concentração de vibração em faixa específica do primeiro estágio no eixo Y = 0.000314, que contribuiu negativamente com cerca de 9.4% na explicação da decisão do modelo nesta janela.",
  "interpretacao_classe_predita": "De forma didática, esse conjunto de variáveis levou o modelo a classificar a janela como Desgaste Superficial, com probabilidade estimada de 96.4% para a classe predita. Essa leitura deve ser interpretada com cautela, porque os indicadores descrevem o sinal medido, mas não confirmam diretamente uma falha específica."
}""",
                """{
  "interpretacao_vibracional": "- nível geral de vibração no eixo Y = 0.170664, que teve participação de cerca de 31.5% na explicação da decisão do modelo nesta janela.\\n- nível geral de vibração no eixo Z = 0.240076, que teve participação de cerca de 18.6% na explicação da decisão do modelo nesta janela.\\n- concentração de vibração em uma faixa específica do segundo estágio no eixo Y = 0.000190, que teve participação de cerca de 10.6% na explicação da decisão do modelo nesta janela.\\n- nível geral de vibração no eixo X = 0.083640, que teve participação de cerca de 7.6% na explicação da decisão do modelo nesta janela.\\n- concentração de vibração em uma faixa específica do primeiro estágio no eixo Y = 0.001764, que contribuiu negativamente com cerca de 6.4% na explicação da decisão do modelo nesta janela.",
  "interpretacao_classe_predita": "De forma didática, esse conjunto de variáveis levou o modelo a classificar a janela como Normal, com probabilidade estimada de 95.1% para a classe predita. Essa leitura deve ser interpretada com cautela, porque os indicadores mostram o comportamento do sinal nesta janela, sem confirmar diretamente uma falha."
}""",
            ],
        }

        assistant_examples = examples_by_audience.get(audience_profile, examples_by_audience["tecnico"])

        return [
            {"role": "user", "content": example_user_1},
            {"role": "assistant", "content": assistant_examples[0]},
            {"role": "user", "content": example_user_2},
            {"role": "assistant", "content": assistant_examples[1]},
            {"role": "user", "content": example_user_3},
            {"role": "assistant", "content": assistant_examples[2]},
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
                f"RMS value on axis {axis} = {value:.6f}; "
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
                f"spectral energy within the ±10 Hz band around the {self._ordinal_order_label(harmonic)}-order harmonic of Fm1 on axis {axis} = {value:.6f}; "
                f"contributed {direction} to the predicted class; "
                f"approximate local importance = {impact_pct:.1f}%."
            )
        if feature.startswith("amp_max_fm1_h"):
            parts = feature.split("_")
            harmonic = parts[3].replace("h", "")
            axis = parts[-1].upper()
            return (
                f"maximum spectral amplitude within the ±10 Hz band around the {self._ordinal_order_label(harmonic)}-order harmonic of Fm1 on axis {axis} = {value:.6f}; "
                f"contributed {direction} to the predicted class; "
                f"approximate local importance = {impact_pct:.1f}%."
            )
        if feature.startswith("energy_fm2_h"):
            parts = feature.split("_")
            harmonic = parts[2].replace("h", "")
            axis = parts[-1].upper()
            return (
                f"spectral energy within the ±10 Hz band around the {self._ordinal_order_label(harmonic)}-order harmonic of Fm2 on axis {axis} = {value:.6f}; "
                f"contributed {direction} to the predicted class; "
                f"approximate local importance = {impact_pct:.1f}%."
            )
        if feature.startswith("amp_max_fm2_h"):
            parts = feature.split("_")
            harmonic = parts[3].replace("h", "")
            axis = parts[-1].upper()
            return (
                f"maximum spectral amplitude within the ±10 Hz band around the {self._ordinal_order_label(harmonic)}-order harmonic of Fm2 on axis {axis} = {value:.6f}; "
                f"contributed {direction} to the predicted class; "
                f"approximate local importance = {impact_pct:.1f}%."
            )
        if feature.startswith("energy_rel_fm2_h"):
            parts = feature.split("_")
            harmonic = parts[3].replace("h", "")
            axis = parts[-1].upper()
            return (
                f"relative spectral energy within the ±10 Hz band around the {self._ordinal_order_label(harmonic)}-order harmonic of Fm2 on axis {axis} = {value:.6f}; "
                f"contributed {direction} to the predicted class; "
                f"approximate local importance = {impact_pct:.1f}%."
            )
        if feature.startswith("ratio_energy_fm2_h"):
            parts = feature.split("_")
            ratio_name = parts[3].upper()
            axis = parts[-1].upper()
            return (
                f"harmonic-band spectral energy ratio {ratio_name} on axis {axis} = {value:.6f}; "
                f"contributed {direction} to the predicted class; "
                f"approximate local importance = {impact_pct:.1f}%."
            )
        if feature.startswith("ratio_amp_fm2_h"):
            parts = feature.split("_")
            ratio_name = parts[3].upper()
            axis = parts[-1].upper()
            return (
                f"harmonic-band maximum spectral amplitude ratio {ratio_name} on axis {axis} = {value:.6f}; "
                f"contributed {direction} to the predicted class; "
                f"approximate local importance = {impact_pct:.1f}%."
            )
        return (
            f"{feature} = {value:.6f}; "
            f"contributed {direction} to the predicted class; "
            f"approximate local importance = {impact_pct:.1f}%."
        )

    def _ordinal_order_label(self, harmonic: str) -> str:
        order = int(harmonic)
        if 10 <= order % 100 <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(order % 10, "th")
        return f"{order}{suffix}"

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

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
        audience_profile: AudienceProfile = "engenharia",
        concise: bool = False,
    ) -> tuple[str, str, list[dict[str, str]]]:
        del condicao_operacao
        del concise
        audience_profile = self._normalize_audience_profile(audience_profile)
        audience_rule = escape(self._audience_profile_rule(audience_profile))
        audience_vibrational_instruction = escape(self._audience_vibrational_instruction(audience_profile))
        audience_mechanical_instruction = escape(self._audience_mechanical_instruction(audience_profile))

        system_prompt = f"""<SYSTEM_PROMPT>
  <ROLE>You are a specialist in vibration analysis for planetary gearboxes.</ROLE>
  <FIXED_CONTEXT>
    <EQUIPMENT>two-stage planetary gearbox</EQUIPMENT>
    <MONITORED_COMPONENT>second-stage sun gear</MONITORED_COMPONENT>
  </FIXED_CONTEXT>
  <RULES>
    <RULE>Use only the listed evidence.</RULE>
    <RULE>Do not invent ranges, thresholds, causes, severity, symptoms, or mechanisms.</RULE>
    <RULE>Do not compare values with normal ranges, baselines, or expected values.</RULE>
    <RULE>Mention only concepts supported by the variable families present in OBSERVED_EVIDENCE.</RULE>
    <RULE>Prefer technically precise wording in Brazilian Portuguese, especially for spectral quantities, harmonic orders, and time-domain metrics.</RULE>
    <RULE>Do not infer a specific damage mechanism solely from one isolated harmonic-band metric.</RULE>
    <RULE>Use cautious wording for mechanical interpretation.</RULE>
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
    <VIBRATIONAL>Use all 5 evidence items listed in OBSERVED_EVIDENCE.</VIBRATIONAL>
    <VIBRATIONAL>Cite variable, axis or harmonic order, value, and approximate local importance.</VIBRATIONAL>
    <VIBRATIONAL>Adapt the wording to AUDIENCE_PROFILE while preserving technical truthfulness.</VIBRATIONAL>
    <VIBRATIONAL>If you summarize, use only concepts supported by the listed evidence.</VIBRATIONAL>
    <VIBRATIONAL>{audience_vibrational_instruction}</VIBRATIONAL>
    <MECHANICAL>State only whether the evidence is compatible with the predicted class, with cautious wording.</MECHANICAL>
    <MECHANICAL>Include the predicted class probability as "probabilidade estimada de X% para a classe predita".</MECHANICAL>
    <MECHANICAL>Adapt the wording to AUDIENCE_PROFILE while avoiding maintenance recommendations.</MECHANICAL>
    <MECHANICAL>{audience_mechanical_instruction}</MECHANICAL>
  </TASK>
  <PROCEDURE>
    <STEP>Read only the evidence items listed in OBSERVED_EVIDENCE.</STEP>
    <STEP>Read AUDIENCE_PROFILE and follow the corresponding style rule.</STEP>
    <STEP>Identify which variable families are present.</STEP>
    <STEP>Use only the concepts supported by those variable families.</STEP>
    <STEP>Write interpretacao_vibracional using all 5 listed evidence items, including value and approximate local importance.</STEP>
    <STEP>Write interpretacao_mecanica only as compatibility with the predicted class, including the predicted probability and cautious wording.</STEP>
    <STEP>Return only the JSON object defined in OUTPUT_FORMAT.</STEP>
  </PROCEDURE>
  <OUTPUT_FORMAT>{{
  "interpretacao_vibracional": "...",
  "interpretacao_mecanica": "..."
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
        audience_profile: AudienceProfile = "engenharia",
    ) -> dict[str, Any]:
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
                    "response_format": "raw_text",
                    "unstructured_response": content.strip(),
                    "interpretacao_vibracional": "",
                    "interpretacao_mecanica": "",
                }
            raise

        return {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "raw_response": raw_response,
            "response_format": "json",
            "unstructured_response": None,
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
                    "interpretacao_mecanica": {"type": "string"},
                },
                "required": [
                    "interpretacao_vibracional",
                    "interpretacao_mecanica",
                ],
                "additionalProperties": False,
            },
        }

    def _normalize_audience_profile(self, audience_profile: AudienceProfile) -> AudienceProfile:
        if audience_profile in {"engenharia", "manutencao", "operacao"}:
            return audience_profile
        return "engenharia"

    def _audience_profile_rule(self, audience_profile: AudienceProfile) -> str:
        rules = {
            "engenharia": (
                "Use the most rigorous technical vocabulary available in the provided evidence. "
                "Distinguish clearly between time-domain metrics and frequency-domain harmonic-band metrics."
            ),
            "manutencao": (
                "Keep technical accuracy, but prefer practical maintenance-oriented wording. "
                "Explain what each relevant quantity represents in practice, such as level of overall vibration, "
                "strongest signal peak, vibration concentrated at a specific harmonic order of the first or second stage, "
                "or maximum intensity observed at that harmonic. Avoid overly academic wording."
            ),
            "operacao": (
                "Use simpler wording, shorter sentences, and minimal jargon. "
                "Do not use the terms RMS, kurtosis, crest factor, harmonic, spectral energy, spectral amplitude, Fm1, or Fm2 in the final text. "
                "Translate them into accessible phrases such as overall vibration level, strongest signal peak, difference between peaks and overall level, "
                "or vibration concentrated in a specific band of the first or second stage. Keep the exact values and the axis or stage reference."
            ),
        }
        return rules[self._normalize_audience_profile(audience_profile)]

    def _audience_vibrational_instruction(self, audience_profile: AudienceProfile) -> str:
        instructions = {
            "engenharia": (
                "Preserve the most rigorous technical wording and explicit distinction between time-domain and harmonic-band evidence."
            ),
            "manutencao": (
                "Explain in practical vibration-monitoring terms what each variable represents and what in the signal deserves technical attention."
            ),
            "operacao": (
                "Explain the same evidence in plain operational language focused on what the machine signal is showing."
            ),
        }
        return instructions[self._normalize_audience_profile(audience_profile)]

    def _audience_mechanical_instruction(self, audience_profile: AudienceProfile) -> str:
        instructions = {
            "engenharia": "Keep the most technical tone.",
            "manutencao": "Keep a practical diagnostic tone centered on what the signal is indicating.",
            "operacao": "Explicitly state in simple language that the listed variables led the model to the predicted class.",
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
      <EVIDENCE index="5">spectral energy within the ±10 Hz band around the 5th-order harmonic of Fm1 on axis Y = 0.000314; contributed positively to the predicted class; approximate local importance = 9.4%.</EVIDENCE>
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
      <EVIDENCE index="5">spectral energy within the ±10 Hz band around the 1st-order harmonic of Fm1 on axis Y = 0.001764; contributed positively to the predicted class; approximate local importance = 6.4%.</EVIDENCE>
  </OBSERVED_EVIDENCE>
</USER_PROMPT>"""

        examples_by_audience = {
            "engenharia": [
                """{
  "interpretacao_vibracional": "A janela apresenta valor RMS no eixo Y = 0.214310, com importância local aproximada de 28.7%, energia espectral na faixa de ±10 Hz em torno da harmônica de 1ª ordem de Fm2 no eixo X = 0.004812, com 24.9%, amplitude espectral máxima na faixa de ±10 Hz em torno da harmônica de 2ª ordem de Fm2 no eixo Z = 0.021334, com 19.8%, fator de crista no eixo Z = 5.184220, com 11.5%, e valor RMS no eixo X = 0.118400, com 8.2%. Essas evidências descrevem valor RMS e nível global de vibração, energia espectral em bandas harmônicas, amplitude espectral máxima em banda harmônica e fator de crista nesta janela.",
  "interpretacao_mecanica": "O conjunto de evidências é compatível com a classe predita de Dente Trincado, com probabilidade estimada de 99.3% para a classe predita. Essa interpretação deve ser vista com cautela, porque as evidências fornecidas descrevem variáveis do sinal e bandas harmônicas relevantes, mas não constituem confirmação direta de um mecanismo específico de falha."
}""",
                """{
  "interpretacao_vibracional": "A janela apresenta valor RMS no eixo X = 0.091390, com importância local aproximada de 24.6%, valor de pico no eixo Y = 0.969533, com 21.4%, energia espectral na faixa de ±10 Hz em torno da harmônica de 2ª ordem de Fm1 no eixo Y = 0.008880, com 15.6%, amplitude espectral máxima na faixa de ±10 Hz em torno da harmônica de 5ª ordem de Fm1 no eixo X = 0.003614, com 9.9%, e energia espectral na faixa de ±10 Hz em torno da harmônica de 5ª ordem de Fm1 no eixo Y = 0.000314, com 9.4%. Essas evidências descrevem valor RMS e nível global de vibração, valor de pico no sinal no tempo, energia espectral em bandas harmônicas e amplitude espectral máxima em banda harmônica nesta janela.",
  "interpretacao_mecanica": "O conjunto de evidências é compatível com a classe predita de Desgaste Superficial, com probabilidade estimada de 96.4% para a classe predita. Essa interpretação deve ser vista com cautela, porque as evidências fornecidas descrevem variáveis do sinal e bandas harmônicas relevantes, mas não constituem confirmação direta de um mecanismo específico de falha."
}""",
                """{
  "interpretacao_vibracional": "A janela apresenta valor RMS no eixo Y = 0.170664, com importância local aproximada de 31.5%, valor RMS no eixo Z = 0.240076, com 18.6%, energia espectral na faixa de ±10 Hz em torno da harmônica de 5ª ordem de Fm2 no eixo Y = 0.000190, com 10.6%, valor RMS no eixo X = 0.083640, com 7.6%, e energia espectral na faixa de ±10 Hz em torno da harmônica de 1ª ordem de Fm1 no eixo Y = 0.001764, com 6.4%. Essas evidências descrevem valor RMS e nível global de vibração, além de energia espectral em bandas harmônicas nesta janela.",
  "interpretacao_mecanica": "O conjunto de evidências é compatível com a classe predita de Normal, com probabilidade estimada de 95.1% para a classe predita. Essa interpretação deve ser vista com cautela, porque as evidências fornecidas descrevem variáveis do sinal e bandas harmônicas relevantes, sem constituir confirmação direta de um mecanismo específico de falha."
}""",
            ],
            "manutencao": [
                """{
  "interpretacao_vibracional": "Na prática, os fatores mais influentes nesta janela foram o nível geral de vibração no eixo Y = 0.214310, com importância local aproximada de 28.7%, a concentração de vibração na harmônica de 1ª ordem do segundo estágio no eixo X = 0.004812, com 24.9%, a maior intensidade observada nessa harmônica do segundo estágio no eixo Z = 0.021334, com 19.8%, a diferença entre picos e nível geral no eixo Z = 5.184220, com 11.5%, e o nível geral de vibração no eixo X = 0.118400, com 8.2%. Essas evidências mostram que a janela combinou aumento do nível geral de vibração com vibração concentrada em harmônicas específicas do segundo estágio.",
  "interpretacao_mecanica": "Para manutenção, o conjunto de evidências é compatível com a classe predita de Dente Trincado, com probabilidade estimada de 99.3% para a classe predita. Essa leitura deve ser tratada com cautela, porque os indicadores apresentados caracterizam o comportamento vibracional observado, mas não confirmam sozinhos um mecanismo específico de falha."
}""",
                """{
  "interpretacao_vibracional": "Na prática, os fatores mais influentes nesta janela foram o nível geral de vibração no eixo X = 0.091390, com importância local aproximada de 24.6%, o pico mais forte do sinal no eixo Y = 0.969533, com 21.4%, a concentração de vibração na harmônica de 2ª ordem do primeiro estágio no eixo Y = 0.008880, com 15.6%, a maior intensidade observada na harmônica de 5ª ordem do primeiro estágio no eixo X = 0.003614, com 9.9%, e outra concentração de vibração na harmônica de 5ª ordem do primeiro estágio no eixo Y = 0.000314, com 9.4%. Essas evidências mostram a combinação de aumento do nível geral de vibração, presença de picos mais fortes e vibração concentrada em harmônicas específicas do primeiro estágio.",
  "interpretacao_mecanica": "Para manutenção, o conjunto de evidências é compatível com a classe predita de Desgaste Superficial, com probabilidade estimada de 96.4% para a classe predita. Essa interpretação deve ser vista com cautela, porque os indicadores descrevem o padrão vibracional observado, sem confirmar diretamente um mecanismo específico de falha."
}""",
                """{
  "interpretacao_vibracional": "Na prática, os fatores mais influentes nesta janela foram o nível geral de vibração no eixo Y = 0.170664, com importância local aproximada de 31.5%, o nível geral de vibração no eixo Z = 0.240076, com 18.6%, a concentração de vibração na harmônica de 5ª ordem do segundo estágio no eixo Y = 0.000190, com 10.6%, o nível geral de vibração no eixo X = 0.083640, com 7.6%, e a concentração de vibração na harmônica de 1ª ordem do primeiro estágio no eixo Y = 0.001764, com 6.4%. Essas evidências mostram predominância do nível geral de vibração, acompanhada por vibração localizada em harmônicas específicas dos dois estágios.",
  "interpretacao_mecanica": "Para manutenção, o conjunto de evidências é compatível com a classe predita de Normal, com probabilidade estimada de 95.1% para a classe predita. Essa leitura deve ser tratada com cautela, porque os indicadores apresentados descrevem o comportamento vibracional da janela, sem constituir confirmação direta de falha."
}""",
            ],
            "operacao": [
                """{
  "interpretacao_vibracional": "Nesta janela, os fatores mais influentes foram o nível geral de vibração no eixo Y = 0.214310, com importância local aproximada de 28.7%, a concentração de vibração em uma faixa específica do segundo estágio no eixo X = 0.004812, com 24.9%, a maior intensidade dessa faixa no eixo Z = 0.021334, com 19.8%, a diferença entre picos e nível geral no eixo Z = 5.184220, com 11.5%, e o nível geral de vibração no eixo X = 0.118400, com 8.2%. Em termos simples, essas variáveis fizeram o modelo reconhecer um padrão de vibração compatível com a classe predita.",
  "interpretacao_mecanica": "Em termos simples, esse conjunto de variáveis levou o modelo a classificar a janela como Dente Trincado, com probabilidade estimada de 99.3% para a classe predita. Essa leitura deve ser interpretada com cautela, porque os indicadores descrevem o comportamento do sinal, mas não confirmam sozinhos uma falha específica."
}""",
                """{
  "interpretacao_vibracional": "Nesta janela, os fatores mais influentes foram o nível geral de vibração no eixo X = 0.091390, com importância local aproximada de 24.6%, o maior pico do sinal no eixo Y = 0.969533, com 21.4%, a concentração de vibração em uma faixa específica do primeiro estágio no eixo Y = 0.008880, com 15.6%, a maior intensidade dessa faixa no eixo X = 0.003614, com 9.9%, e outra concentração de vibração em faixa específica do primeiro estágio no eixo Y = 0.000314, com 9.4%. Em termos simples, essas variáveis fizeram o modelo reconhecer um padrão de vibração compatível com a classe predita.",
  "interpretacao_mecanica": "Em termos simples, esse conjunto de variáveis levou o modelo a classificar a janela como Desgaste Superficial, com probabilidade estimada de 96.4% para a classe predita. Essa leitura deve ser interpretada com cautela, porque os indicadores descrevem o sinal medido, mas não confirmam diretamente uma falha específica."
}""",
                """{
  "interpretacao_vibracional": "Nesta janela, os fatores mais influentes foram o nível geral de vibração no eixo Y = 0.170664, com importância local aproximada de 31.5%, o nível geral de vibração no eixo Z = 0.240076, com 18.6%, a concentração de vibração em uma faixa específica do segundo estágio no eixo Y = 0.000190, com 10.6%, o nível geral de vibração no eixo X = 0.083640, com 7.6%, e a concentração de vibração em uma faixa específica do primeiro estágio no eixo Y = 0.001764, com 6.4%. Em termos simples, essas variáveis fizeram o modelo reconhecer um padrão de vibração mais próximo da condição Normal.",
  "interpretacao_mecanica": "Em termos simples, esse conjunto de variáveis levou o modelo a classificar a janela como Normal, com probabilidade estimada de 95.1% para a classe predita. Essa leitura deve ser interpretada com cautela, porque os indicadores mostram o comportamento do sinal nesta janela, sem confirmar diretamente uma falha."
}""",
            ],
        }

        assistant_examples = examples_by_audience.get(audience_profile, examples_by_audience["engenharia"])

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

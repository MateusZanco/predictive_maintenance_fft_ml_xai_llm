# Sistema Embarcado de Diagnóstico Explicável por Vibração

Aplicação para análise de sinais de vibração de uma caixa de engrenagens planetárias de dois estágios, com:

- extração de sinais no domínio do tempo e da frequência;
- classificação automática da janela de vibração;
- explicabilidade local por contribuição de features;
- geração de interpretação textual com LLM;
- interface web para inspeção do sinal bruto, FFT, classe predita, contribuições locais e tempos de processamento.

O foco do repositório, hoje, está na aplicação embarcada executada na Rock Pi. Os notebooks continuam no projeto como apoio para treinamento, validação e exportação dos artefatos do modelo.

## Visão Geral

O pipeline da aplicação segue esta sequência:

1. carregar uma amostra `.npz` com os sinais dos eixos `x`, `y` e `z`;
2. segmentar o sinal em janelas;
3. extrair features no tempo e em bandas harmônicas centradas em `Fm1` e `Fm2`;
4. executar o modelo tabular treinado;
5. calcular as contribuições locais das features;
6. selecionar as `top_k` features mais influentes;
7. montar o prompt para a LLM;
8. devolver a interpretação textual e os metadados de auditoria para a interface.

## Arquitetura da Aplicação

### Backend

O backend usa `FastAPI` e está em [`app/backend`](app/backend).

Componentes principais:

- [`main.py`](app/backend/main.py): expõe a API HTTP e orquestra o fluxo completo.
- [`model_service.py`](app/backend/model_service.py): carrega o modelo serializado, extrai features, calcula predição e explicabilidade local.
- [`fft_service.py`](app/backend/fft_service.py): funções de FFT, métricas no tempo e extração das bandas harmônicas.
- [`llm_service.py`](app/backend/llm_service.py): monta o prompt e chama o servidor LLM compatível com `/v1/chat/completions`.
- [`sample_service.py`](app/backend/sample_service.py): leitura dos arquivos de amostra.
- [`system_metrics.py`](app/backend/system_metrics.py): telemetria da placa.
- [`schemas.py`](app/backend/schemas.py): contratos da API.

### Frontend

O frontend usa `React + Vite` e está em [`app/frontend`](app/frontend).

A interface permite:

- visualizar o sinal bruto da janela;
- visualizar a FFT;
- consultar a classe predita e as probabilidades;
- inspecionar as contribuições locais das features;
- gerar explicações LLM com diferentes estratégias e perfis de público;
- visualizar tempos de processamento e status da placa;
- auditar o `system prompt`, o `user prompt` e a resposta bruta do modelo.

### LLM

A aplicação não sobe o servidor da LLM internamente. Ela espera um servidor compatível com o endpoint OpenAI-style:

`http://127.0.0.1:8080/v1/chat/completions`

No `docker-compose` atual, o backend aponta por padrão para:

- `LLM_CHAT_COMPLETIONS_URL=http://127.0.0.1:8080/v1/chat/completions`
- `LLM_MODEL=Qwen3-0.6B-Q8_0.gguf`

Esse valor pode ser alterado por variável de ambiente sem mudar o código.

## Estrutura do Repositório

```text
app/
  backend/
    fft_service.py
    llm_service.py
    main.py
    model_service.py
    sample_service.py
    schemas.py
    system_metrics.py
  frontend/
    src/
      api.js
      App.jsx
docker/
  Dockerfile
  docker-compose.yml
notebooks/
outputs/
  model_artifacts_rockpi_simplificado/
  rockpi_test_samples/
README.md
```

## Requisitos

### Infraestrutura

- Docker e Docker Compose para execução embarcada;
- servidor LLM compatível com `/v1/chat/completions`;
- artefatos do modelo já exportados;
- amostras `.npz` já preparadas para a aplicação.

### Dependências do backend

Definidas em [`app/backend/requirements.txt`](app/backend/requirements.txt):

- `fastapi`
- `uvicorn`
- `numpy`
- `pydantic`
- `joblib`
- `pandas`
- `lightgbm`
- `scikit-learn`
- `xgboost`
- `psutil`
- `shap`

## Dados e Artefatos Esperados

### Amostras de inferência

O backend espera as amostras em:

- `outputs/rockpi_test_samples`

No container, esse diretório é montado como:

- `/app/runtime_samples`

### Artefatos do modelo

O backend espera os artefatos em:

- `outputs/model_artifacts_rockpi_simplificado`

Arquivos principais:

- `modelo_rockpi_simplificado.joblib`
- `feature_columns_rockpi_simplificado.json`
- `model_metadata_rockpi_simplificado.json`

No container, esse diretório é montado como:

- `/app/model_artifacts`

## Como Executar

### 1. Garantir os diretórios de entrada

Antes de subir a aplicação, confirme que estes diretórios existem:

- `outputs/rockpi_test_samples`
- `outputs/model_artifacts_rockpi_simplificado`

### 2. Garantir o servidor da LLM

O backend da aplicação depende de um servidor LLM externo disponível no host, por padrão em:

`http://127.0.0.1:8080/v1/chat/completions`

### 3. Subir a aplicação

No diretório [`docker`](docker):

```bash
cd docker
docker compose up --build
```

### 4. Acessar a interface

Com a aplicação em execução, a interface fica disponível pela porta definida em:

- `APP_PORT`

Valor padrão atual:

- `8000`

## Configuração por Variáveis de Ambiente

As principais variáveis hoje são:

### Aplicação

- `APP_PORT`: porta do backend HTTP.
- `ROCKPI_SAMPLE_DIR`: diretório das amostras.
- `ROCKPI_MODEL_ARTIFACTS_DIR`: diretório dos artefatos do modelo.
- `ROCKPI_STATIC_DIR`: diretório do frontend compilado.

### LLM

- `LLM_CHAT_COMPLETIONS_URL`
- `LLM_MODEL`
- `LLM_TEMPERATURE`
- `LLM_TOP_P`
- `LLM_MAX_TOKENS`
- `LLM_TIMEOUT_SECONDS`

### Explicabilidade local

- `ROCKPI_SHAP_BACKEND`

Valores suportados no backend atual:

- `tree_explainer`
- `native`

O valor padrão atual em [`main.py`](app/backend/main.py) é:

- `tree_explainer`

## Endpoints Principais

### Saúde e telemetria

- `GET /api/health`
- `GET /api/system-status`

### Dados da amostra

- `GET /api/samples`
- `GET /api/samples/{sample_id}/meta`

### Sinal e espectro

- `POST /api/signal`
- `POST /api/fft`
- `POST /api/features`

### Inferência e explicabilidade

- `POST /api/predict`
- `POST /api/shap`
- `POST /api/explain`

## Explicabilidade Local

O backend calcula as contribuições locais a partir do modelo tabular e ranqueia as features por impacto absoluto local.

Hoje, o percentual mostrado para cada feature representa:

`|contribuição da feature| / soma(|contribuições de todas as features|) * 100`

Ou seja:

- não é porcentagem da probabilidade;
- não é porcentagem da falha física;
- é participação relativa no impacto explicativo local absoluto da classe predita.

## Prompt da Aplicação

Esta é a parte mais importante do fluxo de explicação da aplicação.

### Objetivo do prompt

O prompt recebe:

- a classe predita;
- a probabilidade da classe predita;
- as `5` features mais influentes na decisão local;
- a direção de cada contribuição, positiva ou negativa;
- o perfil de público da resposta.

Ele deve devolver um JSON com:

- `interpretacao_vibracional`
- `interpretacao_classe_predita`

### Estratégias de prompt

O endpoint `/api/explain` aceita:

- `prompt_strategy = "few_shot"`
- `prompt_strategy = "zero_shot"`

#### `few_shot`

Usa:

- `system prompt`
- `3` exemplos do perfil ativo
- `user prompt` da janela atual

Os `3` exemplos atuais cobrem:

- `Dente Trincado`
- `Desgaste Superficial`
- `Normal`

Dois desses exemplos incluem contribuição negativa, para ensinar o modelo a lidar com evidência contrária à classe predita.

#### `zero_shot`

Usa:

- `system prompt`
- `user prompt` da janela atual

Sem exemplos.

Se a resposta vier sem JSON válido:

- no `zero_shot`, o backend retorna `response_format = "raw_text"`;
- no `few_shot`, o backend trata isso como erro.

### Perfis de público

O prompt adapta o estilo para três perfis:

- `engenharia`
- `manutencao`
- `operacao`

#### Engenharia

Foco:

- terminologia técnica rigorosa;
- distinção explícita entre tempo e frequência;
- linguagem mais próxima de análise vibracional e espectral.

Percentuais positivos:

- `participação relativa de X% no impacto explicativo local absoluto`

Percentuais negativos:

- `participação relativa de X% no impacto explicativo local absoluto, atuando em sentido oposto à classe predita`

#### Manutenção

Foco:

- precisão técnica com linguagem mais prática;
- explicar o que a variável representa em termos de monitoramento vibracional.

Percentuais positivos:

- `respondeu por cerca de X% do impacto local da classificação`

Percentuais negativos:

- `com participação relativa negativa de X% do impacto local da classificação`

#### Operação

Foco:

- linguagem mais simples;
- menos jargão;
- ênfase no que o sinal mostrou e no que levou a decisão do modelo.

Percentuais positivos:

- `teve participação de cerca de X% na explicação da decisão do modelo nesta janela`

Percentuais negativos:

- `contribuiu negativamente com cerca de X% na explicação da decisão do modelo nesta janela`

### Estrutura do prompt

O prompt é montado em XML simplificado.

#### `system prompt`

O `system prompt` contém:

- `<ROLE>`
- `<FIXED_CONTEXT>`
- `<RULES>`
- `<LEGEND>`
- `<ACTIVE_AUDIENCE_PROFILE>`
- `<TASK>`
- `<PROCEDURE>`
- `<OUTPUT_FORMAT>`

#### `user prompt`

O `user prompt` contém somente a parte variável:

- `<AUDIENCE_PROFILE>`
- `<PREDICTED_CLASS>`
- `<PREDICTED_CLASS_PROBABILITY>`
- `<OBSERVED_EVIDENCE>`

Cada evidência é enviada como:

```xml
<EVIDENCE index="1">
RMS value on axis X = 0.091390; contributed positively to the predicted class; approximate local importance = 18.5%.
</EVIDENCE>
```

### Exemplo completo do prompt em XML

O exemplo abaixo mostra o formato completo do `system prompt` e do `user prompt` enviados à aplicação para uma janela, sem incluir os exemplos adicionais do modo `few_shot`.

```xml
<SYSTEM_PROMPT>
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
  <ACTIVE_AUDIENCE_PROFILE name="engenharia">Use the most rigorous technical vocabulary available in the provided evidence. Distinguish clearly between time-domain metrics and frequency-domain harmonic-band metrics.</ACTIVE_AUDIENCE_PROFILE>
  <TASK>
    <VIBRATIONAL>Write interpretacao_vibracional as a single JSON string with exactly 5 bullet lines, one line for each evidence item, each starting with "- ".</VIBRATIONAL>
    <VIBRATIONAL>In each bullet, cite the variable, axis or harmonic order, value, and relative percentage contribution for the current window.</VIBRATIONAL>
    <VIBRATIONAL>When citing each percentage, use wording such as "participação relativa de X% no impacto explicativo local absoluto". When an evidence item contributed negatively, use wording such as "participação relativa de X% no impacto explicativo local absoluto, atuando em sentido oposto à classe predita".</VIBRATIONAL>
    <VIBRATIONAL>Do not add an opening paragraph or a final synthesis paragraph.</VIBRATIONAL>
    <VIBRATIONAL>Preserve the most rigorous technical wording and explicit distinction between time-domain and harmonic-band evidence.</VIBRATIONAL>
    <PREDICTED_CLASS_INTERPRETATION>Write interpretacao_classe_predita only as cautious compatibility with the predicted class, including "probabilidade estimada de X% para a classe predita".</PREDICTED_CLASS_INTERPRETATION>
    <PREDICTED_CLASS_INTERPRETATION>Keep the most technical tone.</PREDICTED_CLASS_INTERPRETATION>
  </TASK>
  <PROCEDURE>
    <STEP>Read AUDIENCE_PROFILE and OBSERVED_EVIDENCE.</STEP>
    <STEP>Identify the present variable families and whether each evidence item contributed positively or negatively.</STEP>
    <STEP>Use all 5 evidence items in interpretacao_vibracional with the audience-specific percentage wording.</STEP>
    <STEP>Return only the JSON object defined in OUTPUT_FORMAT.</STEP>
  </PROCEDURE>
  <OUTPUT_FORMAT>{
  "interpretacao_vibracional": "...",
  "interpretacao_classe_predita": "..."
}</OUTPUT_FORMAT>
</SYSTEM_PROMPT>

<USER_PROMPT>
  <AUDIENCE_PROFILE>engenharia</AUDIENCE_PROFILE>
  <PREDICTED_CLASS>Desgaste Superficial</PREDICTED_CLASS>
  <PREDICTED_CLASS_PROBABILITY>0.8318</PREDICTED_CLASS_PROBABILITY>
  <OBSERVED_EVIDENCE>
    <EVIDENCE index="1">RMS value on axis X = 0.091390; contributed positively to the predicted class; approximate local importance = 18.5%.</EVIDENCE>
    <EVIDENCE index="2">spectral energy within the ±10 Hz band around the 5th-order harmonic of Fm2 on axis Y = 0.000100; contributed positively to the predicted class; approximate local importance = 15.5%.</EVIDENCE>
    <EVIDENCE index="3">spectral energy within the ±10 Hz band around the 5th-order harmonic of Fm2 on axis Z = 0.000422; contributed positively to the predicted class; approximate local importance = 15.0%.</EVIDENCE>
    <EVIDENCE index="4">spectral energy within the ±10 Hz band around the 5th-order harmonic of Fm1 on axis X = 0.000023; contributed negatively to the predicted class; approximate local importance = 7.0%.</EVIDENCE>
    <EVIDENCE index="5">maximum spectral amplitude within the ±10 Hz band around the 5th-order harmonic of Fm2 on axis Y = 0.003877; contributed positively to the predicted class; approximate local importance = 6.4%.</EVIDENCE>
  </OBSERVED_EVIDENCE>
</USER_PROMPT>
```

No modo `few_shot`, esse par `system + user` é precedido por exemplos fixos do perfil selecionado.

### Sequência completa do modo `few_shot`

Na implementação atual, os exemplos `few_shot` não são inseridos dentro do `system prompt`. A sequência real enviada ao servidor LLM é:

1. `system prompt`
2. `user` de exemplo 1
3. `assistant` de exemplo 1
4. `user` de exemplo 2
5. `assistant` de exemplo 2
6. `user` de exemplo 3
7. `assistant` de exemplo 3
8. `user` da janela atual

Abaixo está um exemplo completo da sequência `few_shot` para o perfil `engenharia`.

#### Exemplo 1: `Dente Trincado`

```xml
<USER_PROMPT>
  <AUDIENCE_PROFILE>engenharia</AUDIENCE_PROFILE>
  <PREDICTED_CLASS>Dente Trincado</PREDICTED_CLASS>
  <PREDICTED_CLASS_PROBABILITY>0.9925</PREDICTED_CLASS_PROBABILITY>
  <OBSERVED_EVIDENCE>
      <EVIDENCE index="1">RMS value on axis Y = 0.214310; contributed positively to the predicted class; approximate local importance = 28.7%.</EVIDENCE>
      <EVIDENCE index="2">spectral energy within the ±10 Hz band around the 1st-order harmonic of Fm2 on axis X = 0.004812; contributed positively to the predicted class; approximate local importance = 24.9%.</EVIDENCE>
      <EVIDENCE index="3">maximum spectral amplitude within the ±10 Hz band around the 2nd-order harmonic of Fm2 on axis Z = 0.021334; contributed positively to the predicted class; approximate local importance = 19.8%.</EVIDENCE>
      <EVIDENCE index="4">crest factor on axis Z = 5.184220; contributed positively to the predicted class; approximate local importance = 11.5%.</EVIDENCE>
      <EVIDENCE index="5">RMS value on axis X = 0.118400; contributed positively to the predicted class; approximate local importance = 8.2%.</EVIDENCE>
  </OBSERVED_EVIDENCE>
</USER_PROMPT>
```

```json
{
  "interpretacao_vibracional": "- valor RMS no eixo Y = 0.214310, com participação relativa de 28.7% no impacto explicativo local absoluto.\n- energia espectral na faixa de ±10 Hz em torno da harmônica de 1ª ordem de Fm2 no eixo X = 0.004812, com participação relativa de 24.9% no impacto explicativo local absoluto.\n- amplitude espectral máxima na faixa de ±10 Hz em torno da harmônica de 2ª ordem de Fm2 no eixo Z = 0.021334, com participação relativa de 19.8% no impacto explicativo local absoluto.\n- fator de crista no eixo Z = 5.184220, com participação relativa de 11.5% no impacto explicativo local absoluto.\n- valor RMS no eixo X = 0.118400, com participação relativa de 8.2% no impacto explicativo local absoluto.",
  "interpretacao_classe_predita": "O conjunto de evidências é compatível com a classe predita de Dente Trincado, com probabilidade estimada de 99.3% para a classe predita. Essa interpretação deve ser vista com cautela, porque as evidências fornecidas descrevem variáveis do sinal e bandas harmônicas relevantes, mas não constituem confirmação direta de um mecanismo específico de falha."
}
```

#### Exemplo 2: `Desgaste Superficial`

```xml
<USER_PROMPT>
  <AUDIENCE_PROFILE>engenharia</AUDIENCE_PROFILE>
  <PREDICTED_CLASS>Desgaste Superficial</PREDICTED_CLASS>
  <PREDICTED_CLASS_PROBABILITY>0.9640</PREDICTED_CLASS_PROBABILITY>
  <OBSERVED_EVIDENCE>
      <EVIDENCE index="1">RMS value on axis X = 0.091390; contributed positively to the predicted class; approximate local importance = 24.6%.</EVIDENCE>
      <EVIDENCE index="2">peak value on axis Y = 0.969533; contributed positively to the predicted class; approximate local importance = 21.4%.</EVIDENCE>
      <EVIDENCE index="3">spectral energy within the ±10 Hz band around the 2nd-order harmonic of Fm1 on axis Y = 0.008880; contributed positively to the predicted class; approximate local importance = 15.6%.</EVIDENCE>
      <EVIDENCE index="4">maximum spectral amplitude within the ±10 Hz band around the 5th-order harmonic of Fm1 on axis X = 0.003614; contributed positively to the predicted class; approximate local importance = 9.9%.</EVIDENCE>
      <EVIDENCE index="5">spectral energy within the ±10 Hz band around the 5th-order harmonic of Fm1 on axis Y = 0.000314; contributed negatively to the predicted class; approximate local importance = 9.4%.</EVIDENCE>
  </OBSERVED_EVIDENCE>
</USER_PROMPT>
```

```json
{
  "interpretacao_vibracional": "- valor RMS no eixo X = 0.091390, com participação relativa de 24.6% no impacto explicativo local absoluto.\n- valor de pico no eixo Y = 0.969533, com participação relativa de 21.4% no impacto explicativo local absoluto.\n- energia espectral na faixa de ±10 Hz em torno da harmônica de 2ª ordem de Fm1 no eixo Y = 0.008880, com participação relativa de 15.6% no impacto explicativo local absoluto.\n- amplitude espectral máxima na faixa de ±10 Hz em torno da harmônica de 5ª ordem de Fm1 no eixo X = 0.003614, com participação relativa de 9.9% no impacto explicativo local absoluto.\n- energia espectral na faixa de ±10 Hz em torno da harmônica de 5ª ordem de Fm1 no eixo Y = 0.000314, com participação relativa de 9.4% no impacto explicativo local absoluto, atuando em sentido oposto à classe predita.",
  "interpretacao_classe_predita": "O conjunto de evidências é compatível com a classe predita de Desgaste Superficial, com probabilidade estimada de 96.4% para a classe predita. Essa interpretação deve ser vista com cautela, porque as evidências fornecidas descrevem variáveis do sinal e bandas harmônicas relevantes, mas não constituem confirmação direta de um mecanismo específico de falha."
}
```

#### Exemplo 3: `Normal`

```xml
<USER_PROMPT>
  <AUDIENCE_PROFILE>engenharia</AUDIENCE_PROFILE>
  <PREDICTED_CLASS>Normal</PREDICTED_CLASS>
  <PREDICTED_CLASS_PROBABILITY>0.9510</PREDICTED_CLASS_PROBABILITY>
  <OBSERVED_EVIDENCE>
      <EVIDENCE index="1">RMS value on axis Y = 0.170664; contributed positively to the predicted class; approximate local importance = 31.5%.</EVIDENCE>
      <EVIDENCE index="2">RMS value on axis Z = 0.240076; contributed positively to the predicted class; approximate local importance = 18.6%.</EVIDENCE>
      <EVIDENCE index="3">spectral energy within the ±10 Hz band around the 5th-order harmonic of Fm2 on axis Y = 0.000190; contributed positively to the predicted class; approximate local importance = 10.6%.</EVIDENCE>
      <EVIDENCE index="4">RMS value on axis X = 0.083640; contributed positively to the predicted class; approximate local importance = 7.6%.</EVIDENCE>
      <EVIDENCE index="5">spectral energy within the ±10 Hz band around the 1st-order harmonic of Fm1 on axis Y = 0.001764; contributed negatively to the predicted class; approximate local importance = 6.4%.</EVIDENCE>
  </OBSERVED_EVIDENCE>
</USER_PROMPT>
```

```json
{
  "interpretacao_vibracional": "- valor RMS no eixo Y = 0.170664, com participação relativa de 31.5% no impacto explicativo local absoluto.\n- valor RMS no eixo Z = 0.240076, com participação relativa de 18.6% no impacto explicativo local absoluto.\n- energia espectral na faixa de ±10 Hz em torno da harmônica de 5ª ordem de Fm2 no eixo Y = 0.000190, com participação relativa de 10.6% no impacto explicativo local absoluto.\n- valor RMS no eixo X = 0.083640, com participação relativa de 7.6% no impacto explicativo local absoluto.\n- energia espectral na faixa de ±10 Hz em torno da harmônica de 1ª ordem de Fm1 no eixo Y = 0.001764, com participação relativa de 6.4% no impacto explicativo local absoluto, atuando em sentido oposto à classe predita.",
  "interpretacao_classe_predita": "O conjunto de evidências é compatível com a classe predita de Normal, com probabilidade estimada de 95.1% para a classe predita. Essa interpretação deve ser vista com cautela, porque as evidências fornecidas descrevem variáveis do sinal e bandas harmônicas relevantes, sem constituir confirmação direta de um mecanismo específico de falha."
}
```

### Regras centrais do prompt

As regras mais relevantes hoje são:

- usar apenas `OBSERVED_EVIDENCE`;
- não inventar faixas, baselines, causas, severidade, sintomas ou mecanismos;
- citar apenas conceitos suportados pelas famílias de variáveis presentes;
- tratar contribuição negativa como evidência contra a classe predita;
- responder somente em português brasileiro;
- retornar somente JSON válido.

### Formato atual da saída

#### `interpretacao_vibracional`

Hoje ela é gerada como:

- uma única string JSON;
- com exatamente `5` linhas;
- cada linha iniciando com `- `;
- uma linha para cada evidência.

Exemplo conceitual:

```json
{
  "interpretacao_vibracional": "- valor RMS no eixo X = ...\n- energia espectral na faixa de ±10 Hz ...\n- ...",
  "interpretacao_classe_predita": "..."
}
```

Na interface, essa string é renderizada como lista com tópicos.

#### `interpretacao_classe_predita`

Hoje ela é:

- um texto curto;
- cauteloso;
- limitado à compatibilidade com a classe predita;
- incluindo a probabilidade estimada do modelo.

### Auditoria do prompt

O endpoint `/api/explain` devolve também:

- `system_prompt`
- `user_prompt`
- `raw_response`

Isso permite auditar exatamente:

- o prompt enviado;
- a estratégia usada;
- o perfil ativo;
- a resposta bruta recebida da LLM.

## Exemplo de Requisição para `/api/explain`

```json
{
  "sample_id": "sample_001",
  "window_index": 0,
  "top_k": 5,
  "prompt_strategy": "few_shot",
  "audience_profile": "engenharia"
}
```

## Treinamento e Notebooks

Os notebooks permanecem no repositório para:

- treinamento de modelos;
- avaliação de features;
- validação experimental;
- exportação dos artefatos usados pela aplicação.

Os artefatos carregados pelo backend são gerados fora da aplicação e depois serializados em `joblib`.

## Observações Importantes

- A aplicação não retreina o modelo em runtime.
- A LLM não substitui a explicabilidade local; ela transforma as contribuições locais em texto.
- O texto retornado não deve ser tratado como confirmação física de mecanismo de falha.
- O modo `few_shot` é o mais estável para o fluxo atual.

## Arquivos Mais Relevantes

- [README.md](README.md)
- [app/backend/main.py](app/backend/main.py)
- [app/backend/model_service.py](app/backend/model_service.py)
- [app/backend/llm_service.py](app/backend/llm_service.py)
- [app/backend/schemas.py](app/backend/schemas.py)
- [app/frontend/src/App.jsx](app/frontend/src/App.jsx)
- [docker/docker-compose.yml](docker/docker-compose.yml)
- [docker/Dockerfile](docker/Dockerfile)

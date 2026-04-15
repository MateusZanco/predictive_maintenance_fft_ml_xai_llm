# predictive_maintenance_fft_ml_xai_llm

## Visão geral

- Este projeto aplica FFT, aprendizado de máquina, XAI e LLM para manutenção preditiva com sinais de vibração de uma caixa de engrenagens planetárias.
- O fluxo principal do notebook cobre carregamento dos sinais, segmentação, análise espectral, extração de features, treino de modelos de classificação, explicabilidade com SHAP e geração opcional de texto técnico com a API da OpenAI.

## Notebook principal

- `predictive_maintenance_fft_ml_xai_llm.ipynb`

## Problema tratado

- O objetivo é classificar a condição da engrenagem solar do segundo estágio da caixa de engrenagens planetárias.

## Mapeamento atual dos rótulos

- Classe 0: Normal
- Classe 1: Desgaste Superficial
- Classe 2: Dente Trincado
- Classe 3: Dente Lascado
- Classe 4: Dente Ausente

## Dados

- Os arquivos do dataset devem ficar dentro da pasta `data`.
- Substitua o campo abaixo pelo link real para download dos arquivos.

Link do Drive:

`https://drive.google.com/drive/folders/1eJWnxC4rEQXuOxAbNJ6IErWKJHqsXry0?usp=drive_link`

### Estrutura esperada

```text
data/
  x_1500_10.npy
  y_1500_10.npy
  z_1500_10.npy
  gt_1500_10.npy
```

### Descrição rápida dos arquivos

- `x_1500_10.npy`: sinal de vibração do eixo X
- `y_1500_10.npy`: sinal de vibração do eixo Y
- `z_1500_10.npy`: sinal de vibração do eixo Z
- `gt_1500_10.npy`: rótulos das classes

## Ambiente

- O projeto usa Python com as dependências listadas em `requirements.txt`.
- Para usar a parte de LLM, copie `.env.example` para `.env` e preencha sua chave da OpenAI.

### Exemplo de `.env`

```text
OPENAI_API_KEY=cole_sua_chave_aqui
OPENAI_MODEL=gpt-5-mini
```

## Como executar

### No Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
Copy-Item .env.example .env
New-Item -ItemType Directory -Path data -Force
jupyter notebook
```

### Depois disso

- Baixe os arquivos do dataset.
- Copie os arquivos para dentro da pasta `data`.
- Abra `predictive_maintenance_fft_ml_xai_llm.ipynb`.
- Execute as células em ordem.

## Etapas do notebook

- Carregamento dos sinais de vibração por eixo
- Visualização no domínio do tempo
- Segmentação em janelas
- FFT clássica por segmento
- Extração de features no tempo e na frequência
- Treino dos modelos RandomForest, XGBoost e SVM
- Matriz de confusão e curva ROC
- Explicabilidade local com SHAP
- Geração opcional de texto técnico com a OpenAI API

## Saídas geradas

- As imagens geradas pelo notebook são salvas na pasta `images`.

### Exemplos de saída

- Gráficos dos sinais por classe
- Gráficos de FFT
- Matrizes de confusão
- Curvas ROC
- Gráficos SHAP locais

## Uso da parte de LLM

- Para executar o notebook sem consumir a API da OpenAI, altere `gerar_explicacoes_llm` para `False` nas células finais.
- Para gerar a explicação, mantenha `gerar_explicacoes_llm` como `True`.
- Selecione a amostra desejada em `amostra_explicacao_llm`.

## Arquivos principais do repositório

- `predictive_maintenance_fft_ml_xai_llm.ipynb`
- `requirements.txt`
- `.env.example`
- `README.md`

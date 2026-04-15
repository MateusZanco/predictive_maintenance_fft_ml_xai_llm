predictive_maintenance_fft_ml_xai_llm

Visão geral

Este projeto aplica FFT, aprendizado de máquina, XAI e LLM para manutenção preditiva com sinais de vibração de uma caixa de engrenagens planetarias.

O fluxo principal do notebook cobre carregamento dos sinais, segmentação, análise espectral, extração de features, treino de modelos de classificação, explicabilidade com SHAP e geração opcional de texto tecnico com a API da OpenAI.

Notebook principal

predictive_maintenance_fft_ml_xai_llm.ipynb

Problema tratado

O objetivo e classificar a condicao da engrenagem solar do segundo estágio da caixa de engrenagens planetarias.

Mapeamento atual dos rótulos

Classe 0 = Normal
Classe 1 = Desgaste Superficial
Classe 2 = Dente Trincado
Classe 3 = Dente Lascado
Classe 4 = Dente Ausente

Dados

Os arquivos do dataset devem ficar dentro da pasta data.

Link do Drive

https://drive.google.com/drive/folders/1eJWnxC4rEQXuOxAbNJ6IErWKJHqsXry0?usp=sharing

Depois de baixar os arquivos, crie a pasta data caso ela não exista e copie os arquivos para dentro dela.

Estrutura esperada

```text
data/
  x_1500_10.npy
  y_1500_10.npy
  z_1500_10.npy
  gt_1500_10.npy
```

Descrição rapida dos arquivos

x_1500_10.npy contem o sinal de vibracao do eixo X
y_1500_10.npy contem o sinal de vibracao do eixo Y
z_1500_10.npy contem o sinal de vibracao do eixo Z
gt_1500_10.npy contem os rotulos das classes

Ambiente

O projeto usa Python com dependencias listadas em requirements.txt.

Para usar a parte de LLM, copie .env.example para .env e preencha sua chave da OpenAI.

Arquivo de exemplo

```text
OPENAI_API_KEY=cole_sua_chave_aqui
OPENAI_MODEL=gpt-5-mini
```

Como executar

No Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
Copy-Item .env.example .env
New-Item -ItemType Directory -Path data -Force
jupyter notebook
```

Depois abra o arquivo predictive_maintenance_fft_ml_xai_llm.ipynb e execute as células em ordem.

Etapas do notebook

Carregamento dos sinais de vibração por eixo
Visualização no dominio do tempo
Segmentação em janelas
FFT classição por segmento
Extração de features no tempo e na frequencia
Treino dos modelos RandomForest, XGBoost e SVM
Matriz de confusão e curva ROC
Explicabilidade local com SHAP
Geração opcional de texto tecnico com a OpenAI API

Saídas geradas

As imagens geradas pelo notebook são salvas na pasta images.

Exemplos de saída

Graficos dos sinais por classe
Graficos de FFT
Matrizes de confusao
Curvas ROC
Graficos SHAP locais

Observação sobre a parte de LLM

Se você quiser executar o notebook sem consumir a API da OpenAI, altere a variavel gerar_explicacoes_llm para False nas celulas finais.

Se você quiser gerar a explicação, mantenha gerar_explicacoes_llm como True e selecione a amostra desejada em amostra_explicacao_llm.

Arquivos principais do repositorio

predictive_maintenance_fft_ml_xai_llm.ipynb
requirements.txt
.env.example
README.md


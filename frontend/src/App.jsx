import { useEffect, useMemo, useState } from 'react'

const API_BASE = 'http://localhost:8000/api'

const defaultAnalysis = {
  dataset_key: '1500_10',
  fs: 10000,
  pontos_por_linha: 200,
  duracao_intervalo_s: 1.0,
  largura_busca_fm_real_hz: 10,
  largura_banda_harmonica_hz: 10,
  ordens_harmonicas_fm: 5,
  ordens_harmonicas_fm1: 5,
  ressonancia_min_hz: 3000,
  ressonancia_max_hz: 5000,
  test_size: 0.3,
  explanation_samples_per_class: 3,
  shap_top_k: 15,
  random_state: 42
}

const defaultExplain = {
  shap_source_model: 'XGBoost',
  shap_top_k: 15,
  generate_llm: true,
  ollama_model: 'llama3.1',
  temperature: 0.2,
  num_predict: 500
}

const metricLabels = [
  ['accuracy', 'Accuracy'],
  ['precision_macro', 'Precision'],
  ['recall_macro', 'Recall'],
  ['f1_macro', 'F1 macro']
]

function formatMetric(value) {
  return Number.isFinite(value) ? value.toFixed(4) : '-'
}

function formatPerf(value, digits = 2) {
  return Number.isFinite(value) ? Number(value).toFixed(digits) : '-'
}

function msToSeconds(value) {
  return Number.isFinite(value) ? Number(value) / 1000 : null
}

function normalizeMatrix(matrix) {
  if (!Array.isArray(matrix)) return []
  return matrix.map((row) => {
    if (Array.isArray(row)) return row
    if (typeof row === 'string') {
      return row
        .trim()
        .split(/\s+/)
        .map((value) => Number(value))
        .filter((value) => Number.isFinite(value))
    }
    return []
  })
}

function MetricBlock({ title, data }) {
  if (!data) return null

  const metricRows = [
    ...metricLabels.map(([key, label]) => ({ key, label, value: data[key] })),
    { key: 'roc_auc_macro_ovr', label: 'ROC AUC macro OVR', value: data.roc_auc_macro_ovr }
  ]

  return (
    <div className="panel metric-panel">
      <h3>{title}</h3>
      <div className="metric-grid">
        {metricRows.map((item) => (
          <div key={item.key} className="metric-row">
            <span className="metric-label">{item.label}</span>
            <strong className="metric-value">{formatMetric(item.value)}</strong>
          </div>
        ))}
      </div>
    </div>
  )
}

function MetricsChart({ model }) {
  if (!model) return null

  const values = metricLabels.flatMap(([key]) => [
    Number(model?.treino?.[key]) || 0,
    Number(model?.teste?.[key]) || 0
  ])
  const maxValue = Math.max(...values, 1)

  return (
    <div className="chart-wrap">
      <svg viewBox="0 0 520 240" className="chart-svg" role="img">
        {metricLabels.map(([key, label], index) => {
          const trainValue = Number(model?.treino?.[key]) || 0
          const testValue = Number(model?.teste?.[key]) || 0
          const x = 20 + index * 120
          const trainHeight = (trainValue / maxValue) * 150
          const testHeight = (testValue / maxValue) * 150

          return (
            <g key={key}>
              <rect x={x} y={190 - trainHeight} width="34" height={trainHeight} rx="8" fill="#b54e2f" />
              <rect x={x + 42} y={190 - testHeight} width="34" height={testHeight} rx="8" fill="#24556d" />
              <text x={x + 38} y="214" textAnchor="middle" className="chart-label">{label}</text>
            </g>
          )
        })}
      </svg>
      <div className="legend-inline">
        <span><i className="dot train"></i>Treino</span>
        <span><i className="dot test"></i>Teste</span>
      </div>
    </div>
  )
}

function ConfusionTable({ matrix }) {
  const normalizedMatrix = normalizeMatrix(matrix)
  if (!normalizedMatrix.length) return <p className="placeholder">Matriz de confusao indisponivel.</p>

  return (
    <div className="confusion-grid">
      {normalizedMatrix.map((row, rowIndex) =>
        row.map((value, colIndex) => (
          <div key={`${rowIndex}-${colIndex}`} className="confusion-cell">
            <span>{rowIndex},{colIndex}</span>
            <strong>{value}</strong>
          </div>
        ))
      )}
    </div>
  )
}

function FFTChart({ fftData, axis }) {
  const values = fftData?.fft_axes?.[axis]
  if (!values?.frequencias_hz?.length) return <p className="placeholder">Sem FFT para o eixo {axis.toUpperCase()}.</p>

  const frequencies = values.frequencias_hz
  const amplitudes = values.amplitudes
  const sortedAmplitudes = [...amplitudes].sort((a, b) => a - b)
  const percentileIndex = Math.min(
    sortedAmplitudes.length - 1,
    Math.max(0, Math.floor(sortedAmplitudes.length * 0.995))
  )
  const robustAmpMax = sortedAmplitudes[percentileIndex] || 1
  const absoluteAmpMax = Math.max(...amplitudes, 1)
  const displayAmpMax = Math.max(robustAmpMax, absoluteAmpMax * 0.08, 1e-12)
  const maxFreq = frequencies[frequencies.length - 1] || 1
  const points = frequencies
    .map((freq, index) => {
      const x = 20 + (freq / maxFreq) * 460
      const normalizedAmplitude = Math.min(amplitudes[index], displayAmpMax) / displayAmpMax
      const boostedAmplitude = Math.sqrt(normalizedAmplitude)
      const y = 180 - boostedAmplitude * 150
      return `${x},${y}`
    })
    .join(' ')

  return (
    <div className="chart-wrap">
      <svg viewBox="0 0 500 210" className="chart-svg" role="img">
        <polyline fill="none" stroke="#0b7a61" strokeWidth="2" points={points} />
        <line x1="20" y1="180" x2="480" y2="180" stroke="rgba(19,33,44,0.2)" />
        <line x1="20" y1="20" x2="20" y2="180" stroke="rgba(19,33,44,0.2)" />
        <text x="16" y="28" textAnchor="end" className="chart-value-label">{displayAmpMax.toExponential(2)}</text>
        <text x="16" y="184" textAnchor="end" className="chart-value-label">0</text>
        <text x="250" y="204" textAnchor="middle" className="chart-label">Frequencia (Hz)</text>
      </svg>
      <p className="chart-note">
        Escala visual ajustada pela amplitude robusta do sinal. Maximo visual: {displayAmpMax.toExponential(2)} |
        pico real: {absoluteAmpMax.toExponential(2)}
      </p>
    </div>
  )
}

function App() {
  const [page, setPage] = useState('treino')
  const [datasets, setDatasets] = useState([])
  const [analysisForm, setAnalysisForm] = useState(defaultAnalysis)
  const [analysisId, setAnalysisId] = useState('')
  const [analysisData, setAnalysisData] = useState(null)
  const [trainingModel, setTrainingModel] = useState('XGBoost')
  const [selectedPredictionModel, setSelectedPredictionModel] = useState('XGBoost')
  const [selectedSplit, setSelectedSplit] = useState('teste')
  const [selectedClass, setSelectedClass] = useState('all')
  const [selectedSample, setSelectedSample] = useState('')
  const [fftData, setFftData] = useState(null)
  const [explainForm, setExplainForm] = useState(defaultExplain)
  const [explainData, setExplainData] = useState(null)
  const [loadingAnalysis, setLoadingAnalysis] = useState(false)
  const [loadingSample, setLoadingSample] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    fetch(`${API_BASE}/datasets`)
      .then((response) => response.json())
      .then((payload) => {
        setDatasets(payload.datasets || [])
        if (payload.datasets?.length) {
          const preferredDataset =
            payload.datasets.find((dataset) => dataset.key === '1500_10')?.key ||
            payload.datasets[0].key
          setAnalysisForm((current) => ({
            ...current,
            dataset_key: preferredDataset
          }))
        }
      })
      .catch(() => setError('Nao foi possivel carregar os datasets disponiveis.'))
  }, [])

  const filteredSamples = useMemo(() => {
    const catalog = analysisData?.prediction_catalog?.[selectedPredictionModel] || []
    return catalog.filter((item) => {
      const splitOk = selectedSplit === 'all' ? true : item.split === selectedSplit
      const classOk = selectedClass === 'all' ? true : String(item.classe_predita) === String(selectedClass)
      return splitOk && classOk
    })
  }, [analysisData, selectedPredictionModel, selectedSplit, selectedClass])

  const currentTrainingModel = useMemo(
    () => analysisData?.metricas_modelos?.find((item) => item.modelo === trainingModel),
    [analysisData, trainingModel]
  )

  useEffect(() => {
    if (!filteredSamples.length) {
      setSelectedSample('')
      return
    }

    const sampleStillExists = filteredSamples.some(
      (item) => String(item.indice_original) === String(selectedSample)
    )
    if (!sampleStillExists) {
      setSelectedSample(String(filteredSamples[0].indice_original))
    }
  }, [filteredSamples, selectedSample])

  async function runTraining(event) {
    event.preventDefault()
    setLoadingAnalysis(true)
    setError('')
    setExplainData(null)
    setFftData(null)

    try {
      const response = await fetch(`${API_BASE}/analysis`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(analysisForm)
      })
      const payload = await response.json()
      if (!response.ok) throw new Error(payload.detail || 'Falha ao executar o treino.')

      setAnalysisId(payload.analysis_id)
      setAnalysisData(payload)
      setTrainingModel('XGBoost')
      setSelectedPredictionModel('XGBoost')
      setSelectedSplit('teste')
      setSelectedClass('all')
      setPage('treino')
    } catch (requestError) {
      setError(requestError.message)
    } finally {
      setLoadingAnalysis(false)
    }
  }

  async function analyzeSample() {
    if (!analysisId || !selectedSample) return

    setLoadingSample(true)
    setError('')

    try {
      const [fftResponse, explainResponse] = await Promise.all([
        fetch(`${API_BASE}/sample-fft`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            analysis_id: analysisId,
            sample_index: Number(selectedSample),
            frequencia_min_hz: 0,
            frequencia_max_hz: 5000
          })
        }),
        fetch(`${API_BASE}/explain`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            analysis_id: analysisId,
            sample_index: Number(selectedSample),
            ...explainForm
          })
        })
      ])

      const fftPayload = await fftResponse.json()
      const explainPayload = await explainResponse.json()

      if (!fftResponse.ok) throw new Error(fftPayload.detail || 'Falha ao gerar a FFT da amostra.')
      if (!explainResponse.ok) throw new Error(explainPayload.detail || 'Falha ao analisar a amostra.')

      setFftData(fftPayload)
      setExplainData(explainPayload)
      setPage('analise')
    } catch (requestError) {
      setError(requestError.message)
    } finally {
      setLoadingSample(false)
    }
  }

  return (
    <main className="shell">
      <section className="hero">
        <p className="eyebrow">Predictive Maintenance Console</p>
        <h1>Treinar, gerar artefatos e depois analisar amostras com FFT, metricas e LLM local.</h1>
        <p className="lede">
          O fluxo do usuario foi separado em duas paginas: primeiro o treino com os mesmos parametros do notebook,
          depois a analise da amostra com FFT de visualizacao e botao de inspecao.
        </p>
      </section>

      <div className="tabs">
        <button className={page === 'treino' ? 'tab active' : 'tab'} onClick={() => setPage('treino')}>Treino</button>
        <button className={page === 'analise' ? 'tab active' : 'tab'} onClick={() => setPage('analise')} disabled={!analysisData}>Analise</button>
      </div>

      {error ? <div className="banner error">{error}</div> : null}

      {page === 'treino' ? (
        <div className="grid train-layout">
          <section className="card tall">
            <div className="card-head">
              <h2>Parametros e treino</h2>
              <span className="chip">um clique gera artefato</span>
            </div>

            <form className="form" onSubmit={runTraining}>
              <label>
                Dataset
                <select value={analysisForm.dataset_key} onChange={(event) => setAnalysisForm((current) => ({ ...current, dataset_key: event.target.value }))}>
                  {datasets.map((dataset) => (
                    <option key={dataset.key} value={dataset.key}>{dataset.label}</option>
                  ))}
                </select>
              </label>

              <div className="row">
                <label>
                  Fs
                  <input type="number" value={analysisForm.fs} onChange={(event) => setAnalysisForm((current) => ({ ...current, fs: Number(event.target.value) }))} />
                </label>
                <label>
                  Pontos por linha
                  <input type="number" value={analysisForm.pontos_por_linha} onChange={(event) => setAnalysisForm((current) => ({ ...current, pontos_por_linha: Number(event.target.value) }))} />
                </label>
              </div>

              <div className="row">
                <label>
                  Duracao do segmento (s)
                  <input type="number" step="0.1" value={analysisForm.duracao_intervalo_s} onChange={(event) => setAnalysisForm((current) => ({ ...current, duracao_intervalo_s: Number(event.target.value) }))} />
                </label>
                <label>
                  Test size
                  <input type="number" step="0.05" value={analysisForm.test_size} onChange={(event) => setAnalysisForm((current) => ({ ...current, test_size: Number(event.target.value) }))} />
                </label>
              </div>

              <div className="row">
                <label>
                  Janela busca Fm real (Hz)
                  <input type="number" value={analysisForm.largura_busca_fm_real_hz} onChange={(event) => setAnalysisForm((current) => ({ ...current, largura_busca_fm_real_hz: Number(event.target.value) }))} />
                </label>
                <label>
                  Banda harmonica (Hz)
                  <input type="number" value={analysisForm.largura_banda_harmonica_hz} onChange={(event) => setAnalysisForm((current) => ({ ...current, largura_banda_harmonica_hz: Number(event.target.value) }))} />
                </label>
              </div>

              <div className="row">
                <label>
                  Harmonicas 2o estagio
                  <input type="number" value={analysisForm.ordens_harmonicas_fm} onChange={(event) => setAnalysisForm((current) => ({ ...current, ordens_harmonicas_fm: Number(event.target.value) }))} />
                </label>
                <label>
                  Harmonicas 1o estagio
                  <input type="number" value={analysisForm.ordens_harmonicas_fm1} onChange={(event) => setAnalysisForm((current) => ({ ...current, ordens_harmonicas_fm1: Number(event.target.value) }))} />
                </label>
              </div>

              <button className="primary" disabled={loadingAnalysis}>
                {loadingAnalysis ? 'Treinando e gerando artefato...' : 'Treinar modelo e gerar artefato'}
              </button>
            </form>
          </section>

          <section className="card tall">
            <div className="card-head">
              <h2>Artefatos e metricas</h2>
              <span className="chip">acuracia, grafico e confusao</span>
            </div>

            {analysisData ? (
              <>
                <div className="stats">
                  <article>
                    <strong>{analysisData.dataset_shape.linhas}</strong>
                    <span>linhas do dataset</span>
                  </article>
                  <article>
                    <strong>{analysisData.segmentacao.amostras_por_intervalo}</strong>
                    <span>amostras por segmento</span>
                  </article>
                  <article>
                    <strong>{analysisData.feature_space.n_features}</strong>
                    <span>features extraidas</span>
                  </article>
                </div>

                <div className="panel">
                  <h3>Artefato gerado</h3>
                  <p><strong>Pasta:</strong> {analysisData.artifact.directory}</p>
                  <p><strong>Arquivos:</strong> {analysisData.artifact.files.join(', ')}</p>
                </div>

                <div className="row">
                  <label>
                    Modelo para visualizar
                    <select value={trainingModel} onChange={(event) => setTrainingModel(event.target.value)}>
                      <option value="RandomForest">RandomForest</option>
                      <option value="XGBoost">XGBoost</option>
                      <option value="SVM">SVM</option>
                    </select>
                  </label>
                </div>

                {currentTrainingModel ? (
                  <>
                    <div className="metric-card">
                      <h3>{currentTrainingModel.modelo}</h3>
                      <MetricBlock title="Treino" data={currentTrainingModel.treino} />
                      <MetricBlock title="Teste" data={currentTrainingModel.teste} />
                    </div>

                    <div className="panel">
                      <h3>Grafico comparativo</h3>
                      <MetricsChart model={currentTrainingModel} />
                    </div>

                    <div className="panel">
                      <h3>Matriz de confusao do teste</h3>
                      <ConfusionTable matrix={currentTrainingModel.teste.confusion_matrix} />
                    </div>
                  </>
                ) : null}
              </>
            ) : (
              <p className="placeholder">Rode o treino para gerar o artefato, as metricas e a matriz de confusao.</p>
            )}
          </section>
        </div>
      ) : (
        <div className="grid analysis-layout">
          <section className="card tall">
            <div className="card-head">
              <h2>Escolha da amostra</h2>
              <span className="chip">predicoes do modelo</span>
            </div>

            {analysisData ? (
              <>
                <div className="row">
                  <label>
                    Modelo das predicoes
                    <select value={selectedPredictionModel} onChange={(event) => setSelectedPredictionModel(event.target.value)}>
                      <option value="RandomForest">RandomForest</option>
                      <option value="XGBoost">XGBoost</option>
                      <option value="SVM">SVM</option>
                    </select>
                  </label>
                  <label>
                    Origem
                    <select value={selectedSplit} onChange={(event) => setSelectedSplit(event.target.value)}>
                      <option value="teste">Teste</option>
                      <option value="treino">Treino</option>
                      <option value="all">Treino + teste</option>
                    </select>
                  </label>
                </div>

                <div className="row">
                  <label>
                    Classe predita
                    <select value={selectedClass} onChange={(event) => setSelectedClass(event.target.value)}>
                      <option value="all">Todas</option>
                      <option value="0">0 - Normal</option>
                      <option value="1">1 - Desgaste Superficial</option>
                      <option value="2">2 - Dente Trincado</option>
                      <option value="3">3 - Dente Lascado</option>
                      <option value="4">4 - Dente Ausente</option>
                    </select>
                  </label>
                  <label>
                    Amostra
                    <select value={selectedSample} onChange={(event) => setSelectedSample(event.target.value)}>
                      {filteredSamples.map((item) => (
                        <option key={`${item.source_model}-${item.split}-${item.indice_original}`} value={item.indice_original}>
                          #{item.indice_original} | {item.split} | real C{item.classe_real} | predita C{item.classe_predita}
                        </option>
                      ))}
                    </select>
                  </label>
                </div>

                <div className="catalog">
                  {filteredSamples.slice(0, 12).map((item) => (
                    <button
                      key={`${item.source_model}-${item.split}-${item.indice_original}`}
                      type="button"
                      className={String(selectedSample) === String(item.indice_original) ? 'catalog-item active' : 'catalog-item'}
                      onClick={() => setSelectedSample(String(item.indice_original))}
                    >
                      <strong>{item.classe_predita_nome}</strong>
                      <span>Indice {item.indice_original}</span>
                      <span>{item.split} | {item.source_model}</span>
                      <span>Real: {item.classe_real_nome}</span>
                    </button>
                  ))}
                </div>

                <div className="form">
                  <div className="row">
                    <label>
                      Modelo fonte do SHAP
                      <select value={explainForm.shap_source_model} onChange={(event) => setExplainForm((current) => ({ ...current, shap_source_model: event.target.value }))}>
                        <option value="XGBoost">XGBoost</option>
                        <option value="RandomForest">RandomForest</option>
                      </select>
                    </label>
                    <label>
                      Ollama model
                      <input value={explainForm.ollama_model} onChange={(event) => setExplainForm((current) => ({ ...current, ollama_model: event.target.value }))} />
                    </label>
                  </div>

                  <div className="row">
                    <label>
                      Temperatura da LLM
                      <div className="inline-control">
                        <input
                          type="range"
                          min="0"
                          max="1.5"
                          step="0.05"
                          value={explainForm.temperature}
                          onChange={(event) =>
                            setExplainForm((current) => ({
                              ...current,
                              temperature: Number(event.target.value)
                            }))
                          }
                        />
                        <strong className="inline-value">{Number(explainForm.temperature).toFixed(2)}</strong>
                      </div>
                    </label>

                    <label>
                      Maximo de tokens
                      <input
                        type="number"
                        min="1"
                        step="1"
                        value={explainForm.num_predict}
                        onChange={(event) =>
                          setExplainForm((current) => ({
                            ...current,
                            num_predict: Number(event.target.value)
                          }))
                        }
                      />
                    </label>
                  </div>

                  <button type="button" className="primary secondary" disabled={!selectedSample || loadingSample} onClick={analyzeSample}>
                    {loadingSample ? 'Analisando amostra...' : 'Analisar amostra'}
                  </button>
                </div>
              </>
            ) : (
              <p className="placeholder">Primeiro treine o modelo na pagina de treino.</p>
            )}
          </section>

          <section className="card tall">
            <div className="card-head">
              <h2>FFT de visualizacao</h2>
              <span className="chip">0 a 5000 Hz</span>
            </div>

            {fftData ? (
              <div className="fft-grid">
                <div className="panel">
                  <h3>Eixo X</h3>
                  <FFTChart fftData={fftData} axis="x" />
                </div>
                <div className="panel">
                  <h3>Eixo Y</h3>
                  <FFTChart fftData={fftData} axis="y" />
                </div>
                <div className="panel">
                  <h3>Eixo Z</h3>
                  <FFTChart fftData={fftData} axis="z" />
                </div>
              </div>
            ) : (
              <p className="placeholder">Escolha a amostra e clique em "Analisar amostra" para visualizar a FFT.</p>
            )}
          </section>

          <section className="card tall analysis-wide">
            <div className="card-head">
              <h2>Explicacao da amostra</h2>
              <span className="chip">LLM local</span>
            </div>

            {explainData ? (
              <>
                <div className="audit-meta">
                  <article>
                    <span>Classe real</span>
                    <strong>{explainData.explanation.sample_metadata.classe_real} - {explainData.explanation.sample_metadata.classe_real_nome}</strong>
                  </article>
                  <article>
                    <span>Classe predita</span>
                    <strong>{explainData.explanation.sample_metadata.classe_predita} - {explainData.explanation.sample_metadata.classe_predita_nome}</strong>
                  </article>
                  <article>
                    <span>Probabilidade</span>
                    <strong>{Number(explainData.explanation.sample_metadata.probabilidade_predita).toFixed(4)}</strong>
                  </article>
                </div>

                <div className="subgrid">
                  <div className="panel">
                    <h3>Top features explicativas</h3>
                    <div className="table">
                      {explainData.explanation.top_features.map((item) => (
                        <div key={item.rank} className="table-row">
                          <span>#{item.rank}</span>
                          <span>{item.feature}</span>
                          <span>{Number(item.valor_feature).toFixed(4)}</span>
                          <span>{Number(item.shap_value).toFixed(4)}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="panel">
                    <h3>Prompt e parametros enviados</h3>
                    <pre>{JSON.stringify(explainData.llm?.audit || {}, null, 2)}</pre>
                  </div>
                </div>

                <div className="panel">
                  <h3>JSON de auditoria das features</h3>
                  <p>
                    <strong>Arquivo:</strong> {explainData.feature_audit?.file_path || 'Nao gerado.'}
                  </p>
                  <pre>{JSON.stringify(explainData.feature_audit?.payload || {}, null, 2)}</pre>
                </div>

                <div className="panel">
                  <h3>Tokens e latencia</h3>
                  <div className="stats llm-stats">
                    <article>
                      <strong>{formatPerf(msToSeconds(explainData.llm?.performance?.latency_ms))}</strong>
                      <span>latencia total (s)</span>
                    </article>
                    <article>
                      <strong>{formatPerf(explainData.llm?.performance?.prompt_tokens, 0)}</strong>
                      <span>tokens do prompt</span>
                    </article>
                    <article>
                      <strong>{formatPerf(explainData.llm?.performance?.response_tokens, 0)}</strong>
                      <span>tokens de resposta</span>
                    </article>
                    <article>
                      <strong>{formatPerf(explainData.llm?.performance?.total_tokens, 0)}</strong>
                      <span>tokens totais</span>
                    </article>
                    <article>
                      <strong>{formatPerf(explainData.llm?.performance?.tokens_per_second)}</strong>
                      <span>tokens por segundo</span>
                    </article>
                    <article>
                      <strong>{formatPerf(msToSeconds(explainData.llm?.performance?.load_duration_ms))}</strong>
                      <span>carga do modelo (s)</span>
                    </article>
                  </div>
                  <pre>{JSON.stringify(explainData.llm?.performance || {}, null, 2)}</pre>
                </div>

                <div className="panel">
                  <h3>Resposta da LLM local</h3>
                  <pre>{explainData.llm?.response_text || 'Geracao desativada.'}</pre>
                  {explainData.llm?.error ? <p className="error-text">Erro do Ollama: {explainData.llm.error}</p> : null}
                </div>
              </>
            ) : (
              <p className="placeholder">A analise textual aparece aqui depois de clicar em "Analisar amostra".</p>
            )}
          </section>
        </div>
      )}
    </main>
  )
}

export default App

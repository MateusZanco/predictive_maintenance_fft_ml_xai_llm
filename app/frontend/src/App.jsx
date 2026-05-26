import { useEffect, useRef, useState } from "react";
import Plotly from "plotly.js-dist-min";
import {
  fetchExplanation,
  fetchFeatures,
  fetchFft,
  fetchPrediction,
  fetchSampleMeta,
  fetchSamples,
  fetchShap,
  fetchSignal,
  fetchSystemStatus
} from "./api";

const fieldStyle = {
  display: "grid",
  gap: 6,
  minWidth: 160
};

const panelStyle = {
  background: "#ffffff",
  border: "1px solid #d7e0e5",
  borderRadius: 16,
  padding: 20,
  boxShadow: "0 12px 30px rgba(16, 33, 43, 0.08)"
};

const featureCardStyle = {
  background: "#f8fbfc",
  border: "1px solid #d7e0e5",
  borderRadius: 14,
  padding: 16,
  display: "grid",
  gap: 6,
  minWidth: 180
};

const predictionPanelStyle = {
  background: "#f3f9f4",
  border: "1px solid #cce3d0",
  borderRadius: 14,
  padding: 16,
  display: "grid",
  gap: 10
};

const shapPanelStyle = {
  background: "#f8fbfc",
  border: "1px solid #d7e0e5",
  borderRadius: 14,
  padding: 16,
  display: "grid",
  gap: 12
};

const explanationPanelStyle = {
  background: "#fffdf5",
  border: "1px solid #ece1a8",
  borderRadius: 14,
  padding: 16,
  display: "grid",
  gap: 12
};

const timingPanelStyle = {
  background: "#f7f9ff",
  border: "1px solid #d7def7",
  borderRadius: 14,
  padding: 16,
  display: "grid",
  gap: 12
};

const telemetryCardStyle = {
  background: "#f5fafb",
  border: "1px solid #d7e0e5",
  borderRadius: 14,
  padding: 16,
  display: "grid",
  gap: 6,
  minWidth: 180
};

function buildHarmonicTraces(baseFrequency, maxFrequency, color, familyName) {
  const traces = [];
  for (let order = 2; order <= 5; order += 1) {
    const harmonicFrequency = baseFrequency * order;
    if (harmonicFrequency > maxFrequency) {
      break;
    }
    traces.push({
      x: [harmonicFrequency, harmonicFrequency],
      y: [0, null],
      type: "scatter",
      mode: "lines",
      name: order === 2 ? familyName : `${familyName} h${order}`,
      showlegend: order === 2,
      hovertemplate: `${familyName} h${order}: ${harmonicFrequency.toFixed(2)} Hz<extra></extra>`,
      line: { color, dash: "dot", width: 2.2 }
    });
  }
  return traces;
}

function FftChart({ data }) {
  const ref = useRef(null);

  useEffect(() => {
    if (!ref.current || !data) return;

    const ymax = Math.max(...data.amp, 0);
    const maxFrequency = Math.max(...data.freq, 0);
    const harmonicTraces = [
      ...buildHarmonicTraces(data.fm1, maxFrequency, "#e03131", "Harmonicas Fm1"),
      ...buildHarmonicTraces(data.fm2, maxFrequency, "#1b5e20", "Harmonicas Fm2")
    ].map((trace) => ({
      ...trace,
      y: [0, ymax]
    }));

    Plotly.react(
      ref.current,
      [
        {
          x: data.freq,
          y: data.amp,
          type: "scatter",
          mode: "lines",
          name: "FFT",
          line: { color: "#0b7285", width: 1.5 }
        },
        {
          x: [data.fm1, data.fm1],
          y: [0, ymax],
          type: "scatter",
          mode: "lines",
          name: "Fm1",
          line: { color: "#e03131", dash: "dash", width: 3 }
        },
        {
          x: [data.fm2, data.fm2],
          y: [0, ymax],
          type: "scatter",
          mode: "lines",
          name: "Fm2",
          line: { color: "#1b5e20", dash: "dash", width: 3 }
        },
        ...harmonicTraces
      ],
      {
        margin: { l: 50, r: 20, t: 30, b: 50 },
        paper_bgcolor: "#ffffff",
        plot_bgcolor: "#ffffff",
        xaxis: { title: "Frequência (Hz)", gridcolor: "#e9eef2" },
        yaxis: { title: "Amplitude", gridcolor: "#e9eef2" },
        legend: { orientation: "h" }
      },
      { responsive: true, displaylogo: false }
    );
  }, [data]);

  return <div ref={ref} style={{ width: "100%", minHeight: 440 }} />;
}

function RawSignalChart({ data }) {
  const ref = useRef(null);

  useEffect(() => {
    if (!ref.current || !data) return;

    Plotly.react(
      ref.current,
      [
        {
          x: data.time,
          y: data.signal,
          type: "scatter",
          mode: "lines",
          name: "Sinal bruto",
          line: { color: "#c2255c", width: 1.2 }
        }
      ],
      {
        margin: { l: 50, r: 20, t: 30, b: 50 },
        paper_bgcolor: "#ffffff",
        plot_bgcolor: "#ffffff",
        xaxis: { title: "Tempo (s)", gridcolor: "#e9eef2" },
        yaxis: { title: "Amplitude", gridcolor: "#e9eef2" },
        legend: { orientation: "h" }
      },
      { responsive: true, displaylogo: false }
    );
  }, [data]);

  return <div ref={ref} style={{ width: "100%", minHeight: 320 }} />;
}

function FeaturePanel({ data }) {
  if (!data) return null;

  return (
    <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
      <div style={featureCardStyle}>
        <strong>RMS</strong>
        <span>{data.rms.toFixed(6)}</span>
      </div>
      <div style={featureCardStyle}>
        <strong>Kurtosis</strong>
        <span>{data.kurtosis.toFixed(6)}</span>
      </div>
      <div style={featureCardStyle}>
        <strong>Peak Value</strong>
        <span>{data.peak_value.toFixed(6)}</span>
      </div>
      <div style={featureCardStyle}>
        <strong>Crest Factor</strong>
        <span>{data.crest_factor.toFixed(6)}</span>
      </div>
    </div>
  );
}

function PredictionPanel({ data, error }) {
  if (error) {
    return (
      <div style={{ ...predictionPanelStyle, background: "#fff5f5", borderColor: "#f1c2c2" }}>
        <strong>Predicao do modelo</strong>
        <span>{error}</span>
      </div>
    );
  }

  if (!data) return null;

  const sortedProbabilities = Object.entries(data.class_probabilities).sort((a, b) => Number(a[0]) - Number(b[0]));

  return (
    <div style={predictionPanelStyle}>
      <div>
        <strong>Classe predita:</strong> {data.predicted_class_name} ({data.predicted_class})
      </div>
      <div>
        <strong>Probabilidade:</strong> {data.predicted_probability.toFixed(4)}
      </div>
      {data.classe_real_nome ? (
        <div>
          <strong>Classe real da amostra exportada:</strong> {data.classe_real_nome} ({data.classe_real})
        </div>
      ) : null}
      <div>
        <strong>Probabilidades por classe:</strong>
      </div>
      <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
        {sortedProbabilities.map(([classId, probability]) => (
          <div key={classId} style={featureCardStyle}>
            <strong>Classe {classId}</strong>
            <span>{Number(probability).toFixed(4)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function ShapPanel({ data, error }) {
  if (error) {
    return (
      <div style={{ ...shapPanelStyle, background: "#fff5f5", borderColor: "#f1c2c2" }}>
        <strong>SHAP da amostra</strong>
        <span>{error}</span>
      </div>
    );
  }

  if (!data) return null;

  return (
    <div style={shapPanelStyle}>
      <div>
        <strong>SHAP local da classe predita:</strong> {data.predicted_class_name} ({data.predicted_class})
      </div>
      <div>
        <strong>Expected value:</strong> {data.expected_value.toFixed(6)}
      </div>
      <table style={{ width: "100%", borderCollapse: "collapse" }}>
        <thead>
          <tr>
            <th style={{ textAlign: "left", padding: "8px 6px", borderBottom: "1px solid #d7e0e5" }}>Rank</th>
            <th style={{ textAlign: "left", padding: "8px 6px", borderBottom: "1px solid #d7e0e5" }}>Feature</th>
            <th style={{ textAlign: "right", padding: "8px 6px", borderBottom: "1px solid #d7e0e5" }}>Valor</th>
            <th style={{ textAlign: "right", padding: "8px 6px", borderBottom: "1px solid #d7e0e5" }}>SHAP</th>
            <th style={{ textAlign: "right", padding: "8px 6px", borderBottom: "1px solid #d7e0e5" }}>|SHAP|</th>
            <th style={{ textAlign: "right", padding: "8px 6px", borderBottom: "1px solid #d7e0e5" }}>%</th>
          </tr>
        </thead>
        <tbody>
          {data.top_contributions.map((item) => (
            <tr key={`${item.rank}-${item.feature}`}>
              <td style={{ padding: "8px 6px", borderBottom: "1px solid #eef3f6" }}>{item.rank}</td>
              <td style={{ padding: "8px 6px", borderBottom: "1px solid #eef3f6" }}>{item.feature}</td>
              <td style={{ padding: "8px 6px", borderBottom: "1px solid #eef3f6", textAlign: "right" }}>
                {item.feature_value.toFixed(6)}
              </td>
              <td style={{ padding: "8px 6px", borderBottom: "1px solid #eef3f6", textAlign: "right" }}>
                {item.shap_value.toFixed(6)}
              </td>
              <td style={{ padding: "8px 6px", borderBottom: "1px solid #eef3f6", textAlign: "right" }}>
                {item.impact_abs.toFixed(6)}
              </td>
              <td style={{ padding: "8px 6px", borderBottom: "1px solid #eef3f6", textAlign: "right" }}>
                {item.impact_pct.toFixed(1)}%
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function TimingPanel({ predictionData, shapData, explanationData, explanationLoading, explanationElapsedSeconds }) {
  const featureTime = shapData?.feature_extraction_seconds ?? predictionData?.feature_extraction_seconds;
  const inferenceTime = shapData?.model_inference_seconds ?? predictionData?.model_inference_seconds;
  const shapTime = shapData?.shap_inference_seconds;
  const llmTime = explanationLoading ? explanationElapsedSeconds : explanationData?.llm_processing_seconds;

  if (featureTime === undefined && inferenceTime === undefined && shapTime === undefined && llmTime === undefined) {
    return null;
  }

  return (
    <div style={timingPanelStyle}>
      <strong>Tempos de processamento</strong>
      <div>Extração de Features: {featureTime !== undefined ? `${Number(featureTime).toFixed(3)} s` : "n/d"}</div>
      <div>Inferência do Modelo: {inferenceTime !== undefined ? `${Number(inferenceTime).toFixed(3)} s` : "n/d"}</div>
      <div>SHAP: {shapTime !== undefined ? `${Number(shapTime).toFixed(3)} s` : "n/d"}</div>
      <div>Geração Explicação LLM: {llmTime !== undefined ? `${Number(llmTime).toFixed(1)} s` : "n/d"}</div>
    </div>
  );
}

function ExplanationPanel({ data, error, loading, onGenerate, disabled, promptStrategy, onPromptStrategyChange }) {
  function formatProcessedAt(value) {
    if (!value) return "";
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) return value;
    return parsed.toLocaleString("pt-BR");
  }

  function formatDuration(value) {
    if (value === null || value === undefined || Number.isNaN(value)) return "";
    return `${Number(value).toFixed(1)} s`;
  }

  return (
    <div style={explanationPanelStyle}>
      <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
        <strong>Explicação com LLM</strong>
        <label style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <span>Estratégia</span>
          <select value={promptStrategy} onChange={(e) => onPromptStrategyChange(e.target.value)} disabled={loading}>
            <option value="few_shot">Few-shot</option>
            <option value="zero_shot">Zero-shot</option>
          </select>
        </label>
        <button type="button" onClick={onGenerate} disabled={disabled || loading}>
          {loading ? "Gerando explicação..." : "Gerar explicação"}
        </button>
      </div>

      {error ? (
        <div style={{ color: "#c92a2a" }}>{error}</div>
      ) : null}

      {data ? (
        <>
          <div>
            <strong>Processado em:</strong> {formatProcessedAt(data.processed_at_iso)}
          </div>
          <div>
            <strong>Estratégia prompt:</strong> {data.prompt_strategy}
          </div>
          <div>
            <strong>Formato resposta:</strong> {data.response_format === "raw_text" ? "texto livre" : "json"}
          </div>
          <div>
            <strong>Janela analisada:</strong> {data.window_start_s.toFixed(2)} s - {data.window_end_s.toFixed(2)} s
          </div>
          {data.response_format === "raw_text" ? (
            <div>
              <strong>Resposta livre do modelo:</strong> {data.unstructured_response || data.raw_response}
            </div>
          ) : (
            <>
              <div>
                <strong>Interpretação Vibracional:</strong> {data.interpretacao_vibracional}
              </div>
              <div>
                <strong>Interpretação Mecânica:</strong> {data.interpretacao_mecanica}
              </div>
            </>
          )}
          <details>
            <summary>Auditoria do prompt</summary>
            <pre style={{ whiteSpace: "pre-wrap" }}>{data.system_prompt}</pre>
            <pre style={{ whiteSpace: "pre-wrap" }}>{data.user_prompt}</pre>
            <pre style={{ whiteSpace: "pre-wrap" }}>{data.raw_response}</pre>
          </details>
        </>
      ) : (
        <div>A explicação ainda não foi gerada para esta janela.</div>
      )}
    </div>
  );
}

function formatMetricValue(value, digits = 1, suffix = "") {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "n/d";
  }
  return `${Number(value).toFixed(digits)}${suffix}`;
}

function SystemStatusPanel({ data, error }) {
  return (
    <section style={{ ...panelStyle, display: "grid", gap: 16 }}>
      <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
        <h2 style={{ margin: 0, fontSize: 20 }}>Status da placa</h2>
        {error ? <span style={{ color: "#c92a2a" }}>{error}</span> : null}
      </div>

      <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
        <div style={telemetryCardStyle}>
          <strong>CPU</strong>
          <span>{formatMetricValue(data?.cpu_percent, 1, "%")}</span>
          <small>{data ? `${data.cpu_busy_cores}/${data.cpu_physical_cores || 0} núcleos físicos com atividade` : "n/d"}</small>
        </div>
        <div style={telemetryCardStyle}>
          <strong>Núcleos</strong>
          <span>{data ? `${data.cpu_physical_cores || "n/d"} físicos` : "n/d"}</span>
          <small>{data ? `limiar de atividade: ${data.cpu_busy_threshold_percent.toFixed(0)}%` : "n/d"}</small>
        </div>
        <div style={telemetryCardStyle}>
          <strong>Memória</strong>
          <span>{formatMetricValue(data?.memory_percent, 1, "%")}</span>
          <small>
            {formatMetricValue(data?.memory_used_mb, 0, " MB")} / {formatMetricValue(data?.memory_total_mb, 0, " MB")}
          </small>
        </div>
        <div style={telemetryCardStyle}>
          <strong>Temperatura</strong>
          <span>{formatMetricValue(data?.temperature_c, 1, " °C")}</span>
          <small>{data?.temperature_source || "fonte indisponível"}</small>
        </div>
      </div>
    </section>
  );
}

function formatSampleLabel(sample) {
  const parts = [sample.dataset_operacao, sample.condicao_operacao];
  if (sample.classe_nome) {
    parts.push(sample.classe_nome);
  }
  return parts.join(" | ");
}

export default function App() {
  const [samples, setSamples] = useState([]);
  const [systemStatus, setSystemStatus] = useState(null);
  const [systemStatusError, setSystemStatusError] = useState("");
  const [selectedSample, setSelectedSample] = useState("");
  const [sampleMeta, setSampleMeta] = useState(null);
  const [axis, setAxis] = useState("x");
  const [windowIndex, setWindowIndex] = useState(0);
  const [fmin, setFmin] = useState(0);
  const [fmax, setFmax] = useState(5000);
  const [applyHann, setApplyHann] = useState(true);
  const [signalResult, setSignalResult] = useState(null);
  const [fftResult, setFftResult] = useState(null);
  const [featureResult, setFeatureResult] = useState(null);
  const [predictionResult, setPredictionResult] = useState(null);
  const [predictionError, setPredictionError] = useState("");
  const [shapResult, setShapResult] = useState(null);
  const [shapError, setShapError] = useState("");
  const [promptStrategy, setPromptStrategy] = useState("few_shot");
  const [explanationResult, setExplanationResult] = useState(null);
  const [explanationError, setExplanationError] = useState("");
  const [explanationLoading, setExplanationLoading] = useState(false);
  const [explanationElapsedSeconds, setExplanationElapsedSeconds] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function loadSystemStatus() {
      try {
        const data = await fetchSystemStatus();
        if (!cancelled) {
          setSystemStatus(data);
          setSystemStatusError("");
        }
      } catch (err) {
        if (!cancelled) {
          setSystemStatusError(String(err));
        }
      }
    }

    loadSystemStatus();
    const intervalId = window.setInterval(loadSystemStatus, 5000);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, []);

  useEffect(() => {
    if (!explanationLoading) return undefined;

    const startedAt = Date.now();
    setExplanationElapsedSeconds(0);

    const intervalId = window.setInterval(() => {
      const elapsed = Math.floor((Date.now() - startedAt) / 1000);
      setExplanationElapsedSeconds(elapsed);
    }, 1000);

    return () => {
      window.clearInterval(intervalId);
    };
  }, [explanationLoading]);

  useEffect(() => {
    fetchSamples()
      .then((data) => {
        setSamples(data);
        if (data.length > 0) {
          setSelectedSample(data[0].sample_id);
        }
      })
      .catch((err) => setError(String(err)));
  }, []);

  useEffect(() => {
    if (!selectedSample) return;
    setSignalResult(null);
    setFftResult(null);
    setFeatureResult(null);
    setPredictionResult(null);
    setPredictionError("");
    setShapResult(null);
    setShapError("");
    setExplanationResult(null);
    setExplanationError("");
    setExplanationLoading(false);
    setExplanationElapsedSeconds(0);
    setError("");
    fetchSampleMeta(selectedSample)
      .then((meta) => {
        setSampleMeta(meta);
        setWindowIndex(0);
      })
      .catch((err) => setError(String(err)));
  }, [selectedSample]);

  async function handleRunFft() {
    setLoading(true);
    setError("");
    try {
      const requestBase = {
        sample_id: selectedSample,
        axis,
        window_index: Number(windowIndex)
      };
      const [signalData, fftData, featureData] = await Promise.all([
        fetchSignal(requestBase),
        fetchFft({
          ...requestBase,
          fmin: Number(fmin),
          fmax: Number(fmax),
          apply_hann: applyHann
        }),
        fetchFeatures(requestBase)
      ]);
      setSignalResult(signalData);
      setFftResult(fftData);
      setFeatureResult(featureData);

      try {
        const predictionData = await fetchPrediction({
          sample_id: selectedSample,
          window_index: Number(windowIndex)
        });
        setPredictionResult(predictionData);
        setPredictionError("");

        try {
          const shapData = await fetchShap({
            sample_id: selectedSample,
            window_index: Number(windowIndex),
            top_k: 5
          });
          setShapResult(shapData);
          setShapError("");
        } catch (localShapErr) {
          setShapResult(null);
          setShapError(String(localShapErr));
        }
      } catch (predictionErr) {
        setPredictionResult(null);
        setPredictionError(String(predictionErr));
        setShapResult(null);
        setShapError(String(predictionErr));
      }
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }

  async function handleGenerateExplanation() {
    if (!selectedSample) return;

    setExplanationLoading(true);
    setExplanationElapsedSeconds(0);
    setExplanationError("");
    try {
      const data = await fetchExplanation({
        sample_id: selectedSample,
        window_index: Number(windowIndex),
        top_k: 5,
        prompt_strategy: promptStrategy
      });
      setExplanationResult(data);
    } catch (err) {
      setExplanationResult(null);
      setExplanationError(String(err));
    } finally {
      setExplanationLoading(false);
    }
  }

  return (
    <main style={{ padding: 24, display: "grid", gap: 20 }}>
      <section style={panelStyle}>
        <h1 style={{ marginTop: 0 }}>Manutenção Preditiva em Redutores Planetários por Análise de Vibração: Classificação Explicável de Falhas Integrando ML, XAI e LLM</h1>
      </section>

      <SystemStatusPanel data={systemStatus} error={systemStatusError} />

      <section style={{ ...panelStyle, display: "grid", gap: 16 }}>
        <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
          <label style={fieldStyle}>
            <span>Amostra</span>
            <select value={selectedSample} onChange={(e) => setSelectedSample(e.target.value)}>
              {samples.map((sample) => (
                <option key={sample.sample_id} value={sample.sample_id}>
                  {formatSampleLabel(sample)}
                </option>
              ))}
            </select>
          </label>

          <label style={fieldStyle}>
            <span>Eixo</span>
            <select value={axis} onChange={(e) => setAxis(e.target.value)}>
              <option value="x">X</option>
              <option value="y">Y</option>
              <option value="z">Z</option>
            </select>
          </label>

          <label style={fieldStyle}>
            <span>Janela de 1 s</span>
            <input
              type="number"
              min={0}
              max={sampleMeta ? Math.max(sampleMeta.available_windows - 1, 0) : 0}
              value={windowIndex}
              onChange={(e) => setWindowIndex(e.target.value)}
            />
          </label>

          <label style={fieldStyle}>
            <span>fmin (Hz)</span>
            <input type="number" value={fmin} onChange={(e) => setFmin(e.target.value)} />
          </label>

          <label style={fieldStyle}>
            <span>fmax (Hz)</span>
            <input type="number" value={fmax} onChange={(e) => setFmax(e.target.value)} />
          </label>

          <label style={{ ...fieldStyle, justifyContent: "end" }}>
            <span>Janela Hann</span>
            <input type="checkbox" checked={applyHann} onChange={(e) => setApplyHann(e.target.checked)} />
          </label>
        </div>

        <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
          <button onClick={handleRunFft} disabled={!selectedSample || loading}>
            {loading ? "Calculando..." : "Calcular FFT"}
          </button>
          {sampleMeta && (
            <span>
              {formatSampleLabel(sampleMeta)} | {sampleMeta.available_windows} janelas disponíveis | Fm1 ={" "}
              {sampleMeta.fm1.toFixed(2)} Hz | Fm2 = {sampleMeta.fm2.toFixed(2)} Hz
            </span>
          )}
        </div>

        {error ? <pre style={{ color: "#c92a2a", margin: 0 }}>{error}</pre> : null}
      </section>

      <section style={panelStyle}>
        {signalResult ? (
          <>
            <div style={{ marginBottom: 12 }}>
              <strong>{sampleMeta ? formatSampleLabel(sampleMeta) : fftResult.dataset_operacao}</strong> | janela{" "}
              {signalResult.window_index} | {signalResult.window_start_s.toFixed(2)} s - {signalResult.window_end_s.toFixed(2)} s
            </div>
            <RawSignalChart data={signalResult} />
            <div style={{ marginTop: 16 }}>
              <FeaturePanel data={featureResult} />
            </div>
          </>
        ) : (
          <p style={{ margin: 0 }}>Selecione uma amostra e calcule o sinal bruto e a FFT.</p>
        )}
      </section>

      <section style={panelStyle}>
        {fftResult ? (
          <>
            <div style={{ marginBottom: 12 }}>
              <strong>{sampleMeta ? formatSampleLabel(sampleMeta) : fftResult.dataset_operacao}</strong> | FFT da janela{" "}
              {fftResult.window_index}
            </div>
            <FftChart data={fftResult} />
          </>
        ) : (
          <p style={{ margin: 0 }}>Selecione uma amostra e calcule o sinal bruto e a FFT.</p>
        )}
      </section>

      <section style={panelStyle}>
        {predictionResult || predictionError ? (
          <PredictionPanel data={predictionResult} error={predictionError} />
        ) : (
          <p style={{ margin: 0 }}>Calcule a janela para visualizar a classe predita.</p>
        )}
      </section>

      <section style={panelStyle}>
        {shapResult || shapError ? (
          <ShapPanel data={shapResult} error={shapError} />
        ) : (
          <p style={{ margin: 0 }}>Calcule a janela para visualizar o SHAP local.</p>
        )}
      </section>

      <section style={panelStyle}>
        <TimingPanel
          predictionData={predictionResult}
          shapData={shapResult}
          explanationData={explanationResult}
          explanationLoading={explanationLoading}
          explanationElapsedSeconds={explanationElapsedSeconds}
        />
        {!predictionResult && !shapResult ? (
          <p style={{ margin: 0 }}>Calcule a janela para visualizar os tempos de extração, inferência e SHAP.</p>
        ) : null}
      </section>

      <section style={panelStyle}>
        {explanationLoading || explanationResult?.llm_processing_seconds !== undefined ? (
          <div style={{ marginBottom: 12 }}>
            <strong>Tempo processamento da LLM:</strong>{" "}
            {explanationLoading
              ? `${explanationElapsedSeconds} s`
              : `${Number(explanationResult.llm_processing_seconds).toFixed(1)} s`}
          </div>
        ) : null}
        <ExplanationPanel
          data={explanationResult}
          error={explanationError}
          loading={explanationLoading}
          onGenerate={handleGenerateExplanation}
          disabled={!predictionResult}
          promptStrategy={promptStrategy}
          onPromptStrategyChange={setPromptStrategy}
        />
      </section>
    </main>
  );
}

import { useEffect, useRef, useState } from "react";
import Plotly from "plotly.js-dist-min";
import { fetchExplanation, fetchFeatures, fetchFft, fetchPrediction, fetchSampleMeta, fetchSamples, fetchShap, fetchSignal } from "./api";

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

const presetButtonStyle = {
  padding: "8px 12px",
  borderRadius: 999,
  border: "1px solid #cbd5db",
  background: "#f8fbfc",
  cursor: "pointer"
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
      line: { color, dash: "dot", width: 1 }
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
      ...buildHarmonicTraces(data.fm1, maxFrequency, "#fab005", "Harmonicas Fm1"),
      ...buildHarmonicTraces(data.fm2, maxFrequency, "#51cf66", "Harmonicas Fm2")
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
          line: { color: "#f08c00", dash: "dash" }
        },
        {
          x: [data.fm2, data.fm2],
          y: [0, ymax],
          type: "scatter",
          mode: "lines",
          name: "Fm2",
          line: { color: "#2f9e44", dash: "dash" }
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

function ExplanationPanel({ data, error, loading, onGenerate, disabled }) {
  return (
    <div style={explanationPanelStyle}>
      <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
        <strong>Explicação com LLM</strong>
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
            <strong>Interpretação Vibracional:</strong> {data.interpretacao_vibracional}
          </div>
          <div>
            <strong>Interpretação Mecânica:</strong> {data.interpretacao_mecanica}
          </div>
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

function formatSampleLabel(sample) {
  const parts = [sample.dataset_operacao, sample.condicao_operacao];
  if (sample.classe_nome) {
    parts.push(sample.classe_nome);
  }
  return parts.join(" | ");
}

export default function App() {
  const [samples, setSamples] = useState([]);
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
  const [explanationResult, setExplanationResult] = useState(null);
  const [explanationError, setExplanationError] = useState("");
  const [explanationLoading, setExplanationLoading] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  function applyFrequencyPreset(preset) {
    if (!sampleMeta) return;

    if (preset === "full") {
      setFmin(0);
      setFmax(5000);
      return;
    }

    if (preset === "fm1") {
      setFmin(Math.max(0, Math.round(sampleMeta.fm1 - 80)));
      setFmax(Math.round(sampleMeta.fm1 + 80));
      return;
    }

    if (preset === "fm2") {
      setFmin(Math.max(0, Math.round(sampleMeta.fm2 - 20)));
      setFmax(Math.round(sampleMeta.fm2 + 20));
    }
  }

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
    setExplanationError("");
    try {
      const data = await fetchExplanation({
        sample_id: selectedSample,
        window_index: Number(windowIndex),
        top_k: 5
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
        <h1 style={{ marginTop: 0 }}>Rock Pi FFT Viewer</h1>
        <p style={{ marginBottom: 0 }}>
          MVP para selecionar uma amostra contínua de teste, escolher uma janela de 1 s e visualizar a FFT.
        </p>
      </section>

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

        {sampleMeta && (
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
            <strong>Presets de faixa:</strong>
            <button type="button" style={presetButtonStyle} onClick={() => applyFrequencyPreset("full")}>
              0-5000 Hz
            </button>
            <button type="button" style={presetButtonStyle} onClick={() => applyFrequencyPreset("fm1")}>
              Fm1 ± 80 Hz
            </button>
            <button type="button" style={presetButtonStyle} onClick={() => applyFrequencyPreset("fm2")}>
              Fm2 ± 20 Hz
            </button>
          </div>
        )}

        {error ? <pre style={{ color: "#c92a2a", margin: 0 }}>{error}</pre> : null}
      </section>

      <section style={panelStyle}>
        {signalResult ? (
          <>
            <div style={{ marginBottom: 12 }}>
              <strong>{sampleMeta ? formatSampleLabel(sampleMeta) : fftResult.dataset_operacao}</strong> | janela{" "}
              {signalResult.window_index} | {signalResult.window_start_s.toFixed(2)} s - {signalResult.window_end_s.toFixed(2)} s
            </div>
            <div style={{ marginBottom: 16 }}>
              <FeaturePanel data={featureResult} />
            </div>
            <div style={{ marginBottom: 16 }}>
              <PredictionPanel data={predictionResult} error={predictionError} />
            </div>
            <div style={{ marginBottom: 16 }}>
              <ShapPanel data={shapResult} error={shapError} />
            </div>
            <div style={{ marginBottom: 16 }}>
              <ExplanationPanel
                data={explanationResult}
                error={explanationError}
                loading={explanationLoading}
                onGenerate={handleGenerateExplanation}
                disabled={!predictionResult}
              />
            </div>
            <RawSignalChart data={signalResult} />
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
    </main>
  );
}

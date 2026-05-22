const API_BASE = "";

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {})
    },
    ...options
  });

  if (!response.ok) {
    const raw = await response.text();
    try {
      const parsed = JSON.parse(raw);
      throw new Error(parsed.detail || raw || `HTTP ${response.status}`);
    } catch {
      throw new Error(raw || `HTTP ${response.status}`);
    }
  }

  return response.json();
}

export function fetchSamples() {
  return request("/api/samples");
}

export function fetchSampleMeta(sampleId) {
  return request(`/api/samples/${sampleId}/meta`);
}

export function fetchFft(payload) {
  return request("/api/fft", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export function fetchSignal(payload) {
  return request("/api/signal", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export function fetchFeatures(payload) {
  return request("/api/features", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export function fetchPrediction(payload) {
  return request("/api/predict", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export function fetchShap(payload) {
  return request("/api/shap", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export function fetchExplanation(payload) {
  return request("/api/explain", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

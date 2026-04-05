const byId = (id) => document.getElementById(id);

const statusEl = byId("status");
const promptEl = byId("prompt");
const maxNewTokensEl = byId("maxNewTokens");
const temperatureEl = byId("temperature");
const topKEl = byId("topK");
const generateBtn = byId("generateBtn");
const quickBenchmarkBtn = byId("quickBenchmarkBtn");
const bdhOutputEl = byId("bdhOutput");
const distilOutputEl = byId("distilOutput");
const metricsWrapEl = byId("metricsTableWrap");
const benchmarkWrapEl = byId("benchmarkWrap");
const artifactsWrapEl = byId("artifactsWrap");
const vizWrapEl = byId("vizWrap");
const modelSpecsWrapEl = byId("modelSpecsWrap");

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.className = isError ? "status error" : "status";
}

function fmtNum(v, digits = 4) {
  if (v === null || v === undefined || Number.isNaN(v)) return "N/A";
  return Number(v).toFixed(digits);
}

function fmtRate(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return "N/A";
  return `${Number(v).toFixed(2)} tok/s`;
}

function fmtMillions(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return "N/A";
  return `${Number(v).toFixed(2)}M`;
}

function toJsonBox(obj) {
  const pre = document.createElement("pre");
  pre.className = "json-box";
  pre.textContent = JSON.stringify(obj, null, 2);
  return pre;
}

function getPayloadFromInputs() {
  return {
    prompt: promptEl.value.trim(),
    max_new_tokens: Number(maxNewTokensEl.value),
    temperature: Number(temperatureEl.value),
    top_k: Number(topKEl.value),
  };
}

function winClass(a, b, lowerIsBetter = false) {
  if (a === null || a === undefined || b === null || b === undefined) return "";
  const av = Number(a);
  const bv = Number(b);
  if (Number.isNaN(av) || Number.isNaN(bv)) return "";
  if (lowerIsBetter) {
    if (av < bv) return "metric-win";
    if (av > bv) return "metric-lose";
    return "";
  }
  if (av > bv) return "metric-win";
  if (av < bv) return "metric-lose";
  return "";
}

function renderMetricsTable(data) {
  const bdh = data.bdh || {};
  const distil = data.distilgpt2 || {};
  const bdhStats = bdh.stats || {};
  const distilStats = distil.stats || {};

  const qualityAdjusted = (stats, ppl) => {
    if (!stats || stats.tokens_per_sec === undefined || !ppl) return null;
    const num = Number(stats.tokens_per_sec);
    const denom = Number(ppl);
    if (Number.isNaN(num) || Number.isNaN(denom) || denom === 0) return null;
    return num / denom;
  };

  const bdhQat = qualityAdjusted(bdhStats, bdh.prompt_perplexity);
  const distilQat = qualityAdjusted(distilStats, distil.prompt_perplexity);

  const pplGapPct =
    bdh.prompt_perplexity !== undefined && distil.prompt_perplexity !== undefined
      ? ((Number(distil.prompt_perplexity) - Number(bdh.prompt_perplexity)) / Number(distil.prompt_perplexity)) * 100
      : null;

  const rows = [
    {
      metric: "Input Tokens",
      bdh: bdhStats.input_tokens,
      distil: distilStats.input_tokens,
      lowerIsBetter: false,
      fmt: (x) => `${x ?? "N/A"}`,
    },
    {
      metric: "Generated Tokens",
      bdh: bdhStats.generated_tokens,
      distil: distilStats.generated_tokens,
      lowerIsBetter: false,
      fmt: (x) => `${x ?? "N/A"}`,
    },
    {
      metric: "Total Tokens",
      bdh: bdhStats.total_tokens,
      distil: distilStats.total_tokens,
      lowerIsBetter: false,
      fmt: (x) => `${x ?? "N/A"}`,
    },
    {
      metric: "Latency (seconds)",
      bdh: bdhStats.elapsed_sec,
      distil: distilStats.elapsed_sec,
      lowerIsBetter: true,
      fmt: (x) => fmtNum(x, 3),
    },
    {
      metric: "Generation Speed",
      bdh: bdhStats.tokens_per_sec,
      distil: distilStats.tokens_per_sec,
      lowerIsBetter: false,
      fmt: (x) => fmtRate(x),
    },
    {
      metric: "Prompt Perplexity (lower better)",
      bdh: bdh.prompt_perplexity,
      distil: distil.prompt_perplexity,
      lowerIsBetter: true,
      fmt: (x) => fmtNum(x, 3),
    },
    {
      metric: "Quality-Adjusted Throughput (tok/s ÷ perplexity)",
      bdh: bdhQat,
      distil: distilQat,
      lowerIsBetter: false,
      fmt: (x) => fmtNum(x, 2),
    },
    {
      metric: "Perplexity Advantage vs Distil (%)",
      bdh: pplGapPct,
      distil: pplGapPct === null ? null : 0,
      lowerIsBetter: false,
      fmt: (x) => (x === null ? "N/A" : `${fmtNum(x, 1)}%`),
    },
  ];

  const table = document.createElement("table");
  table.innerHTML = `
    <thead>
      <tr>
        <th>Metric</th>
        <th>BDH</th>
        <th>DistilGPT2</th>
      </tr>
    </thead>
    <tbody></tbody>
  `;
  const tbody = table.querySelector("tbody");

  rows.forEach((r) => {
    const tr = document.createElement("tr");
    const bdhCls = winClass(r.bdh, r.distil, r.lowerIsBetter);
    const distilCls = winClass(r.distil, r.bdh, r.lowerIsBetter);
    tr.innerHTML = `
      <td>${r.metric}</td>
      <td class="${bdhCls}">${r.fmt(r.bdh)}</td>
      <td class="${distilCls}">${r.fmt(r.distil)}</td>
    `;
    tbody.appendChild(tr);
  });

  const summary = document.createElement("p");
  const comp = data.comparison || {};
  const summaryBits = [];
  if (comp.faster_model) summaryBits.push(`Faster model: ${comp.faster_model}`);
  if (comp.lower_prompt_perplexity_model)
    summaryBits.push(`Lower prompt perplexity: ${comp.lower_prompt_perplexity_model}`);
  if (bdhQat !== null && distilQat !== null)
    summaryBits.push(`Quality-adjusted throughput winner: ${bdhQat >= distilQat ? "bdh" : "distilgpt2"}`);
  if (pplGapPct !== null) summaryBits.push(`BDH perplexity gap: ${fmtNum(pplGapPct, 1)}% better vs DistilGPT2`);
  summary.textContent = summaryBits.join(" | ") || "No metrics yet.";

  metricsWrapEl.innerHTML = "";
  metricsWrapEl.appendChild(summary);
  metricsWrapEl.appendChild(table);
}

async function postJson(path, payload) {
  const resp = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const body = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    throw new Error(body.error || `Request failed: ${resp.status}`);
  }
  return body;
}

async function getJson(path) {
  const resp = await fetch(path);
  const body = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(body.error || `Request failed: ${resp.status}`);
  return body;
}

async function generateCompare() {
  const payload = getPayloadFromInputs();
  if (!payload.prompt) {
    setStatus("Prompt is required.", true);
    return;
  }

  generateBtn.disabled = true;
  quickBenchmarkBtn.disabled = true;
  setStatus("Running BDH + DistilGPT2 generation...");

  try {
    const data = await postJson("/api/generate", payload);
    bdhOutputEl.textContent = data.bdh?.text || "No BDH output";
    distilOutputEl.textContent = data.distilgpt2?.text || "No DistilGPT2 output";
    renderMetricsTable(data);
    setStatus("Comparison complete.");
  } catch (err) {
    setStatus(String(err), true);
  } finally {
    generateBtn.disabled = false;
    quickBenchmarkBtn.disabled = false;
  }
}

async function runQuickBenchmark() {
  quickBenchmarkBtn.disabled = true;
  generateBtn.disabled = true;
  setStatus("Running quick benchmark on TinyStories sample...");
  try {
    const data = await postJson("/api/benchmark/quick", {
      sample_count: 30,
      max_len: Number(maxNewTokensEl.value) + 112,
      seed: 42,
    });
    benchmarkWrapEl.innerHTML = "";
    benchmarkWrapEl.appendChild(toJsonBox(data));
    setStatus("Quick benchmark complete.");
  } catch (err) {
    setStatus(String(err), true);
  } finally {
    quickBenchmarkBtn.disabled = false;
    generateBtn.disabled = false;
  }
}

function renderArtifacts(data) {
  const artifacts = data.artifacts || [];
  if (!artifacts.length) {
    artifactsWrapEl.textContent = "No benchmark artifact JSON files found.";
    return;
  }
  const ul = document.createElement("ul");
  artifacts.forEach((a) => {
    const li = document.createElement("li");
    li.textContent = `${a.name} (${a.size_bytes} bytes)`;
    ul.appendChild(li);
  });
  artifactsWrapEl.innerHTML = "";
  artifactsWrapEl.appendChild(ul);
}

function renderVisualizations(data) {
  const visuals = data.visualizations || [];
  if (!visuals.length) {
    vizWrapEl.textContent = "No visualization images found.";
    return;
  }
  vizWrapEl.innerHTML = "";
  visuals.forEach((v) => {
    const a = document.createElement("a");
    a.href = v.url;
    a.target = "_blank";
    a.rel = "noopener noreferrer";
    a.innerHTML = `
      <img src="${v.url}" alt="${v.name}" />
      <div>${v.name}</div>
    `;
    vizWrapEl.appendChild(a);
  });
}

async function bootstrap() {
  try {
    const health = await getJson("/api/health");
    setStatus(`Server ready on ${health.device}. Models loaded: BDH=${health.models_loaded?.bdh}, DistilGPT2=${health.models_loaded?.distilgpt2}`);
  } catch (err) {
    setStatus(`Health check failed: ${err}`, true);
  }

  try {
    const cfg = await getJson("/api/config");
    renderModelCards(cfg.model_cards);
  } catch (err) {
    modelSpecsWrapEl.textContent = `Failed to load model specs: ${err}`;
  }

  try {
    const artifacts = await getJson("/api/artifacts");
    renderArtifacts(artifacts);
    renderVisualizations(artifacts);
  } catch (err) {
    artifactsWrapEl.textContent = `Failed to load artifacts: ${err}`;
    vizWrapEl.textContent = `Failed to load visualizations: ${err}`;
  }

  try {
    const latest = await getJson("/api/benchmark/latest");
    benchmarkWrapEl.innerHTML = "";
    benchmarkWrapEl.appendChild(toJsonBox(latest));
  } catch {
    benchmarkWrapEl.textContent = "No previous benchmark found yet.";
  }
}

generateBtn.addEventListener("click", generateCompare);
quickBenchmarkBtn.addEventListener("click", runQuickBenchmark);

bootstrap();

function renderModelCards(cards) {
  if (!cards) {
    modelSpecsWrapEl.textContent = "Model specs unavailable.";
    return;
  }

  const grid = document.createElement("div");
  grid.className = "spec-grid";

  const renderCard = (key, data) => {
    if (!data) return;
    const card = document.createElement("div");
    card.className = "spec-card";
    const title = data.name || key.toUpperCase();

    card.innerHTML = `
      <h3>${title}</h3>
      <div class="spec-meta">
        <span>${fmtMillions(data.params_m)} params</span>
        <span>${data.max_seq_len ? `${data.max_seq_len} ctx` : "ctx n/a"}</span>
      </div>
    `;

    const details = document.createElement("div");
    const addKv = (label, value) => {
      if (value === null || value === undefined) return;
      const row = document.createElement("div");
      row.className = "kv";
      row.innerHTML = `<span>${label}</span><span>${value}</span>`;
      details.appendChild(row);
    };

    addKv("Layers", data.n_layer);
    addKv("Heads", data.n_head);
    addKv("Hidden Size", data.n_embd);
    addKv("FFN Dim", data.ffn_dim);
    addKv("Logical Layers", data.logical_layers);
    addKv("Model Vocab", data.model_vocab_size);
    addKv("Tokenizer Vocab", data.tokenizer_vocab_size);
    addKv("Vocab Padding", data.vocab_padding_tokens);
    if (data.teacher_model) addKv("Teacher", data.teacher_model);
    if (data.step !== null && data.step !== undefined) addKv("Checkpoint Step", data.step);
    card.appendChild(details);

    if (Array.isArray(data.features) && data.features.length) {
      const pills = document.createElement("div");
      pills.className = "pill-row";
      data.features.forEach((f) => {
        const span = document.createElement("span");
        span.className = "pill";
        span.textContent = f;
        pills.appendChild(span);
      });
      card.appendChild(pills);
    }

    grid.appendChild(card);
  };

  renderCard("bdh", cards.bdh);
  renderCard("distilgpt2", cards.distilgpt2);

  modelSpecsWrapEl.innerHTML = "";
  modelSpecsWrapEl.appendChild(grid);
}


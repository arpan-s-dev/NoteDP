const rosterEl = document.getElementById("roster");
const noteEl = document.getElementById("note");
const epsEl = document.getElementById("eps");
const epsOut = document.getElementById("epsOut");
const metricsEl = document.getElementById("metrics");
const sanitizedEl = document.getElementById("sanitized");
const cleanEl = document.getElementById("clean");
const privateEl = document.getElementById("private");
const runBtn = document.getElementById("run");
const statusEl = document.getElementById("status");

let selected = null;
let charts = [];

epsEl.addEventListener("input", () => {
  epsOut.textContent = Number(epsEl.value).toFixed(2);
});

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === tab));
    const name = tab.dataset.tab;
    document.getElementById("panel-original").classList.toggle("hidden", name !== "original");
    document.getElementById("panel-sanitized").classList.toggle("hidden", name !== "sanitized");
    document.getElementById("panel-model").classList.toggle("hidden", name !== "model");
  });
});

function tag(risk) {
  return `<span class="tag ${risk}">${risk}</span>`;
}

function initials(name) {
  return name
    .split(" ")
    .map((p) => p[0])
    .join("")
    .slice(0, 2);
}

function highlight(note, highlights) {
  let html = note
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
  const unique = [...new Set(highlights.map((h) => h.text).filter(Boolean))];
  unique.sort((a, b) => b.length - a.length);
  for (const text of unique) {
    const esc = text.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
    if (!esc) continue;
    html = html.split(esc).join(`<mark>${esc}</mark>`);
  }
  return html;
}

function showTab(name) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  document.getElementById("panel-original").classList.toggle("hidden", name !== "original");
  document.getElementById("panel-sanitized").classList.toggle("hidden", name !== "sanitized");
  document.getElementById("panel-model").classList.toggle("hidden", name !== "model");
}

async function loadRoster() {
  const res = await fetch("/api/charts");
  const data = await res.json();
  charts = data.charts;
  document.getElementById("count").textContent = String(charts.length);
  rosterEl.innerHTML = charts
    .map(
      (c) => `
      <button class="card" data-id="${c.id}" type="button">
        <div class="name">${c.display_name} ${tag(c.risk_tag)}</div>
        <div class="meta">${c.mrn} · ${c.age}${c.sex} · ${c.specialty}<br/>${c.one_line}</div>
      </button>`
    )
    .join("");
  rosterEl.querySelectorAll(".card").forEach((btn) => {
    btn.addEventListener("click", () => select(btn.dataset.id));
  });
  if (charts[0]) select(charts[0].id);
}

async function select(id) {
  selected = id;
  rosterEl.querySelectorAll(".card").forEach((b) => b.classList.toggle("active", b.dataset.id === id));
  const res = await fetch(`/api/charts/${id}`);
  const chart = await res.json();
  document.getElementById("avatar").textContent = initials(chart.display_name);
  document.getElementById("kicker").textContent = `${chart.note_type} · ${chart.specialty}`;
  document.getElementById("ptName").textContent = chart.display_name;
  document.getElementById("ptMeta").textContent = `${chart.facility}`;
  document.getElementById("ptStats").innerHTML = `
    <div><dt>MRN</dt><dd>${chart.mrn}</dd></div>
    <div><dt>Age / sex</dt><dd>${chart.age} / ${chart.sex}</dd></div>
    <div><dt>Encounter</dt><dd>${chart.encounter_date}</dd></div>`;
  noteEl.innerHTML = highlight(chart.note, chart.highlights);
  sanitizedEl.textContent = chart.sanitized;
  cleanEl.textContent = "Run the model to fill this pane.\nThe excerpt is a few sentences, not the full note.";
  privateEl.textContent = "";
  metricsEl.innerHTML = "";
  statusEl.textContent = "";
  showTab("original");
}

function metric(label, value) {
  return `<div class="metric"><b>${label}</b>${value}</div>`;
}

runBtn.addEventListener("click", async () => {
  if (!selected) return;
  runBtn.disabled = true;
  runBtn.textContent = "Running on CPU…";
  statusEl.textContent = "Loading checkpoint and applying noise. First run is slower.";
  try {
    const res = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ chart_id: selected, epsilon: Number(epsEl.value) }),
    });
    const out = await res.json();
    if (!res.ok) throw new Error(out.detail || "Run failed");
    noteEl.innerHTML = highlight(out.original_note, out.highlights);
    sanitizedEl.textContent = out.sanitized;
    cleanEl.textContent = `EXCERPT FED TO MODEL\n${out.excerpt}\n\nRECONSTRUCTION (no noise)\n${out.non_private}`;
    privateEl.textContent = out.private;
    metricsEl.innerHTML = [
      metric("ANC profile", out.profile),
      metric("Leak risk R(y)", out.risk.toFixed(3)),
      metric("σ_emb / σ_att", `${out.sigma_emb.toFixed(3)} / ${out.sigma_att.toFixed(3)}`),
      metric("Embedding cosine", out.embedding_cosine.toFixed(3)),
      metric("BLEU-4", out.bleu.toFixed(3)),
      metric("ROUGE-L", out.rouge.toFixed(3)),
      metric("Latency", `${out.latency_ms.toFixed(0)} ms`),
      metric("ε remaining", out.epsilon_remaining.toFixed(4)),
    ].join("");
    statusEl.textContent = "Done. Model excerpt tab shows clean vs noisy reconstructions.";
    showTab("model");
  } catch (err) {
    privateEl.textContent = String(err);
    statusEl.textContent = String(err);
    showTab("model");
  } finally {
    runBtn.disabled = false;
    runBtn.textContent = "Run";
  }
});

loadRoster();

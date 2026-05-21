"use strict";

const app = {
  data: null,
  activeTab: "pipeline",
  selectedId: null,
  detail: null,
  loadingDetail: false,
  filters: { family: "all", symbol: "all", gate: "all", minScore: 0 },
  runArgs: {
    use_llm: false,
    iterations: 1,
    hypothesis_count: 5,
    max_workers: 1,
    tail: 50000,
    archive_top: 3,
    research_brief: "",
  },
  toast: "",
  archives: null,
};

const tabDefs = [
  { id: "pipeline", label: "Pipeline", icon: "flow" },
  { id: "experiments", label: "Experiments", icon: "flask" },
  { id: "diagnostics", label: "Diagnostics", icon: "shield" },
  { id: "hypotheses", label: "Hypotheses", icon: "brain" },
  { id: "candidates", label: "Candidate Lab", icon: "list" },
  { id: "methods", label: "Methods", icon: "table" },
  { id: "ledger", label: "Trial Ledger", icon: "shield" },
  { id: "data", label: "Data", icon: "database" },
  { id: "archive", label: "Archive", icon: "archive" },
];

const stageOrder = ["hypothesis", "candidates", "backtests", "gatecheck", "research", "survivor", "hardscore", "archive"];

const fmt = {
  num(value, digits = 2) {
    const n = Number(value);
    return Number.isFinite(n) ? n.toFixed(digits) : "n/a";
  },
  signed(value, digits = 2) {
    const n = Number(value);
    if (!Number.isFinite(n)) return "n/a";
    return `${n > 0 ? "+" : ""}${n.toFixed(digits)}`;
  },
  pct(value, digits = 2) {
    const n = Number(value);
    if (!Number.isFinite(n)) return "n/a";
    return `${n > 0 ? "+" : ""}${(n * 100).toFixed(digits)}%`;
  },
  int(value) {
    const n = Number(value);
    return Number.isFinite(n) ? Math.round(n).toLocaleString("en-US") : "n/a";
  },
  compact(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return "n/a";
    if (Math.abs(n) >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
    if (Math.abs(n) >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
    if (Math.abs(n) >= 1e3) return `${(n / 1e3).toFixed(1)}k`;
    return n.toFixed(0);
  },
  time(value) {
    if (!value) return "--:--:--";
    const d = new Date(value);
    if (Number.isNaN(d.getTime())) return String(value).slice(0, 8);
    return d.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  },
  date(value) {
    if (!value) return "n/a";
    const d = new Date(value);
    if (Number.isNaN(d.getTime())) return String(value);
    return d.toISOString().slice(0, 19).replace("T", " ");
  },
};

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "\"": "&quot;",
    "'": "&#039;",
  }[char]));
}

function icon(name, size = 14) {
  const common = `width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"`;
  const paths = {
    activity: '<path d="M3 12h4l3-8 4 16 3-8h4"/>',
    archive: '<rect x="3" y="4" width="18" height="4" rx="1"/><path d="M5 8v11a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V8M10 12h4"/>',
    brain: '<path d="M9 4a3 3 0 0 0-3 3 3 3 0 0 0-2 5 3 3 0 0 0 2 5 3 3 0 0 0 3 3h1V4H9zM15 4a3 3 0 0 1 3 3 3 3 0 0 1 2 5 3 3 0 0 1-2 5 3 3 0 0 1-3 3h-1V4h1z"/>',
    check: '<circle cx="12" cy="12" r="9"/><path d="m8 12 3 3 5-6"/>',
    close: '<path d="M6 6l12 12M18 6 6 18"/>',
    database: '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/>',
    flow: '<circle cx="6" cy="6" r="2"/><circle cx="6" cy="18" r="2"/><circle cx="18" cy="12" r="2"/><path d="M8 6h2a4 4 0 0 1 4 4M8 18h2a4 4 0 0 0 4-4"/>',
    flask: '<path d="M9 3h6M10 3v6L4 19a2 2 0 0 0 2 3h12a2 2 0 0 0 2-3l-6-10V3"/>',
    list: '<path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01"/>',
    logo: '<rect x="3" y="3" width="18" height="18" rx="3" fill="currentColor" opacity=".12"/><path d="M7 17V7l5 5 5-5v10"/>',
    play: '<path d="M7 5v14l12-7z"/>',
    refresh: '<path d="M21 12a9 9 0 1 1-3-6.7"/><path d="M21 4v6h-6"/>',
    settings: '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 0 1-4 0v-.1a1.7 1.7 0 0 0-1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1A1.7 1.7 0 0 0 4.6 15a1.7 1.7 0 0 0-1.5-1H3a2 2 0 0 1 0-4h.1a1.7 1.7 0 0 0 1.5-1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 0 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 0 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/>',
    shield: '<path d="M12 3 4 6v6c0 5 3.5 8 8 9 4.5-1 8-4 8-9V6l-8-3z"/><path d="m9 12 2 2 4-4"/>',
    stop: '<rect x="6" y="6" width="12" height="12" rx="1"/>',
    table: '<rect x="3" y="4" width="18" height="16" rx="1"/><path d="M3 10h18M9 4v16"/>',
    trend: '<path d="m3 17 6-6 4 4 8-8"/><path d="M14 7h7v7"/>',
    warn: '<circle cx="12" cy="12" r="9"/><path d="M12 8v4M12 16h.01"/>',
    xcircle: '<circle cx="12" cy="12" r="9"/><path d="m9 9 6 6M15 9l-6 6"/>',
  };
  return `<svg ${common}>${paths[name] || ""}</svg>`;
}

async function getJson(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.message || response.statusText);
  return data;
}

async function loadState() {
  try {
    app.data = await getJson("/api/state");
    if (app.data.active_run && app.data.active_run.args) {
      Object.assign(app.runArgs, app.data.active_run.args);
    }
    loadArchives();
    render();
  } catch (error) {
    document.getElementById("app").innerHTML = `<div class="boot">Dashboard error: ${esc(error.message)}</div>`;
  }
}

function setToast(message) {
  app.toast = message;
  render();
  setTimeout(() => {
    if (app.toast === message) {
      app.toast = "";
      render();
    }
  }, 4200);
}

function kpis() {
  const data = app.data;
  const experiments = data.experiments || [];
  const gates = {
    pass: experiments.filter((row) => row.gate === "pass").length,
    warn: experiments.filter((row) => row.gate === "warn").length,
    fail: experiments.filter((row) => row.gate === "fail").length,
  };
  gates.accepted = gates.pass + gates.warn;
  const top = experiments[0] || {};
  const sharpes = experiments.map((row) => Number(row.sharpe)).filter(Number.isFinite).sort((a, b) => a - b);
  const medianSharpe = sharpes.length ? sharpes[Math.floor(sharpes.length / 2)] : 0;
  return {
    hypotheses: data.bundle.hypotheses.length,
    candidates: data.bundle.candidates.length,
    backtests: data.bundle.backtests.length,
    archived: experiments.filter((row) => Number(row.hardscore) > 0).slice(0, 3).length,
    gates,
    top,
    topScore: Number(top.hardscore || 0),
    medianSharpe,
  };
}

function gateDiagnostics() {
  return (app.data && app.data.bundle && app.data.bundle.gatecheck_diagnostics) || {};
}

function statusInfo() {
  const active = app.data.active_run;
  const latest = app.data.latest_run;
  if (active) {
    return { label: `${active.status} · ${shortId(active.run_id)}`, className: "running", icon: "activity" };
  }
  if (latest) {
    const cls = latest.status === "completed" ? "ok" : latest.status === "failed" ? "fail" : "idle";
    const iconName = latest.status === "completed" ? "check" : latest.status === "failed" ? "xcircle" : "activity";
    return { label: `${latest.status} · ${shortId(latest.run_id)}`, className: cls, icon: iconName };
  }
  return { label: "idle", className: "idle", icon: "activity" };
}

function render() {
  if (!app.data) return;
  var screenEl = document.querySelector(".screen.scrollbar");
  var scrollTop = screenEl ? screenEl.scrollTop : (app._scrollByTab || {})[app.activeTab] || 0;
  document.documentElement.dataset.theme = "dark";
  document.documentElement.dataset.density = "comfortable";
  document.getElementById("app").className = "";
  document.getElementById("app").innerHTML = `
    <div class="app">
      ${renderTopbar()}
      ${renderRail()}
      <main class="main">
        ${renderViewHeader()}
        <div class="screen scrollbar">${renderActiveTab()}</div>
      </main>
      ${app.selectedId ? renderDrawer() : ""}
      ${app.toast ? `<div class="toast">${esc(app.toast)}</div>` : ""}
    </div>
  `;
  requestAnimationFrame(function() {
    var newScreen = document.querySelector(".screen.scrollbar");
    if (newScreen) newScreen.scrollTop = scrollTop;
  });
}

function saveScroll() {
  var screenEl = document.querySelector(".screen.scrollbar");
  if (!app._scrollByTab) app._scrollByTab = {};
  if (screenEl) app._scrollByTab[app.activeTab] = screenEl.scrollTop;
}

function renderTopbar() {
  const data = app.data;
  const latest = data.latest_run;
  const active = data.active_run;
  const status = statusInfo();
  const runLabel = latest ? `${shortId(latest.run_id)} / ${fmt.date(latest.started_at)}` : "no recorded run";
  const runIcon = active ? "stop" : "play";
  const runAction = active ? "stop-run" : "run-start";
  const runText = active ? "Stop" : "Run pipeline";
  return `
    <header class="topbar">
      <div class="brand">
        <div class="brand-mark">fm</div>
        <div class="brand-text">
          <div class="brand-title">Factor Mining</div>
          <div class="brand-sub">v1 · local · ${(data.settings.symbols || []).join("/") || "BTC/ETH"} ${esc(data.settings.default_interval || "")}</div>
        </div>
      </div>
      <div class="crumb">
        <span>run</span><span class="sep">/</span><span class="now">${esc(runLabel)}</span><span class="sep">·</span><span>${data.bundle.optimization_history.length || 0} rounds</span>
      </div>
      <span class="topbar-spacer"></span>
      ${active ? `<span class="poll-indicator"><span class="dot"></span>live</span>` : ""}
      <div class="status-pill ${esc(status.className)}">
        <span class="status-dot"></span>${icon(status.icon, 11)}<span>${esc(status.label)}</span>
      </div>
      <button class="btn ${active ? "danger" : "primary"}" data-action="${runAction}" title="${esc(runText)}">${icon(runIcon, 12)}${esc(runText)}</button>
      <button class="btn icon" data-action="refresh" title="Refresh">${icon("refresh", 13)}</button>
    </header>
  `;
}

function renderRail() {
  const data = app.data;
  const active = data.active_run;
  const symbols = data.settings.symbols || [];
  return `
    <aside class="rail">
      <div class="rail-scroll scrollbar">
        <section class="rail-section">
          <div class="eyebrow"><span class="dot"></span>Navigate</div>
          <div class="nav-list">
            ${tabDefs.map((tab) => `
              <button class="nav-item ${tab.id === app.activeTab ? "active" : ""}" data-tab="${esc(tab.id)}">
                <span class="icon">${icon(tab.icon, 14)}</span>
                <span>${esc(tab.label)}</span>
                <span class="count mono">${esc(navCount(tab.id))}</span>
              </button>
            `).join("")}
          </div>
        </section>
        <section class="rail-section">
          <div class="eyebrow"><span class="dot"></span>Universe</div>
          <div class="universe-card">
            ${symbols.map((symbol) => `
              <div class="universe-row">
                <span class="sym mono">${esc(symbol)}</span>
                <span class="meta mono">${esc((data.settings.markets || []).join(" · ")) || "spot"} · ${esc(data.settings.default_interval || "")}</span>
              </div>
            `).join("")}
            <div class="muted mono" style="font-size:10.5px;margin-top:2px">${esc(data.settings.trial_window_days)}d trial window · local SQLite</div>
          </div>
        </section>
        <section class="rail-section">
          <div class="eyebrow"><span class="dot"></span>Run control</div>
          <div class="run-card">
            ${active ? `<span class="badge info">${icon("activity", 11)}${esc(active.status)} ${esc(shortId(active.run_id))}</span>` : `<span class="muted mono">No active run.</span>`}
            <label class="toggle-row"><span>Use DeepSeek hypotheses</span><input data-run-field="use_llm" type="checkbox" ${app.runArgs.use_llm ? "checked" : ""} ${active ? "disabled" : ""}></label>
            <div class="field-row">
              ${field("Rounds", "iterations", "number", 1, 7, active)}
              ${app.runArgs.use_llm ? field("Hypotheses", "hypothesis_count", "number", 1, 10, active) : `<span class="muted" style="font-size:11px;align-self:center">19 built-in</span>`}
            </div>
            <div class="field-row">
              ${field("Workers", "max_workers", "number", 1, 64, active)}
              ${field("Archive top", "archive_top", "number", 0, 10, active)}
            </div>
            ${field("Tail bars", "tail", "number", 0, 2000000, active)}
            <div class="field">
              <label>Research brief</label>
              <textarea data-run-field="research_brief" ${active ? "disabled" : ""}>${esc(app.runArgs.research_brief)}</textarea>
            </div>
            <div class="run-actions">
              <button class="btn primary" data-action="run-start" ${active ? "disabled" : ""}>${icon("play", 12)}Run</button>
              <button class="btn" data-action="${active ? "stop-run" : "hosted-start"}">${icon(active ? "stop" : "refresh", 12)}${active ? "Stop" : "Hosted"}</button>
            </div>
          </div>
        </section>
        <section class="rail-section">
          <div class="eyebrow"><span class="dot"></span>Storage</div>
          <div class="mono muted" style="overflow-wrap:anywhere;font-size:11px">${esc(data.settings.sqlite_path)}</div>
        </section>
      </div>
    </aside>
  `;
}

function navCount(tabId) {
  const data = app.data;
  const diag = gateDiagnostics();
  const map = {
    pipeline: data.bundle.optimization_history.length || data.experiments.length,
    experiments: data.experiments.length,
    diagnostics: (diag.rows || []).length,
    hypotheses: data.bundle.hypotheses.length,
    candidates: data.bundle.candidates.length,
    methods: data.methods.length,
    ledger: (data.ledger || []).reduce((sum, row) => sum + Number(row.family_trials || 0), 0),
    data: data.coverage.length,
    archive: (data.archives || []).length,
  };
  return map[tabId] ?? "";
}

function pipelineStep(label, value, iconName) {
  const done = Number(value) > 0 || String(value).includes("/");
  return `
    <div class="pipeline-step ${done ? "done" : ""}">
      ${icon(iconName, 14)}
      <span>${esc(label)}</span>
      <span class="mono">${esc(value)}</span>
    </div>
  `;
}

function field(label, key, type, min, max, disabled) {
  return `
    <div class="field">
      <label>${esc(label)}</label>
      <input data-run-field="${esc(key)}" type="${type}" min="${min}" max="${max}" value="${esc(app.runArgs[key])}" ${disabled ? "disabled" : ""}>
    </div>
  `;
}

function renderViewHeader() {
  const meta = {
    pipeline: {
      title: "Pipeline overview",
      subtitle: `${shortId((app.data.latest_run || {}).run_id || "no run", 24)} · ${app.data.experiments.length} experiments · workflow-centric mining view`,
    },
    experiments: {
      title: "Experiments",
      subtitle: "Ranked by HardScore · drill into any row for detail artifacts",
    },
    diagnostics: {
      title: "Diagnostics",
      subtitle: "GateCheck failures, warnings, research survivors, and near-miss repair hints",
    },
    hypotheses: {
      title: "Hypotheses",
      subtitle: "Economic mechanism, null hypothesis, and expected IC assumptions",
    },
    candidates: {
      title: "Candidate lab",
      subtitle: "Generated parameter specs across symbols, markets, intervals, and methods",
    },
    methods: {
      title: "Method registry",
      subtitle: "Implemented, planned, blocked, and schedulable mining methods",
    },
    ledger: {
      title: "Trial ledger",
      subtitle: `${app.data.settings.trial_window_days} day rolling window by hypothesis family`,
    },
    data: {
      title: "Data coverage",
      subtitle: "Local warehouse coverage, run records, and storage paths",
    },
  }[app.activeTab] || { title: app.activeTab, subtitle: "" };
  return `
    <div class="view-header">
      <div>
        <div class="view-title">${esc(meta.title)}</div>
        <div class="view-sub">${esc(meta.subtitle)}</div>
      </div>
      <div class="view-tools">
        ${app.activeTab === "pipeline" ? `<span class="kbd">live</span>` : ""}
        <button class="btn icon" data-action="refresh" title="Refresh">${icon("refresh", 13)}</button>
      </div>
    </div>
  `;
}

function renderActiveTab() {
  if (app.activeTab === "pipeline") return renderPipeline();
  if (app.activeTab === "experiments") return renderExperiments();
  if (app.activeTab === "diagnostics") return renderDiagnostics();
  if (app.activeTab === "hypotheses") return renderHypotheses();
  if (app.activeTab === "candidates") return renderCandidates();
  if (app.activeTab === "methods") return renderMethods();
  if (app.activeTab === "ledger") return renderLedger();
  if (app.activeTab === "data") return renderData();
  if (app.activeTab === "archive") return renderArchive();
  return "";
}

function renderPipeline() {
  const data = app.data;
  const k = kpis();
  const latest = data.latest_run;
  return `
    <div class="screen-grid">
      ${renderFlowHero()}
      <div class="grid-4">
        ${kpiTile("Gate accepted", `${k.gates.accepted}/${data.experiments.length}`, `${k.gates.warn} diagnostic · ${k.gates.fail} fail`, "pass")}
        ${kpiTile("Top HardScore", fmt.num(k.topScore, 1), `Sharpe ${fmt.signed(k.top.sharpe || 0)}`, "info")}
        ${kpiTile("Research survivors", data.bundle.research_survivors.length, "optimizer candidate pool", "violet")}
        ${kpiTile("Latest run", latest ? latest.status : "idle", latest ? shortId(latest.run_id, 18) : "no run")}
      </div>
      <div class="two-col">
        ${panel("Funnel", "Counts retained through the mining workflow", renderFunnel())}
        ${panel("Run log", `${data.events.length} events`, renderRunLog(), true)}
      </div>
      ${panel("Gate diagnostics", gateDiagnosticsSubtitle(), renderGateDiagnostics(), true)}
      ${panel("Top experiments", "Ranked by HardScore (DSR-adjusted, haircut, NW-FDR-aware)", renderExperimentTable(data.experiments.slice(0, 12), true), true)}
      <div class="two-col">
        ${panel("Activity by hypothesis family", "Trials and survivors in this run", renderFamilyActivity())}
        ${panel("Optimization history", `${data.bundle.optimization_history.length} rounds`, renderOptimizationHistory(), true)}
      </div>
    </div>
  `;
}

function workflowStages() {
  const data = app.data;
  const k = kpis();
  const diag = gateDiagnostics();
  const liveStage = currentStageId();
  const liveIndex = stageOrder.indexOf(liveStage);
  const researchGate = data.bundle.research_gate || [];
  const survivorStore = data.bundle.research_survivor_store || [];
  const researchGateCounts = researchGate.reduce((acc, row) => {
    const key = row.status || row.research_gate_status || "unknown";
    acc[key] = (acc[key] || 0) + 1;
    return acc;
  }, {});
  const survivors = data.bundle.research_survivors.length || survivorStore.filter((row) => (row.status || "active") === "active").length;
  const totalGate = Number(diag.total || data.experiments.length || 0);
  const stages = [
    { id: "hypothesis", label: "Hypothesis", count: data.bundle.hypotheses.length, prev: null, color: "var(--c-blue)", detail: "LLM or deterministic seed hypotheses", state: data.bundle.hypotheses.length ? "done" : "warn" },
    { id: "candidates", label: "Candidates", count: data.bundle.candidates.length, prev: data.bundle.hypotheses.length, color: "var(--c-blue)", detail: "parameter specs by symbol, market, and method", state: data.bundle.candidates.length ? "done" : "warn" },
    { id: "backtests", label: "Backtests", count: data.bundle.backtests.length, prev: data.bundle.candidates.length, color: "var(--stage-discovery)", detail: "primary backtest artifacts with OOS metrics", state: data.bundle.backtests.length ? "done" : "warn" },
    { id: "gatecheck", label: "GateCheck", count: `${k.gates.accepted}/${totalGate || data.experiments.length}`, prev: data.bundle.backtests.length, color: "var(--stage-gate)", detail: "full pass plus warning-tier acceptance", state: k.gates.fail ? "warn" : "done" },
    { id: "research", label: "Research gate", count: `${researchGateCounts.production_passed || 0}/${researchGateCounts.research_survivor || 0}/${researchGateCounts.rejected || 0}`, prev: totalGate, color: "var(--stage-research)", detail: "production · survivor · rejected", state: researchGate.length ? "done" : "warn" },
    { id: "survivor", label: "Survivors", count: survivors, prev: totalGate, color: "var(--c-violet)", detail: "paper-trade candidates kept for recheck", state: survivors ? "done" : "warn" },
    { id: "hardscore", label: "HardScore", count: fmt.num(k.topScore, 1), prev: k.gates.accepted, color: "var(--stage-final)", detail: "10-dim: DSR+haircut+PBO+PSR+IC+FDR+trades+regime+calib+purity", state: k.topScore > 0 ? "done" : "warn" },
    { id: "archive", label: "Archive", count: k.archived, prev: k.gates.accepted, color: "var(--stage-archive)", detail: "reproducible top-ranked bundles", state: k.archived ? "done" : "warn" },
  ];
  if (!data.active_run || liveIndex < 0) return stages;
  return stages.map((stage, index) => {
    if (stage.id === liveStage) return { ...stage, state: "live" };
    if (index < liveIndex) return { ...stage, state: "done" };
    return { ...stage, state: "pending" };
  });
}

function currentStageId() {
  if (!app.data || !app.data.active_run) return null;
  const events = app.data.events || [];
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const stage = stageFromEvent(events[index]);
    if (stage) return stage;
  }
  return "hypothesis";
}

function stageFromEvent(event) {
  const phase = String((event && event.phase) || "").toLowerCase();
  const message = String((event && event.message) || "").trim().toLowerCase();
  if (!message) return null;
  if (phase === "step") {
    if (message.includes("hypoth")) return "hypothesis";
    if (message.includes("building candidates")) return "candidates";
    if (message.includes("mining round")) return "backtests";
    return null;
  }
  if (message.includes("pipeline complete") || message.includes("archived:")) return "archive";
  if (message.includes("hardscore") || message.startsWith("score=")) return "hardscore";
  if (message.includes("research gate") || message.includes("near miss")) return "research";
  if (message.includes("research survivors:")) return "survivor";
  if (
    message.includes("gatecheck") ||
    message.includes("gate diagnostics") ||
    message.startsWith("fail ") ||
    message.startsWith("pass ") ||
    message.startsWith("cond ")
  ) return "gatecheck";
  if (
    message.includes("backtest") ||
    message.includes("repair validation") ||
    message.includes("final oos") ||
    message.startsWith("split:")
  ) return "backtests";
  if (
    message.includes("candidate") ||
    message.startsWith("data ") ||
    message.startsWith("features ") ||
    message.startsWith("funding ") ||
    message.startsWith("regime ") ||
    message.includes("hmm ")
  ) return "candidates";
  if (message.includes("hypoth") || message.includes("deepseek") || message.includes("falling back")) return "hypothesis";
  return null;
}

function renderFlowHero() {
  const stages = workflowStages();
  return `
    <div class="flow">
      ${stages.map((stage, index) => {
        const numeric = numericStageCount(stage.count);
        const prev = stage.prev == null ? null : Number(stage.prev);
        const kept = prev && Number.isFinite(prev) && prev > 0 && Number.isFinite(numeric) && numeric <= prev
          ? Math.max(0, Math.min(100, numeric / prev * 100))
          : null;
        const delta = prev && numeric > prev ? "expanded" : kept == null ? "workflow input" : `${kept.toFixed(0)}% retained`;
        return `
          <button class="flow-stage ${stage.state === "live" ? "active" : ""}" data-state="${esc(stage.state)}" style="--stage-color:${stage.color}">
            <span class="stage-marker">${String(index + 1).padStart(2, "0")}</span>
            <div class="stage-label"><span class="accent"></span>${esc(stage.label)}</div>
            <div class="stage-count">${esc(stage.count)}</div>
            <div class="stage-delta ${kept == null || kept >= 65 ? "kept" : "drop"}">${esc(delta)}</div>
            <div class="stage-detail">${esc(stage.detail)}</div>
          </button>
        `;
      }).join("")}
    </div>
  `;
}

function renderFunnel() {
  const stages = workflowStages();
  const numericStages = stages.map((stage) => ({ ...stage, numeric: numericStageCount(stage.count) || 0 }));
  const max = Math.max(1, ...numericStages.map((stage) => stage.numeric));
  return `
    <div class="funnel">
      ${numericStages.map((stage, index) => {
        const prev = index === 0 ? null : numericStages[index - 1].numeric;
        const width = Math.max(6, stage.numeric / max * 100);
        const kept = prev ? stage.numeric / Math.max(prev, 1) * 100 : 100;
        const expanded = prev && stage.numeric > prev;
        return `
          <div class="funnel-row">
            <div class="lbl">
              <div class="name">${esc(stage.label)}</div>
              <div class="sub">${esc(stage.detail.split("·")[0].trim())}</div>
            </div>
            <div class="funnel-bar" style="--stage-color:${stage.color}">
              <div class="fill" style="width:${width}%"></div>
              <div class="label mono">${esc(stage.count)}</div>
            </div>
            <div class="figure">
              <div class="mono">${prev ? (expanded ? `${(stage.numeric / Math.max(prev, 1)).toFixed(1)}x` : `${kept.toFixed(0)}%`) : "100%"}</div>
              <div class="pct mono">${prev ? (expanded ? "expanded" : `${(100 - kept).toFixed(0)}% drop`) : "of input"}</div>
            </div>
          </div>
        `;
      }).join("")}
    </div>
  `;
}

function numericStageCount(value) {
  if (typeof value === "number") return value;
  const text = String(value || "");
  if (text.includes("/")) {
    const parts = text.split("/").map(Number).filter(Number.isFinite);
    return parts.length === 2 ? parts[0] : parts.reduce((sum, item) => sum + item, 0);
  }
  const parsed = Number(text);
  return Number.isFinite(parsed) ? parsed : 0;
}

function kpiTile(label, value, sub, tone = "") {
  return `
    <div class="kpi">
      <div class="label-eyebrow">${esc(label)}</div>
      <div class="kpi-value ${esc(tone)}">${esc(value)}</div>
      <div class="kpi-sub">${esc(sub)}</div>
    </div>
  `;
}

function panel(title, subtitle, body, flush = false, right = "") {
  return `
    <section class="panel">
      <header class="panel-header">
        <div><div class="panel-title">${esc(title)}</div><div class="panel-subtitle">${esc(subtitle || "")}</div></div>
        ${right}
      </header>
      <div class="panel-body ${flush ? "flush" : ""}">${body}</div>
    </section>
  `;
}

function renderWorkflow() {
  const data = app.data;
  const k = kpis();
  const researchGate = (data.bundle && data.bundle.research_gate) || [];
  const survivorStore = (data.bundle && data.bundle.research_survivor_store) || [];
  const researchGateCounts = researchGate.reduce((acc, row) => {
    const key = row.status || "unknown";
    acc[key] = (acc[key] || 0) + 1;
    return acc;
  }, {});
  const activeSurvivors = survivorStore.filter((row) => (row.status || "active") === "active").length;
  const stages = [
    ["Hypotheses", k.hypotheses, "DeepSeek or defaults", "brain"],
    ["Candidates", k.candidates, "symbols x methods", "flask"],
    ["Backtests", k.backtests, "vol-target sized", "activity"],
    ["GateCheck", `${k.gates.accepted}/${data.experiments.length}`, "accepted / blocking fail", "shield"],
    ["Research Gate", `${researchGateCounts.production_passed || 0}/${researchGateCounts.research_survivor || 0}/${researchGateCounts.rejected || 0}`, "prod / survivor / reject", "shield"],
    ["Survivor Store", activeSurvivors, "paper-trade recheck", "archive"],
    ["HardScore", fmt.num(k.topScore, 1), "haircut / DSR", "trend"],
    ["Archive", k.archived, "reproducible", "archive"],
  ];
  return `
    <div class="workflow">
      ${stages.map(([label, value, sub, iconName]) => `
        <div class="stage-card">
          <div class="label-eyebrow">${icon(iconName, 13)} ${esc(label)}</div>
          <div class="stage-value">${esc(value)}</div>
          <div class="stage-sub">${esc(sub)}</div>
        </div>
      `).join("")}
    </div>
  `;
}

function gateDiagnosticsSubtitle() {
  const diag = gateDiagnostics();
  if (!diag || !Number.isFinite(Number(diag.total))) return "No diagnostics artifact yet";
  return `${diag.passed || 0}/${diag.total || 0} passed / ${((diag.rows || []).length)} diagnostic rows`;
}

function renderGateDiagnostics() {
  const diag = gateDiagnostics();
  const rows = diag.rows || [];
  const survivors = (app.data.bundle && app.data.bundle.research_survivors) || [];
  const survivorStore = (app.data.bundle && app.data.bundle.research_survivor_store) || [];
  const evidence = (app.data.bundle && app.data.bundle.factor_evidence) || [];
  const researchGate = (app.data.bundle && app.data.bundle.research_gate) || [];
  const nearMisses = (app.data.bundle && app.data.bundle.near_misses) || [];
  const researchGateCounts = researchGate.reduce((acc, row) => {
    const key = row.status || "unknown";
    acc[key] = (acc[key] || 0) + 1;
    return acc;
  }, {});
  if (!rows.length) {
    return `<div class="empty">No GateCheck diagnostics artifact yet. Run the pipeline again to populate failure counts, gross/net Sharpe, turnover, and cost margin.</div>`;
  }
  const summary = diag.metric_summary || {};
  const topFailures = diag.failure_counts || [];
  const topWarnings = diag.warning_counts || [];
  const topRows = diag.top_by_net_sharpe || [];
  const underpoweredSurvivors = nearMisses.filter((row) => row.primary_reason === "statistically_underpowered_survivor").length;
  return `
    <div style="display:flex;flex-direction:column;gap:14px;padding:14px 16px">
      <div class="kpi-grid">
        ${kpiTile("Gate accepted", `${esc(diag.passed || 0)}/${esc(diag.total || rows.length)}`, "blocking gates clear", "pass")}
        ${kpiTile("Evidence reports", evidence.length, "IC, decay, regime, funding")}
        ${kpiTile("Research gate", `${researchGateCounts.production_passed || 0}/${researchGateCounts.research_survivor || 0}/${researchGateCounts.rejected || 0}`, "production / survivor / rejected")}
        ${kpiTile("Near-miss repairs", nearMisses.filter((row) => row.actionable).length, `${nearMisses.length} analyzed`)}
        ${kpiTile("Research survivors", survivors.length, "optimizer input when GateCheck is zero")}
        ${kpiTile("Survivor store", survivorStore.filter((row) => (row.status || "active") === "active").length, "active longitudinal rechecks")}
        ${kpiTile("Underpowered", underpoweredSurvivors, "NW FDR + low-trade diagnostics")}
        ${kpiTile("Best net SR", fmt.signed((topRows[0] || {}).net_sharpe), `gross ${formatOptionalSigned((topRows[0] || {}).gross_sharpe)}`)}
        ${kpiTile("Median net SR", fmt.signed((summary.net_sharpe || {}).median), `p75 ${fmt.signed((summary.net_sharpe || {}).p75)}`)}
        ${kpiTile("Median cost margin", `${fmt.signed((summary.cost_margin_bps || {}).median)} bps`, "break-even minus required cost")}
      </div>
      <div>
        <div class="label-eyebrow">Blocking failures</div>
        <div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:8px">
          ${topFailures.length ? topFailures.slice(0, 10).map((item) => `<span class="badge fail">${icon("xcircle", 11)}${esc(item.rule_id)} ${esc(item.count)}</span>`).join("") : `<span class="muted">No hard failures recorded.</span>`}
        </div>
      </div>
      <div>
        <div class="label-eyebrow">Diagnostic warnings</div>
        <div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:8px">
          ${topWarnings.length ? topWarnings.slice(0, 10).map((item) => `<span class="badge warn">${icon("warn", 11)}${esc(item.rule_id)} ${esc(item.count)}</span>`).join("") : `<span class="muted">No diagnostic warnings recorded.</span>`}
        </div>
      </div>
      ${simpleTable(topRows.slice(0, 10).map((row) => ({
        candidate_id: shortId(row.candidate_id, 14),
        family: row.hypothesis_family,
        variant: row.search_variant,
        net_sharpe: fmt.signed(row.net_sharpe),
        gross_sharpe: formatOptionalSigned(row.gross_sharpe),
        cost_drag: formatOptionalSigned(row.cost_drag_sharpe),
        cost_margin: `${fmt.signed(row.cost_margin_bps)} bps`,
        turnover: fmt.num(row.factor_turnover, 4),
        blocking_failures: (row.failures || []).join(", "),
        diagnostics: (row.warnings || []).join(", "),
      })), [
        ["candidate_id", "Candidate"],
        ["family", "Family"],
        ["variant", "Variant"],
        ["net_sharpe", "Net SR"],
        ["gross_sharpe", "Gross SR"],
        ["cost_drag", "Cost drag"],
        ["cost_margin", "Cost margin"],
        ["turnover", "Turnover"],
        ["blocking_failures", "Blocking"],
        ["diagnostics", "Diagnostics"],
      ])}
      ${survivors.length ? `
        <div>
          <div class="label-eyebrow">Optimizer survivors</div>
          ${simpleTable(survivors.slice(0, 10).map((row) => ({
            candidate_id: shortId(row.candidate_id, 14),
            family: row.hypothesis_family,
            status: row.status || row.research_gate_status || "survivor",
            score: fmt.signed(row.research_score),
            net_sharpe: fmt.signed(row.sharpe),
            gross_sharpe: formatOptionalSigned(row.gross_sharpe),
            turnover: fmt.num(row.factor_turnover, 4),
            subtype: (row.reasons || []).includes("statistically_underpowered_survivor") ? "underpowered" : "edge",
            reason: row.survivor_reason || "ranked",
          })), [
            ["candidate_id", "Candidate"],
            ["family", "Family"],
            ["status", "Status"],
            ["score", "Score"],
            ["net_sharpe", "Net SR"],
            ["gross_sharpe", "Gross SR"],
            ["turnover", "Turnover"],
            ["subtype", "Subtype"],
            ["reason", "Reason"],
          ])}
        </div>
      ` : ""}
      ${nearMisses.length ? `
        <div>
          <div class="label-eyebrow">Near-miss reasons</div>
          ${simpleTable(nearMisses.slice(0, 10).map((row) => ({
            candidate_id: shortId(row.candidate_id, 14),
            reason: row.primary_reason,
            actionable: row.actionable ? "yes" : "no",
            actions: (row.repair_actions || []).join(", "),
          })), [
            ["candidate_id", "Candidate"],
            ["reason", "Reason"],
            ["actionable", "Actionable"],
            ["actions", "Repair"],
          ])}
        </div>
      ` : ""}
    </div>
  `;
}

function formatOptionalSigned(value, digits = 2) {
  return value == null ? "n/a" : fmt.signed(value, digits);
}

function renderRunLog() {
  const latest = app.data.latest_run;
  const events = app.data.events || [];
  if (!latest) return `<div class="empty">No pipeline runs recorded.</div>`;
  if (!events.length) return `<div class="empty">${esc(shortId(latest.run_id))} has no events yet.</div>`;
  return `
    <div class="log-list scrollbar">
      ${events.slice(-80).reverse().map((event) => `
        <div class="log-row ${esc(event.level)}">
          <span class="muted">${fmt.time(event.created_at)}</span>
          <span class="phase">[${esc(event.phase || "log")}]</span>
          <span>${esc(event.message || "")}</span>
        </div>
      `).join("")}
    </div>
  `;
}

function gateStrip(row) {
  const discovery = Number(row.ic_tstat || row.rank_ic_tstat || 0) > 0 ? "pass" : "fail";
  const repair = row.search_variant && row.search_variant !== "unknown" && !String(row.search_variant).includes("base") ? "cond" : "skip";
  const validation = Number(row.pbo || 0) > 0 && Number(row.pbo || 0) <= 0.6 ? "pass" : Number(row.pbo || 0) ? "cond" : "skip";
  const final = Number(row.sharpe || 0) > 0 ? "pass" : "fail";
  const gate = row.gate === "pass" ? "pass" : row.gate === "warn" ? "cond" : row.gate === "fail" ? "fail" : "skip";
  const research = (row.failures || []).length && Number(row.gross_sharpe || row.sharpe || 0) > 0 ? "surv" : gate;
  const cells = [
    ["D", discovery, "Discovery"],
    ["R", repair, "Repair"],
    ["V", validation, "Validation"],
    ["O", final, "Final OOS"],
    ["G", gate, "GateCheck"],
    ["S", research, "Research"],
  ];
  return `<span class="gate-strip">${cells.map(([label, tone, title]) => `<span class="gate-cell ${tone}" title="${esc(title)}: ${esc(tone)}">${label}</span>`).join("")}</span>`;
}

function renderExperimentTable(rows, compact = false) {
  if (!rows.length) return `<div class="empty">No backtest artifacts yet. Run the pipeline from the CLI or dashboard.</div>`;
  return `
    <div class="table-wrap scrollbar">
      <table class="fm">
        <thead>
          <tr>
            <th>Rank</th>
            <th>Experiment</th>
            <th>Hypothesis</th>
            <th>Journey</th>
            <th class="right">SR</th>
            <th class="right">IC t</th>
            <th class="right">NW FDR-p</th>
            <th class="right">Perm-p</th>
            <th class="right">DSR</th>
            <th class="right">Score</th>
            ${compact ? `<th>Equity</th>` : `<th>Variant</th><th class="right">Gross SR</th><th class="right">Cost margin</th><th class="right">Trades</th>`}
            <th>Gate</th>
          </tr>
        </thead>
        <tbody>
          ${rows.map((row, index) => `
            <tr class="row clickable" data-exp-id="${esc(row.experiment_id)}">
              <td class="mono muted">#${String(index + 1).padStart(2, "0")}</td>
              <td>
                <div class="id">${esc(shortId(row.experiment_id, 18))}</div>
                <div class="id-sub">${esc(row.symbol)} · ${esc(row.market)} · ${esc(row.method)}</div>
              </td>
              <td style="max-width:300px">${esc(row.family)}<div class="muted">${esc(row.hypothesis || "No hypothesis text")}</div></td>
              <td>${gateStrip(row)}</td>
              <td class="right mono ${Number(row.sharpe) >= 0 ? "metric-value pass" : "metric-value fail"}">${fmt.signed(row.sharpe)}</td>
              <td class="right mono">${fmt.num(row.ic_tstat)}</td>
              <td class="right mono">${fmt.num(row.fdr_p, 3)}</td>
              <td class="right mono">${fmt.num(row.perm_p, 3)}</td>
              <td class="right mono">${fmt.signed(row.dsr, 3)}</td>
              <td class="right mono" style="color:var(--fg-0)">${fmt.num(row.hardscore, 1)}</td>
              ${compact ? `<td>${sparkline(syntheticSpark(row), Number(row.sharpe) >= 0 ? "var(--accent-green)" : "var(--accent-red)")}</td>` : `<td>${esc(row.search_variant || "unknown")}</td><td class="right mono">${formatOptionalSigned(row.gross_sharpe)}</td><td class="right mono">${fmt.signed(row.cost_margin_bps)} bps</td><td class="right mono">${fmt.int(row.trades)}</td>`}
              <td>${statusBadge(row.gate)}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>
  `;
}

async function loadArchives() {
  if (app.archives) return;
  try {
    const data = await getJson("/api/archives");
    app.archives = data.archives || [];
  } catch (error) {
    app.archives = [];
  }
}

function renderArchive() {
  const archives = app.archives || [];
  if (!archives.length) {
    return `
      <div class="screen-grid">
        ${panel("Archived experiments", "Permanent archives across runs", `<div class="empty">No archives found in archives/ directory. Run the pipeline with archive_top > 0 to populate.</div>`)}
      </div>
    `;
  }
  const valid = archives.filter(function(a) { return a.integrity === "valid"; }).length;
  const invalid = archives.filter(function(a) { return a.integrity === "invalid" || a.integrity === "corrupt"; }).length;
  return `
    <div class="screen-grid">
      <div class="kpi-grid">
        ${kpiTile("Total archived", archives.length, "across all runs", "info")}
        ${kpiTile("Integrity valid", valid, "checksum matches", "pass")}
        ${kpiTile("Integrity invalid", invalid, "checksum mismatch or corrupt", invalid ? "fail" : "")}
        ${kpiTile("Top score", fmt.num(Math.max.apply(null, archives.map(function(a) { return Number(a.hardscore || 0); })), 1), "best HardScore")}
      </div>
      ${panel("Archive index", `${archives.length} experiments`, `
        <div class="table-wrap scrollbar">
          <table class="fm">
            <thead>
              <tr>
                <th>Experiment</th>
                <th>Archived</th>
                <th class="right">Score</th>
                <th class="right">Sharpe</th>
                <th>Gate</th>
                <th>Tier</th>
                <th>Integrity</th>
              </tr>
            </thead>
            <tbody>
              ${archives.map(function(row) { return `
                <tr class="row clickable" data-exp-id="${esc(row.experiment_id)}">
                  <td>
                    <div class="id">${esc(shortId(row.experiment_id, 24))}</div>
                    <div class="id-sub">${esc(row.git_sha ? shortId(row.git_sha, 12) : "no git sha")}</div>
                  </td>
                  <td class="mono muted">${esc(row.created_at ? row.created_at.slice(0, 19).replace("T", " ") : "n/a")}</td>
                  <td class="right mono" style="color:var(--fg-0)">${fmt.num(row.hardscore, 1)}</td>
                  <td class="right mono ${Number(row.sharpe) >= 0 ? "metric-value pass" : "metric-value fail"}">${fmt.signed(row.sharpe)}</td>
                  <td>${statusBadge(row.gate_passed ? "pass" : "fail")}</td>
                  <td><span class="badge ${row.risk_tier === 'full_pass' ? 'pass' : row.risk_tier === 'conditional_pass' ? 'warn' : ''}">${esc(row.risk_tier || "unknown")}</span></td>
                  <td>${statusBadge(row.integrity === "valid" ? "pass" : row.integrity === "invalid" ? "fail" : "warn", row.integrity)}</td>
                </tr>
              `; }).join("")}
            </tbody>
          </table>
        </div>
      `, true)}
    </div>
  `;
}

function renderExperiments() {
  const rows = filteredExperiments();
  const families = uniq(app.data.experiments.map((row) => row.family));
  const symbols = uniq(app.data.experiments.map((row) => row.symbol));
  return `
    <div class="screen-grid">
      <div class="filters">
        ${selectFilter("family", "Family", families)}
        ${selectFilter("symbol", "Symbol", symbols)}
        ${selectFilter("gate", "Gate", ["pass", "warn", "fail", "missing"])}
        <div class="field">
          <label>Minimum HardScore</label>
          <input data-filter="minScore" type="number" min="0" max="100" step="1" value="${esc(app.filters.minScore)}">
        </div>
      </div>
      ${panel("Experiments", `${rows.length} of ${app.data.experiments.length} rows`, renderExperimentTable(rows), true)}
    </div>
  `;
}

function renderDiagnostics() {
  const diag = gateDiagnostics();
  const rows = diag.rows || [];
  if (!rows.length) {
    return `
      <div class="screen-grid">
        ${panel("Gate diagnostics", "No diagnostic artifact", `<div class="empty">Run the pipeline again to generate GateCheck diagnostics for the current code path.</div>`)}
      </div>
    `;
  }
  const summary = diag.metric_summary || {};
  return `
    <div class="screen-grid">
      ${panel("Gate diagnostics", gateDiagnosticsSubtitle(), renderGateDiagnostics(), true)}
      <div class="two-col">
        ${panel("Metric summary", "Distribution across diagnostic rows", simpleTable([
          summaryRow("Net Sharpe", summary.net_sharpe),
          summaryRow("Gross Sharpe", summary.gross_sharpe),
          summaryRow("Cost drag", summary.cost_drag_sharpe),
          summaryRow("Turnover", summary.factor_turnover),
          summaryRow("Break-even bps", summary.break_even_cost_bps),
          summaryRow("Actual cost bps", summary.actual_cost_bps),
          summaryRow("Cost margin bps", summary.cost_margin_bps),
          summaryRow("OOS trades", summary.oos_trade_count),
        ], [
          ["metric", "Metric"],
          ["median", "Median"],
          ["p25", "P25"],
          ["p75", "P75"],
          ["min", "Min"],
          ["max", "Max"],
          ["count", "Count"],
        ]), true)}
        ${panel("Top cost margin", "Closest to clearing cost gate", simpleTable((diag.top_by_cost_margin || []).slice(0, 12).map((row) => ({
          candidate_id: shortId(row.candidate_id, 14),
          family: row.hypothesis_family,
          variant: row.search_variant,
          net_sharpe: fmt.signed(row.net_sharpe),
          gross_sharpe: formatOptionalSigned(row.gross_sharpe),
          cost_margin: `${fmt.signed(row.cost_margin_bps)} bps`,
          be_cost: `${fmt.num(row.break_even_cost_bps, 2)} bps`,
          actual_cost: `${fmt.num(row.actual_cost_bps, 2)} bps`,
          blocking_failures: (row.failures || []).join(", "),
          diagnostics: (row.warnings || []).join(", "),
        })), [
          ["candidate_id", "Candidate"],
          ["family", "Family"],
          ["variant", "Variant"],
          ["net_sharpe", "Net SR"],
          ["gross_sharpe", "Gross SR"],
          ["cost_margin", "Cost margin"],
          ["be_cost", "Break-even"],
          ["actual_cost", "Actual cost"],
          ["blocking_failures", "Blocking"],
          ["diagnostics", "Diagnostics"],
        ]), true)}
      </div>
    </div>
  `;
}

function summaryRow(metric, values = {}) {
  return {
    metric,
    median: fmt.num(values.median),
    p25: fmt.num(values.p25),
    p75: fmt.num(values.p75),
    min: fmt.num(values.min),
    max: fmt.num(values.max),
    count: values.count || 0,
  };
}

function selectFilter(key, label, options) {
  return `
    <div class="field">
      <label>${esc(label)}</label>
      <select data-filter="${esc(key)}">
        <option value="all">All</option>
        ${options.map((option) => `<option value="${esc(option)}" ${app.filters[key] === option ? "selected" : ""}>${esc(option)}</option>`).join("")}
      </select>
    </div>
  `;
}

function filteredExperiments() {
  return (app.data.experiments || []).filter((row) => {
    if (app.filters.family !== "all" && row.family !== app.filters.family) return false;
    if (app.filters.symbol !== "all" && row.symbol !== app.filters.symbol) return false;
    if (app.filters.gate !== "all" && row.gate !== app.filters.gate) return false;
    return Number(row.hardscore || 0) >= Number(app.filters.minScore || 0);
  });
}

function renderHypotheses() {
  const rows = app.data.bundle.hypotheses || [];
  if (!rows.length) return `<div class="empty">No hypothesis artifact yet.</div>`;
  return `
    <div class="cards">
      ${rows.map((hyp) => `
        <article class="hyp-card">
          <div>
            <span class="chip">${esc(hyp.hypothesis_family || "unknown")}</span>
            <span class="mono muted">${esc(hyp.hypothesis_id || "")}</span>
          </div>
          <div class="hyp-title">${esc(hyp.title || hyp.hypothesis_id || "Hypothesis")}</div>
          <div class="hyp-body">${esc(hyp.economic_mechanism || "")}</div>
          <div class="three-col">
            ${miniFact("Prediction", hyp.testable_prediction)}
            ${miniFact("Null", hyp.null_hypothesis)}
            ${miniFact("Expected IC", `${JSON.stringify(hyp.expected_ic_range || [])} / ${hyp.expected_decay_halflife_bars || "n/a"} bars`)}
          </div>
        </article>
      `).join("")}
    </div>
  `;
}

function miniFact(label, value) {
  return `<div><div class="label-eyebrow">${esc(label)}</div><div class="muted">${esc(value || "n/a")}</div></div>`;
}

function renderCandidates() {
  const rows = app.data.bundle.candidates || [];
  if (!rows.length) return `<div class="empty">No candidates yet.</div>`;
  return panel("Candidate parameter specs", `${rows.length} generated rows`, simpleTable(rows, [
    ["candidate_id", "Candidate"],
    ["hypothesis_id", "Hypothesis"],
    ["hypothesis_family", "Family"],
    ["method_id", "Method"],
    ["symbol", "Symbol"],
    ["market", "Market"],
    ["interval", "Interval"],
    ["params", "Params"],
  ]), true);
}

function renderMethods() {
  const methods = app.data.methods || [];
  const counts = methods.reduce((acc, method) => {
    acc[method.status] = (acc[method.status] || 0) + 1;
    return acc;
  }, {});
  return `
    <div class="screen-grid">
      <div class="kpi-grid">
        ${kpiTile("Implemented", counts.implemented || 0, "usable methods", "pass")}
        ${kpiTile("Planned", counts.planned || 0, "registry roadmap", "info")}
        ${kpiTile("Blocked v1", counts.blocked_v1 || 0, "needs more data", "warn")}
        ${kpiTile("Schedulable", app.data.schedulable_method_count, "current universe", "pass")}
      </div>
      ${panel("Method registry", `${methods.length} methods`, simpleTable(methods, [
        ["method_id", "Method ID"],
        ["hypothesis_family", "Family"],
        ["display_name", "Display name"],
        ["status", "Status"],
        ["v1_schedulable", "Schedulable"],
        ["blocked_reason", "Blocked reason"],
      ]), true)}
    </div>
  `;
}

function renderLedger() {
  return `
    <div class="screen-grid">
      ${panel("Trial counts", `${app.data.settings.trial_window_days} day rolling window`, simpleTable(app.data.ledger, [
        ["family", "Family"],
        ["family_trials", "Family trials"],
        ["rolling_trials", "Rolling trials"],
        ["global_trials", "Global trials"],
        ["effective_trials", "Effective trials"],
      ]), true)}
      ${panel("Recent trials", `${app.data.recent_trials.length} rows`, simpleTable(app.data.recent_trials, [
        ["trial_id", "Trial ID"],
        ["candidate_id", "Candidate"],
        ["experiment_id", "Experiment"],
        ["hypothesis_family", "Family"],
        ["method_id", "Method"],
        ["evaluated_at", "Evaluated at"],
      ]), true)}
    </div>
  `;
}

function renderData() {
  const settings = app.data.settings;
  return `
    <div class="screen-grid">
      <div class="three-col">
        ${kpiTile("Symbols", (settings.symbols || []).join(", "), "configured universe")}
        ${kpiTile("Markets", (settings.markets || []).join(", "), "local archives")}
        ${kpiTile("Interval", settings.default_interval, "default bar size")}
      </div>
      ${panel("Data coverage", `${app.data.coverage.length} records`, simpleTable(app.data.coverage, [
        ["coverage_id", "Coverage ID"],
        ["symbol", "Symbol"],
        ["market", "Market"],
        ["dataset", "Dataset"],
        ["interval", "Interval"],
        ["start", "Start"],
        ["end", "End"],
        ["created_at", "Created at"],
      ]), true)}
      ${panel("Pipeline runs", `${app.data.runs.length} records`, simpleTable(app.data.runs, [
        ["run_id", "Run ID"],
        ["status", "Status"],
        ["started_at", "Started"],
        ["ended_at", "Ended"],
        ["args", "Args"],
        ["error", "Error"],
      ]), true)}
      ${panel("Paths", "Local-first storage", `<pre>${esc(JSON.stringify({ sqlite_path: settings.sqlite_path, parquet_dir: settings.parquet_dir }, null, 2))}</pre>`)}
    </div>
  `;
}

function simpleTable(rows, columns) {
  if (!rows || !rows.length) return `<div class="empty">No rows.</div>`;
  return `
    <div class="table-wrap scrollbar">
      <table class="fm">
        <thead><tr>${columns.map(([, label]) => `<th>${esc(label)}</th>`).join("")}</tr></thead>
        <tbody>
          ${rows.map((row) => `
            <tr>${columns.map(([key]) => `<td>${cell(row[key])}</td>`).join("")}</tr>
          `).join("")}
        </tbody>
      </table>
    </div>
  `;
}

function cell(value) {
  if (value == null || value === "") return `<span class="muted">n/a</span>`;
  if (typeof value === "boolean") return statusBadge(value ? "pass" : "missing", value ? "yes" : "no");
  if (typeof value === "object") return `<span class="mono">${esc(JSON.stringify(value))}</span>`;
  return esc(value);
}

function renderFamilyActivity() {
  const ledger = app.data.ledger || [];
  const maxTrials = Math.max(1, ...ledger.map((row) => Number(row.family_trials || 0)));
  return `
    <div style="display:flex;flex-direction:column;gap:10px">
      ${ledger.map((row) => {
        const pass = app.data.experiments.filter((exp) => exp.family === row.family && exp.gate === "pass").length;
        const inRun = app.data.experiments.filter((exp) => exp.family === row.family).length;
        const pct = Math.max(2, Math.min(100, Number(row.family_trials || 0) / maxTrials * 100));
        return `
          <div style="display:grid;grid-template-columns:150px minmax(0,1fr) 78px;gap:12px;align-items:center">
            <div><div class="mono" style="color:var(--fg-0)">${esc(row.family)}</div><div class="muted">${fmt.int(row.rolling_trials)} rolling</div></div>
            <div style="height:18px;border-radius:999px;background:var(--bg-2);overflow:hidden"><div style="width:${pct}%;height:100%;background:color-mix(in oklch,var(--accent-blue) 65%,transparent)"></div></div>
            <div class="mono right"><span style="color:var(--accent-green)">${pass}</span><span class="muted">/${inRun}</span></div>
          </div>
        `;
      }).join("")}
    </div>
  `;
}

function renderOptimizationHistory() {
  const history = app.data.bundle.optimization_history || [];
  if (!history.length) return `<div class="empty">No optimization history artifact yet.</div>`;
  return `
    <div>
      ${history.map((item, index) => `
        <div style="padding:12px 16px;border-bottom:1px solid var(--border-1)">
          <div style="display:flex;justify-content:space-between;gap:8px">
            <span class="badge info">${icon("refresh", 11)}ROUND ${esc(item.round || index + 1)}</span>
            <span class="mono muted">${fmt.num(item.elapsed_s || item.duration_s || 0, 1)}s</span>
          </div>
          <div class="mono" style="display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-top:8px">
            <span><span class="muted">cands </span>${esc(item.num_candidates || 0)}</span>
            <span><span class="muted">tests </span>${esc(item.num_backtests || 0)}</span>
            <span><span class="muted">passed </span><span style="color:var(--accent-green)">${esc(item.num_gatecheck_passed || 0)}</span></span>
            <span><span class="muted">survive </span>${esc(item.num_research_survivors || 0)}</span>
            <span><span class="muted">new </span>${esc(item.new_candidates_count || 0)}</span>
          </div>
          <div class="muted" style="margin-top:8px">${esc(item.summary || item.action || "")}</div>
        </div>
      `).join("")}
    </div>
  `;
}

function renderDrawer() {
  const row = (app.detail && app.detail.row) || app.data.experiments.find((item) => item.experiment_id === app.selectedId) || {};
  if (app.loadingDetail) {
    return `<div class="drawer-backdrop" data-action="close-detail"></div><aside class="drawer"><div class="drawer-body"><div class="drawer-loading"><span class="spinner"></span><span>Loading experiment detail...</span></div></div></aside>`;
  }
  const detail = app.detail && app.detail.detail;
  const hyp = app.detail && app.detail.hypothesis ? app.detail.hypothesis : {};
  return `
    <div class="drawer-backdrop" data-action="close-detail"></div>
    <aside class="drawer">
      <header class="drawer-header">
        <div class="drawer-title-row">
          ${statusBadge(row.gate)}
          <span class="badge info">${icon("flask", 11)}${esc(row.family || "unknown")}</span>
          <span style="flex:1"></span>
          <button class="icon-btn" data-action="close-detail" title="Close">${icon("close", 16)}</button>
        </div>
        <div>
          <h2 class="drawer-title">${esc(row.experiment_id || app.selectedId)}</h2>
          <div class="muted">${esc(row.symbol)} / ${esc(row.market)} / ${esc(row.interval)} / ${esc(row.method)}</div>
        </div>
      </header>
      <div class="drawer-body scrollbar">
        <section class="drawer-section">
          <div class="label-eyebrow">Hypothesis</div>
          <div class="hyp-title">${esc(hyp.hypothesis_id || row.hypothesis_id || "n/a")}</div>
          <div class="hyp-body">${esc(hyp.economic_mechanism || row.hypothesis || "No hypothesis artifact linked.")}</div>
        </section>
        <section class="drawer-section">
          <div class="label-eyebrow">Performance</div>
          <div class="metric-grid">
            ${metric("Sharpe", fmt.signed(row.sharpe), Number(row.sharpe) >= 0 ? "pass" : "fail")}
            ${metric("Gross SR", formatOptionalSigned(row.gross_sharpe), row.gross_sharpe != null && Number(row.gross_sharpe) >= 0 ? "pass" : "")}
            ${metric("Cost drag", formatOptionalSigned(row.cost_drag_sharpe))}
            ${metric("HardScore", fmt.num(row.hardscore, 1), "info")}
            ${metric("DSR", fmt.signed(row.dsr, 3), Number(row.dsr) > 0 ? "pass" : "warn")}
            ${metric("NW FDR p", fmt.num(row.fdr_p, 4), Number(row.fdr_p) <= 0.05 ? "pass" : "warn")}
            ${metric("Perm p", fmt.num(row.perm_p, 4))}
            ${metric("Variant", row.search_variant || "unknown")}
            ${metric("Cost margin", `${fmt.signed(row.cost_margin_bps)} bps`, Number(row.cost_margin_bps) > 0 ? "pass" : "fail")}
            ${metric("Trades", fmt.int(row.trades))}
            ${metric("Capacity", `$${fmt.compact(row.capacity_usd)}`)}
            ${metric("Break-even", `${fmt.num(row.break_even_cost_bps, 1)} bps`)}
          </div>
        </section>
        ${detail ? renderDetailPayload(detail, row) : `<div class="empty">This experiment has no persisted chart detail yet. Run a fresh pipeline to populate K-line, curves, trades, and GateCheck rows.</div>`}
        <section class="drawer-section">
          <div class="label-eyebrow">Candidate params</div>
          <pre>${esc(JSON.stringify(row.params || {}, null, 2))}</pre>
        </section>
      </div>
    </aside>
  `;
}

function renderDetailPayload(detail) {
  const gatecheck = detail.gatecheck || {};
  const gateItems = (gatecheck.items || []).map(normalizeGateItem);
  const tierReasons = gatecheck.tier_reasons || [];
  const regimes = ((detail.summary || {}).regime_conditional_metrics) || {};
  return `
    <section class="drawer-section">
      <div class="label-eyebrow">K-line with buy/sell markers</div>
      <div class="chart-box">${candleChart(detail)}</div>
    </section>
    <section class="drawer-section">
      <div class="label-eyebrow">Backtest curves</div>
      <div class="chart-box">${lineChart(detail.series || [], "equity", "var(--accent-green)", 120)}</div>
      <div class="chart-box">${lineChart(detail.series || [], "drawdown", "var(--accent-red)", 84)}</div>
    </section>
    <section class="drawer-section">
      <div class="label-eyebrow">Trade events</div>
      ${simpleTable((detail.trades || []).slice(-40), [
        ["open_time", "Open time"],
        ["side", "Side"],
        ["price", "Price"],
        ["delta", "Delta"],
        ["position", "Position"],
        ["signal", "Signal"],
        ["equity", "Equity"],
      ])}
    </section>
    <section class="drawer-section">
      <div class="label-eyebrow">GateCheck items</div>
      ${tierReasons.length ? `
        <div class="muted" style="margin-bottom:10px">
          <span class="mono">${esc(gatecheck.risk_tier || "unclassified")}</span>
          ${tierReasons.map((reason) => `<span class="badge info" style="margin-left:6px">${esc(reason)}</span>`).join("")}
        </div>
      ` : ""}
      ${simpleTable(gateItems, [
        ["rule_id", "Rule"],
        ["status", "Status"],
        ["observed", "Observed"],
        ["threshold", "Threshold"],
        ["message", "Message"],
      ])}
    </section>
    <section class="drawer-section">
      <div class="label-eyebrow">Regime metrics</div>
      ${simpleTable(Object.entries(regimes).map(([regime, values]) => ({ regime, ...values })), [
        ["regime", "Regime"],
        ["sharpe", "Sharpe"],
        ["total_return", "Return"],
        ["max_drawdown", "Max DD"],
        ["trade_count", "Trades"],
      ])}
    </section>
    ${exitSection(detail)}
  `;
}

function normalizeGateItem(item) {
  const observed = item.observed == null ? item.value : item.observed;
  return {
    ...item,
    observed: formatGateScalar(observed),
    threshold: formatGateScalar(item.threshold),
  };
}

function formatGateScalar(value) {
  if (value == null || value === "") return value;
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return null;
    const abs = Math.abs(value);
    if (abs !== 0 && abs < 0.0001) return value.toExponential(3);
    if (abs >= 1000) return fmt.compact(value);
    return Number(value.toFixed(6));
  }
  return value;
}

function exitSection(detail) {
  const row = detail.row || {};
  const exit = row.exit || {};
  const keys = Object.keys(exit);
  if (!keys.length) return "";
  const rows = [{
    param: "stop_loss_pct",
    value: exit.stop_loss_pct != null ? fmt.pct(exit.stop_loss_pct) : "off",
    desc: "Hard stop-loss"
  }, {
    param: "max_hold_bars",
    value: exit.max_hold_bars ? fmt.int(exit.max_hold_bars) + " bars" : "off",
    desc: "Max hold duration"
  }, {
    param: "tp_tiers",
    value: (exit.tp_tiers || []).length
      ? (exit.tp_tiers || []).map(function(t) { return fmt.pct(t[0]) + " → close " + fmt.pct(t[1] || 0); }).join(", ")
      : "off",
    desc: "Batch take-profit"
  }, {
    param: "trailing_stop_pct",
    value: exit.trailing_stop_pct ? fmt.pct(exit.trailing_stop_pct) : "off",
    desc: exit.trailing_after_first_tp ? "Trailing stop (after first TP)" : "Trailing stop (from entry)"
  }];
  return `
    <section class="drawer-section">
      <div class="label-eyebrow">Exit rules</div>
      ${simpleTable(rows, [
        ["param", ""],
        ["value", "Value"],
        ["desc", "Description"],
      ])}
    </section>
  `;
}

function metric(label, value, tone = "") {
  return `<div class="metric-tile"><div class="label-eyebrow">${esc(label)}</div><div class="metric-value ${esc(tone)}">${esc(value)}</div></div>`;
}

function statusBadge(status, label) {
  const map = {
    pass: ["pass", "PASS", "check"],
    warn: ["warn", "WARN", "warn"],
    fail: ["fail", "FAIL", "xcircle"],
    missing: ["", "MISSING", "warn"],
    running: ["info", "RUNNING", "activity"],
  };
  const [klass, defaultLabel, iconName] = map[status] || ["", String(status || "n/a").toUpperCase(), "warn"];
  return `<span class="badge ${klass}">${icon(iconName, 11)}${esc(label || defaultLabel)}</span>`;
}

function sparkline(values, color) {
  if (!values.length) return "";
  const width = 82;
  const height = 22;
  const points = scaledPoints(values, width, height, 1);
  return `<svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" aria-hidden="true"><path d="${points.path}" fill="none" stroke="${color}" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
}

function syntheticSpark(row) {
  const seed = Number(row.hardscore || 1) * 7.13 + Number(row.sharpe || 0) * 11;
  const values = [];
  let acc = 100;
  for (let i = 0; i < 60; i += 1) {
    const noise = Math.sin(seed + i * 0.4) * 0.7 + Math.sin(seed * 1.3 + i * 0.09) * 1.4;
    acc *= 1 + Number(row.ann_return || 0) / 1800 + noise * 0.0012;
    values.push(acc);
  }
  return values;
}

function lineChart(rows, key, color, height = 130) {
  const values = rows.map((row) => Number(row[key])).filter(Number.isFinite);
  if (values.length < 2) return `<div class="empty">No ${esc(key)} series.</div>`;
  const width = 520;
  const points = scaledPoints(values, width, height, 12);
  return `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(key)} chart">
      <path d="M12 ${height - 12} H${width - 12}" stroke="var(--border-2)" stroke-width="1"/>
      <path d="${points.path}" fill="none" stroke="${color}" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
  `;
}

function scaledPoints(values, width, height, pad) {
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const step = (width - pad * 2) / Math.max(values.length - 1, 1);
  const coords = values.map((value, index) => {
    const x = pad + index * step;
    const y = height - pad - ((value - min) / range) * (height - pad * 2);
    return [x, y];
  });
  return { coords, path: `M${coords.map(([x, y]) => `${x.toFixed(2)},${y.toFixed(2)}`).join(" L")}` };
}

function candleChart(detail) {
  const rows = (detail.ohlcv || []).slice(-140);
  if (!rows.length) return `<div class="empty">No OHLCV rows.</div>`;
  const width = 520;
  const height = 220;
  const pad = 18;
  const lows = rows.map((row) => Number(row.low)).filter(Number.isFinite);
  const highs = rows.map((row) => Number(row.high)).filter(Number.isFinite);
  const min = Math.min(...lows);
  const max = Math.max(...highs);
  const range = max - min || 1;
  const step = (width - pad * 2) / rows.length;
  const y = (value) => height - pad - ((Number(value) - min) / range) * (height - pad * 2);
  const timeMin = Number(rows[0].open_time);
  const timeMax = Number(rows[rows.length - 1].open_time);
  const xForTime = (time) => pad + ((Number(time) - timeMin) / Math.max(timeMax - timeMin, 1)) * (width - pad * 2);
  const candles = rows.map((row, index) => {
    const x = pad + index * step + step / 2;
    const open = y(row.open);
    const close = y(row.close);
    const high = y(row.high);
    const low = y(row.low);
    const up = Number(row.close) >= Number(row.open);
    const color = up ? "var(--accent-green)" : "var(--accent-red)";
    const rectY = Math.min(open, close);
    const rectH = Math.max(1, Math.abs(close - open));
    return `<line x1="${x.toFixed(2)}" y1="${high.toFixed(2)}" x2="${x.toFixed(2)}" y2="${low.toFixed(2)}" stroke="${color}" stroke-width="1"/><rect x="${(x - Math.max(2, step * 0.32)).toFixed(2)}" y="${rectY.toFixed(2)}" width="${Math.max(2, step * 0.64).toFixed(2)}" height="${rectH.toFixed(2)}" fill="${color}" opacity=".75"/>`;
  }).join("");
  const markers = (detail.trades || []).map((trade) => {
    const x = xForTime(trade.open_time);
    if (x < pad || x > width - pad) return "";
    const yy = y(trade.price);
    const buy = trade.side === "buy";
    const color = buy ? "var(--accent-green)" : "var(--accent-red)";
    const points = buy
      ? `${x},${yy - 8} ${x - 5},${yy + 2} ${x + 5},${yy + 2}`
      : `${x},${yy + 8} ${x - 5},${yy - 2} ${x + 5},${yy - 2}`;
    return `<polygon points="${points}" fill="${color}" stroke="var(--bg-0)" stroke-width="1"/>`;
  }).join("");
  return `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="K-line chart">
      <rect x="0" y="0" width="${width}" height="${height}" fill="transparent"/>
      <path d="M${pad} ${height - pad} H${width - pad}" stroke="var(--border-2)" stroke-width="1"/>
      ${candles}
      ${markers}
    </svg>
  `;
}

function uniq(values) {
  return [...new Set(values.filter(Boolean))].sort();
}

function shortId(value, len = 12) {
  const text = String(value || "");
  return text.length > len ? text.slice(0, len) : text;
}

async function openDetail(experimentId) {
  app.selectedId = experimentId;
  app.loadingDetail = true;
  app.detail = null;
  render();
  try {
    app.detail = await getJson(`/api/detail?id=${encodeURIComponent(experimentId)}`);
  } catch (error) {
    setToast(error.message);
  } finally {
    app.loadingDetail = false;
    render();
  }
}

async function postAction(path) {
  try {
    const result = await getJson(path, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(app.runArgs),
    });
    setToast(result.message || "Done.");
    await loadState();
  } catch (error) {
    setToast(error.message);
  }
}

document.addEventListener("click", (event) => {
  const tab = event.target.closest("[data-tab]");
  if (tab) {
    saveScroll();
    app.activeTab = tab.dataset.tab;
    render();
    if (app.activeTab === "archive") {
      loadArchives().then(render);
    }
    return;
  }
  const row = event.target.closest("[data-exp-id]");
  if (row) {
    openDetail(row.dataset.expId);
    return;
  }
  const action = event.target.closest("[data-action]");
  if (!action) return;
  const name = action.dataset.action;
  if (name === "close-detail") {
    app.selectedId = null;
    app.detail = null;
    render();
  } else if (name === "refresh") {
    loadState();
  } else if (name === "run-start") {
    postAction("/api/run");
  } else if (name === "hosted-start") {
    postAction("/api/hosted");
  } else if (name === "stop-run") {
    postAction("/api/stop");
  }
});

document.addEventListener("change", (event) => {
  const filter = event.target.closest("[data-filter]");
  if (filter) {
    app.filters[filter.dataset.filter] = filter.type === "number" ? Number(filter.value || 0) : filter.value;
    render();
    return;
  }
  const field = event.target.closest("[data-run-field]");
  if (field) updateRunField(field);
});

document.addEventListener("input", (event) => {
  const field = event.target.closest("[data-run-field]");
  if (field) updateRunField(field);
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && app.selectedId) {
    app.selectedId = null;
    app.detail = null;
    render();
  }
});

function updateRunField(field) {
  const key = field.dataset.runField;
  if (field.type === "checkbox") {
    app.runArgs[key] = field.checked;
  } else if (field.type === "number") {
    app.runArgs[key] = Number(field.value || 0);
  } else {
    app.runArgs[key] = field.value;
  }
}

loadState();
setInterval(() => {
  if (app.data && app.data.active_run) loadState();
}, 3500);

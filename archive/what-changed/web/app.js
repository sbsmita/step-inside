const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];

function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v);
  }
  for (const c of children.flat()) if (c != null) el.append(c);
  return el;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const store = {
  get(k, d) { try { return localStorage.getItem(k) ?? d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
};

// ---------- voice, spatial ping, haptics ----------
const voice = {
  muted: false,
  last: "",
  speak(text, { hazard = false } = {}) {
    const box = $("#say");
    box.textContent = text;
    box.classList.toggle("hazard", hazard);
    this.last = text;
    if (this.muted || !("speechSynthesis" in window)) return;
    speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 1.0;
    speechSynthesis.speak(u);
  },
};

let audioCtx;
function ping(bearingDeg, hazard = false) {
  if (voice.muted) return;
  try {
    audioCtx ??= new AudioContext();
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    const panner = new PannerNode(audioCtx, { panningModel: "HRTF", distanceModel: "inverse" });
    const rad = (bearingDeg * Math.PI) / 180;
    panner.positionX.value = Math.sin(rad) * 2;
    panner.positionZ.value = -Math.cos(rad) * 2;
    osc.frequency.value = hazard ? 880 : 620;
    gain.gain.setValueAtTime(0.0001, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.4, audioCtx.currentTime + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + 0.35);
    osc.connect(gain).connect(panner).connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + 0.4);
  } catch { /* audio unavailable */ }
}
const buzz = (p) => navigator.vibrate?.(p);

// ---------- API ----------
async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail ?? detail; } catch { /* not json */ }
    const err = new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    err.detail = detail;
    err.status = r.status;
    throw err;
  }
  return r.json();
}

function postForm(path, fields) {
  const fd = new FormData();
  for (const [k, v] of Object.entries(fields)) {
    if (v instanceof Blob) fd.append(k, v, `${k}.jpg`);
    else if (v != null) fd.append(k, String(v));
  }
  return api(path, { method: "POST", body: fd });
}

const fetchBlob = async (url) => (await fetch(url)).blob();

// ---------- drawing ----------
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

function colorFor(item) {
  if (item.verdict?.verdict === "hazard") return css("--hazard");
  if (item.kind === "moved") return css("--moved");
  return css("--useful");
}

function polyPath(ctx, pts, w, h) {
  ctx.beginPath();
  pts.forEach(([x, y], i) => (i ? ctx.lineTo(x * w, y * h) : ctx.moveTo(x * w, y * h)));
  ctx.closePath();
}

function drawItems(ctx, items, w, h) {
  ctx.lineWidth = Math.max(3, w / 250);
  ctx.font = `600 ${Math.max(14, w / 45)}px system-ui, sans-serif`;
  for (const it of items) {
    const color = colorFor(it);
    ctx.strokeStyle = color;
    polyPath(ctx, it.box_current, w, h);
    ctx.stroke();
    if (it.from_box_current) {
      ctx.setLineDash([10, 8]);
      polyPath(ctx, it.from_box_current, w, h);
      ctx.stroke();
      ctx.setLineDash([]);
      const c = (pts) => pts.reduce(([a, b], [x, y]) => [a + x / 4, b + y / 4], [0, 0]);
      const [fx, fy] = c(it.from_box_current);
      const [tx, ty] = c(it.box_current);
      arrow(ctx, fx * w, fy * h, tx * w, ty * h, color);
    }
    const [lx, ly] = it.box_current[0];
    const text = it.verdict?.label ?? it.kind;
    const tw = ctx.measureText(text).width + 12;
    ctx.fillStyle = color;
    ctx.fillRect(lx * w, Math.max(0, ly * h - 28), tw, 26);
    ctx.fillStyle = "#fff";
    ctx.fillText(text, lx * w + 6, Math.max(19, ly * h - 9));
  }
}

function arrow(ctx, x0, y0, x1, y1, color) {
  const a = Math.atan2(y1 - y0, x1 - x0);
  ctx.strokeStyle = ctx.fillStyle = color;
  ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x1 - 18 * Math.cos(a - 0.4), y1 - 18 * Math.sin(a - 0.4));
  ctx.lineTo(x1 - 18 * Math.cos(a + 0.4), y1 - 18 * Math.sin(a + 0.4));
  ctx.closePath(); ctx.fill();
}

async function drawImageResult(canvas, blob, items) {
  const bmp = await createImageBitmap(blob);
  canvas.width = bmp.width;
  canvas.height = bmp.height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(bmp, 0, 0);
  drawItems(ctx, items, bmp.width, bmp.height);
}

// ---------- result panels ----------
function speakAnalysis(analysis) {
  const hazard = analysis.items?.some((i) => i.verdict.verdict === "hazard");
  voice.speak(analysis.say, { hazard });
  if (hazard) buzz([120, 80, 120]);
  (analysis.items ?? []).slice(0, 3).forEach((it, i) =>
    setTimeout(() => ping(it.bearing_deg, it.verdict.verdict === "hazard"), i * 450));
}

function renderTrace(ol, analysis, scan) {
  ol.replaceChildren();
  const vision = scan
    ? `${scan.status === "ok" ? `matched "${scan.keyframe ?? "baseline"}"` : "view not recognized"}, ` +
      `${scan.alignment.inliers} inlier matches, ${scan.changes.length} change regions, ${scan.timing_ms} ms`
    : "";
  if (scan) ol.append(h("li", {}, h("span", { class: "who" }, "OpenCV 5 "), vision));
  for (const t of analysis.trace ?? []) {
    let text = "";
    if (t.agent === "scout") text = t.action === "rescan" ? `asked for a better view (${t.reason})` : "frame is usable";
    else if (t.agent === "judge") text = `${t.verdicts.length} verdicts: ` +
      t.verdicts.map((v) => `${v.label} → ${v.verdict}`).join("; ");
    else if (t.agent === "narrator") text = "composed speech";
    else if (t.agent === "escalator") text = t.sent ? "alerted helper by SMS/SNS" : "would alert helper (no topic set)";
    else if (t.agent === "planner") text = "wrote fix-it steps";
    else if (t.agent === "verifier") text = `${t.from} → ${t.to}`;
    ol.append(h("li", {}, h("span", { class: "who" }, `${t.agent[0].toUpperCase()}${t.agent.slice(1)} `), text,
      t.source ? h("span", { class: "src" }, ` (${t.source})`) : null));
  }
}

function renderItems(ul, analysis, home, onFix) {
  ul.replaceChildren();
  const items = analysis.items ?? [];
  if (!items.length) {
    ul.append(h("li", { class: "empty" }, analysis.action === "rescan" ? "Waiting for a better view." : "No changes."));
    return;
  }
  for (const it of items) {
    const v = it.verdict;
    const cls = v.verdict === "hazard" ? "hazard" : it.kind === "moved" ? "moved" : "";
    const where = it.kind === "moved" ? `now ${it.direction}; was ${it.from_direction}` : `${it.direction}, ${it.height}`;
    const li = h("li", { class: cls },
      h("div", { class: "label" }, v.label),
      h("div", { class: "meta" }, `${v.verdict} · ${where}`),
      h("div", { class: "meta" }, v.reason));
    if (onFix && ["rearrange", "spill"].includes(v.fix)) {
      li.append(h("button", { onclick: () => onFix(it, li) }, "Guide me to fix it"));
    }
    ul.append(li);
  }
}

function fixStates(li, change) {
  const names = change.verdict.fix === "spill" ? ["clean", "done"]
    : change.kind === "moved" ? ["pickup", "place", "done"] : ["pickup", "done"];
  const box = h("div", { class: "states" }, names.map((n) => h("span", { "data-s": n }, n)));
  li.querySelector(".states")?.remove();
  li.append(box);
  return (state) => {
    const idx = names.indexOf(state);
    $$("span", box).forEach((s, i) => { s.className = i < idx ? "past" : i === idx ? "on" : ""; });
  };
}

async function startFix(home, change, li) {
  const start = await api(`/api/homes/${home}/fix`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ change_id: change.id }),
  });
  if (!start.allowed) { voice.speak(start.say, { hazard: true }); return null; }
  return { session: start.session, show: fixStates(li, change) };
}

function handleFixStep(res, show, li) {
  show(res.state);
  if (res.speak) voice.speak(res.say);
  if (res.done) { li.className = "done"; buzz([60, 40, 60, 40, 200]); }
}

// ---------- demo ----------
async function initDemo() {
  const index = await (await fetch("demo/index.json")).json();
  const cards = $("#scenarios");
  for (const s of index.scenarios) {
    const card = h("button", { class: "card", "aria-pressed": "false", onclick: () => runScenario(s, card) },
      h("img", { src: s.image, alt: "" }), h("span", {}, s.title));
    cards.append(card);
  }

  async function runScenario(s, card) {
    $$(".card", cards).forEach((c) => c.setAttribute("aria-pressed", String(c === card)));
    voice.speak("Looking…");
    const blob = await fetchBlob(s.image);
    const { scan, analysis } = await postForm(`/api/homes/${index.home}/scan`, { image: blob });
    await drawImageResult($("#demo-canvas"), blob, analysis.items ?? []);
    renderItems($("#demo-items"), analysis, index.home, s.name === index.fix.scenario ? demoFix : null);
    renderTrace($("#demo-trace"), analysis, scan);
    speakAnalysis(analysis);
  }

  async function demoFix(change, li) {
    const fix = await startFix(index.home, change, li);
    if (!fix) return;
    voice.speak("Okay, I'll guide you. I'll watch through the camera and check each step.");
    await sleep(2200);
    for (const frame of index.fix.frames) {
      const blob = await fetchBlob(frame);
      const res = await postForm(`/api/fix/${fix.session}/frame`, { image: blob });
      const shown = res.done ? [] : [{ ...change, box_current: change.box_current }];
      await drawImageResult($("#demo-canvas"), blob, shown);
      handleFixStep(res, fix.show, li);
      if (res.trace) renderTrace($("#demo-trace"), { trace: res.trace });
      await sleep(3200);
    }
  }
}

// ---------- camera ----------
async function startCamera(video) {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 960 } }, audio: false,
  });
  video.srcObject = stream;
  await video.play();
  return stream;
}

function grab(video) {
  const c = document.createElement("canvas");
  c.width = video.videoWidth;
  c.height = video.videoHeight;
  c.getContext("2d").drawImage(video, 0, 0);
  return new Promise((r) => c.toBlob(r, "image/jpeg", 0.85));
}

function drawOverlay(canvas, video, items) {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawItems(ctx, items, canvas.width, canvas.height);
}

function homeInput(el) {
  el.value = store.get("home", `home-${Math.random().toString(36).slice(2, 8)}`);
  store.set("home", el.value);
  el.addEventListener("change", () => store.set("home", el.value.trim()));
  return () => el.value.trim();
}

function initLive() {
  const video = $("#live-video");
  const getHome = homeInput($("#live-home"));
  let busy = false, timer = null, fixing = null, lastSpoken = 0;

  $("#live-start").onclick = async () => {
    try {
      await startCamera(video);
      $("#live-scan").disabled = false;
      voice.speak("Camera on. Ask me what's different.");
    } catch (e) { voice.speak(`I can't open the camera: ${e.message}`); }
  };

  async function check() {
    if (busy || fixing || !video.videoWidth) return;
    busy = true;
    try {
      const { scan, analysis } = await postForm(`/api/homes/${getHome()}/scan`, { image: await grab(video) });
      drawOverlay($("#live-overlay"), video, analysis.items ?? []);
      renderItems($("#live-items"), analysis, getHome(), liveFix);
      renderTrace($("#live-trace"), analysis, scan);
      const now = Date.now();
      if (analysis.say !== voice.last || now - lastSpoken > 8000) { speakAnalysis(analysis); lastSpoken = now; }
    } catch (e) {
      voice.speak(e.status === 409 ? "This home has no baseline yet. Set it up first." : `Something went wrong: ${e.message}`);
    } finally { busy = false; }
  }

  async function liveFix(change, li) {
    const fix = await startFix(getHome(), change, li);
    if (!fix) return;
    fixing = fix;
    voice.speak("Okay, I'll guide you. Keep the camera pointed at the area.");
    while (fixing) {
      const res = await postForm(`/api/fix/${fix.session}/frame`, { image: await grab(video) });
      drawOverlay($("#live-overlay"), video, res.done ? [] : [change]);
      handleFixStep(res, fix.show, li);
      if (res.done) { if (res.trace) renderTrace($("#live-trace"), { trace: res.trace }); fixing = null; break; }
      await sleep(700);
    }
  }

  $("#live-scan").onclick = check;
  $("#live-auto").onchange = (e) => {
    clearInterval(timer);
    if (e.target.checked) timer = setInterval(check, 1500);
  };
}

// ---------- compare ----------
function initCompare() {
  $("#cmp-run").onclick = async () => {
    const before = $("#cmp-before").files[0], after = $("#cmp-after").files[0];
    if (!before || !after) { voice.speak("Choose both a before and an after photo."); return; }
    voice.speak("Comparing…");
    try {
      const { scan, analysis } = await postForm("/api/compare", { baseline: before, current: after });
      await drawImageResult($("#cmp-canvas"), after, analysis.items ?? []);
      renderItems($("#cmp-items"), analysis, null, null);
      renderTrace($("#cmp-trace"), analysis, scan);
      speakAnalysis(analysis);
    } catch (e) { voice.speak(`Couldn't compare: ${e.message}`); }
  };
}

// ---------- setup ----------
function initSetup() {
  const video = $("#setup-video");
  const getHome = homeInput($("#setup-home"));

  async function refresh() {
    const list = $("#setup-list");
    list.replaceChildren();
    let frames = [];
    try { frames = await api(`/api/homes/${getHome()}/keyframes`); } catch { /* new home */ }
    if (!frames.length) list.append(h("li", { class: "empty" }, "No views saved yet."));
    for (const kf of frames) {
      list.append(h("li", {},
        h("img", { src: `${kf.url}?t=${Date.now()}`, alt: `Saved view ${kf.name}` }),
        h("span", {}, kf.name),
        h("button", { onclick: async () => {
          await api(`/api/homes/${getHome()}/keyframes/${kf.name}`, { method: "DELETE" });
          refresh();
        } }, "Remove")));
    }
  }

  async function save(blob) {
    const name = ($("#setup-name").value.trim() || "room").replace(/\s+/g, "_");
    try {
      const res = await postForm(`/api/homes/${getHome()}/keyframes`, { image: blob, name });
      voice.speak(`Saved ${name}, with ${res.features} recognizable features.`);
      refresh();
    } catch (e) {
      const d = e.detail;
      if (d?.reason === "quality") voice.speak(`That view isn't clear enough: ${d.quality.issues.join(", ")}. Try again.`);
      else voice.speak(d?.say ?? `Couldn't save: ${e.message}`);
    }
  }

  $("#setup-start").onclick = async () => {
    try { await startCamera(video); $("#setup-capture").disabled = false; }
    catch (e) { voice.speak(`I can't open the camera: ${e.message}`); }
  };
  $("#setup-capture").onclick = async () => save(await grab(video));
  $("#setup-file").onchange = (e) => e.target.files[0] && save(e.target.files[0]);
  $("#setup-home").addEventListener("change", refresh);
  refresh();
}

// ---------- shell ----------
function initShell() {
  $$(".tabs button").forEach((b) => b.addEventListener("click", () => {
    $$(".tabs button").forEach((x) => x.setAttribute("aria-selected", String(x === b)));
    $$(".panel").forEach((p) => { p.hidden = p.id !== `tab-${b.dataset.tab}`; });
  }));
  const lv = $("#lowvision");
  lv.checked = store.get("lowvision", "0") === "1";
  document.body.classList.toggle("lowvision", lv.checked);
  lv.onchange = () => { document.body.classList.toggle("lowvision", lv.checked); store.set("lowvision", lv.checked ? "1" : "0"); };
  const mute = $("#mute");
  mute.onchange = () => { voice.muted = mute.checked; if (mute.checked) speechSynthesis?.cancel(); };
  api("/api/health").then((hl) => {
    $("#health").textContent = `OpenCV ${hl.opencv} · agents: ${hl.llm === "off" ? "rule-based" : hl.llm}`;
  }).catch(() => { $("#health").textContent = "Service unreachable"; });
}

initShell();
initDemo();
initLive();
initCompare();
initSetup();

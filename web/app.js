import { Viewer } from "./viewer.js";
import { Theatre } from "./theatre.js";

const $ = (s) => document.querySelector(s);

function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v);
  }
  for (const c of children.flat()) if (c != null) el.append(c);
  return el;
}

function svg(tag, attrs = {}, ...children) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  for (const c of children.flat()) if (c != null) el.append(c);
  return el;
}

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail ?? msg; } catch { /* not json */ }
    const err = new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
    err.status = r.status;
    throw err;
  }
  return r.json();
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const secs = (v) => `${Number(v).toFixed(1)} s`;
const liveUrl = (id, name) => `/api/live/${encodeURIComponent(id)}/files/${encodeURIComponent(name)}`;

function speak(text) {
  if (!text || !("speechSynthesis" in window)) return;
  speechSynthesis.cancel();
  speechSynthesis.speak(new SpeechSynthesisUtterance(text));
}

const viewer = new Viewer($("#stage"));
$("#vr-slot").append(viewer.vrButton());

// ---------- the story: home <-> experience ----------
const PLAY_LABEL = { playing: "❚❚", holding: "❚❚", paused: "▶", blocked: "▶", ready: "▶" };
const PLAY_NAME = { playing: "Pause", holding: "Pause", paused: "Resume", blocked: "Play", ready: "Play" };
let live = { id: null, line: -1, shot: -1, made: {} };
let captionKey = null;
const mine = new Set(); // stories made from this browser (others, e.g. from a phone, open here by themselves)

const theatre = new Theatre(viewer, {
  onState: renderLive,
  onCaption: renderCaption,
  onPhase: renderPhase,
  onLine: (i) => { live.line = i; markLine(); },
  onShot: (k) => { live.shot = k; markShot(); },
  onClip: (k, what) => markClip(k, what),
  onExit: () => exitStory(),
});
viewer.onEnterVR = () => { if (["ready", "blocked"].includes(theatre.phase)) theatre.resume(); };
Object.assign(window, { __theatre: theatre, __viewer: viewer }); // for browser tests (theatre.debug())

function showExperience(on) {
  $("#home").hidden = on;
  $("#experience").hidden = !on;
  document.querySelector(".sky").hidden = on;
  if (on) viewer.renderer.domElement.focus({ preventScroll: true }); // Space / → / R / WASD work straight away
}

function openLive(id, opts = {}) {
  stopLook();
  live = { id, line: -1, shot: -1, made: {} };
  captionKey = null;
  $("#story-title").textContent = "Bringing your page to life…";
  $("#story-stage").textContent = "";
  for (const s of ["#made-page", "#made-shots", "#made-models", "#made-timings"]) $(s).replaceChildren();
  $("#made-check").textContent = $("#made-world").textContent = "";
  showExperience(true);
  history.replaceState(null, "", `#live/${id}`);
  theatre.start(id, opts);
}

function exitStory() {
  stopLook();
  try { viewer.renderer.xr.getSession()?.end(); } catch { /* not in VR */ }
  theatre.stop();
  viewer.clear();
  $("#made").hidden = true;
  $("#info-toggle").setAttribute("aria-expanded", "false");
  showExperience(false);
  history.replaceState(null, "", location.pathname);
  refreshStories();
}

$("#exit").onclick = exitStory;
$("#info-toggle").onclick = () => {
  const open = $("#made").hidden;
  $("#made").hidden = !open;
  $("#info-toggle").setAttribute("aria-expanded", String(open));
};
$("#t-play").onclick = () => { theatre.unlock(); theatre.togglePause(); };

// The caption box: small, in a corner, and moved anywhere by its handle (the spot is remembered in this browser).
(function dockDrag() {
  const dock = $("#dock"), grip = $("#dock-grip"), KEY = "step-inside-dock";
  const place = (x, y) => {
    const r = dock.getBoundingClientRect(), W = innerWidth, H = innerHeight;
    x = Math.min(Math.max(4, x), W - r.width - 4);
    y = Math.min(Math.max(56, y), H - r.height - 4);
    Object.assign(dock.style, { left: `${x}px`, top: `${y}px`, bottom: "auto", right: "auto" });
    return [x, y];
  };
  try {
    const saved = JSON.parse(localStorage.getItem(KEY) || "null");
    if (Array.isArray(saved)) requestAnimationFrame(() => place(saved[0] * innerWidth, saved[1] * innerHeight));
  } catch { /* private window: the default corner */ }
  let start = null;
  grip.addEventListener("pointerdown", (e) => {
    const r = dock.getBoundingClientRect();
    start = { dx: e.clientX - r.left, dy: e.clientY - r.top };
    grip.setPointerCapture(e.pointerId);
    dock.classList.add("dragging");
    e.preventDefault();
    e.stopPropagation();
  });
  grip.addEventListener("pointermove", (e) => {
    if (start) place(e.clientX - start.dx, e.clientY - start.dy);
  });
  const end = () => {
    if (!start) return;
    start = null;
    dock.classList.remove("dragging");
    const r = dock.getBoundingClientRect();
    try { localStorage.setItem(KEY, JSON.stringify([r.left / innerWidth, r.top / innerHeight])); } catch { /* fine */ }
  };
  grip.addEventListener("pointerup", end);
  grip.addEventListener("pointercancel", end);
  addEventListener("resize", () => { const r = dock.getBoundingClientRect(); if (dock.style.top) place(r.left, r.top); });
})();
$("#t-next").onclick = () => theatre.nextLine();
$("#step-in").onclick = () => theatre.stepInside();
$("#t-replay").onclick = () => theatre.replay();
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !$("#experience").hidden) exitStory(); });

function renderLive(st) {
  if (st.id && st.id !== live.id) return;
  $("#story-title").textContent = st.title || "Bringing your page to life…";
  $("#story-stage").textContent = st.status === "failed" ? `Couldn't finish: ${st.error || "something went wrong"}`
    : st.status === "running" || st.upgrading ? `${st.stage || "Making your story"}` +
      (theatre.progress().waiting ? ` · ${theatre.progress().percent}%` : "")
    : st.timings?.film != null ? `Filmed in ${secs(st.timings.film)}`
    : st.timings?.playable != null ? `Ready in ${secs(st.timings.playable)}` : "";
  renderMade(st);
}

function renderPhase(phase) {
  const on = !["stopped", "loading", "failed"].includes(phase);
  $("#controls").hidden = !on || ["door", "entering"].includes(phase);
  $("#step-in").hidden = phase !== "door";
  $("#t-play").textContent = PLAY_LABEL[phase] ?? "▶";
  $("#t-play").setAttribute("aria-label", PLAY_NAME[phase] ?? "Play");
  $("#t-play").title = PLAY_NAME[phase] ?? "Play";
  $("#t-play").disabled = !(phase in PLAY_LABEL);
  $("#t-play").hidden = phase === "ended"; // at the end only Replay: it replays what was made, nothing is remade
  $("#t-next").disabled = !["playing", "holding", "paused", "blocked"].includes(phase);
  $("#t-replay").disabled = !["playing", "holding", "paused", "blocked", "ended"].includes(phase);
}

/** The caption under the scene mirrors the one in 3D, word by word. */
function renderCaption(cap) {
  const el = $("#caption");
  if (!cap) { el.hidden = true; captionKey = null; return; }
  el.hidden = false;
  if (cap.key !== captionKey) {
    captionKey = cap.key;
    el.classList.toggle("note", !cap.words);
    if (cap.words) {
      el.replaceChildren(h("span", { class: "who" }, cap.speaker),
        ...cap.words.flatMap((w, i) => [...(i ? [" "] : []), h("span", { class: "w" }, w)]));
    } else {
      el.replaceChildren(...[cap.title && h("strong", {}, cap.title), cap.sub && h("span", {}, cap.sub)].filter(Boolean));
    }
  }
  if (cap.words) el.querySelectorAll(".w").forEach((s, i) => s.classList.toggle("now", i === cap.word));
}

// ---------- "How it was made": the evidence, rebuilt only when its data changes ----------
const MODEL_ROLES = [["reader", "Read the page"], ["planner", "Story agent"], ["painter", "Painted"], ["voice", "Voices"],
  ["depth", "Depth"], ["actors", "Moving characters"], ["animator", "Animated"], ["critic", "Clips checked by"]];
const TIMINGS = [["read", "Page read"], ["plan", "Story written"], ["world", "Quick world"], ["first_shot", "First moment"],
  ["playable", "Painted and voiced"], ["world_full", "Full 360° world"], ["first_clip", "First video clip"],
  ["film", "Playing as film"], ["clips", "Every moment filmed"], ["all", "Story ready"],
  ["upgrades", "Everything"]];
const OPTIONAL = new Set(["world_full", "first_clip", "film", "clips", "upgrades"]);

/** What happened to a moment beyond its painting: its moving character and its video clip, with OpenCV's verdict. */
function clipNote(s) {
  const a = s.action ?? {}, c = s.clip, notes = [];
  const who = String(c?.model ?? "").startsWith("amazon.nova-reel") ? "Nova Reel" : "Luma Ray2";
  if (a.path && a.path !== "still") {
    notes.push(`${a.subject || "Something"}: ${a.motion || a.path}`);
    if (s.actor) notes.push(`cut out by OpenCV (${s.actor_check?.label || "figure"}), moves: ${a.path}`);
    else if (s.actor_check && !s.actor_check.ok) notes.push(`no cut-out: ${s.actor_check.reason}`);
  }
  if (c?.status === "queued") notes.push(`Video: waiting for ${who}…`);
  else if (c?.status === "generating") notes.push(`Video: ${who} is filming this moment…`);
  else if (c?.status === "checking") notes.push("Video: OpenCV is checking the clip…");
  else if (c?.status === "ready") {
    const sc = c.verdict?.scores ?? {};
    notes.push(`Video by ${who} · OpenCV: kept the painting (${sc.fidelity_inliers ?? "?"} matching features)` +
      `${c.verdict?.observed && c.verdict.observed !== "unclear" ? `, ${c.verdict.observed} verified by optical flow` : ""}` +
      ` · ${c.loop_at != null ? `loops at ${Number(c.loop_at).toFixed(1)} s` : "holds its last frame"}` +
      `${c.attempts > 1 ? ` · try ${c.attempts}` : ""}`);
  } else if (c?.status === "rejected") notes.push(`Video rejected by OpenCV: ${c.reason}; the living painting stays`);
  else if (c?.status === "off" && c.reason && c.reason !== "a still moment") notes.push(`No video: ${c.reason}`);
  return notes.join(" · ");
}
const lineRange = (ls) => !ls?.length ? "" : ls.length === 1 ? `line ${ls[0] + 1}` : `lines ${ls[0] + 1}–${ls[ls.length - 1] + 1}`;

function renderMade(st) {
  const id = live.id, m = live.made;
  const changed = (key, value) => {
    const sig = JSON.stringify(value);
    if (m[key] === sig) return false;
    m[key] = sig;
    return true;
  };
  if (changed("page", [st.page, (st.lines ?? []).map((l) => l.box)])) {
    $("#made-page").replaceChildren(...pageFigure(id, st));
    markLine();
  }
  if (changed("check", st.check)) {
    const c = st.check;
    $("#made-check").textContent = !c ? "Reading…" : `OpenCV read ${c.ocr_lines} printed lines and agreed with the story agent on ` +
      `${Math.round(c.agreement * 100)}% of the text; it put back ${c.restored} skipped line${c.restored === 1 ? "" : "s"}` +
      ` and removed ${c.dropped ?? 0} repeated.`;
  }
  if (changed("shots", (st.shots ?? []).map((s) => [s.ready, s.image, s.caption, s.lines, s.moving, s.source,
    s.action, s.actor, s.actor_check, s.clip?.status, s.clip?.attempts, s.world?.quality, s.world?.upgrading]))) {
    const shots = st.shots ?? [];
    const busy = (s) => !s.ready || ["queued", "generating", "checking"].includes(s.clip?.status);
    $("#made-shots").replaceChildren(...(shots.length ? shots.map((s) => h("li", { "data-k": s.i, class: busy(s) ? "busy" : "" },
      s.ready && s.image ? h("img", { src: liveUrl(id, s.image), alt: "", loading: "lazy" }) : h("span", { class: "thumb" }, "…"),
      h("span", {}, s.caption || `Moment ${s.i + 1}`, h("small", {}, [lineRange(s.lines),
        s.moving ? Object.entries(s.moving).filter(([, v]) => v > 0.02).map(([k]) => k).join(", ") || null : null,
        s.ready ? (s.from_page ? "from the page's own picture" : s.source) : "painting…",
        s.world?.quality === "full" ? "own 360° world (full)"
          : s.world?.upgrading ? "own 360° world (repainting…)" : s.world ? "own 360° world" : null]
        .filter(Boolean).join(" · ")), clipNote(s) ? h("small", { class: "clip" }, clipNote(s)) : null)))
      : [h("li", {}, "Planning the moments…")]));
    markShot();
  }
  if (changed("world", st.world)) {
    const w = st.world ?? {};
    $("#made-world").textContent = !w.ready ? "Painting…" : !w.image ? "A soft sky (no painter available)."
      : w.quality === "full" ? `Full 360° world: painted outward from the first moment in ${w.tiles ?? 5} tiles, checked by OpenCV` +
        `${w.stray_figures_removed ? `, ${w.stray_figures_removed} stray figures repainted` : ""}; depth and motion by OpenCV.`
      : `Quick world (one wide painting wrapped to 360°${w.figures_erased ? `, ${w.figures_erased} figures erased by OpenCV` : ""})` +
        `${w.upgrading ? "; the full 360° world is being painted…" : "."}`;
    const L = w.life;
    if (L) {
      const bits = [`wind ${L.wind.strength >= 0.75 ? "strong" : L.wind.strength >= 0.4 ? "steady" : "light"}` +
        ` along the sky's brush strokes (${Math.round(L.wind.angle_deg)}°)`];
      if (L.light) bits.push(`${L.time === "night" ? "moon" : "sun"} found and made to glow`);
      if (L.lamps?.length) bits.push(`${L.lamps.length} lamp${L.lamps.length === 1 ? "" : "s"} flickering`);
      if (L.haze?.amount) bits.push("haze at the horizon");
      $("#made-world").textContent += ` OpenCV brought it to life: ${bits.join(", ")}. Each moment also gets its own 360° world.`;
    }
  }
  if (changed("models", st.models)) {
    $("#made-models").replaceChildren(...MODEL_ROLES.filter(([k]) => st.models?.[k])
      .flatMap(([k, label]) => [h("dt", {}, label), h("dd", {}, String(st.models[k]))]));
  }
  if (changed("timings", [st.timings, st.status])) {
    $("#made-timings").replaceChildren(...TIMINGS.flatMap(([k, label]) => {
      const v = st.timings?.[k];
      return v == null && OPTIONAL.has(k) && st.status !== "running" && !st.upgrading ? [] :
        [h("dt", {}, label), h("dd", {}, v != null ? secs(v) : st.status === "running" ? "…" : "–")];
    }));
  }
}

function pageFigure(id, st) {
  if (!st.page) return [h("figcaption", {}, "Reading the page…")];
  const w = Number(st.page.w) || 1, ph = Number(st.page.h) || 1;
  const boxes = (st.lines ?? []).filter((l) => Array.isArray(l.box));
  const overlay = svg("svg", { viewBox: `0 0 ${w} ${ph}`, preserveAspectRatio: "none", "aria-hidden": "true" },
    boxes.map((l) => svg("polygon", { points: l.box.map(([x, y]) => `${Number(x)},${Number(y)}`).join(" "), "data-i": l.i })));
  return [
    h("div", { class: "page-box", style: `aspect-ratio: ${w} / ${ph}` },
      h("img", { src: liveUrl(id, st.page.image), alt: "The page as OpenCV flattened it" }), overlay),
    h("figcaption", {}, boxes.length ? `OpenCV found ${boxes.length} lines of text; the one being read lights up.`
      : "Looking for the lines of text…"),
  ];
}

function markLine() {
  document.querySelectorAll("#made-page polygon").forEach((p) => p.classList.toggle("now", p.dataset.i === String(live.line)));
}

function markClip(k, what) {
  const li = document.querySelector(`#made-shots li[data-k="${k}"]`);
  if (li) li.dataset.playing = what || "";
}

function markShot() {
  document.querySelectorAll("#made-shots li[data-k]").forEach((li) => li.setAttribute("aria-current", String(li.dataset.k === String(live.shot))));
}

// ---------- three ways in: a photo, a video, the camera ----------
/** Phone photos are often 4000+ px and several MB; the page is read at 1280 px, so send at most 2048 px. */
async function shrink(file, side = 2048) {
  try {
    const bmp = await createImageBitmap(file);
    const s = side / Math.max(bmp.width, bmp.height);
    if (s >= 1 && file.size < 3e6) { bmp.close(); return file; }
    const c = document.createElement("canvas");
    c.width = Math.round(bmp.width * Math.min(1, s));
    c.height = Math.round(bmp.height * Math.min(1, s));
    c.getContext("2d").drawImage(bmp, 0, 0, c.width, c.height);
    bmp.close();
    return (await new Promise((r) => c.toBlob(r, "image/jpeg", 0.9))) ?? file;
  } catch {
    return file; // the server can still try to decode it
  }
}

// ---------- one shared password: needed to make a story, not to watch one ----------
let auth = { required: false, ok: true };
async function refreshAuth() {
  try { auth = await api("/api/auth"); } catch { /* server unreachable: leave as is */ }
  const btn = $("#login-btn");
  btn.hidden = !auth.required;
  btn.textContent = auth.ok ? "✓ Logged in · Log out" : "🔒 Log in";
  document.body.classList.toggle("locked", auth.required && !auth.ok);
  $("#home-status").textContent = auth.required && !auth.ok ? "Log in to bring your own page to life. You can watch any story below." : "";
}
function askLogin() {
  $("#login-msg").textContent = "";
  $("#login-pass").value = "";
  $("#login").showModal();
  $("#login-pass").focus();
}
$("#login-btn").onclick = async () => {
  if (auth.ok) { await api("/api/logout", { method: "POST" }).catch(() => {}); await refreshAuth(); return; }
  askLogin();
};
$("#login-cancel").onclick = () => $("#login").close();
$("#login-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  try {
    await api("/api/login", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password: $("#login-pass").value }) });
    $("#login").close();
    await refreshAuth();
  } catch (err) {
    $("#login-msg").textContent = err.status === 429 ? "Too many tries. Wait a minute." : "That password is not right.";
  }
});
/** Before making anything: logged in (or no password needed)? Otherwise ask. */
function mayCreate() {
  if (!auth.required || auth.ok) return true;
  askLogin();
  return false;
}
refreshAuth();

async function send(field, blob, name) {
  $("#home-status").textContent = field === "video" ? "Uploading the video: OpenCV will pick the clearest frame…" : "Sending the page…";
  try {
    const fd = new FormData();
    fd.append(field, blob, name);
    const st = await api("/api/live", { method: "POST", body: fd });
    mine.add(st.id);
    $("#home-status").textContent = "";
    openLive(st.id, { autoplay: true, state: st.reused ? null : st });
  } catch (err) {
    $("#home-status").textContent = {
      429: err.message || "Busy… try again in a minute.",
      413: "That file is too big.",
      415: "That file isn't an image we can read.",
      422: err.message,
      401: "Log in first: creating a story needs the password.",
    }[err.status] ?? `Couldn't send it: ${err.message}`;
    if (err.status === 401) { await refreshAuth(); askLogin(); }
  }
}

for (const id of ["#pick-image", "#pick-video"]) { // no file chooser before logging in
  $(id).addEventListener("click", (e) => { if (!mayCreate()) e.preventDefault(); });
}
$("#pick-image").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  e.target.value = "";
  if (!file) return;
  theatre.unlock(); // still inside the tap: phones only let a page speak after one
  const page = await shrink(file);
  send("page", page, page === file ? file.name : "page.jpg");
});

$("#pick-video").addEventListener("change", (e) => {
  const file = e.target.files[0];
  e.target.value = "";
  if (!file) return;
  theatre.unlock();
  send("video", file, file.name || "page.mp4");
});

// The camera: OpenCV coaches the shot, recognizes pages it has seen, then captures by itself.
let look = null;

function frameBlob(video, side) {
  const s = Math.min(1, side / Math.max(video.videoWidth, video.videoHeight));
  const c = document.createElement("canvas");
  c.width = Math.round(video.videoWidth * s);
  c.height = Math.round(video.videoHeight * s);
  c.getContext("2d").drawImage(video, 0, 0, c.width, c.height);
  return new Promise((r) => c.toBlob(r, "image/jpeg", 0.88));
}

/** Show the captured photo and wait for "Use this page" (true) or "Retake" / Cancel (false). */
function confirmPage(blob, tell) {
  const still = $("#look-still"), ask = $("#look-confirm"), url = URL.createObjectURL(blob);
  still.src = url;
  still.hidden = ask.hidden = false;
  $("#look-box").hidden = true;
  tell("Is this the page? Tap “Use this page” or “Retake”.");
  return new Promise((done) => {
    const finish = (ok) => {
      still.hidden = ask.hidden = true;
      still.removeAttribute("src");
      URL.revokeObjectURL(url);
      $("#look-use").onclick = $("#look-retake").onclick = null;
      if (look) look.confirm = null;
      done(ok);
    };
    $("#look-use").onclick = () => { tell("Bringing it to life…"); finish(true); };
    $("#look-retake").onclick = () => { tell("Show me the page again."); finish(false); };
    if (look) look.confirm = finish;
  });
}

function stopLook() {
  if (!look) return;
  look.confirm?.(false);
  look.running = false;
  look.stream.getTracks().forEach((t) => t.stop());
  look = null;
  $("#look-overlay").hidden = true;
  $("#look-box").hidden = true;
}

/** Outline the text OpenCV found (green when it's good to read), mapped onto the cover-fitted video. */
function showLookBox(q) {
  const box = $("#look-box"), v = $("#look-video");
  if (!q.box || !v.videoWidth) { box.hidden = true; return; }
  const s = Math.max(v.clientWidth / v.videoWidth, v.clientHeight / v.videoHeight);
  const ox = (v.clientWidth - v.videoWidth * s) / 2, oy = (v.clientHeight - v.videoHeight * s) / 2;
  const [x0, y0, x1, y1] = q.box, W = v.videoWidth * s, H = v.videoHeight * s;
  Object.assign(box.style, { left: `${ox + x0 * W}px`, top: `${oy + y0 * H}px`,
    width: `${(x1 - x0) * W}px`, height: `${(y1 - y0) * H}px` });
  box.classList.toggle("ok", q.ok);
  box.hidden = false;
}

$("#look-start").onclick = async () => {
  if (!mayCreate()) return;
  theatre.unlock();
  stopLook();
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment", width: { ideal: 1920 }, height: { ideal: 1080 } }, audio: false });
  } catch (err) {
    $("#home-status").textContent = `Camera unavailable (${err.message}). You can upload a photo instead.`;
    return;
  }
  theatre.stop();
  $("#story-title").textContent = "Show me a page";
  $("#story-stage").textContent = "";
  $("#controls").hidden = $("#caption").hidden = true;
  showExperience(true);
  const video = $("#look-video");
  video.srcObject = stream;
  await video.play();
  $("#look-overlay").hidden = false;
  const local = look = { stream, running: true, good: 0, said: "", saidAt: 0 };
  const tell = (msg) => {
    $("#look-say").textContent = msg;
    if (msg !== local.said && performance.now() - local.saidAt > 2500) { // spoken too, for a headset
      speak(msg);
      local.said = msg;
      local.saidAt = performance.now();
    }
  };
  while (local.running) {
    const fd = new FormData();
    fd.append("image", await frameBlob(video, 960), "frame.jpg");
    let res;
    try { res = await api("/api/look", { method: "POST", body: fd }); } catch { await sleep(600); continue; }
    if (!local.running) return;
    showLookBox(res.quality);
    if (res.match) { // a page that already has a story: start it now
      tell(`Found it: ${res.match.title || "your story"}`);
      openLive(res.match.id, { autoplay: true });
      return;
    }
    local.good = res.quality.ok ? local.good + 1 : 0;
    if (local.good >= 2) { // readable two frames in a row: take the full-size photo, then ask before making a story
      const page = await frameBlob(video, 2048);
      if (!local.running) return;
      if (await confirmPage(page, tell)) {
        stopLook();
        send("page", page, "camera.jpg");
        return;
      }
      if (!local.running) return;
      local.good = 0;
      continue;
    }
    tell(res.quality.ok ? "Hold still…" : res.quality.say);
    await sleep(200);
  }
};
$("#look-stop").onclick = () => { stopLook(); exitStory(); };

// ---------- the shelf: your stories ----------
function when(created) {
  const t = typeof created === "number" ? created * 1000 : Date.parse(created);
  return Number.isFinite(t) ? new Date(t).toLocaleString([], { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }) : "";
}

async function refreshStories() {
  let list = [];
  try { list = await api("/api/live"); } catch { /* service unreachable */ }
  $("#mine").hidden = !list.length;
  $("#stories").replaceChildren(...list.map((s) => h("li", {},
    h("button", { type: "button", title: when(s.created), onclick: () => { theatre.unlock(); openLive(s.id); } },
      s.thumb ? h("img", { src: liveUrl(s.id, s.thumb), alt: "" }) : h("span", { class: "blank" }),
      h("span", {}, s.title || "Untitled story")))));
  return list;
}

const stories = await refreshStories();
const link = location.hash.match(/^#live\/([a-f0-9]{12})$/);
// Nothing plays by itself on load: browsers block sound before a tap, so a linked story waits for Play.
if (link) openLive(link[1], { autoplay: false });

// A story made on another device (a phone pointed at a page) opens here by itself, e.g. in a headset whose browser
// can't use its cameras. It waits for Play (or the in-world Play button in VR).
let newest = stories[0]?.id ?? null;
setInterval(async () => {
  if (document.hidden || look) return;
  let list;
  try { list = await api("/api/live?limit=1"); } catch { return; }
  const id = list[0]?.id;
  if (!id || id === newest) return;
  newest = id;
  refreshStories();
  if (mine.has(id) || theatre.id === id || ["playing", "holding", "paused"].includes(theatre.phase)) return;
  openLive(id, { autoplay: false });
  $("#story-stage").textContent = "A new story just arrived from another device. Press Play.";
}, 4000);

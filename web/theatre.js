import * as THREE from "three";

// Story theatre: one photographed page, told line by line inside its painted world. The backend streams the story
// (lines, moments, pictures, voices) while it is being made; playback starts as soon as the first moment is ready.

const EYE = 1.6;
const WORLD_R = 60;
const R = 7;                                    // shot canvas radius (m)
const SPAN = THREE.MathUtils.degToRad(150);     // its horizontal arc
const SEG_X = 192, SEG_Y = 80;
const DEPTH_GAIN = 2.2;                         // m of relief between the median depth and the nearest point
const FADE_S = 0.9;
const GAP_MS = 250;
const POLL_MS = 400;
const DRIFT_YAW = THREE.MathUtils.degToRad(1.5);
const PAGE_YAW = THREE.MathUtils.degToRad(55), PAGE_DIST = 2.2, PAGE_H = 1.1;
const FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';
const ACCENT = "#8fb4ff", GOLD = "#ffd36b", INK = "#f4f1ea", MUTED = "#a9b0c3", OK = "#6fd39a";
// Transparent layers are drawn back to front by these, not by distance: the canvas origin sits at the viewer's
// eye, so depth sorting would put it in front of the panels.
const ORDER = { world: 0, stars: 1, canvas: 2, particles: 3, panel: 4 };
const PLAYING = ["playing", "holding"];
const STEPS = [
  ["read", "Reading the page (OpenCV)"], ["plan", "Writing the story"], ["world", "Painting the world"],
  ["shots", "Painting the moments"], ["voices", "Giving it voices"],
].map(([name, label]) => ({ name, label, done: false, seconds: null }));
const PENDING_CLIP = ["queued", "generating", "checking"];
const FINAL_CLIP = ["ready", "rejected", "off"];
const SLOW_POLL_MS = 3000;              // after the story is ready, while its moments are still being upgraded
const POLL_MAX_MS = 30 * 60 * 1000;
const LATE_CLIP = 0.85;                 // a clip arriving later than this in its moment waits for the replay
const WEATHER_FX = { rain: ["rain"], snow: ["snow"], storm: ["rain"] };

const fileUrl = (id, name) => `/api/live/${encodeURIComponent(id)}/files/${encodeURIComponent(name)}`;
const clamp01 = (t) => Math.min(1, Math.max(0, t));
const ease = (t) => t * t * (3 - 2 * t);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const tokenize = (text) => [...text.matchAll(/\S+/g)].map((m) => ({ w: m[0], s: m.index, e: m.index + m[0].length }));

async function loadImage(url) {
  const img = new Image();
  img.src = url;
  await img.decode(); // rejects if it can't be fetched or decoded
  return img;
}

/** A five-pointed star (outer radius r, inner ri), filled with the current fill style. */
function star(ctx, cx, cy, r, ri) {
  ctx.beginPath();
  for (let i = 0; i < 10; i++) {
    const a = -Math.PI / 2 + (i * Math.PI) / 5, k = i % 2 ? ri : r;
    ctx[i ? "lineTo" : "moveTo"](cx + Math.cos(a) * k, cy + Math.sin(a) * k);
  }
  ctx.closePath();
  ctx.fill();
}

/** Free GPU memory: geometry buffers and every texture a material references. */
function dispose(root) {
  root?.traverse((o) => {
    o.geometry?.dispose();
    for (const m of [o.material].flat()) {
      if (!m) continue;
      for (const v of Object.values(m)) if (v?.isTexture && !v.userData?.keep) v.dispose();
      m.dispose();
    }
  });
}

function rounded(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

let silentUrl = null;
/** A 0.1 s silent WAV: playing it inside a tap unlocks the shared audio element on phones that need that. */
function silence() {
  if (silentUrl) return silentUrl;
  const n = 800, v = new DataView(new ArrayBuffer(44 + n * 2));
  const str = (o, s) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  str(0, "RIFF"); v.setUint32(4, 36 + n * 2, true); str(8, "WAVE"); str(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, 8000, true);
  v.setUint32(28, 16000, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true); str(36, "data");
  v.setUint32(40, n * 2, true);
  return (silentUrl = URL.createObjectURL(new Blob([v.buffer], { type: "audio/wav" })));
}

/** A canvas-backed plane for text in 3D (HTML overlays don't exist inside a VR headset). */
function panel(w, h, cw, ch) {
  const canvas = document.createElement("canvas");
  canvas.width = cw;
  canvas.height = ch;
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(w, h),
    new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false, toneMapped: false }));
  mesh.renderOrder = ORDER.panel;
  return { canvas, ctx: canvas.getContext("2d"), tex, mesh };
}

/** Dark blue sky to dark ground, shown until (or instead of) the painted world. */
function backdrop() {
  const r = WORLD_R + 2;
  const geo = new THREE.SphereGeometry(r, 48, 24);
  const pos = geo.attributes.position, col = new Float32Array(pos.count * 3);
  const top = new THREE.Color("#0b1534"), mid = new THREE.Color("#1d2c52"), low = new THREE.Color("#07080d");
  const c = new THREE.Color();
  for (let i = 0; i < pos.count; i++) {
    const t = pos.getY(i) / r;
    if (t >= 0) c.copy(mid).lerp(top, Math.sqrt(t)); else c.copy(mid).lerp(low, Math.min(1, -t * 3));
    c.toArray(col, i * 3);
  }
  geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
  const mesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({
    vertexColors: true, side: THREE.BackSide, toneMapped: false, depthWrite: false }));
  mesh.name = "theatre-backdrop";
  mesh.position.y = EYE;
  return mesh;
}

/** Depth PNG sampled once per canvas vertex (row-major from the top, like PlaneGeometry), lightly blurred so a
 *  depth edge stretches over a few triangles instead of tearing. 0 = far, 1 = near. */
function depthGrid(img) {
  const w = SEG_X + 1, h = SEG_Y + 1;
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(img, 0, 0, w, h);
  const px = ctx.getImageData(0, 0, w, h).data;
  let d = new Float32Array(w * h);
  for (let i = 0; i < d.length; i++) d[i] = px[i * 4] / 255;
  for (const [dx, dy] of [[1, 0], [0, 1]]) {
    const out = new Float32Array(d.length);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const a = d[Math.max(0, y - dy) * w + Math.max(0, x - dx)];
        const b = d[Math.min(h - 1, y + dy) * w + Math.min(w - 1, x + dx)];
        out[y * w + x] = (a + 2 * d[y * w + x] + b) / 4;
      }
    }
    d = out;
  }
  return d;
}

/** The 150° cylinder segment, each vertex pushed along its ray from the eye by the depth: from the centre the
 *  picture looks unchanged, and moving your head (VR) reveals the relief. Coordinates are relative to the
 *  canvas centre (0, 0, -R), so scaling the mesh zooms the picture. */
function bentGeometry(aspect, grid, midIn = null) {
  const arc = R * SPAN;
  const geo = new THREE.PlaneGeometry(arc, arc / aspect, SEG_X, SEG_Y);
  const pos = geo.attributes.position;
  let mid = midIn ?? 0;
  if (grid && midIn == null) {
    const sorted = Float32Array.from(grid).sort();
    mid = sorted[sorted.length >> 1];
  }
  geo.userData.mid = mid;
  for (let i = 0; i < pos.count; i++) {
    const a = pos.getX(i) / R, y = pos.getY(i);
    const r = grid ? THREE.MathUtils.clamp(R - DEPTH_GAIN * (grid[i] - mid), R - 3, R + 2) : R;
    pos.setXYZ(i, Math.sin(a) * r, y * (r / R), R - Math.cos(a) * r);
  }
  geo.computeBoundingSphere();
  return geo;
}

/** The depth grid with the character's area flattened to the scenery around it, so once the cut-out moves away
 *  no bump in its shape is left on the screen (visible when you move your head in VR). Box in picture pixels. */
function relaxedGrid(grid, box, W, H) {
  const gw = SEG_X + 1, gh = SEG_Y + 1, out = Float32Array.from(grid);
  const gx0 = Math.max(0, Math.floor((box[0] / W) * SEG_X) - 1), gx1 = Math.min(gw - 1, Math.ceil((box[2] / W) * SEG_X) + 1);
  const gy0 = Math.max(0, Math.floor((box[1] / H) * SEG_Y) - 1), gy1 = Math.min(gh - 1, Math.ceil((box[3] / H) * SEG_Y) + 1);
  const ring = [];
  for (let y = Math.max(0, gy0 - 2); y <= Math.min(gh - 1, gy1 + 2); y++) {
    for (let x = Math.max(0, gx0 - 2); x <= Math.min(gw - 1, gx1 + 2); x++) {
      if (x < gx0 || x > gx1 || y < gy0 || y > gy1) ring.push(grid[y * gw + x]);
    }
  }
  if (!ring.length) return out;
  ring.sort((a, b) => a - b);
  const level = ring[ring.length >> 1];
  for (let y = gy0; y <= gy1; y++) for (let x = gx0; x <= gx1; x++) out[y * gw + x] = level;
  return out;
}

/** Living paintings: a motion map from OpenCV (red = sky, green = water, blue = leaves) moves only those parts of
 *  the picture. Sky and water flow in a seamless loop (two samples a cycle apart, cross-faded), water also ripples,
 *  leaves sway. Patched into MeshBasicMaterial so the alpha feather, opacity fades and colour handling stay. */
function animate(mat, motion, clock, life = null) {
  mat.motionMap = motion; // a material property, so dispose() frees it with the rest
  mat.customProgramCacheKey = () => "living-painting";
  // The wind OpenCV read from the sky's own brush strokes: the sky drifts along them, leaves sway with its strength.
  const w = life?.wind, s = w?.strength ?? 0.5, a = THREE.MathUtils.degToRad(w?.angle_deg ?? 0);
  const wind = new THREE.Vector2(Math.cos(a), -Math.sin(a)).multiplyScalar(0.022 * (0.5 + s));
  mat.userData.wind = wind;
  mat.onBeforeCompile = (shader) => {
    shader.uniforms.motionMap = { value: motion };
    shader.uniforms.uTime = clock;
    shader.uniforms.uWind = { value: wind };
    shader.uniforms.uLeaf = { value: 0.5 + s };
    shader.fragmentShader = shader.fragmentShader
      .replace("#include <map_pars_fragment>",
        "#include <map_pars_fragment>\nuniform sampler2D motionMap;\nuniform float uTime;\nuniform vec2 uWind;\nuniform float uLeaf;")
      .replace("#include <map_fragment>", `
        vec3 mm = texture2D(motionMap, vMapUv).rgb;
        float ph = fract(uTime / 10.0);
        vec2 flow = uWind * mm.r + vec2(0.012, 0.0) * mm.g;
        vec2 wobble = vec2(0.0016 * uLeaf * sin(uTime * (1.2 + uLeaf) + vMapUv.y * 60.0) * mm.b,
                           0.0022 * sin(uTime * 2.4 + vMapUv.x * 90.0 + vMapUv.y * 40.0) * mm.g);
        vec4 sampledDiffuseColor = mix(texture2D(map, vMapUv + flow * ph + wobble),
                                       texture2D(map, vMapUv + flow * (ph - 1.0) + wobble), ph);
        diffuseColor *= sampledDiffuseColor;`);
  };
}

/** The 360 world as a sphere whose radius follows OpenCV's depth map: near scenery (the ground at your feet, a tree
 *  close by) is up to half as far as the sky, so gliding through the world shows real parallax. No displacement at
 *  the poles (which would pinch), and the seam column samples the same depth on both sides. */
function worldGeometry(depthImg) {
  const geo = new THREE.SphereGeometry(WORLD_R, 192, 96);
  geo.scale(-1, 1, 1); // seen from inside
  if (!depthImg) return geo;
  const radius = worldRadius(depthImg);
  const pos = geo.attributes.position, uv = geo.attributes.uv;
  for (let i = 0; i < pos.count; i++) {
    const k = radius(uv.getX(i), 1 - uv.getY(i));
    pos.setXYZ(i, pos.getX(i) * k, pos.getY(i) * k, pos.getZ(i) * k);
  }
  geo.computeBoundingSphere();
  return geo;
}

/** (u, v from the top) -> how far the world's surface is there, as a share of WORLD_R (from its depth map). */
function worldRadius(depthImg) {
  if (!depthImg) return () => 1;
  const W = 512, H = 256, c = document.createElement("canvas");
  c.width = W;
  c.height = H;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.filter = "blur(2px)"; // a depth edge stretches over a few triangles instead of tearing
  ctx.drawImage(depthImg, 0, 0, W, H);
  const px = ctx.getImageData(0, 0, W, H).data;
  return (u, v) => {
    const x = ((Math.floor(u * W) % W) + W) % W, y = Math.min(H - 1, Math.max(0, Math.floor(v * H)));
    return 1 - 0.5 * (px[(y * W + x) * 4] / 255) * Math.sin(Math.PI * v); // 0 at the poles, full at the horizon
  };
}

/** A point of the world sphere, in the sphere mesh's own coordinates (the same mapping SphereGeometry uses). */
function worldPoint(u, v, r) {
  const phi = u * Math.PI * 2, theta = v * Math.PI;
  return new THREE.Vector3(Math.cos(phi) * Math.sin(theta) * r, Math.cos(theta) * r, Math.sin(phi) * Math.sin(theta) * r);
}

let glowTex = null;
/** A soft round glow: the sun, the moon, a lamp. */
function glow() {
  if (glowTex) return glowTex;
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const g = c.getContext("2d");
  const grd = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  grd.addColorStop(0, "rgba(255,255,255,1)");
  grd.addColorStop(0.18, "rgba(255,255,255,0.85)");
  grd.addColorStop(0.45, "rgba(255,255,255,0.22)");
  grd.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = grd;
  g.fillRect(0, 0, 128, 128);
  glowTex = new THREE.CanvasTexture(c);
  glowTex.colorSpace = THREE.SRGBColorSpace;
  glowTex.userData.keep = true; // shared by every glow: dispose() leaves it
  return glowTex;
}

function glowSprite(color, size, opacity) {
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: glow(), color, transparent: true, opacity,
    blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false, toneMapped: false }));
  s.scale.setScalar(size);
  s.renderOrder = ORDER.world + 0.7;
  s.userData.base = opacity;
  return s;
}

/** What OpenCV found alive in a world, put back into it: a glow (and slow rays) where the sun or moon is painted,
 *  flickering lamps at night, and a band of haze where land meets sky. Children of the world sphere, so they fade
 *  and leave with it. */
function worldLife(life, radius) {
  const group = new THREE.Group();
  group.name = "theatre-world-life";
  const lamps = [], rays = [];
  if (!life) return { group, lamps, rays };
  const at = (u, v, pad = 0.96) => worldPoint(u, v, WORLD_R * radius(u, v) * pad);
  if (life.light) {
    const { u, v, size, warm } = life.light, night = life.time === "night";
    const color = warm ? "#ffcf8a" : night ? "#cfdcff" : "#fff6dc";
    // The glow is a few times the painted disc (its angular size is `size` of the full circle), within reason.
    const sun = glowSprite(color, THREE.MathUtils.clamp(size * 2 * Math.PI * WORLD_R * 2.5, 4, 18), night ? 0.5 : 0.65);
    sun.position.copy(at(u, v));
    group.add(sun);
    if (!night) {
      for (let i = 0; i < 6; i++) { // slow god-rays: long, faint, turning
        const ray = glowSprite(color, 1, 0.08);
        ray.scale.set(WORLD_R * 0.05, WORLD_R * 0.6, 1);
        ray.position.copy(at(u, v, 0.94));
        ray.material.rotation = (i / 6) * Math.PI;
        rays.push(ray);
        group.add(ray);
      }
    }
  }
  for (const [u, v, size] of life.lamps ?? []) {
    const lamp = glowSprite("#ffb255", Math.max(0.8, size * WORLD_R * 2 * Math.PI * 4), 0.6);
    lamp.position.copy(at(u, v, 0.97));
    lamp.userData.seed = Math.random() * 100;
    lamps.push(lamp);
    group.add(lamp);
  }
  if (life.haze?.amount) {
    const [r, g, b] = life.haze.color ?? [200, 210, 225];
    const c = document.createElement("canvas");
    c.width = 4;
    c.height = 128;
    const ctx = c.getContext("2d");
    const grd = ctx.createLinearGradient(0, 0, 0, 128);
    grd.addColorStop(0, `rgba(${r},${g},${b},0)`);
    grd.addColorStop(0.5, `rgba(${r},${g},${b},1)`);
    grd.addColorStop(1, `rgba(${r},${g},${b},0)`);
    ctx.fillStyle = grd;
    ctx.fillRect(0, 0, 4, 128);
    const tex = new THREE.CanvasTexture(c);
    tex.colorSpace = THREE.SRGBColorSpace;
    const ring = new THREE.Mesh(new THREE.CylinderGeometry(WORLD_R * 0.45, WORLD_R * 0.45, WORLD_R * 0.22, 64, 1, true),
      new THREE.MeshBasicMaterial({ map: tex, transparent: true, opacity: life.haze.amount, side: THREE.BackSide,
        depthWrite: false, depthTest: false, toneMapped: false }));
    ring.name = "theatre-haze";
    ring.renderOrder = ORDER.world + 0.6;
    ring.userData.base = life.haze.amount;
    group.add(ring);
  }
  return { group, lamps, rays };
}

const DEG = Math.PI / 180;
const easeIn = (t) => t * t;
/** How the cut-out character moves along the story agent's path. t: seconds since it started, u: share of the
 *  moment's spoken time (0..1), side: +1 toward the screen's right, h: its height (m), D: how far a fall may drop.
 *  Returns a rotation (rad, + = counter-clockwise), offsets (m, along the screen and up), an extra opacity, and a
 *  vertical squash. */
function pose(path, t, u, { side, h, D, loop, span }) {
  const p = { rot: 0, dx: 0, dy: 0, da: 0, extra: 1, sy: 1 };
  const wave = (period) => Math.sin((2 * Math.PI * t) / period);
  if (path === "sway") {
    p.rot = 4 * DEG * wave(3.2);
    p.dy = 0.015 * h * wave(1.6);
  } else if (path === "fall") {
    if (u < 0.15) p.rot = DEG * wave(2.5);                    // sitting, the faintest wobble
    else if (u < 0.35) {                                        // the wobble grows
      const q = (u - 0.15) / 0.2;
      p.rot = -side * 6 * DEG * Math.sin(6 * Math.PI * q) * q;
    } else if (u < 0.75) {                                      // over it goes
      const f = easeIn((u - 0.35) / 0.4);
      p.rot = -side * 110 * DEG * f;
      p.dx = side * 0.25 * h * f;
      p.dy = -D * f;
    } else {                                                    // a bump on landing, then still
      const q = clamp01((u - 0.75) / 0.1);
      p.rot = -side * (110 - 8 * Math.sin(Math.PI * q)) * DEG;
      p.dx = side * 0.25 * h;
      p.dy = -D + 0.05 * h * Math.sin(Math.PI * q);
    }
  } else if (path === "walk-left" || path === "walk-right") {
    const dir = path === "walk-left" ? -1 : 1;
    const go = loop ? (t / 8) % 1 : ease(u);
    p.da = dir * 0.4 * span * go;
    p.dy = 0.03 * h * Math.abs(wave(1));
    p.rot = 3 * DEG * wave(1);
  } else if (path === "rise") {
    p.dy = D * ease(u) + 0.02 * h * wave(2); // D here: the room above it
    p.rot = 3 * DEG * wave(2);
    p.extra = 1 - clamp01((u - 0.6) / 0.4);
  } else if (path === "jump") {
    const phase = loop ? (t / 1.2) % 1 : (u * 2) % 1;
    const air = 4 * phase * (1 - phase);
    p.dy = 0.35 * h * air;
    p.sy = 1 - 0.1 * (1 - air) * (phase < 0.12 || phase > 0.88 ? 1 : 0);
  }
  return p;
}

/** Alpha falls to 0 over the outer 8% (sides) and 12% (top, bottom): the picture melts into the world. */
function featherMap() {
  const w = 256, h = 128;
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d");
  const img = ctx.createImageData(w, h);
  const ramp = (t, edge) => ease(clamp01(t / edge));
  for (let y = 0; y < h; y++) {
    const v = (y + 0.5) / h, ay = ramp(Math.min(v, 1 - v), 0.12);
    for (let x = 0; x < w; x++) {
      const u = (x + 0.5) / w, a = Math.round(255 * ay * ramp(Math.min(u, 1 - u), 0.08));
      const o = (y * w + x) * 4;
      img.data[o] = img.data[o + 1] = img.data[o + 2] = a;
      img.data[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return new THREE.CanvasTexture(c);
}

function sprite() {
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const g = c.getContext("2d");
  const grd = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  grd.addColorStop(0, "rgba(255,255,255,1)");
  grd.addColorStop(0.3, "rgba(255,255,255,0.75)");
  grd.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = grd;
  g.fillRect(0, 0, 64, 64);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

function points(n, size, { additive = true, order = ORDER.particles } = {}) {
  const geo = new THREE.BufferGeometry();
  const pos = new Float32Array(n * 3), col = new Float32Array(n * 3);
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3).setUsage(THREE.DynamicDrawUsage));
  geo.setAttribute("color", new THREE.BufferAttribute(col, 3).setUsage(THREE.DynamicDrawUsage));
  const obj = new THREE.Points(geo, new THREE.PointsMaterial({
    size, map: sprite(), vertexColors: true, transparent: true, opacity: 0, depthWrite: false, toneMapped: false,
    blending: additive ? THREE.AdditiveBlending : THREE.NormalBlending }));
  obj.frustumCulled = false; // points move every frame; a stale bounding sphere would cull them
  obj.renderOrder = order;
  return { obj, pos, col, geo, n, base: 1 };
}

/** Random point in a ring around the viewer (or a slice of it in front: yaw within ±arc of -Z). */
function around(rMin, rMax, arc = Math.PI) {
  const a = (Math.random() * 2 - 1) * arc, r = rMin + Math.sqrt(Math.random()) * (rMax - rMin);
  return [Math.sin(a) * r, -Math.cos(a) * r];
}

/** Falling particles (snow, leaves): sway around a fixed column, wrap from the floor back to the top. */
function falling(n, size, colors, { additive, speed: [s0, s1], sway: [w0, w1], top = 7 }) {
  const p = points(n, size, { additive });
  const bx = new Float32Array(n), bz = new Float32Array(n), sp = new Float32Array(n), ph = new Float32Array(n), amp = new Float32Array(n);
  const c = new THREE.Color();
  for (let i = 0; i < n; i++) {
    [bx[i], bz[i]] = around(0.8, 7.5);
    p.pos[i * 3 + 1] = Math.random() * top;
    sp[i] = s0 + Math.random() * (s1 - s0);
    ph[i] = Math.random() * 6.283;
    amp[i] = w0 + Math.random() * (w1 - w0);
    c.set(colors[i % colors.length]).multiplyScalar(0.75 + Math.random() * 0.25).toArray(p.col, i * 3);
  }
  p.update = (dt, t) => {
    for (let i = 0; i < n; i++) {
      let y = p.pos[i * 3 + 1] - sp[i] * dt;
      if (y < -0.2) y += top + 0.2;
      p.pos[i * 3] = bx[i] + Math.sin(t * 0.9 + ph[i]) * amp[i];
      p.pos[i * 3 + 1] = y;
      p.pos[i * 3 + 2] = bz[i] + Math.cos(t * 0.7 + ph[i]) * amp[i] * 0.6;
    }
    p.geo.attributes.position.needsUpdate = true;
  };
  return p;
}

/** Points that stay put and change brightness (stars twinkle, sparkles glint, fireflies pulse while drifting). */
function glowing(n, size, place, tint, { order, wave, drift = 0 }) {
  const p = points(n, size, { order });
  const home = new Float32Array(n * 3), base = new Float32Array(n * 3), ph = new Float32Array(n * 4), sp = new Float32Array(n);
  const c = new THREE.Color();
  for (let i = 0; i < n; i++) {
    home.set(place(), i * 3);
    p.pos.set(home.subarray(i * 3, i * 3 + 3), i * 3);
    c.set(tint(i)).toArray(base, i * 3);
    for (let k = 0; k < 4; k++) ph[i * 4 + k] = Math.random() * 6.283;
    sp[i] = 0.5 + Math.random() * 1.5;
  }
  p.update = (dt, t) => {
    for (let i = 0; i < n; i++) {
      const k = wave(Math.sin(t * sp[i] + ph[i * 4]));
      p.col[i * 3] = base[i * 3] * k;
      p.col[i * 3 + 1] = base[i * 3 + 1] * k;
      p.col[i * 3 + 2] = base[i * 3 + 2] * k;
      if (drift) {
        p.pos[i * 3] = home[i * 3] + Math.sin(t * 0.31 * sp[i] + ph[i * 4 + 1]) * drift;
        p.pos[i * 3 + 1] = home[i * 3 + 1] + Math.sin(t * 0.23 * sp[i] + ph[i * 4 + 2]) * drift * 0.6;
        p.pos[i * 3 + 2] = home[i * 3 + 2] + Math.cos(t * 0.27 * sp[i] + ph[i * 4 + 3]) * drift;
      }
    }
    p.geo.attributes.color.needsUpdate = true;
    if (drift) p.geo.attributes.position.needsUpdate = true;
  };
  return p;
}

const EFFECTS = {
  stars: () => glowing(900, 0.55, () => {
    const az = Math.random() * 6.283, el = Math.asin(0.08 + Math.random() * 0.9), r = 42 + Math.random() * 14;
    return [Math.cos(el) * Math.sin(az) * r, EYE + Math.sin(el) * r, -Math.cos(el) * Math.cos(az) * r];
  }, (i) => ["#ffffff", "#cfe0ff", "#fff1c9"][i % 3], {
    order: ORDER.stars, wave: (s) => 0.5 + 0.45 * s }),
  sparkles: () => glowing(380, 0.14, () => {
    const [x, z] = around(4.6, 6.6, 1.3);
    return [x, EYE - 2.6 + Math.random() * 5.6, z];
  }, () => "#ffd77a", { wave: (s) => Math.max(0, s) ** 10, drift: 0.08 }),
  fireflies: () => glowing(170, 0.17, () => {
    const [x, z] = around(3.2, 6.2, 1.25);
    return [x, 0.25 + Math.random() * 2.8, z];
  }, () => "#ffd25a", { wave: (s) => 0.12 + 0.88 * Math.max(0, s) ** 2, drift: 0.4 }),
  snow: () => falling(1400, 0.07, ["#ffffff", "#e8f0ff"], { additive: false, speed: [0.35, 0.8], sway: [0.15, 0.4] }),
  leaves: () => falling(340, 0.11, ["#d9692b", "#e8a33d", "#b8452a", "#8fae3e", "#c9a227"],
    { additive: false, speed: [0.3, 0.6], sway: [0.4, 0.9] }),
  rain() {
    const n = 700, top = 8, len = 0.32;
    const geo = new THREE.BufferGeometry(), pos = new Float32Array(n * 6), sp = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const [x, z] = around(0.6, 7);
      const y = Math.random() * top;
      pos.set([x, y + len, z, x + 0.03, y, z], i * 6);
      sp[i] = 8 + Math.random() * 3;
    }
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3).setUsage(THREE.DynamicDrawUsage));
    const obj = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({
      color: "#a9c4ff", transparent: true, opacity: 0, depthWrite: false, toneMapped: false }));
    obj.frustumCulled = false;
    obj.renderOrder = ORDER.particles;
    return { obj, base: 0.45, update(dt) {
      for (let i = 0; i < n; i++) {
        let y = pos[i * 6 + 4] - sp[i] * dt;
        if (y < -0.4) y += top;
        pos[i * 6 + 1] = y + len;
        pos[i * 6 + 4] = y;
      }
      geo.attributes.position.needsUpdate = true;
    } };
  },
};

export class Theatre {
  #viewer;
  #cb;
  #gen = 0;            // bumps on start()/stop(): work from an older session is dropped when it resumes
  #run = 0;            // bumps on every (re)start of the timeline: an older timeline exits at its next await
  #lineToken = 0;      // bumps whenever speech stops: late media/speech events from an old line are ignored
  #id = null;
  #state = null;
  #stateAt = 0;
  #autoplay = true;
  #phase = "stopped";
  #paused = false;
  #resumeTo = "playing";
  #skipRest = false;
  #waiters = new Set();
  #root = null;
  #ui = null;          // caption, page, buttons: travels with the viewer as they glide
  #buttons = null;     // in-world VR controls
  #backdrop = null;
  #world = null;
  #worldName = null;
  #oldWorlds = [];
  #worldReq = null;
  #pageReq = null;
  #canvas = null;      // the moment on screen: { k, pivot, mesh, mat, fade, life, dur, dir }
  #fading = [];
  #shotIndex = 0;
  #assets = new Map(); // shot index -> Promise<{ tex, grid, aspect } | null>, until it goes on screen
  #voices = new Map(); // line index -> Promise<object URL | url>
  #effects = new Map();
  #captionPanel = null;
  #pagePanel = null;
  #progressPanel = null;
  #cap = null;
  #current = null;     // the line being told: { line, words, marks, word, audio, speech, boundary, done, finish }
  #player = new Audio(); // one element for every line: never two voices at once, and phones unlock it once
  #utter = null;
  #unlocked = false;
  #time = 0;
  #clock = { value: 0 }; // seconds, shared by every living painting's shader
  #hooks;
  #media = new Map();  // shot index -> { actor: {p, value}, clip: {p, value, failed} }: the moment's upgrades
  #videos = [];        // two <video> elements, reused: a clip on screen and one fading out
  #flash = { next: 0, until: 0 };
  #amb = null;          // the world's sound: { ctx, master, wind, rain, water }, started inside a tap
  #door = null;         // the magic door the ready story waits behind
  #doorAsset = null;
  #veilMesh = null;     // a white veil on the eyes while stepping through
  #ambFor = null;       // the world sphere it was last tuned to

  constructor(viewer, callbacks = {}) {
    this.#viewer = viewer;
    this.#cb = callbacks;
    this.#player.preload = "auto";
    for (let i = 0; i < 2; i++) {
      const v = document.createElement("video");
      Object.assign(v, { muted: true, playsInline: true, preload: "auto", disablePictureInPicture: true });
      v.setAttribute("playsinline", "");
      v.setAttribute("muted", "");
      v.style.cssText = "position:absolute;width:1px;height:1px;opacity:0;pointer-events:none;left:0;top:0";
      v.owner = null;
      (viewer.container || document.body).appendChild(v);
      this.#videos.push(v);
    }
    this.#hooks = {
      onTick: (dt) => this.#tick(dt),
      onPrimary: () => this.nextLine(),      // A / X in VR
      onSecondary: () => this.togglePause(), // B / Y in VR
    };
    viewer.container.addEventListener("keydown", (e) => {
      if (!this.#id || e.altKey || e.ctrlKey || e.metaKey || e.target.closest("button, input, select, textarea, a")) return;
      if (e.code === "Space") { e.preventDefault(); if (!e.repeat) this.togglePause(); }
      else if (e.key === "ArrowRight") { e.preventDefault(); this.nextLine(); }
      else if (e.key === "r" || e.key === "R") { if (!e.repeat) this.replay(); }
    });
  }

  get id() { return this.#id; }
  get phase() { return this.#phase; }
  get state() { return this.#state; }

  /** Seconds since the page was uploaded, ticking between polls while the story is being made. */
  elapsed() {
    const st = this.#state;
    if (!st) return 0;
    if (st.status !== "running" && !(this.#filming(st) && this.#progressPanel)) return st.timings?.all ?? st.elapsed_s ?? 0;
    return (st.elapsed_s ?? 0) + (performance.now() - this.#stateAt) / 1000;
  }

  /** Follow a live story: show its progress, then play it as soon as it is playable (or, with autoplay false, show
   *  its first moment and wait for play()). `state` is the response that created it, if any. */
  start(id, { autoplay = true, state = null } = {}) {
    this.stop();
    const gen = this.#gen;
    this.#id = id;
    this.#autoplay = autoplay;
    this.#viewer.stage({ vfov: 70, minHfov: 76, move: 3 });
    this.#root = new THREE.Group();
    this.#root.name = "theatre";
    this.#viewer.scene.add(this.#root);
    this.#ui = new THREE.Group();
    this.#ui.name = "theatre-ui";
    this.#root.add(this.#ui);
    this.#buttons = this.#makeButtons();
    this.#ui.add(this.#buttons.group);
    this.#backdrop = backdrop();
    this.#root.add(this.#backdrop);
    this.#captionPanel = panel(0.4, 0.083, 1024, 213); // small, in the dock under its buttons
    this.#captionPanel.mesh.name = "theatre-caption";
    this.#captionPanel.mesh.position.set(0.01, -0.08, 0.002);
    this.#captionPanel.mesh.userData.dragTarget = this.#buttons.group;
    this.#captionPanel.mesh.visible = false;
    this.#buttons.group.add(this.#captionPanel.mesh);
    this.#viewer.draggables.push(this.#captionPanel.mesh);
    this.#progressPanel = panel(2.4, 1.41, 1024, 602); // big enough to read the scenes being filmed
    this.#progressPanel.mesh.name = "theatre-progress";
    this.#progressPanel.mesh.position.set(0, EYE + 0.05, -2.6);
    this.#ui.add(this.#progressPanel.mesh);
    Object.assign(this.#viewer, this.#hooks);
    this.#setPhase("loading");
    if (state) this.#accept(state);
    this.#drawProgress();
    this.#poll(gen);
    this.#prepare(gen);
  }

  /** Stop playing and polling, and free everything the theatre put in the scene. */
  stop() {
    this.#gen++;
    this.#run++;
    this.#abortLine();
    this.#paused = false;
    this.#wake();
    const a = this.#player;
    if (a.getAttribute("src")) { a.removeAttribute("src"); a.load(); }
    for (const p of this.#assets.values()) p.then((asset) => { asset?.tex.dispose(); asset?.motion?.dispose(); });
    this.#assets.clear();
    for (const p of this.#voices.values()) p.then((url) => url?.startsWith("blob:") && URL.revokeObjectURL(url));
    this.#voices.clear();
    for (const c of [this.#canvas, ...this.#fading]) if (c) this.#releaseClip(c);
    for (const k of [...this.#media.keys()]) this.#dropMedia(k);
    if (this.#amb) { this.#amb.ctx.close().catch(() => {}); this.#amb = null; this.#ambFor = null; }
    if (this.#door) {
      this.#viewer.buttons = this.#viewer.buttons.filter((b) => !this.#door.pressable.includes(b));
      if (this.#door.from) this.#viewer.dolly.position.copy(this.#door.from);
      this.#door = null;
    }
    this.#doorAsset = null;
    this.#veilOut = 0;
    if (this.#veilMesh) { this.#veilMesh.removeFromParent(); dispose(this.#veilMesh); this.#veilMesh = null; }
    if (this.#root) {
      this.#root.removeFromParent();
      dispose(this.#root);
    }
    if (this.#buttons) {
      this.#viewer.buttons = this.#viewer.buttons.filter((b) => !this.#buttons.list.includes(b));
      const mine = [this.#buttons.grip, this.#buttons.back, this.#captionPanel?.mesh];
      this.#viewer.draggables = this.#viewer.draggables.filter((d) => !mine.includes(d));
    }
    this.#root = this.#ui = this.#buttons = this.#backdrop = this.#world = this.#worldReq = this.#pageReq = this.#canvas = null;
    this.#worldName = null;
    this.#oldWorlds = [];
    this.#captionPanel = this.#pagePanel = this.#progressPanel = null;
    this.#fading = [];
    this.#effects.clear();
    this.#shotIndex = 0;
    for (const [k, fn] of Object.entries(this.#hooks)) if (this.#viewer[k] === fn) this.#viewer[k] = null;
    const was = this.#id;
    this.#id = this.#state = this.#cap = null;
    if (was) {
      this.#cb.onCaption?.(null);
      this.#setPhase("stopped");
    }
  }

  /** Call inside a tap/click: phones only let audio play that was first started by a gesture. */
  unlock() {
    this.#startAmbience();
    for (const v of this.#videos) { // a muted video started inside a tap can play later without one
      if (v.unlocked || v.owner) continue;
      v.unlocked = true;
      v.src = silence();
      v.play().then(() => v.pause(), () => {}).finally(() => { if (!v.owner) v.removeAttribute("src"); });
    }
    if (this.#unlocked || (this.#current && !this.#current.done)) return;
    const a = this.#player;
    a.src = silence();
    a.play().then(() => { this.#unlocked = true; }, () => {});
    try { window.speechSynthesis?.speak(new SpeechSynthesisUtterance("")); } catch { /* no speech engine */ }
  }

  pause() {
    if (!PLAYING.includes(this.#phase)) return;
    this.#resumeTo = this.#phase;
    this.#paused = true;
    const cur = this.#current;
    if (cur && !cur.done) {
      if (cur.audio) this.#player.pause();
      else if (cur.speech) this.#hush(cur); // speech can't pause reliably everywhere: the line restarts on resume
    }
    this.#setPhase("paused");
  }

  resume() {
    if (!this.#id) return;
    if (this.#phase === "door") { this.stepInside(); return; }
    if (this.#phase === "ready") { this.unlock(); this.#begin(); return; }
    if (this.#phase === "ended") { this.replay(); return; }
    if (!this.#paused) return;
    this.#paused = false;
    const cur = this.#current;
    if (cur && !cur.done) {
      if (cur.audio) this.#startAudio(cur, this.#lineToken);
      else this.#speak(cur);
    }
    this.#setPhase(this.#resumeTo);
    this.#wake();
  }

  togglePause() {
    if (this.#paused || ["ready", "ended", "door"].includes(this.#phase)) this.resume();
    else this.pause();
  }

  /** Skip the rest of the line being told (and the pause after it). */
  nextLine() {
    if (!["playing", "holding", "paused", "blocked"].includes(this.#phase)) return;
    if (this.#paused) {
      this.#paused = false;
      this.#setPhase(this.#resumeTo);
    }
    this.#skipRest = true;
    const cur = this.#current;
    if (cur && !cur.done) {
      this.#silence();
      cur.finish();
    }
    this.#wake();
  }

  /** Tell the story again from the first line, with what has already been made (nothing is regenerated). */
  replay() {
    if (!["ready", "playing", "holding", "paused", "blocked", "ended"].includes(this.#phase)) return;
    this.unlock();
    this.#begin();
  }

  // ---------- following the story as it is made ----------
  /** Still worth asking the server: the story is being made, or its moments are still being upgraded. */
  #busy(st) {
    return st.status === "running" || !!st.upgrading || !!st.world?.upgrading
      || (st.shots ?? []).some((s) => s.world?.upgrading || PENDING_CLIP.includes(s.clip?.status));
  }

  async #poll(gen) {
    let fails = 0;
    const t0 = performance.now();
    while (gen === this.#gen) {
      if (this.#state && (!this.#busy(this.#state) || performance.now() - t0 > POLL_MAX_MS)) return;
      const wait = fails ? Math.min(4000, POLL_MS * 2 ** fails) : this.#state?.status === "running" ? POLL_MS : SLOW_POLL_MS;
      if (this.#state || fails) await sleep(wait);
      if (gen !== this.#gen) return;
      try {
        const r = await fetch(`/api/live/${encodeURIComponent(this.#id)}`, { cache: "no-store" });
        if (gen !== this.#gen) return;
        if (r.status === 404) {
          this.#accept({ ...this.#state, status: "failed", error: "This story doesn't exist any more." });
          return;
        }
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const st = await r.json();
        if (gen !== this.#gen) return;
        fails = 0;
        this.#accept(st);
      } catch {
        if (gen !== this.#gen) return;
        if (++fails > 12) {
          this.#accept({ ...this.#state, status: "failed", error: "Lost contact with the server." });
          return;
        }
      }
    }
  }

  #accept(st) {
    const before = this.#state;
    this.#state = st;
    this.#stateAt = performance.now();
    this.#preload();
    if (before?.lines?.length !== st.lines?.length) this.#drawPage();
    this.#drawProgress();
    this.#cb.onState?.(st);
    this.#reconcile();
    this.#wake();
  }

  /** The world to stand in now: the playing moment's own world once it exists, else the story's world. */
  #wantedWorld(st) {
    const own = st.shots?.[this.#shotIndex]?.world;
    if (own?.image && (this.#canvas || this.#shotIndex === 0)) return own;
    return st.world?.ready && st.world.image ? st.world : null;
  }

  /** Start fetching whatever is ready and will be needed soon: the world, the page, the moment playing and the
   *  next one, and their voices. */
  #preload() {
    const st = this.#state, gen = this.#gen;
    if (!st || !this.#root) return;
    const world = this.#wantedWorld(st);
    if (world && world.image !== this.#worldName) {
      // The quick world first; each moment's own world, and its full repainting, replace it as they arrive.
      this.#worldName = world.image;
      const req = this.#loadWorld(world, gen);
      this.#worldReq ??= req;
    }
    if (!this.#pageReq && st.page?.image) this.#pageReq = this.#loadPage(st.page, gen);
    for (const k of [this.#shotIndex, this.#shotIndex + 1]) {
      if (this.#canvas?.k !== k) this.#shotAsset(k, gen);
      for (const li of st.shots?.[k]?.lines ?? []) this.#voice(li, gen);
      this.#actorAsset(k, gen);
      this.#clipAsset(k, gen);
      const next = st.shots?.[k]?.world;
      if (next?.image && k !== this.#shotIndex) loadImage(fileUrl(this.#id, next.image)).catch(() => {}); // warm the cache
    }
    for (const k of [...this.#media.keys()]) if (Math.abs(k - this.#shotIndex) > 1) this.#dropMedia(k);
  }

  // ---------- a moment's upgrades: the moving character (OpenCV) and the video clip (Luma Ray2) ----------
  #mediaOf(k) {
    if (!this.#media.has(k)) this.#media.set(k, {});
    return this.#media.get(k);
  }

  #dropMedia(k) {
    const m = this.#media.get(k);
    if (!m || [this.#canvas, ...this.#fading].some((c) => c?.k === k && (c.clip || c.actor))) return;
    m.clip?.p?.then((v) => v?.url && URL.revokeObjectURL(v.url));
    m.actor?.p?.then((v) => { v?.tex.dispose(); v?.plateTex.dispose(); });
    this.#media.delete(k);
  }

  #actorAsset(k, gen) {
    const shot = this.#state?.shots?.[k], m = this.#mediaOf(k);
    if (m.actor || !shot?.actor || !shot.plate || !Array.isArray(shot.actor_box) || !shot.action
        || shot.action.path === "still") return m.actor?.p;
    const id = this.#id;
    const p = Promise.all([loadImage(fileUrl(id, shot.actor)), loadImage(fileUrl(id, shot.plate))]).then(([a, b]) => {
      if (gen !== this.#gen) return null;
      const tex = new THREE.Texture(a), plateTex = new THREE.Texture(b);
      for (const t of [tex, plateTex]) {
        t.colorSpace = THREE.SRGBColorSpace;
        t.needsUpdate = true;
        this.#viewer.renderer.initTexture(t);
      }
      return { tex, plateTex, box: shot.actor_box, action: shot.action, facing: shot.actor_check?.facing ?? null };
    }).catch(() => null);
    m.actor = { p };
    p.then((v) => { m.actor.value = v; this.#reconcile(); });
    return p;
  }

  #clipAsset(k, gen) {
    const shot = this.#state?.shots?.[k], m = this.#mediaOf(k), c = shot?.clip;
    if (m.clip || c?.status !== "ready" || !c.video) return m.clip?.p;
    const url = fileUrl(this.#id, c.video);
    // Fetched whole (no range requests needed, and seeking to the loop point is instant).
    const get = () => fetch(url).then((r) => (r.ok ? r.blob() : Promise.reject(new Error(`HTTP ${r.status}`))));
    const p = get().catch(() => sleep(2000).then(get)).then((blob) => {
      if (gen !== this.#gen) return null;
      return { url: URL.createObjectURL(blob), loopAt: c.loop_at ?? null, hold: !!c.hold };
    }).catch(() => null);
    m.clip = { p };
    p.then((v) => { m.clip.value = v; this.#reconcile(); });
    return p;
  }

  /** The one place that decides what the moment on screen shows: its clip when one is ready (unless the moment is
   *  nearly over), else its moving character, else the living painting. */
  #reconcile() {
    const c = this.#canvas, st = this.#state;
    if (!c || !st || !this.#root) return;
    const shot = st.shots?.[c.k], m = this.#media.get(c.k);
    if (!shot) return;
    if (!c.clip && m?.clip?.value && !m.clip.failed && shot.clip?.status === "ready" && c.life / c.dur < LATE_CLIP) {
      this.#attachClip(c, m.clip.value, m.clip);
    } else if (!c.clip && !c.actor && m?.actor?.value && shot.clip?.status !== "ready") {
      this.#attachActor(c, m.actor.value);
    }
  }

  #order(c, base) {
    c.base = base;
    c.mesh.renderOrder = base;
    if (c.plate) c.plate.mesh.renderOrder = base + 0.05;
    if (c.actor) c.actor.sprite.renderOrder = base + 0.1;
    if (c.clip?.mesh) c.clip.mesh.renderOrder = base + 0.15;
  }

  #attachActor(c, a) {
    const W = c.imgW, H = c.imgH, path = a.action.path;
    const [x0, y0, x1, y1] = a.box;
    const plateMat = new THREE.MeshBasicMaterial({ map: a.plateTex, alphaMap: c.mat.alphaMap, transparent: true,
      opacity: 0, depthWrite: false, side: THREE.DoubleSide, toneMapped: false });
    if (c.motion) animate(plateMat, c.motion, this.#clock, c.wl); // the living sky stays on the plate
    const plateMesh = new THREE.Mesh(c.mesh.geometry, plateMat);
    plateMesh.name = "theatre-plate";
    plateMesh.userData.sharedGeometry = true;
    c.mesh.add(plateMesh);
    // The cut-out stands on the curved screen exactly where it was painted, a little in front of it.
    const arc = R * SPAN, Hm = arc / c.aspect, ra = R - 0.3, s = ra / R;
    const uc = (x0 + x1) / (2 * W), vb = y1 / H;
    const w = ((x1 - x0) / W) * arc * s, h = ((y1 - y0) / H) * Hm * s;
    const room = path === "rise" ? (y0 / H) * Hm * s * 0.9 : Math.min(0.9 * h, Math.max(0, (1 - vb - 0.08) * Hm * s));
    const geo = new THREE.PlaneGeometry(1, 1).translate(0, 0.5, 0); // origin at its feet: it topples about them
    const mat = new THREE.MeshBasicMaterial({ map: a.tex, transparent: true, opacity: 0, depthWrite: false,
      depthTest: false, side: THREE.DoubleSide, toneMapped: false, alphaTest: 0.02 });
    const sprite = new THREE.Mesh(geo, mat);
    sprite.name = "theatre-actor";
    sprite.scale.set(w, h, 1);
    const slot = new THREE.Group();
    slot.add(sprite);
    c.mesh.add(slot);
    if (c.grid) { // flatten the screen's relief where the character was painted: it now stands in front of it
      const relaxed = bentGeometry(c.aspect, relaxedGrid(c.grid, a.box, W, H), c.mesh.geometry.userData.mid);
      c.mesh.geometry.attributes.position.array.set(relaxed.attributes.position.array);
      c.mesh.geometry.attributes.position.needsUpdate = true;
      c.mesh.geometry.computeBoundingSphere();
      relaxed.dispose();
    }
    c.plate = { mesh: plateMesh, mat: plateMat, fade: c.fade < 0.05 ? 1 : 0 };
    c.actor = { slot, sprite, mat, fade: c.fade < 0.05 ? 1 : 0, path, loop: !!a.action.loop, t0: c.life + 0.3,
      side: 1 - x1 / W >= x0 / W ? 1 : -1, a0: (uc - 0.5) * SPAN, yb: (0.5 - vb) * Hm * s, w, h, ra, room,
      half: w / 2 / ra, floor: -Hm * s / 2 + 0.08 * Hm * s, extra: 1,
      // A character walking the other way from where it looks (OpenCV read its face) is mirrored, not walked backwards.
      mirror: (path === "walk-left" && a.facing === "right") || (path === "walk-right" && a.facing === "left") };
    this.#order(c, c.base);
    this.#placeActor(c);
    this.#cb.onClip?.(c.k, "actor");
  }

  #placeActor(c) {
    // Timed to the moment's spoken lines (the fall happens while "had a great fall" is read), finishing during
    // the crossfade to the next moment at the latest.
    const A = c.actor, t = Math.max(0, c.life - A.t0), T = Math.max(1.6, c.speak - 1 - A.t0);
    const p = pose(A.path, t, clamp01(t / T), { side: A.side, h: A.h, D: A.room, loop: A.loop, span: SPAN });
    const lim = SPAN / 2 - 0.06 * SPAN - A.half;
    const a = THREE.MathUtils.clamp(A.a0 + p.da + p.dx / A.ra, -lim, lim);
    A.slot.position.set(Math.sin(a) * A.ra, Math.max(A.floor, A.yb + p.dy), R - Math.cos(a) * A.ra);
    A.slot.rotation.y = -a;
    A.sprite.rotation.z = p.rot;
    A.sprite.scale.set(A.mirror ? -A.w : A.w, A.h * p.sy, 1);
    A.extra = p.extra;
  }

  #takeVideo(c) {
    let v = this.#videos.find((x) => !x.owner);
    if (!v) { // both busy: take the one under the oldest fading moment (its still shows for the rest of its fade)
      const old = this.#fading.find((f) => f.clip);
      if (old) this.#releaseClip(old);
      v = this.#videos.find((x) => !x.owner);
    }
    if (v) v.owner = c;
    return v;
  }

  async #attachClip(c, media, entry) {
    const v = this.#takeVideo(c);
    if (!v) return;
    const gen = this.#gen;
    c.clip = { video: v, state: "loading", mix: 0, loopAt: media.loopAt, hold: media.hold, shown: -1 };
    v.loop = !media.hold && media.loopAt == null;
    v.src = media.url;
    v.load();
    const ok = await new Promise((resolve) => {
      const done = (r) => { v.onloadeddata = v.onerror = null; resolve(r); };
      v.onloadeddata = () => done(true);
      v.onerror = () => done(false);
      setTimeout(() => done(v.readyState >= 2), 8000);
    });
    const live = gen === this.#gen && (this.#canvas === c || this.#fading.includes(c)) && c.clip?.video === v;
    if (!ok || !live) {
      if (!ok) entry.failed = true;
      if (c.clip?.video === v) this.#releaseClip(c);
      else { v.owner = null; v.removeAttribute("src"); v.load(); }
      if (!ok) this.#reconcile(); // the moving character (or the painting) carries on
      return;
    }
    const tex = new THREE.VideoTexture(v);
    tex.colorSpace = THREE.SRGBColorSpace;
    this.#viewer.renderer.initTexture(tex);
    const mat = new THREE.MeshBasicMaterial({ map: tex, alphaMap: c.mat.alphaMap, transparent: true, opacity: 0,
      depthWrite: false, side: THREE.DoubleSide, toneMapped: false });
    const mesh = new THREE.Mesh(c.mesh.geometry, mat);
    mesh.name = "theatre-clip";
    mesh.userData.sharedGeometry = true;
    c.mesh.add(mesh);
    Object.assign(c.clip, { tex, mat, mesh, state: "ready", mix: c.fade < 0.05 ? 1 : 0 });
    this.#order(c, c.base);
    this.#cb.onClip?.(c.k, "clip");
  }

  /** Each frame: play the clip while the story plays, loop or hold as the critic decided, and upload its frame. */
  #syncClip(c, live, dt) {
    const k = c.clip;
    if (!k || k.state === "loading") return;
    const v = k.video;
    if (live && (k.state === "ready" || k.state === "paused")) {
      k.state = "playing";
      v.play().catch((err) => { if (err?.name !== "AbortError" && c.clip === k) k.state = "paused"; });
    } else if (!live && k.state === "playing") {
      k.state = "paused";
      v.pause();
    }
    if (k.state === "playing") k.mix = Math.min(1, k.mix + dt / FADE_S);
    if (k.loopAt != null && v.currentTime >= k.loopAt - dt) v.currentTime = 0;
    if (k.hold && v.ended) k.state = "held";
    // VideoTexture marks itself dirty from requestVideoFrameCallback, which a headset's immersive session may
    // throttle: mark it from the render loop whenever the video has moved on.
    if (v.readyState >= 2 && v.currentTime !== k.shown) {
      k.shown = v.currentTime;
      k.tex.needsUpdate = true;
    }
  }

  #releaseClip(c) {
    const k = c.clip;
    if (!k) return;
    const v = k.video;
    v.pause();
    v.removeAttribute("src");
    v.load();
    v.owner = null;
    k.tex?.dispose();
    k.mat?.dispose();
    k.mesh?.removeFromParent();
    c.clip = null;
    this.#cb.onClip?.(c.k, c.actor ? "actor" : null);
  }

  /** Opacity of every layer of a moment: the picture, the plate behind the character, the character, the clip. */
  #shade(c) {
    const k = ease(c.fade), mix = c.clip?.mat ? ease(c.clip.mix) : 0, plate = c.plate ? ease(c.plate.fade) : 0;
    c.mat.opacity = k;
    c.mat.visible = !(mix >= 1 || plate >= 1);
    if (c.plate) {
      c.plate.mat.opacity = k * plate;
      c.plate.mat.visible = mix < 1;
    }
    if (c.actor) c.actor.mat.opacity = k * ease(c.actor.fade) * (1 - mix) * c.actor.extra;
    if (c.clip?.mat) c.clip.mat.opacity = k * mix;
  }

  /** A snapshot for tests and debugging. */
  debug() {
    const c = this.#canvas;
    return { phase: this.#phase, k: c?.k ?? null, life: c?.life ?? null, fade: c?.fade ?? null,
      progress: !!this.#progressPanel, filming: this.#filming(this.#state),
      door: this.#door ? { t: +this.#door.t.toFixed(2) } : null, veil: this.#veilMesh ? +this.#veilMesh.material.opacity.toFixed(2) : 0,
      dollyZ: +this.#viewer.dolly.position.z.toFixed(2),
      sound: this.#amb ? { state: this.#amb.ctx.state, master: +this.#amb.master.gain.value.toFixed(3),
        wind: +this.#amb.wind.gain.gain.value.toFixed(3), rain: +this.#amb.rain.gain.gain.value.toFixed(3),
        water: +this.#amb.water.gain.gain.value.toFixed(3) } : null,
      fading: this.#fading.length, world: this.#worldName,
      clip: c?.clip ? { state: c.clip.state, mix: c.clip.mix, t: c.clip.video.currentTime, paused: c.clip.video.paused,
        loopAt: c.clip.loopAt, hold: c.clip.hold, mapIsVideo: !!c.clip.tex?.isVideoTexture } : null,
      actor: c?.actor ? { path: c.actor.path, x: c.actor.slot.position.x, y: c.actor.slot.position.y,
        rot: c.actor.sprite.rotation.z, opacity: c.actor.mat.opacity } : null,
      fadingActors: this.#fading.filter((f) => f.actor).map((f) => ({ k: f.k, rot: f.actor.sprite.rotation.z,
        y: f.actor.slot.position.y, life: f.life })),
      videos: this.#videos.map((v) => ({ hasSrc: v.hasAttribute("src"), paused: v.paused })) };
  }

  #worldSettled(st) {
    const w = st.world;
    if (w?.ready && w.image) return "painted";
    if (w?.ready || st.status !== "running" || st.steps?.find((s) => s.name === "world")?.done) return "failed";
    return null;
  }

  /** The world is settled, the first moment is painted and all of its lines are voiced (or the story is as finished
   *  as it will get). */
  #playable(st) {
    const lines = st.lines ?? [], shots = st.shots ?? [];
    // Waiting for the film: "all" = only once the whole story is made (every moment painted, voiced, and filmed or
    // given up, so a failed clip still lets it start on the painting); "clip" = once the first moment is settled.
    if (st.play_when === "all") {
      return st.status !== "running" && shots.length > 0 && lines.length > 0
        && shots.every((s) => this.#clipSettled(st, s)) && lines.every((l) => l.ready);
    }
    if (st.play_when === "clip" && shots.length && !this.#clipSettled(st, shots[0])) return false;
    if (st.status !== "running") return lines.length > 0 && shots.length > 0;
    if (!this.#worldSettled(st)) return false;
    return !!shots[0]?.ready && shots[0].lines.every((i) => lines[i]?.ready);
  }

  /** A moment's video is decided: playing it (ready), or not getting one (rejected, off, or never painted). */
  #clipSettled(st, shot) {
    if (!shot) return st.status !== "running";
    if (shot.clip) return FINAL_CLIP.includes(shot.clip.status);
    return shot.ready && (shot.source === "opencv" || st.status !== "running");
  }

  /** The story is waiting for its film to be made (the progress panel shows the scenes being filmed). */
  #filming(st) {
    if (!st || !["clip", "all"].includes(st.play_when) || !st.shots?.length) return false;
    return st.play_when === "all" ? !this.#playable(st) : !this.#clipSettled(st, st.shots[0]);
  }

  /** How much of the story is made, 0..100 (100 only when it can start). */
  #percent(st) {
    if (!st) return 0;
    if (this.#playable(st) && !this.#filming(st)) return 100;
    const shots = st.shots ?? [], lines = st.lines ?? [], step = (n) => st.steps?.find((x) => x.name === n)?.done;
    const filmed = ["clip", "all"].includes(st.play_when);
    const parts = [[0.05, step("read") ? 1 : 0], [0.1, step("plan") ? 1 : 0], [0.05, st.world?.ready ? 1 : 0],
      [0.25, shots.length ? shots.filter((x) => x.ready).length / shots.length : 0],
      [0.1, lines.length ? lines.filter((x) => x.ready).length / lines.length : 0]];
    if (filmed) {
      const todo = shots.filter((x) => x.clip && x.clip.status !== "off");
      const done = shots.length ? shots.filter((x) => this.#clipSettled(st, x)).length / shots.length : 0;
      parts.push([0.45, todo.length || shots.length ? done : 0]);
    }
    const total = parts.reduce((a, [w]) => a + w, 0);
    return Math.min(99, Math.floor((100 * parts.reduce((a, [w, v]) => a + w * v, 0)) / total));
  }

  /** For the page: how far along the story is, and whether it is still being made. */
  progress() {
    const st = this.#state;
    return { percent: this.#percent(st), waiting: !!this.#progressPanel };
  }

  async #prepare(gen) {
    for (;;) {
      if (gen !== this.#gen) return;
      const st = this.#state;
      if (st && this.#playable(st)) break;
      if (st?.status === "failed") { this.#fail(st.error); return; }
      await this.#next();
    }
    // Wait for the pictures too (and the first clip, when the story waits for its film), so the first frame of
    // the story is complete.
    const wantClip = ["clip", "all"].includes(this.#state.play_when) && this.#state.shots?.[0]?.clip?.status === "ready";
    const [asset] = await Promise.all([this.#shotAsset(0, gen), this.#worldReq,
      wantClip ? Promise.race([this.#clipAsset(0, gen), sleep(15000)]) : null]);
    if (gen !== this.#gen) return;
    const st = this.#state;
    if (["clip", "all"].includes(st.play_when) && this.#progressPanel) {
      this.#drawProgress(); // 100%: show it, then the door appears
      this.#cb.onState?.(st);
      await sleep(1200);
      if (gen !== this.#gen) return;
    }
    this.#removeProgress();
    // The story is ready and waits behind a door: it starts only when the viewer chooses to step inside.
    this.#doorAsset = { asset, wantClip, gen };
    this.#door = this.#makeDoor();
    this.#root.add(this.#door.group);
    this.#setCaption({ key: "door", title: st.title || "Your story", sub: "Your story is ready. Step inside." });
    this.#setPhase("door");
  }

  /** Open the door and carry the viewer through it into the story (no walking: the view glides). */
  stepInside() {
    const d = this.#door;
    if (this.#phase !== "door" || !d || d.t >= 0) return;
    this.unlock();
    d.t = 0;
    d.from = this.#viewer.dolly.position.clone();
    this.#setCaption(null);
    this.#setPhase("entering");
  }

  /** Behind the door: the first moment, its clip if there is one, then the story. */
  async #enterStory() {
    const { asset, wantClip, gen } = this.#doorAsset ?? {};
    if (gen !== this.#gen) return;
    const st = this.#state;
    this.#showShot(0, st.shots[0], asset);
    if (wantClip) { // start on the moving picture, not the still: wait for the video's first frame
      const t0 = performance.now();
      while (gen === this.#gen && performance.now() - t0 < 8000) {
        const clip = this.#canvas?.clip;
        if ((clip && clip.state !== "loading") || this.#media.get(0)?.clip?.failed) break;
        if (!clip) this.#reconcile();
        await sleep(50);
      }
      if (gen !== this.#gen) return;
    }
    this.#begin();
  }

  #shotAsset(k, gen) {
    if (this.#assets.has(k)) return this.#assets.get(k);
    const shot = this.#state?.shots?.[k];
    if (!shot?.ready || !shot.image) return null;
    const p = this.#loadShot(shot, gen).catch(() => null); // null: told over the previous picture
    this.#assets.set(k, p);
    return p;
  }

  async #loadShot(shot, gen) {
    const id = this.#id;
    const get = (name) => loadImage(fileUrl(id, name)).catch(() => sleep(600).then(() => loadImage(fileUrl(id, name))));
    const [img, depth, moving] = await Promise.all([get(shot.image), shot.depth ? get(shot.depth).catch(() => null) : null,
      shot.motion ? get(shot.motion).catch(() => null) : null]);
    if (gen !== this.#gen) return null;
    const tex = new THREE.Texture(img);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.anisotropy = Math.min(8, this.#viewer.renderer.capabilities.getMaxAnisotropy());
    tex.needsUpdate = true;
    this.#viewer.renderer.initTexture(tex); // upload now, not in the first frame of the crossfade
    let motion = null;
    if (moving) {
      motion = new THREE.Texture(moving); // data, not colour: no colour-space conversion
      motion.needsUpdate = true;
      this.#viewer.renderer.initTexture(motion);
    }
    return { tex, motion, grid: depth ? depthGrid(depth) : null, aspect: shot.aspect || img.naturalWidth / img.naturalHeight };
  }

  #voice(li, gen) {
    if (this.#voices.has(li)) return this.#voices.get(li);
    const line = this.#state?.lines?.[li];
    if (!line?.ready || !line.audio) return null;
    const url = fileUrl(this.#id, line.audio);
    // Fetched whole ahead of time, so a line starts the moment the previous one ends.
    const p = fetch(url).then((r) => (r.ok ? r.blob() : null)).then((blob) => {
      if (gen !== this.#gen) return null;
      return blob ? URL.createObjectURL(blob) : url;
    }, () => url);
    this.#voices.set(li, p);
    return p;
  }

  async #loadWorld(world, gen) {
    try {
      const id = this.#id, opt = (name) => name ? loadImage(fileUrl(id, name)).catch(() => null) : null;
      const [img, depth, moving] = await Promise.all([loadImage(fileUrl(id, world.image)), opt(world.depth), opt(world.motion)]);
      if (gen !== this.#gen || !this.#root) return;
      const tex = new THREE.Texture(img);
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.anisotropy = Math.min(8, this.#viewer.renderer.capabilities.getMaxAnisotropy());
      tex.needsUpdate = true;
      this.#viewer.renderer.initTexture(tex);
      const mat = new THREE.MeshBasicMaterial({ map: tex, toneMapped: false, transparent: true, opacity: 0,
        depthWrite: false, depthTest: false });
      if (moving) {
        const motion = new THREE.Texture(moving);
        motion.needsUpdate = true;
        animate(mat, motion, this.#clock, world.life); // the world's sky drifts with its wind, its leaves sway
      }
      const sphere = new THREE.Mesh(worldGeometry(depth), mat);
      sphere.name = "theatre-world";
      sphere.position.y = EYE;
      sphere.rotation.y = -Math.PI / 2; // image centre straight ahead (-Z)
      const life = worldLife(world.life, worldRadius(depth));
      sphere.add(life.group);
      sphere.userData.life = life;
      sphere.userData.lifeInfo = world.life ?? null;
      sphere.userData.moving = world.moving ?? null;
      sphere.renderOrder = ORDER.world + (this.#world ? 0.5 : 0); // a better world fades in over the quick one
      sphere.userData.fade = 0;
      this.#root.add(sphere);
      if (this.#world) {
        this.#world.renderOrder = ORDER.world;
        this.#oldWorlds.push(this.#world);
      }
      this.#world = sphere;
    } catch { /* the gradient backdrop (or the quick world) stays */ }
  }

  /** Replay, Pause/Play, Next and Exit as buttons inside the world, for a headset; pressed by pointing a controller
   *  and pulling the trigger, by a pinch, or by a click. Shown only in VR (the page has its own buttons). */
  /** The VR dock: a small panel close to the viewer, low and to the left so it never covers the story, holding
   *  Replay / Play-Pause / Next / Exit and the caption. Grab its handle (or the caption) with the trigger or a pinch
   *  and move it anywhere; it keeps facing you. */
  #makeButtons() {
    const group = new THREE.Group();
    group.name = "theatre-dock";
    group.position.set(-0.6, EYE - 0.36, -0.58); // ~45° to the left and below eye level: never in front of the story
    group.lookAt(0, EYE, 0);
    // The whole dock's background is a handle too, so it is easy to grab and move anywhere.
    const back = panel(0.5, 0.2, 500, 200);
    back.ctx.fillStyle = "rgba(9, 12, 24, 0.55)";
    rounded(back.ctx, 2, 2, 496, 196, 28);
    back.ctx.fill();
    back.ctx.strokeStyle = "rgba(255,255,255,0.18)";
    back.ctx.lineWidth = 3;
    back.ctx.stroke();
    back.tex.needsUpdate = true;
    back.mesh.name = "theatre-dock-back";
    back.mesh.position.set(0.0, -0.035, -0.004);
    back.mesh.userData.dragTarget = group;
    back.mesh.renderOrder = ORDER.panel - 0.01;
    group.add(back.mesh);
    const grip = panel(0.07, 0.07, 128, 128);
    grip.ctx.fillStyle = "rgba(255, 211, 107, 0.22)";
    rounded(grip.ctx, 4, 4, 120, 120, 30);
    grip.ctx.fill();
    grip.ctx.fillStyle = INK;
    grip.ctx.font = `700 78px ${FONT}`;
    grip.ctx.textAlign = "center";
    grip.ctx.textBaseline = "middle";
    grip.ctx.fillText("⠿", 64, 66);
    grip.tex.needsUpdate = true;
    grip.mesh.name = "theatre-dock-grip";
    grip.mesh.position.set(-0.2, 0, 0.002);
    grip.mesh.userData.dragTarget = group;
    group.add(grip.mesh);
    const specs = [["replay", "↺", () => this.replay()], ["play", "❚❚", () => { this.unlock(); this.togglePause(); }],
      ["next", "⏭", () => this.nextLine()], ["exit", "✕", () => this.#cb.onExit?.()]];
    const list = [], byName = {};
    specs.forEach(([name, label, onPress], i) => {
      const b = panel(0.06, 0.05, 144, 120);
      b.mesh.position.set(-0.115 + i * 0.075, 0, 0.002);
      b.mesh.userData.onPress = onPress;
      b.mesh.name = `theatre-button-${name}`;
      b.label = null;
      b.draw = (text) => {
        if (b.label === text) return;
        b.label = text;
        const { ctx, canvas: { width: W, height: H } } = b;
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = name === "play" ? GOLD : "rgba(9, 12, 24, 0.82)";
        rounded(ctx, 3, 3, W - 6, H - 6, 40);
        ctx.fill();
        ctx.strokeStyle = "rgba(255,255,255,0.35)";
        ctx.lineWidth = 3;
        ctx.stroke();
        ctx.fillStyle = name === "play" ? "#241a05" : INK;
        ctx.font = `700 64px ${FONT}`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(text, W / 2, H / 2 + 3);
        b.tex.needsUpdate = true;
      };
      b.draw(label);
      group.add(b.mesh);
      list.push(b.mesh);
      byName[name] = b;
    });
    this.#viewer.buttons.push(...list);
    this.#viewer.draggables.push(grip.mesh, back.mesh);
    return { group, list, byName, grip: grip.mesh, back: back.mesh };
  }

  async #loadPage(page, gen) {
    try {
      const img = await loadImage(fileUrl(this.#id, page.image));
      if (gen !== this.#gen) return;
      const w = page.w || img.naturalWidth, h = page.h || img.naturalHeight;
      const s = Math.min(1.5, 1024 / Math.max(w, h)); // canvas pixels per page pixel
      const p = panel(PAGE_H * (w / h), PAGE_H, Math.round(w * s), Math.round(h * s));
      p.img = img;
      p.s = s;
      p.mesh.name = "theatre-page";
      // To the left-front, turned to face the viewer.
      p.mesh.position.set(-Math.sin(PAGE_YAW) * PAGE_DIST, EYE - 0.15, -Math.cos(PAGE_YAW) * PAGE_DIST);
      p.mesh.rotation.y = PAGE_YAW;
      this.#ui.add(p.mesh);
      this.#pagePanel = p;
      this.#drawPage();
    } catch { /* no page panel */ }
  }

  // ---------- the timeline ----------
  #begin() {
    const run = ++this.#run;
    this.#abortLine();
    this.#paused = false;
    this.#shotIndex = 0;
    this.#wake();
    this.#timeline(this.#gen, run).catch((err) => console.warn("theatre:", err));
  }

  async #timeline(gen, run) {
    const alive = () => gen === this.#gen && run === this.#run;
    this.#setPhase("playing");
    for (let k = 0; ; k++) {
      const got = await this.#awaitShot(k, alive);
      if (!alive()) return;
      if (!got) break;
      await this.#unpaused(alive);
      if (!alive()) return;
      this.#shotIndex = k;
      this.#showShot(k, got.shot, got.asset);
      this.#setPhase(this.#paused ? "paused" : "playing");
      this.#preload();
      for (const li of got.shot.lines) {
        const line = await this.#awaitLine(li, alive);
        if (!alive()) return;
        if (!line) continue;
        await this.#unpaused(alive);
        if (!alive()) return;
        await this.#say(line, gen, alive);
        if (!alive()) return;
        await this.#rest(GAP_MS, alive);
        if (!alive()) return;
      }
    }
    this.#current = null;
    this.#setCaption({ key: "end", title: "The end", sub: "Replay to watch it again" });
    this.#drawPage();
    this.#setPhase("ended");
  }

  /** Moment k once it is painted and loaded; null when the story has no moment k. Holds the current picture with a
   *  note while the next one is still being painted. */
  async #awaitShot(k, alive) {
    let holding = false;
    for (;;) {
      if (!alive()) return null;
      const st = this.#state, shot = st.shots?.[k], over = st.status !== "running";
      if (shot && this.#canvas?.k === k) return { shot, asset: null };
      if (shot?.ready && shot.image) {
        const asset = await this.#shotAsset(k, this.#gen);
        return alive() ? { shot, asset } : null;
      }
      if (!shot && (over || st.steps?.find((s) => s.name === "plan")?.done)) return null;
      if (shot && over) return { shot, asset: null }; // never painted: told over the last picture
      if (!holding && k > 0) {
        holding = true;
        this.#current = null;
        this.#setCaption({ key: `hold${k}`, sub: "Painting the next moment…" });
        this.#setPhase(this.#paused ? "paused" : "holding");
        if (this.#paused) this.#resumeTo = "holding";
      }
      await this.#next();
    }
  }

  async #awaitLine(li, alive) {
    for (;;) {
      if (!alive()) return null;
      const st = this.#state, line = st.lines?.[li];
      if (line?.ready) return line;
      if (st.status !== "running") return line ?? null; // never voiced: the browser reads it
      await this.#next();
    }
  }

  async #unpaused(alive) {
    while (this.#paused && alive()) await this.#next();
  }

  /** A pause that stops counting while paused and ends early on nextLine(). */
  async #rest(ms, alive) {
    this.#skipRest = false;
    let left = ms, last = performance.now();
    while (left > 0 && alive() && !this.#skipRest) {
      await this.#next(Math.min(left, 60));
      const now = performance.now();
      if (!this.#paused) left -= now - last;
      last = now;
    }
  }

  #next(timeout = 0) {
    return new Promise((resolve) => {
      const done = () => { this.#waiters.delete(done); clearTimeout(timer); resolve(); };
      const timer = timeout ? setTimeout(done, timeout) : 0;
      this.#waiters.add(done);
    });
  }

  #wake() {
    for (const w of [...this.#waiters]) w();
  }

  // ---------- telling a line ----------
  async #say(line, gen, alive) {
    const src = line.audio ? await this.#voice(line.i, gen) : null;
    if (!alive()) return;
    this.#silence();
    const token = this.#lineToken;
    const words = tokenize(line.text);
    const cur = { line, words, marks: line.marks?.length ? line.marks : null, word: -1, audio: !!src,
      speech: null, boundary: -1, done: false, heard: 0, stall: performance.now() };
    this.#current = cur;
    this.#cap = { key: `line${line.i}.${this.#run}`, speaker: line.speaker || "Narrator", words, word: -1 };
    this.#renderCaption();
    this.#drawPage();
    this.#cb.onLine?.(line.i, this.#shotIndex);
    await new Promise((resolve) => {
      cur.finish = () => { if (!cur.done) { cur.done = true; resolve(); } };
      if (src) this.#playAudio(src, token, cur);
      else this.#speak(cur);
    });
  }

  #playAudio(src, token, cur) {
    const a = this.#player;
    a.onended = () => { if (token === this.#lineToken) cur.finish(); };
    a.onerror = () => {
      if (token !== this.#lineToken || cur.done) return;
      cur.audio = false;
      this.#speak(cur);
    };
    a.src = src;
    if (!this.#paused) this.#startAudio(cur, token);
  }

  #startAudio(cur, token) {
    this.#player.play().then(() => { this.#unlocked = true; }, (err) => {
      if (token !== this.#lineToken || cur.done) return;
      if (err?.name === "NotAllowedError") {
        // No tap yet (e.g. opened from a link): wait for Play.
        this.#resumeTo = PLAYING.includes(this.#phase) ? this.#phase : "playing";
        this.#paused = true;
        this.#setPhase("blocked");
      } else if (err?.name !== "AbortError") {
        cur.audio = false;
        this.#speak(cur);
      }
    });
  }

  /** The browser's own voice, for lines without audio (or whose audio failed). */
  #speak(cur) {
    cur.speech = { started: 0, est: Math.max(1500, cur.line.text.length * 75), mute: false };
    cur.boundary = -1;
    const synth = window.speechSynthesis;
    if (!synth || typeof SpeechSynthesisUtterance === "undefined") {
      cur.speech.mute = true; // no speech engine: the caption stays up for a reading time
      if (!this.#paused) cur.speech.started = performance.now();
      return;
    }
    const u = new SpeechSynthesisUtterance(cur.line.text);
    u.lang = "en-GB";
    u.rate = 0.95;
    u.onboundary = (e) => { if (this.#utter === u) cur.boundary = e.charIndex; };
    u.onend = u.onerror = () => {
      if (this.#utter !== u) return;
      this.#utter = null;
      // Ended at once: there is no voice to speak with. Keep the line up for a reading time instead.
      if (performance.now() - cur.speech.started < cur.speech.est * 0.4) cur.speech.mute = true;
      else cur.finish();
    };
    this.#utter = u;
    if (this.#paused) return;
    synth.cancel();
    synth.speak(u);
    cur.speech.started = performance.now();
  }

  #hush(cur) {
    this.#utter = null;
    window.speechSynthesis?.cancel();
    cur.speech.started = 0;
    cur.boundary = -1;
  }

  #silence() {
    this.#lineToken++;
    const a = this.#player;
    a.onended = a.onerror = null;
    if (!a.paused) a.pause();
    if (this.#utter) {
      this.#utter = null;
      window.speechSynthesis?.cancel();
    }
  }

  /** Stop the line being told and release the timeline waiting on it (which then sees it was superseded). */
  #abortLine() {
    this.#silence();
    const cur = this.#current;
    this.#current = null;
    cur?.finish?.();
  }

  // ---------- pictures and particles ----------
  #showShot(k, shot, asset) {
    this.#shotIndex = k;
    this.#setEffects(shot.effects);
    this.#cb.onShot?.(k);
    const same = this.#canvas?.k === k;
    if (same) { // told again (replay): the clip and the character start over
      const c = this.#canvas;
      c.life = 0;
      if (c.clip?.video) { c.clip.video.currentTime = 0; if (c.clip.state === "held") c.clip.state = "ready"; }
      if (c.actor) c.actor.t0 = 0.3;
    }
    if (!asset || same || !this.#root) { this.#reconcile(); return; }
    this.#assets.delete(k); // the canvas owns the texture now; it is disposed when the canvas is retired
    const st = this.#state;
    const wl = shot.world?.life ?? st.world?.life ?? null;
    const mat = new THREE.MeshBasicMaterial({
      map: asset.tex, alphaMap: featherMap(), transparent: true, opacity: 0, depthWrite: false,
      side: THREE.DoubleSide, toneMapped: false });
    if (asset.motion) animate(mat, asset.motion, this.#clock, wl);
    const mesh = new THREE.Mesh(bentGeometry(asset.aspect, asset.grid), mat);
    mesh.name = "theatre-shot";
    mesh.position.z = -R;
    const pivot = new THREE.Group();
    pivot.position.y = EYE;
    pivot.add(mesh);
    this.#root.add(pivot);
    const secs = shot.lines.reduce((t, li) => {
      const l = st.lines?.[li];
      return t + (l?.duration ?? (l ? l.text.length * 0.075 : 2.5)) + GAP_MS / 1000;
    }, 1);
    if (this.#canvas) this.#fading.push(this.#canvas);
    for (const f of this.#fading) this.#order(f, ORDER.canvas);
    const img = asset.tex.image;
    this.#canvas = { k, pivot, mesh, mat, fade: 0, life: 0, dur: Math.max(4, secs), speak: secs, dir: k % 2 ? -1 : 1,
      motion: asset.motion, aspect: asset.aspect, grid: asset.grid, imgW: img.naturalWidth || img.width,
      imgH: img.naturalHeight || img.height, wl, plate: null, actor: null, clip: null, base: 0 };
    this.#order(this.#canvas, ORDER.canvas + 0.5); // the incoming picture draws over the outgoing one
    this.#preload();
    this.#reconcile();
  }

  #setEffects(names) {
    const st = this.#state;
    const weather = [...(WEATHER_FX[st?.weather] ?? []), ...(st?.time === "night" ? ["stars"] : [])];
    const want = new Set([...(names ?? []), ...weather].filter((n) => Object.hasOwn(EFFECTS, n)));
    for (const n of want) {
      if (this.#effects.has(n) || !this.#root) continue;
      const fx = EFFECTS[n]();
      fx.fade = 0;
      this.#effects.set(n, fx);
      this.#root.add(fx.obj);
    }
    for (const [n, fx] of this.#effects) fx.target = want.has(n) ? 1 : 0;
  }

  #tick(dt) {
    if (!this.#root) return;
    const t = (this.#time += dt);
    this.#clock.value = t;
    const d = this.#viewer.dolly.position;
    this.#ui.position.set(d.x, 0, d.z); // captions and buttons travel with the viewer
    this.#ui.rotation.y = this.#viewer.dolly.rotation.y; // and turn with them
    const vr = this.#viewer.renderer.xr.isPresenting;
    this.#buttons.group.visible = vr;
    for (const b of this.#buttons.list) {
      b.visible = (!this.#progressPanel || b.name === "theatre-button-exit")
        && !(b.name === "theatre-button-play" && this.#phase === "ended"); // at the end only Replay (nothing is remade)
    }
    if (this.#captionPanel) this.#captionPanel.mesh.visible = vr && !!this.#cap; // on a screen the page shows its own caption
    if (vr) this.#buttons.byName.play.draw(["paused", "ready", "blocked", "ended", "door"].includes(this.#phase) ? "▶" : "❚❚");
    const w = this.#world;
    if (w && w.userData.fade < 1) {
      w.userData.fade = Math.min(1, w.userData.fade + dt / (this.#oldWorlds.length ? 2.5 : 1.2));
      w.material.opacity = ease(w.userData.fade);
      if (w.userData.fade === 1) {
        // Fully in: draw it opaque, and drop what it now hides (the backdrop, an older world).
        w.material.transparent = false;
        w.material.needsUpdate = true;
        for (const old of [this.#backdrop, ...this.#oldWorlds]) {
          if (!old) continue;
          old.removeFromParent();
          dispose(old);
        }
        this.#backdrop = null;
        this.#oldWorlds = [];
      }
    }
    const live = PLAYING.includes(this.#phase);
    const c = this.#canvas;
    if (c) {
      if (c.fade < 1) c.fade = Math.min(1, c.fade + dt / FADE_S);
      if (live) c.life += dt;
      // Gentle life: a slow push-in and a small pan, the camera itself stays still (VR comfort).
      const p = ease(clamp01(c.life / c.dur));
      c.mesh.scale.setScalar(1 + 0.05 * p);
      c.pivot.rotation.y = c.dir * DRIFT_YAW * p;
      this.#layers(c, live, dt);
    }
    for (const f of [...this.#fading]) {
      f.fade = Math.max(0, f.fade - dt / FADE_S);
      if (live) f.life += dt;
      this.#layers(f, live, dt);
      if (f.fade === 0) {
        this.#releaseClip(f);
        f.pivot.removeFromParent();
        dispose(f.pivot);
        this.#fading.splice(this.#fading.indexOf(f), 1);
      }
    }
    this.#worldAlive(dt, t);
    this.#doorTick(dt, t);
    for (const fx of this.#effects.values()) {
      fx.fade += Math.sign(fx.target - fx.fade) * Math.min(Math.abs(fx.target - fx.fade), dt / FADE_S);
      fx.obj.visible = fx.fade > 0.001;
      if (!fx.obj.visible) continue;
      fx.obj.material.opacity = fx.base * ease(fx.fade);
      fx.update(dt, t);
    }
    this.#follow();
    if (this.#progressPanel && this.elapsed().toFixed(1) !== this.#progressPanel.shown) this.#drawProgress();
  }

  #layers(c, live, dt) {
    if (c.plate && c.plate.fade < 1) c.plate.fade = Math.min(1, c.plate.fade + dt / 0.5);
    if (c.actor) {
      if (c.actor.fade < 1) c.actor.fade = Math.min(1, c.actor.fade + dt / 0.5);
      this.#placeActor(c);
    }
    this.#syncClip(c, live, dt);
    this.#shade(c);
  }

  /** The world's own life, each frame: lamps flicker, rays turn slowly, a storm flashes now and then; all of it
   *  fades with its world. */
  #worldAlive(dt, t) {
    const w = this.#world;
    this.#tuneAmbience(w);
    for (const sphere of [w, ...this.#oldWorlds]) {
      const L = sphere?.userData.life;
      if (!L) continue;
      const k = sphere === w ? ease(sphere.userData.fade) : 1 - ease(w?.userData.fade ?? 0);
      L.group.traverse((o) => { if (o.material && o.userData.base != null) o.material.opacity = o.userData.base * k; });
      for (const lamp of L.lamps) {
        const s = lamp.userData.seed;
        lamp.material.opacity = lamp.userData.base * k * (0.6 + 0.25 * Math.sin(t * 7 + s) + 0.15 * Math.sin(t * 13.7 + s * 2));
      }
      L.rays.forEach((ray, i) => { ray.material.rotation = (i / L.rays.length) * Math.PI + t * 0.02; });
    }
    if (!w) return;
    if (this.#state?.weather === "storm") {
      if (t > this.#flash.next) {
        this.#flash.until = t + 0.08;
        this.#flash.next = t + 7 + Math.random() * 8;
      }
      w.material.color.setScalar(t < this.#flash.until ? 1.8 : 1);
    }
  }

  /** The world's sound, made in the browser from noise (no files): wind shaped by the wind OpenCV read from the
   *  painting, rain when it rains, running water when the world has water. Quiet, under the voices. */
  #startAmbience() {
    if (this.#amb || !this.#root) return;
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return;
    let ctx;
    try { ctx = new AC(); } catch { return; }
    const buf = ctx.createBuffer(1, ctx.sampleRate * 2, ctx.sampleRate);
    const data = buf.getChannelData(0);
    let b = 0;
    for (let i = 0; i < data.length; i++) { b = 0.97 * b + 0.03 * (Math.random() * 2 - 1); data[i] = b * 6; } // soft noise
    const master = ctx.createGain();
    master.gain.value = 0;
    master.connect(ctx.destination);
    const voice = (type, freq, q = 0.7) => {
      const src = ctx.createBufferSource();
      src.buffer = buf;
      src.loop = true;
      src.playbackRate.value = 0.8 + Math.random() * 0.4;
      const filter = ctx.createBiquadFilter();
      filter.type = type;
      filter.frequency.value = freq;
      filter.Q.value = q;
      const gain = ctx.createGain();
      gain.gain.value = 0;
      src.connect(filter).connect(gain).connect(master);
      src.start();
      return { src, filter, gain };
    };
    const wind = voice("bandpass", 420, 0.5), rain = voice("highpass", 2200), water = voice("lowpass", 380);
    const lfo = ctx.createOscillator(), depth = ctx.createGain(); // gusts: the wind's colour drifts slowly
    lfo.frequency.value = 0.09;
    depth.gain.value = 160;
    lfo.connect(depth).connect(wind.filter.frequency);
    lfo.start();
    this.#amb = { ctx, master, wind, rain, water };
    this.#ambFor = null;
    ctx.resume?.().catch(() => {});
  }

  #tuneAmbience(world) {
    const A = this.#amb;
    if (!A) return;
    const playing = PLAYING.includes(this.#phase) || this.#phase === "ready";
    const now = A.ctx.currentTime;
    A.master.gain.setTargetAtTime(playing ? 0.15 : 0.04, now, 0.6);
    if (!world || world === this.#ambFor) return;
    this.#ambFor = world;
    const life = world.userData.lifeInfo ?? {}, moving = world.userData.moving ?? {}, st = this.#state ?? {};
    const weather = life.weather ?? st.weather, strength = life.wind?.strength ?? 0.4;
    A.wind.gain.gain.setTargetAtTime(0.25 + 0.75 * strength, now, 1.5);
    A.rain.gain.gain.setTargetAtTime(weather === "rain" || weather === "storm" ? 0.7 : 0, now, 1.5);
    A.water.gain.gain.setTargetAtTime((moving.water ?? 0) > 0.02 ? 0.8 : 0, now, 1.5);
  }

  /** A golden arched door with two star-carved leaves and warm light behind it, 2.4 m ahead. Pressable (door,
   *  sign) in VR; the page has its own "Step inside" button. */
  #makeDoor() {
    const group = new THREE.Group();
    group.name = "theatre-door";
    group.position.set(0, 0, -2.4);
    const W = 1.3, H = 2.25, top = 0.32; // opening; the top is an arch
    const arch = (ctx, x, y, w, h, r) => {
      ctx.beginPath();
      ctx.moveTo(x, y + h);
      ctx.lineTo(x, y + r);
      ctx.quadraticCurveTo(x, y, x + w / 2, y);
      ctx.quadraticCurveTo(x + w, y, x + w, y + r);
      ctx.lineTo(x + w, y + h);
      ctx.closePath();
    };
    // the light behind the doors
    const light = panel(W, H, 260, 450);
    {
      const { ctx } = light, g = ctx.createRadialGradient(130, 250, 10, 130, 250, 260);
      g.addColorStop(0, "rgba(255,255,240,1)");
      g.addColorStop(0.35, "rgba(255,226,150,0.95)");
      g.addColorStop(1, "rgba(255,180,80,0.0)");
      ctx.fillStyle = g;
      arch(ctx, 0, 0, 260, 450, 70);
      ctx.fill();
      light.tex.needsUpdate = true;
    }
    light.mesh.material.blending = THREE.AdditiveBlending;
    light.mesh.material.opacity = 0.35;
    light.mesh.position.set(0, H / 2, -0.04);
    light.mesh.renderOrder = ORDER.panel - 0.03;
    group.add(light.mesh);
    // the frame
    const frame = panel(W + 0.34, H + 0.22, 520, 860);
    {
      const { ctx } = frame, s = 520 / (W + 0.34), x0 = 0.17 * s, w = W * s, h = H * s, y0 = 0.22 * s;
      const gold = ctx.createLinearGradient(0, 0, 520, 860);
      gold.addColorStop(0, "#fff1b8");
      gold.addColorStop(0.5, "#d9a43c");
      gold.addColorStop(1, "#8a5a17");
      ctx.lineJoin = "round";
      ctx.strokeStyle = gold;
      ctx.lineWidth = 34;
      arch(ctx, x0, y0, w, h + 40, top * s);
      ctx.stroke();
      ctx.lineWidth = 6;
      ctx.strokeStyle = "rgba(255,245,210,0.9)";
      arch(ctx, x0 - 22, y0 - 22, w + 44, h + 62, top * s + 22);
      ctx.stroke();
      ctx.fillStyle = "#ffe9a8";
      for (let i = 0; i < 9; i++) { // little stars along the arch
        const a = Math.PI * (0.08 + 0.84 * i / 8), cx = 260 - Math.cos(a) * (w / 2 + 34), cy = y0 + 40 - Math.sin(a) * (top * s + 18);
        star(ctx, cx, cy, 10, 4.5);
      }
      frame.tex.needsUpdate = true;
    }
    frame.mesh.position.set(0, (H + 0.22) / 2 - 0.02, 0.01);
    frame.mesh.renderOrder = ORDER.panel - 0.01;
    group.add(frame.mesh);
    // two leaves on hinges
    const leaf = (side) => {
      const p = panel(W / 2, H, 260, 900);
      const { ctx } = p;
      const wood = ctx.createLinearGradient(0, 0, 260, 900);
      wood.addColorStop(0, "#2b2a6b");
      wood.addColorStop(1, "#151436");
      ctx.save();
      ctx.beginPath();
      if (side < 0) { ctx.moveTo(0, 900); ctx.lineTo(0, 140); ctx.quadraticCurveTo(0, 0, 260, 0); ctx.lineTo(260, 900); }
      else { ctx.moveTo(0, 900); ctx.lineTo(0, 0); ctx.quadraticCurveTo(260, 0, 260, 140); ctx.lineTo(260, 900); }
      ctx.closePath();
      ctx.fillStyle = wood;
      ctx.fill();
      ctx.clip();
      ctx.strokeStyle = "rgba(255, 215, 120, 0.7)";
      ctx.lineWidth = 6;
      ctx.strokeRect(30, 220, 200, 280);
      ctx.strokeRect(30, 560, 200, 280);
      ctx.fillStyle = "rgba(255, 230, 160, 0.9)";
      for (let i = 0; i < 14; i++) star(ctx, 40 + ((i * 97) % 180), 240 + ((i * 151) % 600), 3 + (i % 3) * 2, 1.6);
      ctx.fillStyle = "#ffd36b";
      ctx.beginPath();
      ctx.arc(side < 0 ? 228 : 32, 520, 14, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
      p.tex.needsUpdate = true;
      const pivot = new THREE.Group(); // the hinge, at the leaf's outer edge
      pivot.position.x = side < 0 ? -W / 2 : W / 2;
      p.mesh.position.set(side < 0 ? W / 4 : -W / 4, H / 2, 0);
      p.mesh.name = "theatre-door-leaf";
      p.mesh.renderOrder = ORDER.panel - 0.02;
      p.mesh.userData.onPress = () => this.stepInside();
      pivot.add(p.mesh);
      group.add(pivot);
      return { pivot, mesh: p.mesh, side };
    };
    const leaves = [leaf(-1), leaf(1)];
    // the sign above
    const sign = panel(1.0, 0.2, 640, 128);
    {
      const { ctx } = sign;
      ctx.fillStyle = "rgba(9, 12, 24, 0.7)";
      rounded(ctx, 4, 4, 632, 120, 60);
      ctx.fill();
      ctx.fillStyle = GOLD;
      ctx.font = `700 60px ${FONT}`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("✨ Step inside", 320, 68);
      sign.tex.needsUpdate = true;
    }
    sign.mesh.name = "theatre-door-sign";
    sign.mesh.position.set(0, H + 0.42, 0.02);
    sign.mesh.userData.onPress = () => this.stepInside();
    group.add(sign.mesh);
    const pressable = [...leaves.map((l) => l.mesh), sign.mesh];
    this.#viewer.buttons.push(...pressable);
    return { group, leaves, light, sign, pressable, t: -1, from: null };
  }

  #doorTick(dt, t) {
    const d = this.#door;
    if (!d) {
      if (this.#veilOut > 0) { // the veil lifts on the first moment of the story
        this.#veilOut = Math.max(0, this.#veilOut - dt / 0.9);
        this.#veil(ease(this.#veilOut));
      }
      return;
    }
    if (d.t < 0) { // closed: light breathing at the seams, the sign bobbing
      d.light.mesh.material.opacity = 0.3 + 0.12 * Math.sin(t * 2.2);
      d.sign.mesh.position.y = 2.67 + 0.025 * Math.sin(t * 1.6);
      return;
    }
    d.t += dt;
    const open = ease(clamp01(d.t / 1.1));
    for (const l of d.leaves) l.pivot.rotation.y = -l.side * 1.85 * open;
    d.light.mesh.material.opacity = 0.35 + 0.65 * open;
    d.light.mesh.scale.setScalar(1 + 0.6 * open);
    const walk = ease(clamp01((d.t - 0.8) / 1.7)); // the view glides through; nobody walks
    this.#viewer.dolly.position.set(d.from.x, d.from.y, d.from.z - 3.3 * walk);
    const veil = clamp01((d.t - 1.9) / 0.5);
    this.#veil(veil);
    if (d.t >= 2.6 && !d.done) {
      d.done = true;
      this.#viewer.buttons = this.#viewer.buttons.filter((b) => !d.pressable.includes(b));
      d.group.removeFromParent();
      dispose(d.group);
      this.#door = null;
      this.#viewer.dolly.position.copy(d.from);
      this.#veilOut = 1;
      this.#enterStory();
    }
  }

  #veilOut = 0;
  /** A warm white veil over the eyes (child of the camera), for the moment of stepping through. */
  #veil(k) {
    if (!this.#veilMesh) {
      const m = new THREE.Mesh(new THREE.PlaneGeometry(4, 4), new THREE.MeshBasicMaterial({
        color: 0xfff6e0, transparent: true, opacity: 0, depthTest: false, depthWrite: false, toneMapped: false }));
      m.name = "theatre-veil";
      m.position.set(0, 0, -0.3);
      m.renderOrder = 999;
      this.#viewer.camera.add(m);
      this.#veilMesh = m;
    }
    this.#veilMesh.material.opacity = k;
    this.#veilMesh.visible = k > 0.001;
  }

  /** Highlight the word being spoken, and move on if a voice has stalled or never started. */
  #follow() {
    const cur = this.#current;
    if (!cur || cur.done) return;
    const text = cur.line.text, now = performance.now();
    let ch = -1;
    if (cur.audio) {
      const a = this.#player, ms = a.currentTime * 1000;
      if (cur.marks) {
        for (const m of cur.marks) { if (m.t <= ms + 30) ch = m.s; else break; }
      } else if (a.currentTime > 0) {
        const dur = Number.isFinite(a.duration) && a.duration > 0 ? a.duration : cur.line.duration;
        if (dur) ch = (a.currentTime / dur) * text.length;
      }
      if (this.#paused || a.ended || a.currentTime !== cur.heard) {
        cur.heard = a.currentTime;
        cur.stall = now;
      } else if (now - cur.stall > 6000) {
        cur.audio = false; // stuck (network): let the browser read it
        this.#silence();
        this.#speak(cur);
      }
    } else if (cur.speech?.started) {
      const ms = now - cur.speech.started;
      ch = cur.boundary >= 0 && !cur.speech.mute ? cur.boundary : (ms / cur.speech.est) * text.length;
      const speaking = !cur.speech.mute && window.speechSynthesis?.speaking;
      if (!this.#paused && ms > (cur.speech.mute ? cur.speech.est : speaking ? cur.speech.est * 2.5 + 3000 : cur.speech.est + 1500)) {
        this.#silence();
        cur.finish();
        return;
      }
    }
    let word = -1;
    if (ch >= 0) for (let i = 0; i < cur.words.length && cur.words[i].s <= ch; i++) word = i;
    if (word !== cur.word && this.#cap?.words === cur.words) {
      cur.word = this.#cap.word = word;
      this.#renderCaption();
    }
  }

  // ---------- panels ----------
  #setCaption(cap) {
    this.#cap = cap;
    this.#renderCaption();
  }

  #renderCaption() {
    this.#drawCaption();
    const c = this.#cap;
    this.#cb.onCaption?.(c && { key: c.key, speaker: c.speaker ?? null, words: c.words?.map((w) => w.w) ?? null,
      word: c.word ?? -1, title: c.title ?? null, sub: c.sub ?? null });
  }

  #drawCaption() {
    const p = this.#captionPanel, cap = this.#cap;
    if (!p) return;
    const { ctx, canvas: { width: W, height: H } } = p;
    ctx.clearRect(0, 0, W, H);
    p.mesh.visible = !!cap && this.#viewer.renderer.xr.isPresenting; // in VR only: a screen shows its own caption
    if (!cap) return;
    ctx.fillStyle = "rgba(9, 12, 24, 0.74)";
    rounded(ctx, 2, 2, W - 4, H - 4, 34);
    ctx.fill();
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    if (!cap.words) {
      if (cap.title) {
        ctx.fillStyle = INK;
        ctx.font = `700 46px ${FONT}`;
        ctx.fillText(cap.title, W / 2, cap.sub ? 84 : H / 2, W - 80);
      }
      if (cap.sub) {
        ctx.fillStyle = cap.title ? MUTED : INK;
        ctx.font = `italic 500 ${cap.title ? 32 : 40}px ${FONT}`;
        ctx.fillText(cap.sub, W / 2, cap.title ? 146 : H / 2, W - 80);
      }
    } else {
      ctx.textAlign = "left";
      ctx.fillStyle = ACCENT;
      ctx.font = `700 24px ${FONT}`;
      ctx.fillText((cap.speaker || "").toUpperCase(), 44, 36, W - 88);
      cap.layout ??= layoutWords(ctx, cap.words, W - 88, H - 82);
      const { size, lh, lines } = cap.layout;
      ctx.font = `600 ${size}px ${FONT}`;
      let y = 62 + (H - 72 - lines.length * lh) / 2 + lh / 2;
      for (const line of lines) {
        const x0 = (W - line.width) / 2;
        for (const it of line.items) {
          if (it.i === cap.word) {
            ctx.fillStyle = GOLD;
            rounded(ctx, x0 + it.x - 8, y - size * 0.62, it.w + 16, size * 1.24, 12);
            ctx.fill();
            ctx.fillStyle = "#1a1408";
          } else ctx.fillStyle = INK;
          ctx.fillText(cap.words[it.i].w, x0 + it.x, y);
        }
        y += lh;
      }
    }
    p.tex.needsUpdate = true;
  }

  #drawPage() {
    const p = this.#pagePanel;
    if (!p) return;
    const { ctx, canvas: { width: W, height: H }, s } = p;
    ctx.drawImage(p.img, 0, 0, W, H);
    ctx.lineJoin = "round";
    const now = this.#current?.line.i;
    for (const ln of this.#state?.lines ?? []) {
      if (!ln.box) continue;
      ctx.beginPath();
      ln.box.forEach(([x, y], i) => (i ? ctx.lineTo(x * s, y * s) : ctx.moveTo(x * s, y * s)));
      ctx.closePath();
      if (ln.i === now) {
        ctx.fillStyle = "rgba(255, 213, 74, 0.42)";
        ctx.fill();
        ctx.strokeStyle = "#ffbf1f";
        ctx.lineWidth = 3;
      } else {
        ctx.strokeStyle = "rgba(108, 156, 255, 0.5)";
        ctx.lineWidth = 1.5;
      }
      ctx.stroke();
    }
    ctx.strokeStyle = "rgba(255, 255, 255, 0.35)";
    ctx.lineWidth = 2;
    ctx.strokeRect(1, 1, W - 2, H - 2);
    p.tex.needsUpdate = true;
  }

  #drawProgress() {
    const p = this.#progressPanel;
    if (!p) return;
    const { ctx, canvas: { width: W, height: H } } = p, st = this.#state;
    p.shown = this.elapsed().toFixed(1);
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "rgba(9, 12, 24, 0.8)";
    rounded(ctx, 2, 2, W - 4, H - 4, 36);
    ctx.fill();
    ctx.textBaseline = "alphabetic";
    ctx.textAlign = "left";
    ctx.fillStyle = INK;
    ctx.font = `700 44px ${FONT}`;
    ctx.fillText(st?.title || "Bringing your page to life", 56, 92, W - 330);
    ctx.fillStyle = MUTED;
    ctx.font = `400 28px ${FONT}`;
    ctx.fillText(st?.status === "failed" ? "Something went wrong" : st?.stage || "Sending the page…", 56, 136, W - 330);
    ctx.textAlign = "right";
    ctx.fillStyle = GOLD;
    ctx.font = `700 52px ${FONT}`;
    ctx.fillText(`${p.shown} s`, W - 56, 96);
    if (st && ["clip", "all"].includes(st.play_when)) this.#drawBar(ctx, W, this.#percent(st));
    if (this.#filming(st) && st.steps?.find((x) => x.name === "plan")?.done) { this.#drawFilming(p, st); return; }
    (st?.steps ?? STEPS).forEach((s, i) => {
      const y = 214 + i * 72, cx = 80, cy = y - 11;
      ctx.lineWidth = 4;
      if (s.done) {
        ctx.fillStyle = OK;
        ctx.beginPath();
        ctx.arc(cx, cy, 19, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "#0d1a12";
        ctx.beginPath();
        ctx.moveTo(cx - 9, cy);
        ctx.lineTo(cx - 2, cy + 7);
        ctx.lineTo(cx + 10, cy - 7);
        ctx.stroke();
      } else {
        ctx.strokeStyle = "rgba(255, 255, 255, 0.25)";
        ctx.beginPath();
        ctx.arc(cx, cy, 17, 0, Math.PI * 2);
        ctx.stroke();
        if (st?.status === "running") {
          const a = performance.now() / 160;
          ctx.strokeStyle = GOLD;
          ctx.beginPath();
          ctx.arc(cx, cy, 17, a, a + 1.6);
          ctx.stroke();
        }
      }
      ctx.textAlign = "left";
      ctx.fillStyle = s.done ? INK : MUTED;
      ctx.font = `500 32px ${FONT}`;
      ctx.fillText(s.label, 122, y, W - 340);
      ctx.textAlign = "right";
      ctx.fillStyle = s.done ? OK : MUTED;
      ctx.font = `500 30px ${FONT}`;
      const note = s.done && s.seconds != null ? `${Number(s.seconds).toFixed(1)} s`
        : s.total ? `${s.done_count ?? 0} of ${s.total}` : "";
      ctx.fillText(note, W - 56, y);
    });
    p.tex.needsUpdate = true;
  }

  #drawBar(ctx, W, pct) {
    const x = 56, y = 152, w = W - 112, h = 14;
    ctx.fillStyle = "rgba(255,255,255,0.12)";
    rounded(ctx, x, y, w, h, 7);
    ctx.fill();
    ctx.fillStyle = pct >= 100 ? OK : GOLD;
    rounded(ctx, x, y, Math.max(h, (w * pct) / 100), h, 7);
    ctx.fill();
    ctx.textAlign = "right";
    ctx.fillStyle = pct >= 100 ? OK : GOLD;
    ctx.font = `700 26px ${FONT}`;
    ctx.fillText(pct >= 100 ? "100% · stepping in" : `${pct}%`, x + w, y - 12);
    ctx.textAlign = "left";
  }

  /** While the story waits for its film: every painted scene as a thumbnail, each marked once its clip is made. */
  #drawFilming(p, st) {
    const { ctx, canvas: { width: W, height: H } } = p, shots = st.shots ?? [];
    const clips = shots.map((s) => s.clip).filter((c) => c && c.status !== "off");
    const done = clips.filter((c) => FINAL_CLIP.includes(c.status)).length;
    ctx.textAlign = "left";
    ctx.fillStyle = INK;
    ctx.font = `600 34px ${FONT}`;
    ctx.fillText(`Filming your story · ${done} of ${clips.length || shots.length} scenes`, 56, 214);
    const n = Math.max(1, shots.length), gap = 18, cols = Math.min(n, 3), rows = Math.ceil(n / cols);
    const tw = (W - 112 - gap * (cols - 1)) / cols, th = Math.min(tw * 9 / 16, (H - 270 - gap * (rows - 1)) / rows);
    shots.forEach((s, i) => {
      const x = 56 + (i % cols) * (tw + gap), y = 240 + Math.floor(i / cols) * (th + gap);
      const img = s.ready && s.image ? this.#thumb(s.image) : null;
      ctx.save();
      rounded(ctx, x, y, tw, th, 14);
      ctx.clip();
      ctx.fillStyle = "rgba(255,255,255,0.08)";
      ctx.fillRect(x, y, tw, th);
      if (img?.complete && img.naturalWidth) {
        const sc = Math.max(tw / img.naturalWidth, th / img.naturalHeight);
        const dw = img.naturalWidth * sc, dh = img.naturalHeight * sc;
        ctx.drawImage(img, x + (tw - dw) / 2, y + (th - dh) / 2, dw, dh);
      }
      const busy = !this.#clipSettled(st, s);
      if (busy) { // still being made: dimmed, with a turning ring
        ctx.fillStyle = "rgba(5, 7, 15, 0.55)";
        ctx.fillRect(x, y, tw, th);
        const a = performance.now() / 300, r = Math.min(tw, th) * 0.14;
        ctx.lineWidth = 6;
        ctx.strokeStyle = "rgba(255,255,255,0.25)";
        ctx.beginPath();
        ctx.arc(x + tw / 2, y + th / 2 - 8, r, 0, Math.PI * 2);
        ctx.stroke();
        ctx.strokeStyle = GOLD;
        ctx.beginPath();
        ctx.arc(x + tw / 2, y + th / 2 - 8, r, a, a + 1.7);
        ctx.stroke();
      }
      ctx.restore();
      const st2 = s.clip?.status, label = !s.ready ? "painting…" : st2 === "ready" ? "filmed ✓"
        : st2 === "checking" ? "checking…" : st2 === "generating" ? "filming…" : st2 === "queued" ? "waiting…"
        : st2 === "rejected" ? "painting kept" : "painted";
      ctx.font = `600 22px ${FONT}`;
      const w = ctx.measureText(label).width + 22;
      ctx.fillStyle = st2 === "ready" ? OK : "rgba(9, 12, 24, 0.75)";
      rounded(ctx, x + 10, y + th - 40, w, 30, 15);
      ctx.fill();
      ctx.fillStyle = st2 === "ready" ? "#0d1a12" : INK;
      ctx.fillText(label, x + 21, y + th - 18);
    });
    p.tex.needsUpdate = true;
  }

  #thumbs = new Map();
  #thumb(name) {
    if (!this.#thumbs.has(name)) {
      const img = new Image();
      img.onload = () => this.#drawProgress();
      img.src = fileUrl(this.#id, name);
      this.#thumbs.set(name, img);
    }
    return this.#thumbs.get(name);
  }

  #removeProgress() {
    if (!this.#progressPanel) return;
    this.#progressPanel.mesh.removeFromParent();
    dispose(this.#progressPanel.mesh);
    this.#progressPanel = null;
  }

  #fail(message) {
    this.#removeProgress();
    this.#setCaption({ key: "fail", title: "Couldn't bring this page to life", sub: message || "Something went wrong." });
    this.#setPhase("failed");
  }

  #setPhase(phase) {
    this.#phase = phase;
    const c = this.#canvas;
    if (c?.clip) this.#syncClip(c, PLAYING.includes(phase), 0); // the film starts (or stops) with the story itself
    this.#cb.onPhase?.(phase);
  }
}

/** Wrap the words to the panel, shrinking the type until they fit. Positions are relative to each line's start. */
function layoutWords(ctx, words, maxW, maxH) {
  let out = null;
  for (const size of [52, 46, 40, 35, 30, 26]) {
    ctx.font = `600 ${size}px ${FONT}`;
    const space = ctx.measureText(" ").width, lh = size * 1.22;
    const lines = [];
    let line = { items: [], width: 0 };
    words.forEach((w, i) => {
      const ww = ctx.measureText(w.w).width;
      if (line.items.length && line.width + space + ww > maxW) {
        lines.push(line);
        line = { items: [], width: 0 };
      }
      const x = line.items.length ? line.width + space : 0;
      line.items.push({ i, x, w: ww });
      line.width = x + ww;
    });
    if (line.items.length) lines.push(line);
    out = { size, lh, lines };
    if (lines.length * lh <= maxH) break;
  }
  return out;
}

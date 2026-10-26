import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { VRButton } from "three/addons/webxr/VRButton.js";
import { XRControllerModelFactory } from "three/addons/webxr/XRControllerModelFactory.js";
import { XRHandModelFactory } from "three/addons/webxr/XRHandModelFactory.js";

const EYE = 1.6;
const MOVE_SPEED = 0.9; // m/s
const SNAP_TURN = Math.PI / 6;

export class Viewer {
  constructor(container) {
    this.container = container;
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.xr.enabled = true;
    this.renderer.xr.setReferenceSpaceType("local-floor");
    container.append(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0e0f12);
    this.camera = new THREE.PerspectiveCamera(60, 1, 0.03, 300);
    this.camera.rotation.order = "YXZ";
    this.camera.position.set(0, EYE, 0);
    this.dolly = new THREE.Group();
    this.dolly.add(this.camera);
    this.scene.add(this.dolly);
    this.world = new THREE.Group();
    this.world.position.y = EYE;
    this.scene.add(this.world);

    this.bounds = { radius: 0.5, forward: 0.8, back: 0.25 };
    this.gen = 0; // bumps on every load(); a load that finishes after a newer one started is thrown away
    this.keys = new Set();
    this.clock = new THREE.Clock();
    this.raycaster = new THREE.Raycaster();
    this.meshes = [];
    this.buttons = []; // in-world buttons (meshes with userData.onPress): pressed by ray + trigger, pinch, or click
    this.draggables = []; // handles (meshes with userData.dragTarget): grab with trigger or pinch, move, let go
    this.marker = new THREE.Mesh(
      new THREE.RingGeometry(0.12, 0.16, 32).rotateX(-Math.PI / 2),
      new THREE.MeshBasicMaterial({ color: 0x6c9cff, side: THREE.DoubleSide }));
    this.marker.visible = false;
    this.scene.add(this.marker);

    this.#setupXR();
    this.#setupDesktop();
    this.renderer.xr.addEventListener("sessionstart", () => this.onEnterVR?.());
    new ResizeObserver(() => this.#resize()).observe(container);
    this.#resize();
    this.renderer.setAnimationLoop(() => this.#tick());
  }

  vrButton() {
    return VRButton.createButton(this.renderer);
  }

  /** Remove the current scene; a scene still loading is dropped when it arrives. */
  clear() {
    this.gen++;
    this.#clear();
    this.mode = null;
  }

  /** An empty stage for content placed straight into this.scene (the story theatre): glide within `move` metres
   *  (thumbstick or W A S D, no teleporting), snap-turn in VR. At least `minHfov` degrees fit across a narrow
   *  (portrait) screen. */
  stage({ vfov = 70, minHfov = 0, move = 0 } = {}) {
    this.clear();
    this.mode = "stage";
    this.stageView = { vfov, minHfov, zoom: 1 };
    this.bounds = { radius: move, forward: move, back: move };
    this.reset();
  }

  #clear() {
    for (const child of [...this.world.children]) {
      this.world.remove(child);
      this.#dispose(child);
    }
    if (this.surround) {
      this.surround.parent?.remove(this.surround);
      this.#dispose(this.surround);
      this.surround = null;
    }
    this.meshes = [];
  }

  /** Free GPU memory: three.js keeps geometry buffers and textures until they are disposed. */
  #dispose(root) {
    root?.traverse((obj) => {
      obj.geometry?.dispose();
      for (const m of [obj.material].flat()) {
        m?.map?.dispose();
        m?.dispose();
      }
    });
  }

  /** Inside-out textured sphere from an equirectangular image, its centre straight ahead (-Z). */
  async #panoramaSphere(imageUrl, radius) {
    const tex = await new THREE.TextureLoader().loadAsync(imageUrl);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.anisotropy = this.renderer.capabilities.getMaxAnisotropy();
    const geo = new THREE.SphereGeometry(radius, 96, 64);
    geo.scale(-1, 1, 1); // view the texture from inside the sphere
    const sphere = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ map: tex, toneMapped: false }));
    sphere.rotation.y = -Math.PI / 2;
    return sphere;
  }

  /** Soft gradient sphere from the photo's own top/bottom colours, so there's no black void around the scene. */
  async #gradientSphere(imageUrl, radius) {
    let top = new THREE.Color(0x2a2d33), bottom = new THREE.Color(0x17181c);
    try {
      const img = await new Promise((res, rej) => {
        const im = new Image();
        im.onload = () => res(im);
        im.onerror = rej;
        im.src = imageUrl;
      });
      const c = document.createElement("canvas");
      c.width = c.height = 48;
      const ctx = c.getContext("2d");
      ctx.drawImage(img, 0, 0, 48, 48);
      const avg = (y0, y1) => {
        const d = ctx.getImageData(0, y0, 48, y1 - y0).data;
        let r = 0, g = 0, b = 0;
        for (let i = 0; i < d.length; i += 4) { r += d[i]; g += d[i + 1]; b += d[i + 2]; }
        const n = d.length / 4;
        return new THREE.Color().setRGB(r / n / 255, g / n / 255, b / n / 255, THREE.SRGBColorSpace);
      };
      top = avg(0, 4);
      bottom = avg(44, 48);
    } catch { /* keep neutral defaults */ }
    const mid = top.clone().lerp(bottom, 0.5).multiplyScalar(0.7);
    const geo = new THREE.SphereGeometry(radius, 48, 32);
    const pos = geo.attributes.position;
    const colors = new Float32Array(pos.count * 3);
    const c = new THREE.Color();
    for (let i = 0; i < pos.count; i++) {
      const t = pos.getY(i) / radius; // -1 bottom .. 1 top
      if (t >= 0) c.copy(mid).lerp(top, t); else c.copy(mid).lerp(bottom, -t);
      c.toArray(colors, i * 3);
    }
    geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    return new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.BackSide, toneMapped: false }));
  }

  /** Show a scene. Everything is fetched first and shown only if no newer load() started meanwhile (the old scene
   *  stays up until then). Returns false if this load was superseded. */
  async load(url, meta, imageUrl, panoramaUrl = null) {
    const gen = ++this.gen;
    if (meta.type === "360") {
      // Full 360 panorama: looked around by rotation only.
      const sphere = await this.#panoramaSphere(imageUrl, 50);
      if (gen !== this.gen) { this.#dispose(sphere); return false; }
      this.#clear();
      this.mode = "360";
      sphere.position.y = EYE;
      this.scene.add(sphere);
      this.surround = sphere;
      this.world.position.y = EYE;
      this.bounds = { radius: 0, forward: 0, back: 0 }; // a photo sphere has no depth to walk through
      this.imageVfov = 75;
      this.reset();
      return true;
    }
    const is3d = meta.type === "3d";
    const extent = is3d
      ? Math.max(...meta.bounds.max.map((v, i) => Math.abs(v - meta.bounds.min[i])))
      : meta.scale.far_m;
    const radius = Math.max(extent * 3, 40);
    // Story pages: the illustration's 3D layers stand inside its full 360 world. Other scenes get a soft gradient.
    const [gltf, surround] = await Promise.all([
      new GLTFLoader().loadAsync(url),
      panoramaUrl ? this.#panoramaSphere(panoramaUrl, radius) : this.#gradientSphere(imageUrl, radius),
    ]);
    if (gen !== this.gen) {
      this.#dispose(gltf.scene);
      this.#dispose(surround);
      return false;
    }
    this.#clear();
    this.mode = meta.type;
    gltf.scene.traverse((obj) => {
      if (obj.isPoints) {
        obj.material = new THREE.PointsMaterial({ size: 0.022, vertexColors: true, sizeAttenuation: true, toneMapped: false });
        this.meshes.push(obj);
        return;
      }
      if (!obj.isMesh) return;
      const old = obj.material;
      if (old.map) old.map.colorSpace = THREE.SRGBColorSpace;
      // Photos must look like photos: unlit, no tone mapping, alpha cut-outs for layers.
      obj.material = new THREE.MeshBasicMaterial({
        map: old.map, side: THREE.DoubleSide, alphaTest: old.alphaTest > 0 ? old.alphaTest : 0,
        transparent: false, toneMapped: false,
      });
      old.dispose(); // the texture lives on in the new material
      this.meshes.push(obj);
    });
    this.world.add(gltf.scene);

    if (is3d) {
      // Reconstructions are metric with the floor below the first camera: put that floor at y = 0.
      this.world.position.y = meta.orient.camera_height_m;
      const { min, max } = meta.bounds;
      this.bounds = { box: { minX: min[0], maxX: max[0], minZ: min[2], maxZ: max[2] } };
      this.imageVfov = 70;
    } else {
      this.world.position.y = EYE;
      const b = meta.bounds ?? {};
      const r = b.move_radius_m ?? 0.5;
      this.bounds = { radius: r, forward: b.forward_m ?? r * 1.4, back: r * 0.5 };
      const { w, h } = meta.image;
      const hfov = THREE.MathUtils.degToRad(meta.camera.hfov_deg);
      this.imageVfov = THREE.MathUtils.radToDeg(2 * Math.atan(Math.tan(hfov / 2) * (h / w)));
    }
    if (panoramaUrl) {
      // Fixed in the world, centred at eye level like the 3D art, so the painted copy of the art stays behind the
      // real layers: a snap turn turns the viewer (the dolly), not the world.
      surround.position.y = EYE;
      this.scene.add(surround);
    } else {
      this.dolly.add(surround); // a plain gradient can follow the viewer, so it always surrounds them
    }
    this.surround = surround;
    this.raycaster.params.Points.threshold = 0.04;
    this.reset();
    return true;
  }

  reset() {
    this.dolly.position.set(0, 0, 0);
    this.dolly.rotation.set(0, 0, 0);
    this.camera.rotation.set(0, 0, 0);
    if (this.mode === "360") this.imageVfov = 75;
    if (this.mode === "stage") this.stageView.zoom = 1;
    this.#resize();
  }

  // ---------- desktop / touch ----------
  #setupDesktop() {
    const el = this.renderer.domElement;
    el.tabIndex = 0;
    let drag = null;
    let moved = 0;
    el.addEventListener("pointerdown", (e) => { drag = { x: e.clientX, y: e.clientY }; moved = 0; el.setPointerCapture(e.pointerId); });
    el.addEventListener("pointerup", (e) => {
      if (drag && moved < 6) { // a click, not a drag: press an in-world button under the pointer
        const r = el.getBoundingClientRect();
        this.raycaster.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1,
          -((e.clientY - r.top) / r.height) * 2 + 1), this.camera);
        this.raycaster.intersectObjects(this.buttons.filter((b) => b.visible), false)[0]?.object.userData.onPress?.();
      }
      drag = null;
    });
    el.addEventListener("pointermove", (e) => {
      if (!drag || this.renderer.xr.isPresenting || this.pinch) return;
      moved += Math.abs(e.clientX - drag.x) + Math.abs(e.clientY - drag.y);
      const k = 0.0035;
      this.camera.rotation.y += (e.clientX - drag.x) * k;
      this.camera.rotation.x = THREE.MathUtils.clamp(this.camera.rotation.x + (e.clientY - drag.y) * k, -1.1, 1.1);
      drag = { x: e.clientX, y: e.clientY };
    });
    el.addEventListener("wheel", (e) => {
      e.preventDefault();
      if (this.mode === "stage") {
        this.stageView.zoom = THREE.MathUtils.clamp(this.stageView.zoom * (1 + Math.sign(e.deltaY) * 0.06), 0.5, 1.35);
        this.#resize();
        return;
      }
      if (this.mode === "360") {
        // Nothing to walk to inside a photo sphere: scroll zooms instead.
        this.imageVfov = THREE.MathUtils.clamp(this.imageVfov + Math.sign(e.deltaY) * 4, 35, 100);
        this.#resize();
        return;
      }
      this.#moveLocal(0, Math.sign(e.deltaY) * 0.08);
    }, { passive: false });
    el.addEventListener("dblclick", (e) => {
      const r = el.getBoundingClientRect();
      const ndc = new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
      this.raycaster.setFromCamera(ndc, this.camera);
      const hit = this.raycaster.intersectObjects(this.meshes, false)[0];
      if (hit) this.#teleportTo(hit.point);
    });
    el.addEventListener("touchstart", (e) => { if (e.touches.length === 2) this.pinch = this.#touchDist(e); }, { passive: true });
    el.addEventListener("touchmove", (e) => {
      if (e.touches.length !== 2 || !this.pinch) return;
      const d = this.#touchDist(e);
      this.#moveLocal(0, (this.pinch - d) * 0.004);
      this.pinch = d;
    }, { passive: true });
    el.addEventListener("touchend", () => { this.pinch = null; });
    el.addEventListener("keydown", (e) => this.keys.add(e.key.toLowerCase()));
    el.addEventListener("keyup", (e) => this.keys.delete(e.key.toLowerCase()));
    el.addEventListener("blur", () => this.keys.clear());
  }

  #touchDist(e) {
    const [a, b] = e.touches;
    return Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
  }

  // ---------- XR: controllers and hands ----------
  #setupXR() {
    const controllerModels = new XRControllerModelFactory();
    const handModels = new XRHandModelFactory();
    const rayGeom = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0, -1)]);
    this.controllers = [0, 1].map((i) => {
      const ray = this.renderer.xr.getController(i);
      const line = new THREE.Line(rayGeom, new THREE.LineBasicMaterial({ color: 0x6c9cff }));
      line.scale.z = 4;
      ray.add(line);
      ray.userData = { aiming: false, source: null, turnReady: true };
      ray.addEventListener("connected", (e) => { ray.userData.source = e.data; });
      ray.addEventListener("disconnected", () => { ray.userData.source = null; });
      // Trigger on a controller and pinch on a tracked hand both arrive as "select".
      ray.addEventListener("selectstart", () => {
        if (!ray.userData.hover && this.#grab(ray)) return; // pointing at a handle: carry it instead
        ray.userData.aiming = true;
      });
      ray.addEventListener("selectend", () => {
        if (ray.userData.drag) { ray.userData.drag = null; return; }
        ray.userData.aiming = false;
        const pressed = ray.userData.hover;
        this.#hover(ray, null);
        if (pressed) pressed.userData.onPress?.();
        else if (this.marker.visible) this.#teleportTo(this.marker.position);
        this.marker.visible = false;
      });
      const grip = this.renderer.xr.getControllerGrip(i);
      grip.add(controllerModels.createControllerModel(grip));
      const hand = this.renderer.xr.getHand(i);
      hand.add(handModels.createHandModel(hand, "mesh"));
      this.dolly.add(ray, grip, hand);
      return ray;
    });
  }

  /** Start carrying the handle this controller points at (keeps its distance along the ray). */
  #grab(ray) {
    const live = this.draggables.filter((d) => d.visible && d.parent?.visible !== false);
    if (!live.length) return false;
    this.#setRay(ray);
    const hit = this.raycaster.intersectObjects(live, false)[0];
    if (!hit) return false;
    ray.userData.drag = { target: hit.object.userData.dragTarget ?? hit.object, dist: Math.max(0.3, hit.distance) };
    return true;
  }

  #setRay(ray) {
    const m = new THREE.Matrix4().extractRotation(ray.matrixWorld);
    this.raycaster.ray.origin.setFromMatrixPosition(ray.matrixWorld);
    this.raycaster.ray.direction.set(0, 0, -1).applyMatrix4(m);
  }

  /** Move a carried handle with the controller, turned to face the viewer's eyes. */
  #carry(ray) {
    const { target, dist } = ray.userData.drag;
    if (!target.parent) { ray.userData.drag = null; return; }
    this.#setRay(ray);
    const p = this.raycaster.ray.at(Math.min(2.5, dist), new THREE.Vector3());
    target.parent.worldToLocal(p);
    target.position.copy(p);
    const eye = new THREE.Vector3().setFromMatrixPosition(this.camera.matrixWorld);
    target.lookAt(eye);
  }

  #xrInput(dt) {
    for (const ray of this.controllers) {
      const src = ray.userData.source;
      if (ray.userData.drag) { this.#carry(ray); continue; }
      this.#pointAtButtons(ray);
      if (ray.userData.aiming && !ray.userData.hover) this.#aim(ray);
      // xr-standard mapping: button 4 = A/X, button 5 = B/Y. Fire once per press.
      const buttons = src?.gamepad?.buttons ?? [];
      for (const [i, cb] of [[4, this.onPrimary], [5, this.onSecondary]]) {
        const down = !!buttons[i]?.pressed;
        if (down && !ray.userData[`b${i}`]) cb?.();
        ray.userData[`b${i}`] = down;
      }
      const axes = src?.gamepad?.axes;
      if (!axes || axes.length < 4) continue;
      const [x, y] = [axes[2], axes[3]];
      if (src.handedness === "left") {
        if (Math.hypot(x, y) > 0.15) this.#moveLocal(x * MOVE_SPEED * dt, y * MOVE_SPEED * dt, true);
      } else if (Math.abs(x) > 0.7 && ray.userData.turnReady) {
        this.#snapTurn(-Math.sign(x) * SNAP_TURN);
        ray.userData.turnReady = false;
      } else if (Math.abs(x) < 0.3) {
        ray.userData.turnReady = true;
      }
    }
  }

  /** Highlight the in-world button a controller (or hand) points at; selectend presses it. */
  #pointAtButtons(ray) {
    const live = this.buttons.filter((b) => b.visible);
    if (!live.length && !this.draggables.length) { this.#hover(ray, null); return; }
    const m = new THREE.Matrix4().extractRotation(ray.matrixWorld);
    this.raycaster.ray.origin.setFromMatrixPosition(ray.matrixWorld);
    this.raycaster.ray.direction.set(0, 0, -1).applyMatrix4(m);
    const button = this.raycaster.intersectObjects(live, false)[0]?.object ?? null;
    this.#hover(ray, button);
    // A handle under the ray (and no button) glows a little, so it is clear it can be grabbed.
    const handles = button ? [] : this.draggables.filter((d) => d.visible && d.parent?.visible !== false);
    const handle = handles.length ? this.raycaster.intersectObjects(handles, false)[0]?.object ?? null : null;
    if (ray.userData.handle !== handle) {
      if (ray.userData.handle?.material) ray.userData.handle.material.opacity = 1;
      if (handle?.material) handle.material.opacity = 0.75;
      ray.userData.handle = handle;
    }
  }

  #hover(ray, button) {
    const was = ray.userData.hover;
    if (was === button) return;
    if (was) was.scale.setScalar(1);
    if (button) button.scale.setScalar(1.12);
    ray.userData.hover = button;
  }

  #aim(ray) {
    const m = new THREE.Matrix4().extractRotation(ray.matrixWorld);
    this.raycaster.ray.origin.setFromMatrixPosition(ray.matrixWorld);
    this.raycaster.ray.direction.set(0, 0, -1).applyMatrix4(m);
    const hit = this.raycaster.intersectObjects(this.meshes, false)[0];
    this.marker.visible = !!hit;
    if (hit) this.marker.position.copy(hit.point);
  }

  #snapTurn(angle) {
    const head = new THREE.Vector3();
    this.camera.getWorldPosition(head);
    this.dolly.position.sub(head).applyAxisAngle(new THREE.Vector3(0, 1, 0), angle).add(head);
    this.dolly.rotation.y += angle;
  }

  // ---------- movement ----------
  #moveLocal(dx, dz, fromHead = false) {
    const yaw = fromHead ? this.#headYaw() : this.camera.rotation.y + this.dolly.rotation.y;
    const v = new THREE.Vector3(dx, 0, dz).applyAxisAngle(new THREE.Vector3(0, 1, 0), yaw);
    this.dolly.position.add(v);
    this.#clamp();
  }

  #headYaw() {
    const q = new THREE.Quaternion();
    this.camera.getWorldQuaternion(q);
    return new THREE.Euler().setFromQuaternion(q, "YXZ").y;
  }

  #teleportTo(point) {
    const head = new THREE.Vector3();
    this.camera.getWorldPosition(head);
    this.dolly.position.x += point.x - head.x;
    this.dolly.position.z += point.z - head.z;
    this.#clamp();
  }

  #clamp() {
    const p = this.dolly.position;
    const box = this.bounds.box;
    if (box) {
      // Real 3D: roam anywhere inside the reconstructed space.
      p.x = THREE.MathUtils.clamp(p.x, box.minX, box.maxX);
      p.z = THREE.MathUtils.clamp(p.z, box.minZ, box.maxZ);
      return;
    }
    // 2.5D scenes only hold up near the original viewpoint; keep the viewer inside the parallax-safe zone.
    p.x = THREE.MathUtils.clamp(p.x, -this.bounds.radius, this.bounds.radius);
    p.z = THREE.MathUtils.clamp(p.z, -this.bounds.forward, this.bounds.back);
  }

  #resize() {
    const { clientWidth: w, clientHeight: h } = this.container;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.fov = this.mode === "stage" ? this.#stageFov() : this.imageVfov ?? 60;
    this.camera.updateProjectionMatrix();
  }

  #stageFov() {
    const { vfov, minHfov, zoom } = this.stageView, deg = THREE.MathUtils;
    const fit = deg.radToDeg(2 * Math.atan(Math.tan(deg.degToRad(minHfov) / 2) / this.camera.aspect));
    return deg.clamp(Math.max(vfov, fit) * zoom, 25, 110);
  }

  #tick() {
    const dt = Math.min(this.clock.getDelta(), 0.05);
    if (this.renderer.xr.isPresenting) {
      this.#xrInput(dt);
    } else {
      const fwd = (this.keys.has("w") || this.keys.has("arrowup") ? -1 : 0) + (this.keys.has("s") || this.keys.has("arrowdown") ? 1 : 0);
      const side = (this.keys.has("d") || this.keys.has("arrowright") ? 1 : 0) + (this.keys.has("a") || this.keys.has("arrowleft") ? -1 : 0);
      if (fwd || side) this.#moveLocal(side * MOVE_SPEED * dt, fwd * MOVE_SPEED * dt);
    }
    this.onTick?.(dt);
    this.renderer.render(this.scene, this.camera);
  }
}

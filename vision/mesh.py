"""Depth-map grid meshes and GLB export (camera at the origin looking down -Z, Y up, meters)."""
import io

import numpy as np
import trimesh
from PIL import Image


def grid_mesh(depth_m: np.ndarray, f: float, cx: float, cy: float, box: tuple, step: int,
              mask: np.ndarray | None = None, cut_ratio: float | None = None, any_corner: bool = False):
    """Back-project a pixel grid inside `box` (x0, y0, x1, y1) to 3D.

    Returns vertices (N,3), faces (M,3) and uv (N,2) relative to the box crop. Triangles are dropped
    outside `mask` (all corners inside, or any corner with `any_corner`, leaving the exact outline to an
    alpha texture), or where corner depths differ by more than `cut_ratio` (rubber-sheet stretches)."""
    x0, y0, x1, y1 = box
    us = np.arange(x0, x1, step, dtype=np.int32)
    vs = np.arange(y0, y1, step, dtype=np.int32)
    if us[-1] != x1 - 1:
        us = np.append(us, x1 - 1)
    if vs[-1] != y1 - 1:
        vs = np.append(vs, y1 - 1)
    uu, vv = np.meshgrid(us, vs)
    z = depth_m[vv, uu]
    verts = np.stack([(uu - cx) * z / f, -(vv - cy) * z / f, -z], axis=-1).reshape(-1, 3)
    uv = np.stack([(uu - x0) / max(x1 - x0 - 1, 1), 1.0 - (vv - y0) / max(y1 - y0 - 1, 1)], axis=-1).reshape(-1, 2)

    rows, cols = uu.shape
    idx = np.arange(rows * cols).reshape(rows, cols)
    a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    c, d = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    tris = np.concatenate([np.stack([a, c, b], 1), np.stack([b, c, d], 1)])

    keep = np.ones(len(tris), bool)
    zf = z.ravel()
    if mask is not None:
        m = mask[vv, uu].ravel()
        keep &= m[tris].any(axis=1) if any_corner else m[tris].all(axis=1)
    if cut_ratio is not None:
        zt = zf[tris]
        keep &= zt.max(axis=1) / np.maximum(zt.min(axis=1), 1e-6) < cut_ratio
    return verts.astype(np.float32), tris[keep].astype(np.uint32), uv.astype(np.float32)


def textured(verts, faces, uv, rgba: np.ndarray, name: str) -> trimesh.Trimesh:
    """rgba: HxWx4 uint8 (RGB order). Alpha < 128 is cut out in the viewer (alphaMode MASK)."""
    has_alpha = rgba.shape[2] == 4 and (rgba[..., 3] < 255).any()
    image = Image.fromarray(rgba if has_alpha else rgba[..., :3])
    material = trimesh.visual.material.PBRMaterial(
        name=name, baseColorTexture=image, metallicFactor=0.0, roughnessFactor=1.0, doubleSided=True,
        alphaMode="MASK" if has_alpha else "OPAQUE", alphaCutoff=0.5 if has_alpha else None)
    visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    return trimesh.Trimesh(vertices=verts, faces=faces, visual=visual, process=False)


def to_glb(meshes: dict[str, trimesh.Trimesh]) -> bytes:
    scene = trimesh.Scene()
    for name, m in meshes.items():
        scene.add_geometry(m, node_name=name, geom_name=name)
    buf = io.BytesIO()
    scene.export(buf, file_type="glb")
    return buf.getvalue()


def jpeg_bytes(rgb: np.ndarray, quality: int = 88) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
